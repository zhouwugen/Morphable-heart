from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import trimesh
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import CardiacHMR
from utils import mask_loss_bce, pca_prior_loss, vertex_l2_loss


def resize_volume(vol: np.ndarray, target_shape: tuple[int, int, int], mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(tensor, size=target_shape, mode=mode, align_corners=False if mode != "nearest" else None).squeeze(0)


class CASMeshDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        seg_dir: Path,
        mesh_dir: Path,
        target_shape: tuple[int, int, int] = (128, 128, 128),
        limit: int = 0,
    ):
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.mesh_dir = mesh_dir
        self.target_shape = target_shape
        self.case_ids = sorted(path.name.replace("_seg.nii.gz", "") for path in seg_dir.glob("CAS_*_seg.nii.gz"))
        if limit > 0:
            self.case_ids = self.case_ids[:limit]

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        case_id = self.case_ids[idx]
        image = nib.load(self.image_dir / f"{case_id}_image.nii.gz").get_fdata().astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        mask = nib.load(self.seg_dir / f"{case_id}_seg.nii.gz").get_fdata().astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        mesh = trimesh.load_mesh(self.mesh_dir / f"{case_id}.obj", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        verts = np.asarray(mesh.vertices, dtype=np.float32)

        image_t = resize_volume(image, self.target_shape, mode="trilinear")
        mask_t = resize_volume(mask, self.target_shape, mode="nearest")

        return {
            "case_id": case_id,
            "volume": image_t,
            "mask": mask_t.clamp(0.0, 1.0),
            "mesh_gt": torch.from_numpy(verts).float(),
        }


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    pattern = re.compile(r"epoch(\d+)\.pth$")
    candidates: list[tuple[int, Path]] = []
    for path in ckpt_dir.glob("finetune_epoch*.pth"):
        match = pattern.search(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def train_one_epoch(model: CardiacHMR, loader: DataLoader, opt: optim.Optimizer, device: str) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        volume = batch["volume"].to(device)
        mask_gt = batch["mask"].to(device)
        mesh_gt = batch["mesh_gt"].to(device)

        out = model(volume)
        loss = 1.0 * vertex_l2_loss(out["verts"], mesh_gt)
        loss = loss + 1.0 * mask_loss_bce(out["mask"], mask_gt)
        loss = loss + 0.01 * pca_prior_loss(out["coeff"])

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cta_root = repo_root.parent / "CTA1100"
    default_pca_dir = repo_root / "PCA" / "pca_results"
    default_pretrain = repo_root / "runs" / "dheart_reg_rebuild" / "pretrain_ckpts" / "best.pth"
    default_save_dir = repo_root / "runs" / "dheart_reg_rebuild" / "finetune_ckpts"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=cta_root / "img_1000")
    parser.add_argument("--seg-dir", type=Path, default=cta_root / "seg_1000")
    parser.add_argument("--mesh-dir", type=Path, default=cta_root / "mesh_1000")
    parser.add_argument("--pca-dir", type=Path, default=default_pca_dir)
    parser.add_argument("--pretrain-ckpt", type=Path, default=default_pretrain)
    parser.add_argument("--save-dir", type=Path, default=default_save_dir)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on CAS cases for smoke tests.")
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean = np.load(args.pca_dir / "mean_shape.npy")
    basis = np.load(args.pca_dir / "pca_components.npy")
    faces = np.load(args.pca_dir / "template_faces.npy")

    dataset = CASMeshDataset(args.image_dir, args.seg_dir, args.mesh_dir, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = CardiacHMR(mean, basis, faces, latent_dim=256, mask_out_shape=(128, 128, 128)).to(device)
    opt = optim.Adam(model.parameters(), args.lr)

    if args.pretrain_ckpt.exists():
        print(f"Loading pretrain checkpoint {args.pretrain_ckpt}")
        checkpoint = torch.load(args.pretrain_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        print(f"Warning: pretrain checkpoint missing at {args.pretrain_ckpt}; starting from scratch.")

    start_epoch = 0
    latest = find_latest_checkpoint(args.save_dir)
    if latest is not None:
        print(f"Resuming finetune from {latest}")
        checkpoint = torch.load(latest, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)
        opt.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1

    best_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        loss = train_one_epoch(model, loader, opt, device)
        print(f"[Finetune] Epoch {epoch}: loss={loss:.6f}", flush=True)
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict()}
        torch.save(ckpt, args.save_dir / f"finetune_epoch{epoch}.pth")
        torch.save(ckpt, args.save_dir / "last.pth")
        if loss < best_loss:
            best_loss = loss
            torch.save(ckpt, args.save_dir / "best.pth")
            print(f"[Finetune] Updated best checkpoint at epoch {epoch}", flush=True)


if __name__ == "__main__":
    main()
