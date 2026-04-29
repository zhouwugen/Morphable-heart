from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import CardiacCTDataset, collate_fn
from models import CardiacHMR
from utils import mask_loss_bce, pca_prior_loss, vertex_l2_loss


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    pattern = re.compile(r"epoch(\d+)\.pth$")
    candidates: list[tuple[int, Path]] = []
    for path in ckpt_dir.glob("cardiac_hmr_epoch*.pth"):
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
        vol = batch["volume"].to(device)
        mesh_gt = batch["mesh_gt"].to(device)
        mask_gt = batch["mask"].to(device)

        out = model(vol)
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
    default_data_dir = repo_root / "runs" / "dheart_reg_rebuild" / "synthetic"
    default_pca_dir = repo_root / "PCA" / "pca_results"
    default_ckpt_dir = repo_root / "runs" / "dheart_reg_rebuild" / "pretrain_ckpts"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--pca-dir", type=Path, default=default_pca_dir)
    parser.add_argument("--save-dir", type=Path, default=default_ckpt_dir)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on synthetic cases for smoke tests.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.save_dir.mkdir(parents=True, exist_ok=True)

    mean = np.load(args.pca_dir / "mean_shape.npy")
    basis = np.load(args.pca_dir / "pca_components.npy")
    faces = np.load(args.pca_dir / "template_faces.npy")

    ids = sorted(path.name.replace("_mask.npy", "") for path in args.data_dir.glob("*_mask.npy"))
    if not ids:
        raise RuntimeError(f"No synthetic samples found in {args.data_dir}")
    if args.limit > 0:
        ids = ids[: args.limit]
        print(f"Using first {len(ids)} synthetic cases for this run")

    ds = CardiacCTDataset(ids, str(args.data_dir))
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    model = CardiacHMR(mean, basis, faces).to(device)
    opt = optim.Adam(model.parameters(), args.lr)

    start_epoch = 0
    latest = find_latest_checkpoint(args.save_dir)
    if latest is not None:
        print(f"Resuming from {latest}")
        checkpoint = torch.load(latest, map_location=device)
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1

    best_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        loss = train_one_epoch(model, dl, opt, device)
        print(f"[Pretrain] Epoch {epoch}: loss={loss:.6f}", flush=True)
        ckpt_path = args.save_dir / f"cardiac_hmr_epoch{epoch}.pth"
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict()}, ckpt_path)
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict()}, args.save_dir / "last.pth")
        if loss < best_loss:
            best_loss = loss
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict()}, args.save_dir / "best.pth")
            print(f"[Pretrain] Updated best checkpoint at epoch {epoch}", flush=True)


if __name__ == "__main__":
    main()
