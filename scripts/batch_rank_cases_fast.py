from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import CardiacHMR


def dice_3d(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float(2.0 * inter / (pred.sum() + gt.sum() + 1e-8))


def iou_3d(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / (union + 1e-8))


def hd95(pred: np.ndarray, gt: np.ndarray, sample_size: int = 5000) -> float:
    p = np.argwhere(pred > 0)
    g = np.argwhere(gt > 0)
    if len(p) == 0 or len(g) == 0:
        return float("nan")
    rng = np.random.default_rng(2026)
    if len(p) > sample_size:
        p = p[rng.choice(len(p), sample_size, replace=False)]
    if len(g) > sample_size:
        g = g[rng.choice(len(g), sample_size, replace=False)]
    tp, tg = cKDTree(p), cKDTree(g)
    d1, _ = tp.query(g)
    d2, _ = tg.query(p)
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def cohort_for_case(case_id: str) -> str:
    if case_id.startswith("CAS_"):
        return "CAS"
    if case_id.startswith("ct_"):
        return "MMWHS"
    return "WHS++"


def resize_volume(vol: np.ndarray, target_shape: tuple[int, int, int], mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(tensor, size=target_shape, mode=mode, align_corners=False if mode != "nearest" else None).squeeze(0)


class CaseDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        seg_dir: Path,
        seg_pattern: str,
        case_id_suffix: str,
        target_shape: tuple[int, int, int] = (128, 128, 128),
        limit: int = 0,
    ):
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.seg_pattern = seg_pattern
        self.case_id_suffix = case_id_suffix
        self.target_shape = target_shape
        self.seg_paths = sorted(seg_dir.glob(seg_pattern))
        if limit > 0:
            self.seg_paths = self.seg_paths[:limit]

    def __len__(self) -> int:
        return len(self.seg_paths)

    def __getitem__(self, idx: int) -> dict[str, object]:
        seg_path = self.seg_paths[idx]
        case_id = seg_path.name.replace(self.case_id_suffix, "")
        image_path = self.image_dir / f"{case_id}_image.nii.gz"
        image = nib.load(image_path).get_fdata().astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        seg = nib.load(seg_path).get_fdata().astype(np.float32)
        gt_binary = (seg > 0).astype(np.float32)

        image_t = resize_volume(image, self.target_shape, mode="trilinear")
        gt_t = resize_volume(gt_binary, self.target_shape, mode="nearest")
        return {
            "case_id": case_id,
            "volume": image_t,
            "gt_mask": gt_t.clamp(0.0, 1.0),
        }


def collate(batch: list[dict[str, object]]) -> dict[str, object]:
    case_ids = [item["case_id"] for item in batch]
    volumes = torch.stack([item["volume"] for item in batch], dim=0)
    gt_masks = torch.stack([item["gt_mask"] for item in batch], dim=0)
    return {"case_ids": case_ids, "volume": volumes, "gt_mask": gt_masks}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_pca_dir = repo_root / "PCA" / "pca_results"
    default_ckpt = repo_root / "runs" / "dheart_reg_rebuild" / "finetune_ckpts" / "best.pth"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--seg-dir", type=Path, required=True)
    parser.add_argument("--seg-pattern", type=str, required=True)
    parser.add_argument("--case-id-suffix", type=str, required=True)
    parser.add_argument("--pca-dir", type=Path, default=default_pca_dir)
    parser.add_argument("--ckpt", type=Path, default=default_ckpt)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean = np.load(args.pca_dir / "mean_shape.npy")
    basis = np.load(args.pca_dir / "pca_components.npy")
    faces = np.load(args.pca_dir / "template_faces.npy")
    ckpt = torch.load(args.ckpt, map_location=device)

    model = CardiacHMR(mean, basis, faces, latent_dim=256, mask_out_shape=(128, 128, 128)).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    ds = CaseDataset(
        image_dir=args.image_dir,
        seg_dir=args.seg_dir,
        seg_pattern=args.seg_pattern,
        case_id_suffix=args.case_id_suffix,
        limit=args.limit,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    rows: list[dict[str, object]] = []
    processed = 0
    total = len(ds)

    with torch.no_grad():
        for batch in dl:
            volume = batch["volume"].to(device)
            gt_masks = batch["gt_mask"].cpu().numpy()[:, 0]
            out = model(volume)
            # Auxiliary mask-head output for quick sanity/ranking only. The
            # reported D-Heart-Reg objective does not use mask supervision.
            pred_prob = out["mask"][:, 0].detach().cpu().numpy()
            pred_masks = (pred_prob >= args.mask_threshold).astype(np.uint8)
            for case_id, pred_mask, gt_mask in zip(batch["case_ids"], pred_masks, gt_masks, strict=True):
                case_dice = dice_3d(pred_mask, gt_mask > 0.5)
                case_iou = iou_3d(pred_mask, gt_mask > 0.5)
                case_hd95 = hd95(pred_mask, gt_mask > 0.5)
                failure_score = float((1.0 - case_dice) * 100.0 + (0.0 if np.isnan(case_hd95) else case_hd95))
                rows.append(
                    {
                        "case_id": case_id,
                        "cohort": cohort_for_case(case_id),
                        "dice": case_dice,
                        "iou": case_iou,
                        "hd95": case_hd95,
                        "failure_score": failure_score,
                    }
                )
                processed += 1
                if processed % 25 == 0 or processed == total:
                    print(
                        f"[{processed}/{total}] latest={case_id} "
                        f"dice={case_dice:.4f} hd95={case_hd95:.3f} failure={failure_score:.3f}",
                        flush=True,
                    )

    rows.sort(key=lambda row: row["failure_score"], reverse=True)
    csv_path = args.out_dir / "external_case_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "cohort", "dice", "iou", "hd95", "failure_score"])
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "top20_failure_cases.json").write_text(json.dumps(rows[:20], indent=2))
    print(f"Saved metrics to {csv_path}")
    print("Top 10 current failure candidates:")
    for row in rows[:10]:
        print(row)


if __name__ == "__main__":
    main()
