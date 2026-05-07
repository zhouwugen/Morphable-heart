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
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

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


def resize_volume(vol: np.ndarray, target_shape: tuple[int, int, int], mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(tensor, size=target_shape, mode=mode, align_corners=False if mode != "nearest" else None).squeeze(0)


def cohort_for_case(case_id: str) -> str:
    if case_id.startswith("CAS_"):
        return "CAS"
    if case_id.startswith("ct_"):
        return "MMWHS"
    return "WHS++"


def save_mid_slice_png(image_3d: np.ndarray, out_path: Path) -> None:
    z = image_3d.shape[0] // 2
    arr = image_3d[z]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    img = Image.fromarray((arr * 255.0).astype(np.uint8))
    img.save(out_path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_pca_dir = repo_root / "PCA" / "pca_results"
    default_ckpt = repo_root / "runs" / "dheart_reg_rebuild" / "finetune_ckpts" / "best.pth"
    default_out_dir = repo_root / "runs" / "dheart_reg_rebuild" / "external_predictions"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing external CTA images, e.g. <case_id>_image.nii.gz.",
    )
    parser.add_argument(
        "--seg-dir",
        type=Path,
        required=True,
        help="Directory containing external four-chamber labels for evaluation/ranking.",
    )
    parser.add_argument("--pca-dir", type=Path, default=default_pca_dir)
    parser.add_argument("--ckpt", type=Path, default=default_ckpt)
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on external cases for smoke tests.")
    parser.add_argument("--seg-pattern", type=str, default="*_label.nii.gz")
    parser.add_argument("--case-id-suffix", type=str, default="_label.nii.gz")
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.metrics_only:
        (args.out_dir / "objs").mkdir(exist_ok=True)
        (args.out_dir / "npy").mkdir(exist_ok=True)
        (args.out_dir / "slices").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean = np.load(args.pca_dir / "mean_shape.npy")
    basis = np.load(args.pca_dir / "pca_components.npy")
    faces = np.load(args.pca_dir / "template_faces.npy")
    template_labels = np.load(args.pca_dir / "template_labels.npy").astype(np.int32)
    ckpt = torch.load(args.ckpt, map_location=device)

    model = CardiacHMR(mean, basis, faces, latent_dim=256, mask_out_shape=(128, 128, 128)).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    color_map = {
        1: np.array([223, 122, 94], dtype=np.uint8),
        2: np.array([244, 241, 222], dtype=np.uint8),
        4: np.array([242, 204, 142], dtype=np.uint8),
        5: np.array([130, 178, 154], dtype=np.uint8),
    }
    vertex_colors = np.zeros((template_labels.shape[0], 4), dtype=np.uint8)
    for label, color in color_map.items():
        vertex_colors[template_labels == label, :3] = color
    vertex_colors[:, 3] = 255

    rows: list[dict[str, object]] = []
    seg_paths = sorted(args.seg_dir.glob(args.seg_pattern))
    if not seg_paths:
        raise RuntimeError(f"No external labels found in {args.seg_dir}")
    if args.case_ids:
        allowed = set(args.case_ids)
        seg_paths = [path for path in seg_paths if path.name.replace(args.case_id_suffix, "") in allowed]
        print(f"Selected {len(seg_paths)} requested cases")
    if args.limit > 0:
        seg_paths = seg_paths[: args.limit]
        print(f"Using first {len(seg_paths)} external labeled cases for this run")

    for idx, seg_path in enumerate(seg_paths, start=1):
        case_id = seg_path.name.replace(args.case_id_suffix, "")
        image_path = args.image_dir / f"{case_id}_image.nii.gz"
        if not image_path.exists():
            print(f"Skipping {case_id}: missing image {image_path}")
            continue

        image = nib.load(image_path).get_fdata().astype(np.float32)
        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
        gt = nib.load(seg_path).get_fdata().astype(np.float32)
        gt_binary = (gt > 0).astype(np.float32)

        volume = resize_volume(image_norm, (128, 128, 128), mode="trilinear").unsqueeze(0).to(device)
        gt_resized = resize_volume(gt_binary, (128, 128, 128), mode="nearest").cpu().numpy()[0]

        with torch.no_grad():
            out = model(volume)
        # Auxiliary mask-head output for quick sanity/ranking only. The reported
        # D-Heart-Reg reconstruction objective does not use mask supervision.
        pred_mask_prob = out["mask"][0, 0].detach().cpu().numpy()
        pred_mask = (pred_mask_prob >= args.mask_threshold).astype(np.uint8)
        pred_verts = out["verts"][0].detach().cpu().numpy()

        if not args.metrics_only:
            mesh = trimesh.Trimesh(vertices=pred_verts, faces=faces, process=False)
            mesh.visual.vertex_colors = vertex_colors
            mesh.export(args.out_dir / "objs" / f"{case_id}_pred_colored.obj")
            np.save(args.out_dir / "npy" / f"{case_id}_pred_mask.npy", pred_mask.astype(np.uint8))
            np.save(args.out_dir / "npy" / f"{case_id}_pred_mask_prob.npy", pred_mask_prob.astype(np.float32))
            np.save(args.out_dir / "npy" / f"{case_id}_gt_mask.npy", gt_resized.astype(np.uint8))
            np.save(args.out_dir / "npy" / f"{case_id}_error_mask.npy", np.abs(pred_mask.astype(np.int16) - (gt_resized > 0.5).astype(np.int16)).astype(np.uint8))
            save_mid_slice_png(image, args.out_dir / "slices" / f"{case_id}_cta_mid.png")

        case_dice = dice_3d(pred_mask, gt_resized > 0.5)
        case_iou = iou_3d(pred_mask, gt_resized > 0.5)
        case_hd95 = hd95(pred_mask, gt_resized > 0.5)
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
        print(
            f"[{idx}/{len(seg_paths)}] {case_id} "
            f"dice={case_dice:.4f} iou={case_iou:.4f} hd95={case_hd95:.3f} failure={failure_score:.3f}",
            flush=True,
        )

    rows.sort(key=lambda row: row["failure_score"], reverse=True)

    csv_path = args.out_dir / "external_case_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "cohort", "dice", "iou", "hd95", "failure_score"])
        writer.writeheader()
        writer.writerows(rows)

    top_k = rows[:20]
    (args.out_dir / "top20_failure_cases.json").write_text(json.dumps(top_k, indent=2))
    print(f"Saved metrics to {csv_path}")
    print("Top 10 current failure candidates:")
    for row in rows[:10]:
        print(row)


if __name__ == "__main__":
    main()
