from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes, binary_closing, generate_binary_structure


def solidify_multiclass_mask(vol: np.ndarray) -> np.ndarray:
    """
    Input: vol in {0,1,2,4,5} on a 64^3 grid.
    Output: hole-filled solid chamber mask.
    """
    solid = np.zeros_like(vol)
    struct = generate_binary_structure(3, 2)

    for cls in [1, 2, 4, 5]:
        mask = vol == cls
        mask = binary_closing(mask, structure=struct, iterations=2)
        mask = binary_fill_holes(mask)
        solid[mask] = cls

    return solid


def synthesize_cases(
    mean: np.ndarray,
    basis: np.ndarray,
    template_labels: np.ndarray,
    num: int = 100,
    save_dir: Path = Path("./synthetic"),
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    num_vertices = mean.shape[0]
    num_basis = basis.shape[0]

    basis = basis.T
    mean_flat = mean.reshape(-1)

    for i in range(num):
        coeff = np.random.randn(num_basis) * 0.5
        delta = basis @ coeff
        verts = (mean_flat + delta).reshape(num_vertices, 3)

        vol = np.zeros((64, 64, 64), np.int32)
        coords = (
            (verts - verts.min(0))
            / (verts.max(0) - verts.min(0) + 1e-6)
            * 63
        ).astype(int)
        coords = np.clip(coords, 0, 63)

        for (x, y, z), cls in zip(coords, template_labels):
            vol[z, y, x] = cls

        solid_vol = solidify_multiclass_mask(vol)

        np.save(save_dir / f"case{i:03d}_mesh.npy", verts)
        np.save(save_dir / f"case{i:03d}_mask.npy", solid_vol)
        np.save(save_dir / f"case{i:03d}_vol.npy", solid_vol)

    print("Done. Example labels:", np.unique(solid_vol))


def resolve_default_pca_dir(repo_root: Path) -> Path:
    candidates = [
        repo_root / "PCA" / "pca_results",
        repo_root / "PCA" / "pca_result_color",
    ]
    for candidate in candidates:
        if (candidate / "mean_shape.npy").exists():
            return candidate
    return candidates[0]


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-dir", type=Path, default=resolve_default_pca_dir(repo_root))
    parser.add_argument("--save-dir", type=Path, default=repo_root / "synthetic")
    parser.add_argument("--num", type=int, default=10000)
    args = parser.parse_args()

    mean = np.load(args.pca_dir / "mean_shape.npy")
    basis = np.load(args.pca_dir / "pca_components.npy")
    template_labels = np.load(args.pca_dir / "template_labels.npy")

    synthesize_cases(mean, basis, template_labels, num=args.num, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
