from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure


def solidify_binary_mask(mask: np.ndarray) -> np.ndarray:
    struct = generate_binary_structure(3, 2)
    mask = binary_closing(mask, structure=struct, iterations=2)
    mask = binary_fill_holes(mask)
    return mask.astype(np.float32)


def rasterize_vertices_to_binary(verts: np.ndarray, grid_size: int) -> np.ndarray:
    vol = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    coords = ((verts - mn) / (mx - mn + 1e-6) * (grid_size - 1)).astype(int)
    coords = np.clip(coords, 0, grid_size - 1)
    for x, y, z in coords:
        vol[z, y, x] = 1.0
    return solidify_binary_mask(vol > 0.5)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_pca_dir = repo_root / "PCA" / "pca_results"
    default_out_dir = repo_root / "runs" / "dheart_reg_rebuild" / "synthetic"

    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-dir", type=Path, default=default_pca_dir)
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--num-cases", type=int, default=10000)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_shape = np.load(args.pca_dir / "mean_shape.npy").astype(np.float32)
    basis = np.load(args.pca_dir / "pca_components.npy").astype(np.float32)
    eigenvalues = np.load(args.pca_dir / "pca_eigenvalues.npy").astype(np.float32)

    num_components = basis.shape[0]
    std = np.sqrt(np.maximum(eigenvalues, 1e-8)) * args.sigma_scale

    for i in range(args.num_cases):
        coeff = rng.normal(size=num_components).astype(np.float32) * std
        verts = mean_shape + (coeff @ basis).reshape(mean_shape.shape)
        binary_mask = rasterize_vertices_to_binary(verts, args.grid_size)
        case_id = f"case{i:05d}"
        np.save(out_dir / f"{case_id}_mesh.npy", verts.astype(np.float32))
        np.save(out_dir / f"{case_id}_mask.npy", binary_mask.astype(np.float32))
        np.save(out_dir / f"{case_id}_vol.npy", binary_mask.astype(np.float32))
        np.save(out_dir / f"{case_id}_coeff.npy", coeff.astype(np.float32))
        if (i + 1) % 1000 == 0 or i == 0:
            print(f"Generated {i + 1}/{args.num_cases} synthetic cases")

    print(f"Synthetic dataset saved to {out_dir}")


if __name__ == "__main__":
    main()
