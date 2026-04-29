from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from sklearn.decomposition import PCA


COLOR_TO_LABEL = {
    (223, 122, 94): 1,   # LV
    (244, 241, 222): 2,  # LA
    (242, 204, 142): 4,  # RA
    (130, 178, 154): 5,  # RV
}


def load_obj_with_color(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if hasattr(mesh.visual, "vertex_colors"):
        colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)[:, :3]
    else:
        colors = np.full((verts.shape[0], 3), 200, dtype=np.uint8)
    return verts, faces, colors


def save_colored_obj(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, path: Path) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rgba = np.concatenate(
        [colors, np.full((colors.shape[0], 1), 255, dtype=np.uint8)],
        axis=1,
    )
    mesh.visual.vertex_colors = rgba
    mesh.export(path)


def map_colors_to_labels(colors: np.ndarray) -> np.ndarray:
    canonical_colors = np.asarray(list(COLOR_TO_LABEL.keys()), dtype=np.int32)
    canonical_labels = np.asarray(list(COLOR_TO_LABEL.values()), dtype=np.int32)
    labels = np.zeros((colors.shape[0],), dtype=np.int32)
    for i, color in enumerate(colors.astype(np.int32)):
        diff = np.abs(canonical_colors - color).sum(axis=1)
        labels[i] = canonical_labels[np.argmin(diff)]
    return labels


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_mesh_dir = repo_root.parent / "CTA1100" / "mesh_1000"
    default_out_dir = repo_root / "PCA" / "pca_results"

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", type=Path, default=default_mesh_dir)
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--n-components", type=float, default=0.98)
    args = parser.parse_args()

    mesh_dir = args.mesh_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_files = sorted(mesh_dir.glob("*.obj"))
    if not obj_files:
        raise RuntimeError(f"No OBJ files found in {mesh_dir}")

    print(f"Found {len(obj_files)} meshes in {mesh_dir}")

    centroids: list[np.ndarray] = []
    radii: list[float] = []
    faces_ref: np.ndarray | None = None
    colors_ref: np.ndarray | None = None
    normalized_samples: list[np.ndarray] = []

    for obj_path in obj_files:
        verts, faces, colors = load_obj_with_color(obj_path)
        centroid = verts.mean(axis=0)
        radius = np.linalg.norm(verts - centroid, axis=1).max()
        centroids.append(centroid)
        radii.append(float(radius))
        if faces_ref is None:
            faces_ref = faces.copy()
            colors_ref = colors.copy()

    mean_scale = float(np.mean(radii))
    np.save(out_dir / "sample_centroids.npy", np.asarray(centroids, dtype=np.float32))
    np.save(out_dir / "sample_scales.npy", np.asarray(radii, dtype=np.float32))
    np.save(out_dir / "global_mean_scale.npy", np.asarray(mean_scale, dtype=np.float32))

    for obj_path, centroid in zip(obj_files, centroids):
        verts, _, _ = load_obj_with_color(obj_path)
        verts_norm = (verts - centroid) / mean_scale
        normalized_samples.append(verts_norm.reshape(-1))

    sample_matrix = np.stack(normalized_samples, axis=0)
    pca = PCA(n_components=args.n_components, svd_solver="full", whiten=False)
    shape_codes = pca.fit_transform(sample_matrix)
    components_norm = pca.components_.astype(np.float32)
    eigenvalues = pca.explained_variance_.astype(np.float32)
    mean_norm = pca.mean_.reshape(-1, 3).astype(np.float32)

    # Save basis already scaled to the real-world coordinate system so that
    # decoder output = mean_shape + coeff @ basis is dimensionally consistent.
    basis_real = (components_norm * mean_scale).astype(np.float32)
    mean_real = (mean_norm * mean_scale).astype(np.float32)

    assert faces_ref is not None and colors_ref is not None
    template_labels = map_colors_to_labels(colors_ref)

    np.save(out_dir / "pca_components.npy", basis_real)
    np.save(out_dir / "mean_shape.npy", mean_real)
    np.save(out_dir / "shape_codes.npy", shape_codes.astype(np.float32))
    np.save(out_dir / "pca_eigenvalues.npy", eigenvalues)
    np.save(out_dir / "template_faces.npy", faces_ref.astype(np.int32))
    np.save(out_dir / "template_labels.npy", template_labels.astype(np.int32))

    save_colored_obj(mean_real, faces_ref, colors_ref, out_dir / "mean.obj")

    metadata = {
        "source_mesh_dir": mesh_dir.name,
        "num_meshes": len(obj_files),
        "num_vertices": int(mean_real.shape[0]),
        "num_faces": int(faces_ref.shape[0]),
        "explained_variance_ratio_top4": [
            float(x) for x in pca.explained_variance_ratio_[:4]
        ],
        "global_mean_scale": mean_scale,
        "pca_basis_saved_in_real_world_scale": True,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(f"Saved rebuilt PCA assets to {out_dir}")


if __name__ == "__main__":
    main()
