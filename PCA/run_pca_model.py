"""
PCA pipeline for colored whole-heart meshes
---------------------------------------------------------
✅ Use “global unified scale” normalization (preserving the real-world size ratios)
✅ The exported mean shape, random samples, and PCA modes are all kept in the real-world scale
✅ Automatically save each sample’s centroid and scale
✅ Additionally export template_faces.npy
"""

import os
import numpy as np
import trimesh
from sklearn.decomposition import PCA
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt

# -------------------- Config --------------------
DATA_DIR = "../cardiac mesh"
OUTPUT_DIR = "./pca_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_COMPONENTS = 0.98
RANDOM_SAMPLES = 20

# -------------------- Color label mapping --------------------
COLOR_TO_LABEL = {
    (223, 122, 94): 1,   # LV
    (244, 241, 222): 2,  # LA
    (242, 204, 142): 4,  # RA
    (130, 178, 154): 5,  # RV
}
canonical_colors = np.array(list(COLOR_TO_LABEL.keys()), dtype=np.int32)
canonical_labels = np.array(list(COLOR_TO_LABEL.values()), dtype=np.int32)

def map_colors_to_labels(vertex_colors, tolerance=10):
    Nv = vertex_colors.shape[0]
    colors = vertex_colors.astype(np.int32)
    labels = np.zeros((Nv,), dtype=np.int32)
    for i in range(Nv):
        diff = np.abs(canonical_colors - colors[i]).sum(axis=1)
        idx = np.argmin(diff)
        labels[i] = canonical_labels[idx]
    return labels

# -------------------- Helper functions --------------------
def load_obj_with_color(path):
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if hasattr(mesh.visual, "vertex_colors"):
        vc = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)
        colors = vc[:, :3]
    else:
        colors = np.ones((verts.shape[0], 3), dtype=np.uint8) * 200
    return verts, faces, colors

def save_whole_heart_obj(vertices, faces, colors, path):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rgba = np.hstack([colors, np.ones((colors.shape[0],1), dtype=np.uint8)*255])
    mesh.visual.vertex_colors = rgba
    mesh.export(path)

# -------------------- Step 1: computer unified scale --------------------
print("Scanning all meshes for global scaling ...")
obj_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".obj")])
if len(obj_files) == 0:
    raise RuntimeError("No OBJ found!")

scales, centroids = [], []
for f in tqdm(obj_files):
    verts, _, _ = load_obj_with_color(os.path.join(DATA_DIR, f))
    centroid = verts.mean(axis=0)
    radius = np.linalg.norm(verts - centroid, axis=1).max()
    scales.append(radius)
    centroids.append(centroid)
mean_scale = float(np.mean(scales))
print(f"[Info] Global mean scale = {mean_scale:.3f}")

np.save(os.path.join(OUTPUT_DIR, "sample_centroids.npy"), np.array(centroids))
np.save(os.path.join(OUTPUT_DIR, "sample_scales.npy"), np.array(scales))

# -------------------- Step 2: load and normalization --------------------
samples = []
faces_ref, colors_ref = None, None

print("Loading and normalizing meshes (using global scale)...")
for i, f in enumerate(tqdm(obj_files)):
    path = os.path.join(DATA_DIR, f)
    verts, faces, colors = load_obj_with_color(path)
    centroid = centroids[i]
    verts = (verts - centroid) / mean_scale
    if faces_ref is None:
        faces_ref, colors_ref = faces.copy(), colors.copy()
        labels = map_colors_to_labels(colors_ref)
        np.save(os.path.join(OUTPUT_DIR, "template_labels.npy"), labels)
    samples.append(verts.flatten())

samples = np.stack(samples, axis=0)
N_samples, vec_dim = samples.shape
print(f"Loaded {N_samples} samples, vector dim={vec_dim}")

# -------------------- Step 3: PCA --------------------
print("Running PCA ...")
pca = PCA(n_components=N_COMPONENTS, svd_solver='full', whiten=False)
shape_codes = pca.fit_transform(samples)
components = pca.components_
mean_flat = pca.mean_
mean_verts = mean_flat.reshape(-1, 3) * mean_scale
eigenvalues = pca.explained_variance_

np.save(os.path.join(OUTPUT_DIR, "pca_components.npy"), components.astype(np.float32))
np.save(os.path.join(OUTPUT_DIR, "mean_shape.npy"), mean_verts.astype(np.float32))
np.save(os.path.join(OUTPUT_DIR, "shape_codes.npy"), shape_codes.astype(np.float32))
np.save(os.path.join(OUTPUT_DIR, "pca_eigenvalues.npy"), eigenvalues.astype(np.float32))

np.save(os.path.join(OUTPUT_DIR, "template_faces.npy"), faces_ref.astype(np.int32))
print(f"[OK] Saved template_faces.npy  (faces={faces_ref.shape})")
print(f"[OK] PCA finished. Components={components.shape}, mean={mean_verts.shape}")

# -------------------- Step 4: Compactness plot --------------------
explained_var_ratio = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var_ratio) * 100
plt.figure(figsize=(6,5))
plt.plot(range(1, len(cumulative_var)+1), cumulative_var, '-o', markersize=3)
plt.xlabel("# Components"); plt.ylabel("% Variability")
plt.title("Compactness (Shape space)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig(os.path.join(OUTPUT_DIR, "compactness_shape.png"), dpi=300)
plt.close()

# -------------------- Step 5: Save mean + PC1~3 + Randoms --------------------
save_whole_heart_obj(mean_verts, faces_ref, colors_ref, os.path.join(OUTPUT_DIR, "mean.obj"))

for i in range(min(3, components.shape[0])):
    std = np.sqrt(pca.explained_variance_[i])
    for sign, s in [(-1, "-"), (1, "+")]:
        vec = mean_flat + sign * 3.0 * std * components[i]
        verts = vec.reshape(-1, 3) * mean_scale
        out_path = os.path.join(OUTPUT_DIR, f"pc{i}_{s}3sigma.obj")
        save_whole_heart_obj(verts, faces_ref, colors_ref, out_path)

print("✅ PCA done with physical scale and template_faces.npy")
