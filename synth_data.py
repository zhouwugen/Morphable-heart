import numpy as np, os
from scipy.ndimage import binary_fill_holes, binary_closing, generate_binary_structure


def solidify_multiclass_mask(vol):
    """
    Input：vol ∈ [0,1,2,4,5] (64³)
    Output：solid_vol (64³)
    """
    solid = np.zeros_like(vol)
    struct = generate_binary_structure(3, 2)

    for cls in [1,2,4,5]:
        mask = (vol == cls)
        mask = binary_closing(mask, structure=struct, iterations=2)
        mask = binary_fill_holes(mask)
        solid[mask] = cls

    return solid


def synthesize_cases(mean, basis, template_labels, num=100, save_dir="./synthetic"):
    os.makedirs(save_dir, exist_ok=True)
    Nv = mean.shape[0]
    n_basis = basis.shape[0]

    basis = basis.T
    mean_flat = mean.reshape(-1)

    for i in range(num):
        coeff = np.random.randn(n_basis) * 0.5
        delta = basis @ coeff
        verts = (mean_flat + delta).reshape(Nv, 3)

        # -----------------------------
        # voxelization → (64³)
        # -----------------------------
        vol = np.zeros((64,64,64), np.int32)
        coords = ((verts - verts.min(0)) /
                  (verts.max(0) - verts.min(0) + 1e-6) * 63).astype(int)
        coords = np.clip(coords, 0, 63)

        for (x,y,z), cls in zip(coords, template_labels):
            vol[z,y,x] = cls

        solid_vol = solidify_multiclass_mask(vol)

        # -----------------------------
        np.save(os.path.join(save_dir, f"case{i:03d}_mesh.npy"), verts)
        np.save(os.path.join(save_dir, f"case{i:03d}_mask.npy"), solid_vol)
        np.save(os.path.join(save_dir, f"case{i:03d}_vol.npy"), solid_vol)
        # np.save(os.path.join(save_dir, f"case{i:03d}_coeff.npy"), coeff)

    print("Done. Example labels:", np.unique(solid_vol))


if __name__=="__main__":

    mean = np.load("/home/zwg/data/CTAdata/pca_results_color_v2/mean_shape.npy")   # (Nv,3)
    basis = np.load("/home/zwg/data/CTAdata/pca_results_color_v2/pca_components.npy")  # (128, Nv*3)
    template_labels = np.load("/home/zwg/data/CTAdata/pca_results_color_v2/template_labels.npy") # (Nv,)

    synthesize_cases(mean, basis, template_labels, num=10000) # generate 10,000 samples

