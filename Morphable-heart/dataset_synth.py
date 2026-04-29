import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class SyntheticNPYDataset(Dataset):
    """
    Dataset for synthetic_v6 generated samples.
    Each case contains:
        caseXXXXX_mask.npy   -> (64,64,64) uint8 {0,1,2,4,5}
        caseXXXXX_vol.npy    -> (64,64,64) float32 (intensity volume / sdf-based)
        caseXXXXX_mesh.npy   -> (Nv,3) float32
        caseXXXXX_coeff.npy  -> (n_comp,) float32
    """

    def __init__(self,
                 root,
                 ids=None,
                 input_mode="volume",
                 augment=False,
                 seed=42,
                 verbose=True):

        self.root = root
        self.input_mode = input_mode
        self.augment = augment
        self.verbose = verbose

        # ------------------------------
        # Collect IDs
        # ------------------------------
        if ids is None:
            files = sorted(f for f in os.listdir(root) if f.endswith("_mask.npy"))
            ids = [f.replace("_mask.npy", "") for f in files]
        self.ids = ids

        if verbose:
            print(f"[SyntheticNPYDataset] Found {len(self.ids)} samples in {root}")

        random.seed(seed)

    # =====================================================
    # Augmentation (only vol & mask)
    # =====================================================
    def _augment(self, vol, mask):
        if not self.augment:
            return vol, mask

        v = vol
        # random flips
        if random.random() < 0.5:
            v = np.flip(v, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() < 0.5:
            v = np.flip(v, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.5:
            v = np.flip(v, axis=3).copy()
            mask = np.flip(mask, axis=2).copy()

        return v, mask

    # =====================================================
    # Data loader
    # =====================================================
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        cid = self.ids[idx]

        # ------------------------
        # Load required files
        # ------------------------
        mask_path = os.path.join(self.root, f"{cid}_mask.npy")
        vol_path  = os.path.join(self.root, f"{cid}_vol.npy")
        mesh_path = os.path.join(self.root, f"{cid}_mesh.npy")
        coeff_path = os.path.join(self.root, f"{cid}_coeff.npy")

        # mask: int64 (64,64,64)
        mask = np.load(mask_path).astype(np.int64)

        # volume: float32 (64,64,64)
        vol_raw = None
        if os.path.exists(vol_path):
            vol_raw = np.load(vol_path).astype(np.float32)

        # ----------------------------------------------------
        # Build network inputs
        # ----------------------------------------------------
        if self.input_mode == "zeros":
            vol = np.zeros((1,) + mask.shape, dtype=np.float32)

        elif self.input_mode == "mask_as_input":
            vol = (mask > 0).astype(np.float32)[None, ...]

        elif self.input_mode == "volume":
            # intensity volume
            assert vol_raw is not None, f"Missing {vol_path}"
            vol = vol_raw[None, ...]

        elif self.input_mode == "mask+volume":
            assert vol_raw is not None, f"Missing {vol_path}"
            ch0 = (mask > 0).astype(np.float32)
            ch1 = vol_raw
            vol = np.stack([ch0, ch1], axis=0)

        else:
            raise ValueError(f"Unknown input_mode: {self.input_mode}")

        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------
        vol, mask = self._augment(vol, mask)

        # ----------------------------------------------------
        # Optional: mesh & PCA coefficients
        # ----------------------------------------------------
        item = {
            "id": cid,
            "volume": torch.from_numpy(vol).float(),
            "mask": torch.from_numpy(mask).long(),
        }

        if os.path.exists(mesh_path):
            verts = np.load(mesh_path).astype(np.float32)
            item["mesh_gt"] = torch.from_numpy(verts).float()

        if os.path.exists(coeff_path):
            coeff = np.load(coeff_path).astype(np.float32)
            item["coeff_gt"] = torch.from_numpy(coeff).float()

        return item
