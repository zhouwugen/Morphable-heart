import os, torch, numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F


class CardiacCTDataset(Dataset):
    """
    Dataset for CTA + optional mask + mesh
    Expect npy files:
      caseID_vol.npy, caseID_mask.npy (optional), caseID_mesh.npy (optional)
    """
    def __init__(self, index_list, root_dir):
        self.index_list = index_list
        self.root = root_dir

    def __len__(self): return len(self.index_list)

    def __getitem__(self, idx):
        key = self.index_list[idx]
        vol = np.load(os.path.join(self.root, f"{key}_vol.npy"))  # (D,H,W)
        item = {"volume": torch.from_numpy(vol[None]).float()}
        mpath = os.path.join(self.root, f"{key}_mask.npy")
        if os.path.exists(mpath):
            item["mask"] = torch.from_numpy(np.load(mpath)[None]).float()
        mpath = os.path.join(self.root, f"{key}_mesh.npy")
        if os.path.exists(mpath):
            item["mesh_gt"] = torch.from_numpy(np.load(mpath)).float()
        return item

def collate_fn(batch):
    maxd = max([b["volume"].shape[1] for b in batch])
    maxh = max([b["volume"].shape[2] for b in batch])
    maxw = max([b["volume"].shape[3] for b in batch])
    vols, masks, meshes = [], [], []
    for b in batch:
        v = F.pad(b["volume"],
                  (0,maxw-b["volume"].shape[3],
                   0,maxh-b["volume"].shape[2],
                   0,maxd-b["volume"].shape[1]))
        vols.append(v)
        if "mask" in b:
            m = F.pad(b["mask"],
                      (0,maxw-b["mask"].shape[3],
                       0,maxh-b["mask"].shape[2],
                       0,maxd-b["mask"].shape[1]))
            masks.append(m)
        if "mesh_gt" in b: meshes.append(b["mesh_gt"])

    out = {"volume": torch.stack(vols)}
    if masks: out["mask"] = torch.stack(masks)
    if meshes: out["mesh_gt"] = torch.stack(meshes)
    
    return out
