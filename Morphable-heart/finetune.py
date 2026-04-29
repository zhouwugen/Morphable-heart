import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from models import CardiacHMR
from utils import mask_loss_bce, pca_prior_loss


def resize_volume(vol, target_shape=(128,128,128)):
    """
    vol: numpy array (D,H,W) 或 torch.Tensor (1,D,H,W)
    target_shape: (D,H,W)
    """
    vol = np.asarray(vol)

    if vol.ndim == 4:  
        vol = np.squeeze(vol)

    if vol.ndim != 3:
        raise ValueError(f"Unexpected volume shape {vol.shape}, expected 3D")
    
    if vol.shape[0] != vol.shape[-1]:  
        vol = np.transpose(vol, (2,0,1))  # (D,H,W)

    vol = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    vol = F.interpolate(vol, size=target_shape, mode='trilinear', align_corners=False)
    return vol.squeeze(0)  # (1,D,H,W)


# =====================
# NIfTI Dataset
# =====================
class CTANiftiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, ids, target_shape=(128,128,128)):
        """
        Args:
            image_dir: CTA image
            mask_dir: mask image
            ids: ID list，such as ["case_001", "case_002"]
            target_shape: desired output shape (D,H,W)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.ids = ids
        self.target_shape = target_shape

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]

        # Read CTA
        img_path = os.path.join(self.image_dir, f"{pid}_image.nii.gz")
        img_nii = nib.load(img_path)
        img = img_nii.get_fdata().astype(np.float32)

        # Normalization to [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Read mask
        mask_path = os.path.join(self.mask_dir, f"{pid}_seg.nii.gz")
        mask_nii = nib.load(mask_path)
        mask = mask_nii.get_fdata().astype(np.float32)

        # Normalization to [0,1]
        mask = mask / mask.max()  # [0., 0., 1.]

        # channel -> (1, D, H, W)
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # resample to a fixed volum
        img = resize_volume(img, self.target_shape)  # (1,D,H,W)
        mask = resize_volume(mask, self.target_shape)  # (1,D,H,W)

        sample = {
            "volume": img,
            "mask": mask }

        return sample


# training function
def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0
    for batch in loader:
        vol = batch["volume"].to(device)
        mask_gt = batch["mask"].to(device)

        out = model(vol)
        loss = 0

        loss += 1.0 * mask_loss_bce(out["mask"], mask_gt)
        loss += 0.01 * pca_prior_loss(out["coeff"])

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


# =====================
# Main
# =====================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="CTA image folder")
    parser.add_argument("--mask_dir", type=str, required=True, help="mask folder")
    parser.add_argument("--ids_file", type=str, required=True, help="ID.txt，every case per ID")
    parser.add_argument("--pretrain_ckpt", type=str, default="./checkpoints/cardiac_hmr_epoch19.pth")
    parser.add_argument("--save_dir", type=str, default="./finetune_ckpts")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================
    # Load PCA model
    # =====================
    mean = np.load("./PCA/pca_results/mean_shape.npy")
    basis = np.load("./PCA/pca_results/pca_components.npy")
    faces = np.load("./PCA/pca_results/template_faces.npy")

    # =====================
    # Model
    # =====================
    model = CardiacHMR(mean, basis, faces, 
                       latent_dim=256, 
                       mask_out_shape=(128,128,128)).to(device)

    # Load pretrained weights (only model)
    if os.path.exists(args.pretrain_ckpt):
        print(f"Loading pretrained checkpoint {args.pretrain_ckpt} ...")
        checkpoint = torch.load(args.pretrain_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        print("⚠️ Warning: Pretrained checkpoint not found, training from scratch!")

    # =====================
    # Dataset & Dataloader
    # =====================
    with open(args.ids_file, "r") as f:
        ids = [line.strip() for line in f.readlines()]

    ds = CTANiftiDataset(args.image_dir, args.mask_dir, ids)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # Optimizer
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # =====================
    # Training loop
    # =====================
    for e in range(args.epochs):
        loss = train_one_epoch(model, dl, opt, device)
        print(f"[FineTune] Epoch {e}: {loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.save_dir, f"finetune_epoch{e}.pth")
        torch.save({"epoch": e, "model": model.state_dict(), "optimizer": opt.state_dict()}, ckpt_path)


if __name__ == "__main__":
    main()
