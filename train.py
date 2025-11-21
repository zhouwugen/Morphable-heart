import os
import torch, torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CardiacCTDataset, collate_fn
from models import CardiacHMR
from utils import vertex_l2_loss, mask_loss_bce, pca_prior_loss
import numpy as np


def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0
    for batch in loader:
        vol = batch["volume"].to(device)
        out = model(vol)
        loss = 0
        if "mesh_gt" in batch:
            loss += 1.0 * vertex_l2_loss(out["verts"], batch["mesh_gt"].to(device))
        if "mask" in batch:
            loss += 1.0 * mask_loss_bce(out["mask"], batch["mask"].to(device))
        loss += 0.01 * pca_prior_loss(out["coeff"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================
    # Load PCA model
    # =====================
    mean = np.load("./PCA/pca_results/mean_shape.npy")
    basis = np.load("./PCA/pca_results/pca_components.npy")
    faces = np.load("./PCA/pca_results/template_faces.npy")

    # =====================
    # Dataset
    # =====================
    ids = [f"case{i:03d}" for i in range(0, 10000)]
    ds = CardiacCTDataset(ids, "./synthetic")
    dl = DataLoader(ds, batch_size=60, collate_fn=collate_fn)

    # =====================
    # Model & Optimizer
    # =====================
    model = CardiacHMR(mean, basis, faces).to(device)
    opt = optim.Adam(model.parameters(), 1e-4)

    # =====================
    # Checkpoint path
    # =====================
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # =====================
    # Resume if checkpoint exists
    # =====================
    start_epoch = 0
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path} ...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # =====================
    # Training loop
    # =====================
    for e in range(start_epoch, 20):
        loss = train_one_epoch(model, dl, opt, device)
        print(f"Epoch {e}: {loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"cardiac_hmr_epoch{e}.pth")
        torch.save(
            {"epoch": e, "model": model.state_dict(), "optimizer": opt.state_dict()},
            ckpt_path,
        )


if __name__ == "__main__":
    main()

