import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np


class Simple3DEncoder(nn.Module):
    def __init__(self, latent_dim=256, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels,16,3,2,1)
        self.conv2 = nn.Conv3d(16,32,3,2,1)
        self.conv3 = nn.Conv3d(32,64,3,2,1)
        self.conv4 = nn.Conv3d(64,128,3,2,1)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, latent_dim)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.gap(x).view(x.shape[0],-1)

        return self.fc(x)


class PCADecoder(nn.Module):
    def __init__(self, mean_vertices, basis):
        super().__init__()

        if isinstance(mean_vertices, np.ndarray):
            mean_vertices = torch.from_numpy(mean_vertices).float()
        if isinstance(basis, np.ndarray):
            basis = torch.from_numpy(basis).float()

        Nv = mean_vertices.shape[0]
        self.Nv = Nv
        self.register_buffer("v_mean", mean_vertices)  # (Nv,3)

        if basis.shape[1] != Nv*3:
            print(f"Warning: basis second dim {basis.shape[1]} != {Nv*3}")
            basis = basis.T

        self.register_buffer("basis", basis)

    def forward(self, coeffs):
        shape = coeffs @ self.basis   # (B, num_components) @ (num_components, Nv*3)
        shape = shape.view(coeffs.shape[0], self.Nv, 3)
        return self.v_mean[None] + shape


class MaskDecoder3D(nn.Module):
    def __init__(self, latent_dim=256, out_shape=(64,64,64)):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256*4*4*4)
        self.up1 = nn.ConvTranspose3d(256,128,4,2,1)
        self.up2 = nn.ConvTranspose3d(128,64,4,2,1)
        self.up3 = nn.ConvTranspose3d(64,32,4,2,1)
        self.up4 = nn.ConvTranspose3d(32,1,4,2,1)
        self.out_shape = out_shape

    def forward(self,z):
        B = z.shape[0]
        x = F.relu(self.fc(z)).view(B,256,4,4,4)
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))
        x = torch.sigmoid(self.up4(x))

        return F.interpolate(x,size=self.out_shape,mode='trilinear',align_corners=False)


class CardiacHMR(nn.Module):
    def __init__(self, mean_vertices, basis, faces, latent_dim=256, mask_out_shape=(64,64,64)):
        super().__init__()
        self.encoder = Simple3DEncoder(latent_dim)
        self.mask_head = MaskDecoder3D(latent_dim,mask_out_shape)
        self.decoder = PCADecoder(mean_vertices,basis)
        self.faces = torch.from_numpy(faces).long()
        self.shape_head = nn.Linear(latent_dim,basis.shape[0])

    def forward(self,vol):
        z = self.encoder(vol)
        coeff = self.shape_head(z)
        verts = self.decoder(coeff)
        mask = self.mask_head(z)

        return {"coeff":coeff,"verts":verts,"mask":mask}
