import torch, torch.nn.functional as F


def vertex_l2_loss(pred,gt): return F.mse_loss(pred,gt)
def mask_loss_ce(pred,gt): return F.cross_entropy(pred,gt)
def pca_prior_loss(coeff): return torch.mean(coeff**2)


def voxelize_points(batch_vertices,image_size=64):
    B,N,_=batch_vertices.shape
    occ = torch.zeros(B,1,image_size,image_size,image_size,device=batch_vertices.device)
    coords=(batch_vertices.clamp(0,1)*(image_size-1)).long()
    for b in range(B):
        for v in range(N):
            z,y,x=coords[b,v]
            occ[b,0,z,y,x]=1.0
    return F.max_pool3d(occ,3,1,1)


def get_coord_map_3d_normalized(shape, affine_mat):
    """
    (Z,Y,X) in [-1,1]
    shape: (D,H,W)
    affine_mat: (B,4,4)
    return: (B, D, H, W, 3)
    """

    D, H, W = shape
    device = affine_mat.device

    z = torch.linspace(-1, 1, D, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)

    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1)  # (D,H,W,3)

    B = affine_mat.shape[0]
    grid = grid.unsqueeze(0).repeat(B,1,1,1,1)

    return grid


def scatter(src, index, dim=0, reduce='mean'):
    """
    src: (N,3)
    index: (N,)
    """
    B = index.max().item() + 1
    out = torch.zeros(B, src.shape[-1], device=src.device)
    count = torch.zeros(B, 1, device=src.device) + 1e-8

    for i in range(src.shape[0]):
        b = index[i]
        out[b] += src[i]
        count[b] += 1

    if reduce == 'mean':
        out = out / count

    return out


def get_4chamberview_frame(vol, mask, spacing=None):
    """
    Align 3D CTA volume to 4-chamber canonical
    Return:
        vol_aligned: (B,1,D,H,W)
        mask_aligned: (B,1,D,H,W)
        affine_mat: (B,4,4)
    """

    B, _, D, H, W = vol.shape
    device = vol.device

    mask_float = mask.float()

    zz, yy, xx = torch.meshgrid(
        torch.arange(D, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    coords = torch.stack([xx, yy, zz], dim=-1).float()   # (D,H,W,3)
    coords = coords.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B,D,H,W,3)

    mask_flat = mask_float.reshape(B, -1)         # (B, N)
    coords_flat = coords.reshape(B, -1, 3)        # (B, N, 3)

    pts = []
    for b in range(B):
        m = mask_flat[b] > 0.5
        pts.append(coords_flat[b][m])

    affine_list = []
    for b in range(B):
        if pts[b].shape[0] < 100:
            affine_list.append(torch.eye(4, device=device))
            continue

        pc = pts[b] - pts[b].mean(dim=0, keepdim=True)
        C = pc.t() @ pc / pc.shape[0]
        U, S, V = torch.svd(C)

        x_axis = U[:, 0]
        y_axis = U[:, 1]
        z_axis = U[:, 2]

        R = torch.stack([x_axis, y_axis, z_axis], dim=1)  # 3×3

        # Construct 4x4 affine
        A = torch.eye(4, device=device)
        A[:3, :3] = R.t()

        # Align the center of mask to the center of volume
        ctr = pts[b].mean(dim=0)
        A[:3, 3] = -R.t() @ ctr

        affine_list.append(A)

    affine_mat = torch.stack(affine_list, dim=0)  # (B,4,4)

    # generate normalized grid
    z = torch.linspace(-1, 1, D, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1)   # (D,H,W,3)
    grid = grid.unsqueeze(0).repeat(B,1,1,1,1)  # (B,D,H,W,3)

    ones = torch.ones(B, D, H, W, 1, device=device)
    homog = torch.cat([grid, ones], dim=-1)   # (B,D,H,W,4)

    affine_mat_T = affine_mat.transpose(1,2)  # (B,4,4)

    new = torch.matmul(homog, affine_mat_T)   # (B,D,H,W,4)
    new = new[...,:3]   # (B,D,H,W,3)

    # grid_sample (B,D,H,W,3) in [-1,1]
    vol_aligned = F.grid_sample(vol, new, mode='bilinear', padding_mode='border')
    mask_aligned = F.grid_sample(mask_float, new, mode='nearest', padding_mode='border')

    return vol_aligned, mask_aligned, affine_mat

