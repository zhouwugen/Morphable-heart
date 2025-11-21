import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.ops import cubify, sample_points_from_meshes, knn_points
from pytorch3d.utils import ico_sphere

from torch_geometric.utils import degree, to_undirected, to_dense_adj, get_laplacian, add_self_loops
from torch_geometric.data import Data
from torch_scatter import scatter

import numpy as np
import trimesh

from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix

from ops.graph_operators import NativeFeaturePropagation, LaplacianSmoothing
from data.dataset import MMWHSDataset_3DLabel
from data.dataset import CTADataset3D

from tqdm import tqdm
from probreg import cpd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

import warnings
warnings.filterwarnings("ignore")

from GHD import GHD_config, GHDmesh, Normal_iterative_GHDmesh
from GHD.GHD_cardiac import GHD_Cardiac

from einops import rearrange, einsum, repeat

from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss

from losses import *
from ops.mesh_geometry import *

import data.data_utils as dut
from ops.medical_related import *
from tqdm import tqdm
from pytorch3d.structures import Meshes


def apply_affine_to_points(points: torch.Tensor, affine: torch.Tensor):
    """
    points: (N,3) torch on same device as affine
    affine: (4,4) torch
    returns: (N,3) torch
    """
    if points.numel() == 0:
        return points
    N = points.shape[0]
    ones = torch.ones((N,1), dtype=points.dtype, device=points.device)
    homog = torch.cat([points, ones], dim=1)
    transformed = (affine @ homog.T).T[:, :3]
    return transformed


def apply_affine_to_mesh(mesh: Meshes, affine: torch.Tensor, device=None):
    """
    mesh: pytorch3d Meshes with single mesh (or list)
    affine: (4,4) torch on desired device
    return: new Meshes with transformed verts, same faces
    """
    verts = mesh.verts_list()[0].to(affine.device)
    faces = mesh.faces_list()[0].to(affine.device)
    verts_h = torch.cat([verts, torch.ones((verts.shape[0],1), device=affine.device)], dim=1)
    verts_t = (affine @ verts_h.T).T[:, :3]
    return Meshes(verts=[verts_t], faces=[faces])


def ensure_mesh_float32(mesh: Meshes, device=torch.device("cpu")):
    verts = mesh.verts_packed().to(torch.float32).to(device)
    faces = mesh.faces_packed().to(device)
    return Meshes(verts=[verts], faces=[faces])


def rigid_register_whole_template_to_points(paraheart_list, target_pcl_np, sample_num_template=8000, update_scale=True):
    """
    Align template to target_pcl_np（numpy array, (N,3)）
    Args:
        paraheart_list: list of GHD_Cardiac (e.g. [para_lv, para_rv, para_la, para_ra])
        target_pcl_np: numpy array (N,3) （ canonical frame）
        sample_num_template: int
        update_scale: bool
    Returns:
        param_dict: {'rot': R (3x3 numpy), 'scale': s (float), 't': T (3,) numpy}
    """
    # 1) Concatenate all the verts/faces of all the template（store them in numpy）
    verts_all = []
    faces_all = []
    v_offset = 0
    for para in paraheart_list:
        vert_np = para.base_shape.verts_packed().detach().cpu().numpy()
        face_np = para.base_shape.faces_packed().detach().cpu().numpy() + v_offset
        verts_all.append(vert_np)
        faces_all.append(face_np)
        v_offset += vert_np.shape[0]
    verts_all = np.vstack(verts_all)
    faces_all = np.vstack(faces_all)

    trimesh_whole = trimesh.Trimesh(vertices=verts_all, faces=faces_all, process=False)

    # 2) From the overall template volume, sample point clouds
    sample_num_template = int(min(sample_num_template, max(1000, target_pcl_np.shape[0])))
    point_cloud_template = trimesh.sample.volume_mesh(trimesh_whole, sample_num_template)
    point_cloud_template = np.asarray(point_cloud_template, dtype=np.float64)
    target_pcl_np = np.asarray(target_pcl_np, dtype=np.float64)

    # 3) Call probreg.cpd to perform RigidCPD
    param_dict_init = {'rot': np.eye(3, dtype=np.float64), 'scale': 1.0, 't': np.zeros(3, dtype=np.float64)}
    rgd_cpd = cpd.RigidCPD(point_cloud_template, tf_init_params=param_dict_init, update_scale=update_scale)
    tf_param, _, _ = rgd_cpd.registration(target_pcl_np)

    R_mat = tf_param.rot   # numpy 3x3
    s_val = float(tf_param.scale)
    T_vec = tf_param.t     # numpy 3
    param_dict = {'rot': R_mat, 'scale': s_val, 't': T_vec}

    # 4) Write back R/s/T to paraheart_list
    for para in paraheart_list:
        device = para.device
        R_axis = matrix_to_axis_angle(torch.from_numpy(R_mat).float().unsqueeze(0).to(device))  # (1,3)
        para.R = torch.nn.Parameter(R_axis.clone())
        para.s = torch.nn.Parameter(torch.tensor([s_val], device=device).float().unsqueeze(0))
        para.T = torch.nn.Parameter(torch.from_numpy(T_vec).float().to(device).unsqueeze(0))
    return param_dict


save_dir = "../cardiac mesh"
os.makedirs(save_dir, exist_ok=True)

base_shape_path = './canonical_shapes/Standard_LV_4055.obj'
bi_ventricle_path = './canonical_shapes/Standard_BiV.obj'

cfg = GHD_config(base_shape_path=base_shape_path,
            num_basis=9**2, mix_laplacian_tradeoff={'cotlap':1.0, 'dislap':0.1, 'stdlap':0.1},
            device='cuda:0',
            if_nomalize=True, if_return_scipy=True, 
            bi_ventricle_path=bi_ventricle_path)

device = cfg.device

dataset = CTADataset3D(dataset_path='path/to/your/CTAdata', mode='train', simple_mode='4chambers',
                       output_shape=(256, 256, 256), load_cache = True)

lv_path = './canonical_shapes/LV.obj'
rv_path = './canonical_shapes/RV.obj'
la_path = './canonical_shapes/LA.obj'
ra_path = './canonical_shapes/RA.obj'

num_basis = 7**2
cfg_lv = GHD_config(base_shape_path=lv_path,
            num_basis=num_basis, mix_laplacian_tradeoff={'cotlap':1.0, 'dislap':0.1, 'stdlap':0.1},
            device=device, if_nomalize=True, if_return_scipy=True, bi_ventricle_path=bi_ventricle_path)
cfg_rv = GHD_config(base_shape_path=rv_path,
            num_basis=num_basis, mix_laplacian_tradeoff={'cotlap':1.0, 'dislap':0.1, 'stdlap':0.1},
            device=device, if_nomalize=True, if_return_scipy=True, bi_ventricle_path=bi_ventricle_path)
cfg_la = GHD_config(base_shape_path=la_path,
            num_basis=num_basis, mix_laplacian_tradeoff={'cotlap':1.0, 'dislap':0.1, 'stdlap':0.1},
            device=device, if_nomalize=True, if_return_scipy=True, bi_ventricle_path=bi_ventricle_path)
cfg_ra = GHD_config(base_shape_path=ra_path,
            num_basis=num_basis, mix_laplacian_tradeoff={'cotlap':1.0, 'dislap':0.1, 'stdlap':0.1},
            device=device, if_nomalize=True, if_return_scipy=True, bi_ventricle_path=bi_ventricle_path)

para_lv = GHD_Cardiac(cfg_lv)
para_rv = GHD_Cardiac(cfg_rv)
para_la = GHD_Cardiac(cfg_la)
para_ra = GHD_Cardiac(cfg_ra)
GHD_list = [para_rv, para_lv, para_la, para_ra]

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

for i, example in enumerate(tqdm(dataloader, desc="Processing hearts", ncols=100)):
    case_id = dataset.cases[i]
    save_path = os.path.join(save_dir, f"{case_id}.obj")

    if os.path.exists(save_path):
        tqdm.write(f"[SKIP] {case_id} already exists.")
        continue

    image_tem = example['image'].to(device).float() # b, 1, z, y, x
    affine_tem = example['affine'].to(device).float()
    label_tem = example['label'].to(device).float()

    B = image_tem.shape[0]
    coordinate_map_tem = dut.get_coord_map_3d_normalized(image_tem.shape[-3:], affine_tem)
    lv_cavity_index =  torch.where(label_tem>0)
    lv_cavity_center = coordinate_map_tem[lv_cavity_index[0],lv_cavity_index[2],lv_cavity_index[3],lv_cavity_index[4],:]
    lv_cavity_center = scatter(lv_cavity_center, lv_cavity_index[0], dim=0, reduce='mean')

    affine_aug = torch.eye(4).to(device).unsqueeze(0).repeat(B,1,1)
    affine_aug[...,:3, 3] = affine_aug[...,:3, 3] + lv_cavity_center
    del lv_cavity_center, lv_cavity_index

    Z_new, Y_new, X_new = 256, 256, 256
    with torch.no_grad():
        new_affine = affine_tem.inverse()@affine_aug

        label_tem = rearrange(label_tem, 'b 1 z y x -> b z y x')
        label_tem = dut.label_to_onehot(label_tem, dataset.label_value)
        label_tem = rearrange(label_tem, 'b z y x c -> b c z y x')
        label_tem = dut.augment_from_affine(label_tem, new_affine, (Z_new, Y_new, X_new), mode='bilinear')
        label_tem = rearrange(label_tem, 'b c z y x -> b z y x c')
        label_tem = dut.onehot_to_label(label_tem, dataset.label_value)
        label_tem = rearrange(label_tem, 'b z y x -> b 1 z y x')
        image_tem = dut.augment_from_affine(image_tem, new_affine, (Z_new, Y_new, X_new), mode='bilinear')

        example['label'] = label_tem
        example['image'] = image_tem

        image_tem = dut.img_standardize(image_tem)
        image_tem = dut.img_normalize(image_tem)
        coordinate_map_tem = dut.get_coord_map_3d_normalized(
            image_tem.shape[-3:], torch.eye(4).to(device).unsqueeze(0).repeat(B,1,1)
        )

        Z_rv, Y_rv, X_rv = torch.where(label_tem[0, 0]==dataset.label_value[5])
        Z_lv, Y_lv, X_lv = torch.where(label_tem[0, 0]==dataset.label_value[1])
        Z_cav, Y_cav, X_cav = torch.where(label_tem[0, 0]==dataset.label_value[3])
        Z_la, Y_la, X_la = torch.where(label_tem[0, 0]==dataset.label_value[2])
        Z_ra, Y_ra, X_ra = torch.where(label_tem[0, 0]==dataset.label_value[4])
        Z_bg, Y_bg, X_bg = torch.where(label_tem[0, 0]==dataset.label_value[0])

        Pt_rv = coordinate_map_tem[0, Z_rv, Y_rv, X_rv]
        Pt_lv = coordinate_map_tem[0, Z_lv, Y_lv, X_lv]
        Pt_cav = coordinate_map_tem[0, Z_cav, Y_cav, X_cav]
        Pt_bg = coordinate_map_tem[0, Z_bg, Y_bg, X_bg]
        Pt_la = coordinate_map_tem[0, Z_la, Y_la, X_la]
        Pt_ra = coordinate_map_tem[0, Z_ra, Y_ra, X_ra]
        
        Pt_list = [Pt_bg, Pt_lv, Pt_la, Pt_cav, Pt_ra, Pt_rv]

        geom_dict = get_4chamberview_frame(Pt_cav, Pt_lv, Pt_rv)
        initial_affine = geom_dict['target_affine'] # canonical->data
        inv_initial_affine = torch.inverse(initial_affine).to(device) # data->canonical

    # make recubified_list order match GHD_list (rv, lv, la, ra)
    recubified_list = [cubify((label_tem==dataset.label_value[k]).squeeze(1).float(), 0.5) for k in [5, 1, 2, 4]]

    for paraheart in GHD_list:
        paraheart.R = torch.zeros_like(paraheart.R)
        paraheart.s = torch.ones_like(paraheart.s)
        paraheart.T = torch.zeros_like(paraheart.T)


    # -------------------- global_registration --------------------
    pts_all = torch.cat([Pt_rv, Pt_lv, Pt_la, Pt_ra], dim=0)
    pts_all_canonical = apply_affine_to_points(pts_all, inv_initial_affine)
    pts_all_np = pts_all_canonical.detach().cpu().numpy()

    sample_num = 8000
    if pts_all_np.shape[0] > sample_num:
        idx = np.random.choice(pts_all_np.shape[0], sample_num, replace=False)
        pts_all_np_sub = pts_all_np[idx]
    else:
        pts_all_np_sub = pts_all_np

    print("🔹Run one global_registration on whole-heart template (LV+RV+LA+RA)")

    param_dict = rigid_register_whole_template_to_points(
                [para_rv, para_lv, para_la, para_ra], 
                pts_all_np_sub,
                update_scale=True)

    print("Whole heart registration completed：", param_dict)
    
    R_mat, s_val, T_vec = param_dict['rot'], param_dict['scale'], param_dict['t']
    print("global reg result: scale=%.4f, T=%s" % (s_val, str(T_vec)))

    R_axis = matrix_to_axis_angle(torch.from_numpy(R_mat).float().unsqueeze(0).to(device))
    s_torch = torch.tensor([s_val], device=device).unsqueeze(0)
    T_torch = torch.from_numpy(T_vec).float().to(device).unsqueeze(0)

    for paraheart in [para_rv, para_lv, para_la, para_ra]:
        paraheart.R = nn.Parameter(R_axis.clone())
        paraheart.s = nn.Parameter(s_torch.clone())
        paraheart.T = nn.Parameter(T_torch.clone())
    #--------------------------------------------------------------------


    para_lv.base_shape = ensure_mesh_float32(para_lv.base_shape, para_lv.device)
    para_rv.base_shape = ensure_mesh_float32(para_rv.base_shape, para_rv.device)
    para_la.base_shape = ensure_mesh_float32(para_la.base_shape, para_la.device)
    para_ra.base_shape = ensure_mesh_float32(para_ra.base_shape, para_ra.device)

    margin = 0.05
    GHD_corespondece = [5, 1, 2, 4]  # [rv, lv, la, ra]
    convergence_list = []
    sample_part_num = 2000

    for j, paraheart in enumerate(GHD_list):
        print('Fitting part ', j)

        # data frame -> canonical frame
        points_in_df = Pt_list[GHD_corespondece[j]]
        points_in = apply_affine_to_points(points_in_df, inv_initial_affine)  # (N,3) canonical

        points_out_df = torch.cat([Pt_list[k] for k in range(6) if k != GHD_corespondece[j]], dim=0)
        points_out = apply_affine_to_points(points_out_df, inv_initial_affine)  # canonical

        if points_in.numel() == 0:
            print(f"⚠️ Part {j} has zero positive points, skip")
            convergence_list.append(paraheart.base_shape)
            continue

        mins = points_in.min(dim=0)[0] - margin
        maxs = points_in.max(dim=0)[0] + margin
        mask = (points_out[:,0] > mins[0]) & (points_out[:,0] < maxs[0]) & \
               (points_out[:,1] > mins[1]) & (points_out[:,1] < maxs[1]) & \
               (points_out[:,2] > mins[2]) & (points_out[:,2] < maxs[2])
        points_neg = points_out[mask]

        recubified_mesh_canonical = apply_affine_to_mesh(recubified_list[j], inv_initial_affine)

        if points_in.shape[0] > sample_part_num:
            idxp = torch.randperm(points_in.shape[0], device=points_in.device)[:sample_part_num]
            points_in = points_in[idxp]
        if points_neg.shape[0] > 3*sample_part_num:
            idxn = torch.randperm(points_neg.shape[0], device=points_neg.device)[:3*sample_part_num]
            points_neg = points_neg[idxn]

        convergence, Loss_dict  = paraheart.fitting2target(
            target_positives=points_in,
            target_negatives=points_neg,
            target_mesh=recubified_mesh_canonical,
            loss_dict={'Loss_occupancy':1.0, 'Loss_Laplacian':0.001, 'Loss_thickness':0.001,
                       'Loss_Chamfer_P0':0.05},
            lr_start=1e-3,
            num_iter=1000,
            if_reset=True,
            if_fit_R=False,
            if_fit_s=True,
            if_fit_T=True,
            record_convergence=0
        )
        convergence_list.append(convergence)

    color_list = [np.array([130, 178, 154]) / 255,
                  np.array([223, 122, 94]) / 255,
                  np.array([244, 241, 222]) / 255,
                  np.array([242, 204, 142]) / 255,]
    all_meshes = []

    for j, mesh in enumerate(convergence_list):
        # mesh = apply_affine_to_mesh(mesh, initial_affine)

        verts = mesh.verts_list()[0].detach().cpu().numpy()
        faces = mesh.faces_list()[0].detach().cpu().numpy()

        print(f"Part {j}: verts={verts.shape}, faces={faces.shape}")
        if verts.shape[0] == 0 or faces.shape[0] == 0:
            print(f"⚠️ Warning: Part {j} is empty, skip")
            continue

        temp_trimesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        face_colors = np.tile((color_list[j] * 255).astype(np.uint8), (faces.shape[0], 1))
        temp_trimesh.visual.face_colors = face_colors
        all_meshes.append(temp_trimesh)

    if len(all_meshes) > 0:
        whole_mesh = trimesh.util.concatenate(all_meshes)
        print("Final mesh empty?", whole_mesh.is_empty)
        whole_mesh.export(save_path)
        print(f"✅ Save sucessfully: {save_path}")
    else:
        print(f"❌ Without valid mesh，skip saving {save_path}")

