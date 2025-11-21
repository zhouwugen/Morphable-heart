# Compare predicted mesh vs GT mesh in canonical space.
# Automatically rigid-align (GT → Pred) using Open3D ICP.
import os
import numpy as np
import torch
import nibabel as nib
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes, binary_closing
from models import CardiacHMR
from scipy.stats import mode


# -------------------------------------------------------------
#                     CONFIG
# -------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEAN_PATH   = "./pca_results_color/mean_shape.npy"
BASIS_PATH  = "./pca_results_color/pca_components.npy"
FACES_PATH  = "./pca_results_color/template_faces.npy"
TEMPLATE_LABEL_PATH = "./pca_results_color/template_labels.npy"

MODEL_WEIGHTS = "../finetune_ckpts/finetune_epoch49.pth"

INPUT_IMG = "./CTAdata/img/ct_train_1009_image.nii.gz"  
INPUT_MASK = "./CTAdata/seg/ct_train_1009_label.nii.gz"
GT_MESH = "./CTAdata/mesh/ct_train_1009.obj"
OUT_OBJ = "./train_1009_pred_fine.obj"

TARGET_RESOLUTION = 8 #128
EVAL_CLASSES = [1,2,4,5]

COLOR_TO_LABEL = {
    (223,122,94):1,
    (244,241,222):2,
    (242,204,142):4,
    (130,178,154):5,
}


# -------------------------------------------------------------
#             METRICS
# -------------------------------------------------------------
def dice_3d(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred,gt).sum()
    return 2*inter/(pred.sum()+gt.sum()+1e-8)


def iou_3d(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred,gt).sum()
    union = np.logical_or(pred,gt).sum()
    return inter/(union+1e-8)

def hd95(pred, gt, sample_size=5000):
    p = np.argwhere(pred)
    g = np.argwhere(gt)
    if len(p)==0 or len(g)==0:
        return np.nan
    if len(p)>sample_size:
        p = p[np.random.choice(len(p),sample_size,False)]
    if len(g)>sample_size:
        g = g[np.random.choice(len(g),sample_size,False)]
    tp, tg = cKDTree(p), cKDTree(g)
    d1,_ = tp.query(g); d2,_ = tg.query(p)
    return np.percentile(np.hstack([d1,d2]),95)

def chamfer(pred, gt):
    tp, tg = cKDTree(pred), cKDTree(gt)
    d1,_ = tp.query(gt); d2,_ = tg.query(pred)
    return d1.mean()+d2.mean(), d1.mean(), d2.mean()

def hd95_mesh(pred, gt):
    tp, tg = cKDTree(pred), cKDTree(gt)
    d1,_ = tp.query(gt); d2,_ = tg.query(pred)
    return np.percentile(d1,95), np.percentile(d2,95)

def vertex_error(pred, gt):
    diff = np.linalg.norm(pred-gt,axis=1)
    return {
        "mean": float(diff.mean()),
        "median": float(np.median(diff)),
        "rmse": float(np.sqrt((diff**2).mean())),
        "p90": float(np.percentile(diff,90))
    }


# -------------------------------------------------------------
#               VOXELIZATION
# -------------------------------------------------------------
def get_bbox_pitch(template, res=128):
    mn = template.min(0); mx = template.max(0)
    pitch = (mx-mn).max()/(res-1)
    return mn, mx, pitch

def voxelize(verts, faces, labels, cls, bbox_min, pitch, res):

    out = np.zeros((res, res, res), dtype=np.uint8)

    face_mask = labels == cls
    faces_cls = faces[face_mask]
    if len(faces_cls) == 0:
        return out 

    # triangle rasterize
    for f in faces_cls:
        tri = verts[f]   # (3,3)
        # triangle bounding box
        vmin = np.floor((tri.min(axis=0) - bbox_min) / pitch).astype(int)
        vmax = np.ceil((tri.max(axis=0) - bbox_min) / pitch).astype(int)

        vmin = np.clip(vmin, 0, res - 1)
        vmax = np.clip(vmax, 0, res - 1)

        out[vmin[0]:vmax[0]+1,
            vmin[1]:vmax[1]+1,
            vmin[2]:vmax[2]+1] = 1

    out = binary_closing(out, iterations=2)
    out = binary_fill_holes(out)

    return out


def compute_face_labels(faces, vertex_labels):
    tri_labels = vertex_labels[faces]       # (N_faces, 3)
    face_labels = mode(tri_labels, axis=1, keepdims=False).mode

    return face_labels.astype(np.int32)


def voxelize_single_class(verts, faces, face_labels, target_cls, bbox_min, pitch, res):

    out = np.zeros((res, res, res), dtype=np.uint8)
    mask = (face_labels == target_cls)
    faces_cls = faces[mask]
    if len(faces_cls) == 0:
        return out

    for f in faces_cls:
        tri = verts[f]   # (3,3)

        vmin = np.floor((tri.min(0) - bbox_min) / pitch).astype(int)
        vmax = np.ceil((tri.max(0) - bbox_min) / pitch).astype(int)

        vmin = np.clip(vmin, 0, res-1)
        vmax = np.clip(vmax, 0, res-1)

        out[
            vmin[0]:vmax[0]+1,
            vmin[1]:vmax[1]+1,
            vmin[2]:vmax[2]+1
        ] = 1

    # morphological closing
    out = binary_closing(out, iterations=1)
    out = binary_fill_holes(out)

    return out


def voxelize_multiclass(verts, faces, vertex_labels, bbox_min, pitch, res, classes=[1,2,4,5]):

    face_labels = compute_face_labels(faces, vertex_labels)
    out = np.zeros((res, res, res), dtype=np.uint8)

    for cls in classes:
        print(f"  voxelize class {cls}")
        m = voxelize_single_class(verts, faces, face_labels, cls,
                                  bbox_min, pitch, res)
        out[m == 1] = cls

    return out


# -------------------------------------------------------------
#            LOAD GT LABELS
# -------------------------------------------------------------
def load_mesh_labels(mesh_path):
    mesh = trimesh.load(mesh_path,process=False)
    if isinstance(mesh,trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    verts = np.asarray(mesh.vertices,float)
    faces = np.asarray(mesh.faces,int)

    base = os.path.splitext(mesh_path)[0]
    for ext in ["_labels.npy","_vertex_labels.npy"]:
        p = base+ext
        if os.path.exists(p):
            lab = np.load(p)
            if len(lab)==len(verts):
                return verts,faces,lab.astype(int)

    # try vertex colors
    if hasattr(mesh.visual,"vertex_colors"):
        vc = np.asarray(mesh.visual.vertex_colors)[:,:3]
        labs = np.zeros(len(verts),int)
        keys = np.array(list(COLOR_TO_LABEL.keys()))
        vals = np.array(list(COLOR_TO_LABEL.values()))
        for i,c in enumerate(vc):
            d=np.abs(keys-c).sum(1)
            labs[i]=vals[np.argmin(d)]
        return verts,faces,labs

    raise RuntimeError("GT labels not found & no vertex color available.")


# -------------------------------------------------------------
#        ICP rigid alignment (GT -> Pred)
# -------------------------------------------------------------
def to_o3d(verts, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    return mesh

def sample_points(mesh, n=6000):
    return mesh.sample_points_uniformly(n)

def rigid_align_icp(gt_verts, gt_faces, pred_verts, pred_faces):
    print("\n=== Rigid ICP aligning GT → Pred ===")

    m_gt = to_o3d(gt_verts, gt_faces)
    m_pr = to_o3d(pred_verts, pred_faces)

    p_gt = sample_points(m_gt, 6000)
    p_pr = sample_points(m_pr, 6000)

    thr_coarse = 10.0
    thr_fine   = 2.0

    # coarse
    reg1 = o3d.pipelines.registration.registration_icp(
        p_gt, p_pr, thr_coarse,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    # fine
    reg2 = o3d.pipelines.registration.registration_icp(
        p_gt, p_pr, thr_fine,
        reg1.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    T = reg2.transformation
    print("ICP finished. Transformation:\n", T)
    gt_aligned = (gt_verts @ T[:3,:3].T) + T[:3,3]

    return gt_aligned, T


# -------------------------------------------------------------
#                    MAIN
# -------------------------------------------------------------
def main():

    # load PCA basis
    mean = np.load(MEAN_PATH)
    comps = np.load(BASIS_PATH)
    faces = np.load(FACES_PATH)

    # load model
    model = CardiacHMR(mean, comps, faces, latent_dim=256)
    sd = torch.load(MODEL_WEIGHTS,map_location="cpu")
    sd = sd["model"] if "model" in sd else sd
    model.load_state_dict(sd,strict=False)
    model.to(DEVICE).eval()

    # infer (dummy volume)
    vol = torch.zeros((1,1,TARGET_RESOLUTION,TARGET_RESOLUTION,TARGET_RESOLUTION),dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        out = model(vol)
        pred = out["verts"][0].cpu().numpy()

    # save pred
    trimesh.Trimesh(pred,faces,process=False).export(OUT_OBJ)
    print("Saved pred mesh:", OUT_OBJ)

    # load template pred labels
    tpl = np.load(TEMPLATE_LABEL_PATH).astype(int)
    if len(tpl)!=len(pred):
        n=min(len(tpl),len(pred))
        tpl, pred = tpl[:n], pred[:n]
        F = faces[faces.max(1)<n]
    else:
        F = faces

    # load GT mesh + labels
    gtv, gtf, gtl = load_mesh_labels(GT_MESH)
    print("GT labels:", np.unique(gtl))

    # ---- Load template pred labels ----
    template_labels = None
    if os.path.exists(TEMPLATE_LABEL_PATH):
        template_labels = np.load(TEMPLATE_LABEL_PATH)

        if template_labels.shape[0] != pred.shape[0]:
            min_len = min(template_labels.shape[0], pred.shape[0])
            template_labels = template_labels[:min_len]
            pred = pred[:min_len]
            F = faces[faces.max(axis=1) < min_len]

        print(f"Loaded template_labels: shape={template_labels.shape}, unique={np.unique(template_labels)}")

        # Color assignment
        color_list = [
            np.array([223,122,94])/255,
            np.array([244,241,222])/255,
            np.array([242,204,142])/255,
            np.array([130,178,154])/255
            ]
        label_to_color = {1:color_list[0],2:color_list[1],4:color_list[2],5:color_list[3]}
        colors = np.zeros((pred.shape[0],3),dtype=np.float32)
        for k,c in label_to_color.items():
            colors[template_labels==k] = c

        color_mesh = trimesh.Trimesh(vertices=pred, faces=F,
                                    vertex_colors=(colors*255).astype(np.uint8), process=False)
        ply_path = OUT_OBJ.replace(".obj","_colored.obj")
        color_mesh.export(ply_path)
        print(f"✅ Saved colored PLY mesh: {ply_path}")

    # ------ RIGID ICP ALIGN GT → PRED ------
    gtv_aligned, T = rigid_align_icp(gtv,gtf,pred,F)

    # save aligned GT mesh
    out_gt_aligned = os.path.splitext(GT_MESH)[0] + "_aligned.obj"
    trimesh.Trimesh(gtv_aligned,gtf,process=False).export(out_gt_aligned)
    print("Saved aligned GT mesh:", out_gt_aligned)

    # save transform
    np.savetxt(out_gt_aligned.replace(".obj","_T.txt"), T)
    print("Saved transform matrix.")

    # ------ canonical voxelization ------
    bbox_min, bbox_max, pitch = get_bbox_pitch(mean, TARGET_RESOLUTION)

    print("\nVoxelizing Pred...")
    mask_pred = voxelize_multiclass(pred,F,tpl,bbox_min,pitch,TARGET_RESOLUTION)

    print("Voxelizing GT aligned...")
    mask_gt = voxelize_multiclass(gtv_aligned,gtf,gtl,bbox_min,pitch,TARGET_RESOLUTION)

    # save NII
    nib.save(nib.Nifti1Image(mask_pred.astype(np.uint8),np.eye(4)), OUT_OBJ.replace(".obj","_mask_pred.nii.gz"))
    nib.save(nib.Nifti1Image(mask_gt.astype(np.uint8),np.eye(4)), out_gt_aligned.replace(".obj","_mask_gt.nii.gz"))

    # ------ Segmentation metrics ------
    print("\n=== Segmentation metrics (canonical) ===")
    for c in EVAL_CLASSES:
        d = dice_3d(mask_pred==c, mask_gt==c)
        i = iou_3d(mask_pred==c, mask_gt==c)
        h = hd95(mask_pred==c, mask_gt==c)
        print(f"Class {c}: Dice={d:.4f} IoU={i:.4f} HD95={h:.2f}")

    # ------ Mesh metrics ------
    print("\n=== Mesh metrics ===")
    cd, p2g, g2p = chamfer(pred, gtv_aligned)
    print(f"Chamfer={cd:.4f}  pred→gt={p2g:.4f}  gt→pred={g2p:.4f}")

    h1, h2 = hd95_mesh(pred, gtv_aligned)
    print(f"HD95 pred→gt={h1:.4f}, gt→pred={h2:.4f}")

    if len(pred)==len(gtv_aligned):
        print("Vertex error:", vertex_error(pred, gtv_aligned))
    else:
        print("Vertex count mismatch → skip vertex error.")

    print("\nDone.")


if __name__ == "__main__":
    main()


