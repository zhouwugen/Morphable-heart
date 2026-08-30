# <div align="center">Morphable Heart Model</div>

This repository provides the public implementation and morphable-model assets
for **D-Heart** and its reference reconstruction baseline,
**D-Heart-Reg**.

- **D-Heart** is a CTA dataset and benchmark for template-consistent,
  template-index four-region heart reconstruction.
- **D-Heart-Reg** is a PCA-native regression baseline that maps a complete
  stored CTA array to raw PCA scores and a centred, fixed-topology mesh.

## D-Heart release scope

The D-Heart package contains:

- **1,100 source-named CTA entries**: 1,000 CAS, 60 MMWHS, and 40 WHS++;
- **1,100 expert-reviewed four-region labels**;
- **1,000 released CAS template-consistent meshes**;
- a CAS-only PCA morphable shape model; and
- manifests, coordinate metadata, configurations, hashes, and verification
  records for benchmark use.

The 100 external source aliases correspond to 80 primary unique label targets
after identity accounting. Primary statistical evaluation should follow the
released unique-target manifest and representative-alias rule. Per-case
external OBJ meshes are not publicly redistributed; source-governed access and
regeneration follow the data package documentation.

- **Dataset:** [D-Heart release on Harvard Dataverse](https://dataverse.harvard.edu/previewurl.xhtml?token=014c4eea-bba5-4fbb-a5e5-550d87f31bab)
- **Code:** [Morphable-heart on GitHub](https://github.com/zhouwugen/Morphable-heart)

This repository includes:

- mesh-fitting and PCA-construction utilities;
- released PCA assets for the D-Heart morphable shape space;
- synthetic initialization, training, fine-tuning, and inference code for
  D-Heart-Reg; and
- scripts for rebuilding the public D-Heart-Reg pipeline from the D-Heart
  package.

<p align="center">
  <img src="Figure/teaser.png" width="900" alt="D-Heart morphable model and reconstruction pipeline">
</p>

## Installation

```bash
conda create -n morphable-heart python=3.8
conda activate morphable-heart
pip install -r requirements.txt
```

## Repository structure

```text
Morphable-heart/
├── Mesh fitting/                # canonical mesh-fitting utilities
├── PCA/
│   ├── pca_result_color/        # PCA construction outputs and legacy layout
│   └── pca_results/             # released PCA assets used by D-Heart-Reg
├── scripts/                     # staged rebuild and reproduction scripts
├── train.py                     # synthetic initialization entry point
├── finetune.py                  # real-CTA fine-tuning entry point
├── inference.py                 # prediction and demo evaluation
└── synth_data.py                # synthetic PCA-sample generation
```

## Data setup

Set the local D-Heart package root before running the examples:

```bash
export DHEART_ROOT=/path/to/D-Heart
```

The commands below expect the released CAS images and meshes at
`$DHEART_ROOT/img_1000` and `$DHEART_ROOT/mesh_1000`. External paths and
manifest locations are supplied to the staged evaluation scripts through
arguments or environment variables; no private local path is hard-coded.

## Mesh fitting and PCA construction

Run the mesh-fitting utility:

```bash
cd "Mesh fitting"
python FourChambers.py
```

Run the PCA utility:

```bash
cd PCA
python run_pca_model.py
```

PCA construction outputs are written under `PCA/pca_result_color/`.

To rebuild the D-Heart-Reg PCA assets directly from the released CAS meshes:

```bash
python scripts/rebuild_pca_assets_from_cta1100.py \
  --mesh-dir "$DHEART_ROOT/mesh_1000" \
  --out-dir ./PCA/pca_results
```

The rebuilt asset directory contains:

- `mean_shape.npy`
- `pca_components.npy`
- `pca_eigenvalues.npy`
- `template_faces.npy`
- `template_labels.npy`
- `mean.obj`

The rebuild script uses `--n-components 0.98` by default. The released CAS PCA
asset retains 98% variance and contains **119 components**. PC1--PC4
visualizations are descriptive shape-mode summaries; D-Heart-Reg uses the full
retained basis unless another PCA asset is explicitly supplied.

## D-Heart-Reg training

The reported D-Heart-Reg objective combines template-index vertex
reconstruction loss with an L2 regularizer on raw PCA scores. Segmentation-mask
supervision is not part of the reported reconstruction objective.

Generate 10,000 PCA-derived synthetic samples:

```bash
python scripts/generate_synthetic_dataset_from_pca.py \
  --pca-dir ./PCA/pca_results \
  --out-dir ./runs/dheart_reg/synthetic \
  --num-cases 10000
```

Run synthetic initialization:

```bash
python scripts/train_pretrain_from_synthetic.py \
  --data-dir ./runs/dheart_reg/synthetic \
  --pca-dir ./PCA/pca_results \
  --save-dir ./runs/dheart_reg/pretrain_ckpts
```

Fine-tune on the 1,000 CAS CTA entries:

```bash
python scripts/finetune_cta1100_cas.py \
  --image-dir "$DHEART_ROOT/img_1000" \
  --mesh-dir "$DHEART_ROOT/mesh_1000" \
  --pca-dir ./PCA/pca_results \
  --pretrain-ckpt ./runs/dheart_reg/pretrain_ckpts/best.pth \
  --save-dir ./runs/dheart_reg/finetune_ckpts \
  --epochs 50 \
  --lr 1e-6
```

The complete staged workflow is available through:

- `scripts/nohup_rebuild_pipeline.sh`
- `scripts/launch_full_rebuild_background.sh`

These scripts use relative or environment-configurable paths. Set
`DHEART_ROOT` for individual commands or `CTA_ROOT` for the staged shell
pipeline.

## Inference

`inference.py` provides a lightweight D-Heart-Reg prediction and evaluation
entry point:

```bash
python inference.py \
  --pca-dir ./PCA/pca_results \
  --model-weights ./runs/dheart_reg/finetune_ckpts/best.pth \
  --input-img "$DHEART_ROOT/img_1000/CAS_0001_image.nii.gz" \
  --gt-mesh "$DHEART_ROOT/mesh_1000/CAS_0001.obj" \
  --out-obj ./prediction.obj
```

The script reports mesh metrics and writes the predicted mesh, a colored
prediction, and an aligned reference mesh. Any mask output is an auxiliary
sanity check and is not part of the reported D-Heart-Reg objective.

## Reproducibility notes

- The repository does not bundle private or source-restricted datasets.
- Source-derived assets remain governed by their original licenses and
  data-use terms.
- Training checkpoints are not assumed to exist; provide checkpoint paths
  explicitly.
- `PCA/pca_results/` is the canonical released morphable-model asset directory.
- Use the scripts in `scripts/` as the primary entry point for rebuilding and
  evaluating D-Heart-Reg.
- Primary cross-cohort statistics should use the released 80-unique-target
  accounting rather than treating all 100 source aliases as independent cases.
