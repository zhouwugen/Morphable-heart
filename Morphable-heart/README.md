# <div align="center">Morphable Heart Model</div>

This repository accompanies the morphable-heart component of **D-Heart** and
the released baseline regression model **D-Heart-Reg**.

More specifically:

- **D-Heart** refers to the dataset/benchmark, the released CAS
  template-consistent mesh assets, and the PCA morphable shape space derived
  from the fitted meshes.
- **D-Heart-Reg** refers to the regression baseline that predicts morphable parameters / meshes from CTA input.

Current public D-Heart package scope:

- 1,100 CTA volumes
- 1,082 released four-chamber labels
- 1,000 released CAS meshes
- no per-case public external OBJ meshes

This repository contains:

- the original mesh fitting and PCA construction code,
- released PCA shape assets for **D-Heart**,
- baseline training / fine-tuning code for **D-Heart-Reg**, and
- utilities for rebuilding the released **D-Heart-Reg** pipeline from the public **D-Heart** package.

<p align="center">
  <img src="Figure/teaser.png" width="900" alt="Morphable Heart teaser">
</p>

## Installation

```bash
conda create -n morphable-heart python==3.8
conda activate morphable-heart
pip install -r requirements.txt
```

## Repository structure

```text
Morphable-heart/
├── Mesh fitting/                # canonical mesh fitting utilities
├── PCA/
│   ├── pca_result_color/        # original PCA assets / legacy layout for the D-Heart shape model
│   └── pca_results/             # released PCA assets used by D-Heart-Reg reproduction
├── scripts/                     # reproducibility scripts for running D-Heart-Reg on the released D-Heart package
├── train.py                     # original synthetic pretraining entry for D-Heart-Reg
├── finetune.py                  # original fine-tuning entry for D-Heart-Reg
├── inference.py                 # prediction / evaluation demo entry for D-Heart-Reg
└── synth_data.py                # synthetic data generation helper
```

## PCA construction

### Original PCA workflow

Cardiac mesh fitting:

```bash
cd "Mesh fitting"
python FourChambers.py
```

Run PCA:

```bash
cd PCA
python run_pca_model.py
```

The original PCA outputs are written under `PCA/pca_result_color/`.

### Rebuilding PCA assets from the released D-Heart package

If you have the released **D-Heart** dataset package available locally, you can
rebuild the PCA assets used by the released **D-Heart-Reg** pipeline from the
public CAS mesh subset:

```bash
python scripts/rebuild_pca_assets_from_cta1100.py \
  --mesh-dir /path/to/CTA1100/mesh_1000 \
  --out-dir ./PCA/pca_results
```

The rebuilt assets include:

- `mean_shape.npy`
- `pca_components.npy`
- `pca_eigenvalues.npy`
- `template_faces.npy`
- `template_labels.npy`
- `mean.obj`

## Synthetic data generation and D-Heart-Reg training

### Original lightweight workflow

Generate synthetic samples:

```bash
python synth_data.py --pca-dir ./PCA/pca_results --save-dir ./synthetic --num 10000
```

Pre-train:

```bash
python train.py
```

Fine-tune:

```bash
./finetune.sh
```

### Released D-Heart-Reg reconstruction pipeline

The `scripts/` directory contains a more explicit reproduction pipeline for **D-Heart-Reg** on top of the released **D-Heart** package:

1. rebuild PCA assets from released D-Heart meshes,
2. generate synthetic PCA samples,
3. pre-train on synthetic data,
4. fine-tune on CAS, and
5. evaluate on the currently released external labeled subset.

Example:

```bash
python scripts/generate_synthetic_dataset_from_pca.py \
  --pca-dir ./PCA/pca_results \
  --out-dir ./runs/dheart_reg_rebuild/synthetic \
  --num-cases 10000

python scripts/train_pretrain_from_synthetic.py \
  --data-dir ./runs/dheart_reg_rebuild/synthetic \
  --pca-dir ./PCA/pca_results \
  --save-dir ./runs/dheart_reg_rebuild/pretrain_ckpts
```

For the full staged pipeline, see:

- `scripts/nohup_rebuild_pipeline.sh`
- `scripts/launch_full_rebuild_background.sh`

These scripts intentionally use relative or environment-configurable paths so they can be adapted to different local setups without editing source code.

## Inference / demo evaluation

`inference.py` is kept as a small demo-style entry point for **D-Heart-Reg**. It now accepts explicit CLI arguments instead of relying on hard-coded local paths.

Example:

```bash
python inference.py \
  --pca-dir ./PCA/pca_results \
  --model-weights ./finetune_ckpts/best.pth \
  --gt-mesh /path/to/example.obj \
  --out-obj ./prediction.obj
```

The script reports mesh and voxelized-mask metrics and writes:

- predicted mesh,
- colored predicted mesh,
- aligned reference mesh,
- voxelized prediction / reference masks.

## Notes

- This repository does **not** bundle private local datasets.
- Training checkpoints are not assumed to exist by default; users should supply their own checkpoint paths.
- The released `PCA/pca_results/` assets correspond to the released **D-Heart** morphable shape space.
- The `scripts/` directory is the recommended starting point for reproducing the public **D-Heart-Reg** baseline on the released **D-Heart** package.
