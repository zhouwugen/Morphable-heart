#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CTA_ROOT="${CTA_ROOT:-}"
PY_TRAIN="${PY_TRAIN:-python}"
PY_PRE="${PY_PRE:-$PY_TRAIN}"
RUN_NAME="${RUN_NAME:-dheart_reg_rebuild}"
RUN_ROOT="$ROOT/runs/$RUN_NAME"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"
export PYTHONUNBUFFERED=1

if [[ -z "$CTA_ROOT" ]]; then
  echo "[Error] CTA_ROOT is not set. Set CTA_ROOT to the local D-Heart dataset root before running this pipeline." >&2
  echo "        Example: CTA_ROOT=./D-Heart bash scripts/nohup_rebuild_pipeline.sh" >&2
  exit 2
fi
if [[ ! -d "$CTA_ROOT" ]]; then
  echo "[Error] CTA_ROOT does not exist: $CTA_ROOT" >&2
  exit 2
fi

SMOKE_MODE="${SMOKE_MODE:-0}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "[Info] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[Info] CTA_ROOT=$CTA_ROOT"
echo "[Info] RUN_ROOT=$RUN_ROOT"
if [[ "$SMOKE_MODE" == "1" ]]; then
  SYNTH_CASES=256
  PRETRAIN_EPOCHS=1
  PRETRAIN_LIMIT=256
  FINETUNE_EPOCHS=1
  FINETUNE_LIMIT=16
  EXTERNAL_LIMIT=8
  NUM_WORKERS=0
else
  SYNTH_CASES=10000
  PRETRAIN_EPOCHS=20
  PRETRAIN_LIMIT=0
  FINETUNE_EPOCHS=50
  FINETUNE_LIMIT=0
  EXTERNAL_LIMIT=0
  NUM_WORKERS=4
fi

echo "[Stage 0] Environment"
"$PY_TRAIN" -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "[Stage 1] Rebuild PCA assets"
"$PY_PRE" -u "$ROOT/scripts/rebuild_pca_assets_from_cta1100.py" \
  --mesh-dir "$CTA_ROOT/mesh_1000" \
  --out-dir "$ROOT/PCA/pca_results"

echo "[Stage 2] Generate synthetic dataset"
"$PY_PRE" -u "$ROOT/scripts/generate_synthetic_dataset_from_pca.py" \
  --pca-dir "$ROOT/PCA/pca_results" \
  --out-dir "$RUN_ROOT/synthetic" \
  --num-cases "$SYNTH_CASES"

echo "[Stage 3] Pretrain from synthetic data"
"$PY_TRAIN" -u "$ROOT/scripts/train_pretrain_from_synthetic.py" \
  --data-dir "$RUN_ROOT/synthetic" \
  --pca-dir "$ROOT/PCA/pca_results" \
  --save-dir "$RUN_ROOT/pretrain_ckpts" \
  --epochs "$PRETRAIN_EPOCHS" \
  --batch-size 40 \
  --lr 1e-4 \
  --num-workers "$NUM_WORKERS" \
  --limit "$PRETRAIN_LIMIT"

echo "[Stage 4] Finetune on CTA1100 CAS"
"$PY_TRAIN" -u "$ROOT/scripts/finetune_cta1100_cas.py" \
  --image-dir "$CTA_ROOT/img_1000" \
  --mesh-dir "$CTA_ROOT/mesh_1000" \
  --pca-dir "$ROOT/PCA/pca_results" \
  --pretrain-ckpt "$RUN_ROOT/pretrain_ckpts/best.pth" \
  --save-dir "$RUN_ROOT/finetune_ckpts" \
  --epochs "$FINETUNE_EPOCHS" \
  --batch-size 4 \
  --lr 1e-6 \
  --num-workers "$NUM_WORKERS" \
  --limit "$FINETUNE_LIMIT"

echo "[Stage 5] Batch predict external labeled subset"
"$PY_TRAIN" -u "$ROOT/scripts/batch_predict_and_rank_external.py" \
  --image-dir "$CTA_ROOT/img_100" \
  --seg-dir "$CTA_ROOT/seg_100" \
  --pca-dir "$ROOT/PCA/pca_results" \
  --ckpt "$RUN_ROOT/finetune_ckpts/best.pth" \
  --out-dir "$RUN_ROOT/external_predictions" \
  --limit "$EXTERNAL_LIMIT"

echo "[Done] Pipeline finished."
