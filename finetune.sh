#!/usr/bin/env bash
set -euo pipefail

: "${DHEART_ROOT:?Set DHEART_ROOT to the local D-Heart dataset root before running finetune.sh}"

python scripts/finetune_cta1100_cas.py \
  --image-dir "$DHEART_ROOT/img_1000" \
  --mesh-dir "$DHEART_ROOT/mesh_1000" \
  --pretrain-ckpt ./checkpoints/cardiac_hmr_epoch19.pth \
  --save-dir ./finetune_ckpts \
  --epochs 50 \
  --batch-size 4 \
  --lr 1e-6
