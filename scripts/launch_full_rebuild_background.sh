#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
GPU_ID="${GPU_ID:-0}"
SMOKE_MODE="${SMOKE_MODE:-0}"
RUN_NAME="${RUN_NAME:-dheart_reg_rebuild_full_${STAMP}}"
RUN_ROOT="$ROOT/runs/$RUN_NAME"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/full_rebuild_background_${STAMP}.log"
PID_FILE="$LOG_DIR/full_rebuild_background_${STAMP}.pid"

cd "$ROOT"

setsid env GPU_ID="$GPU_ID" SMOKE_MODE="$SMOKE_MODE" RUN_NAME="$RUN_NAME" CTA_ROOT="${CTA_ROOT:-}" \
  bash "$ROOT/scripts/nohup_rebuild_pipeline.sh" \
  > "$LOG_FILE" 2>&1 < /dev/null &
RUN_PID=$!
echo "$RUN_PID" > "$PID_FILE"

echo "PID=$RUN_PID"
echo "RUN_NAME=$RUN_NAME"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG=$LOG_FILE"
echo "PID_FILE=$PID_FILE"
