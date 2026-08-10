#!/usr/bin/env bash
# Serve a fine-tuned student checkpoint over an OpenAI-compatible endpoint.
#
# Same arrangement as serve_teacher.sh and for the same reason: vLLM pins torch
# 2.10 while this project needs 2.11+cu128, so it runs from the rlvr-transfer
# venv and is reached over HTTP.
#
# The student MUST be evaluated over the same transport as the teacher was. The
# frozen v4 numbers came from /chat/completions; scoring the student through the
# in-process transformers path instead would change the prompt rendering and the
# decoding parameters at the same time as the model, leaving nothing attributable.
#
# Accepts a local checkpoint directory or a Hugging Face model id, so the same
# script serves the fine-tuned student and the off-the-shelf control arm.
#
# Usage:  bash scripts/19_serve_student.sh <checkpoint-dir|model-id> [PORT] [GPU] [SERVED_NAME]
set -euo pipefail

CKPT="${1:?usage: 19_serve_student.sh <checkpoint-dir|model-id> [port] [gpu] [served-name]}"
PORT="${2:-8001}"
GPU="${3:-7}"
SERVED_NAME="${4:-sft-student}"

export HF_HOME=/data/huangyanyu/hf_cache
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="$GPU"

VENV=/data/huangyanyu/rlvr-transfer/.venv-rlvr

if [ ! -x "$VENV/bin/vllm" ]; then
  echo "vLLM not found at $VENV/bin/vllm" >&2
  exit 1
fi
# A path must actually hold a checkpoint; a bare model id is left to vLLM and the
# offline cache to resolve.
case "$CKPT" in
  ./*|/*|data/*)
    if [ ! -f "$CKPT/config.json" ]; then
      echo "No config.json in $CKPT — not a checkpoint directory." >&2
      exit 1
    fi
    ;;
esac

# Only the target card has to be free. The teacher occupies GPUs 0-1 and stays up,
# so serve_teacher.sh's "refuse if any GPU is busy" check is wrong here.
USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v g="$GPU" -F', *' '$1==g {print $2}')
if [ "${USED:-0}" -gt 2000 ]; then
  echo "Refusing to start: GPU $GPU already holds ${USED} MiB." >&2
  exit 1
fi

echo "Serving $CKPT on port $PORT (GPU $GPU) as $SERVED_NAME"
exec "$VENV/bin/vllm" serve "$CKPT" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --served-model-name "$SERVED_NAME"
