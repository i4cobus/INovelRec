#!/usr/bin/env bash
# Serve the Qwen3-32B teacher over an OpenAI-compatible endpoint.
#
# vLLM runs from /data/huangyanyu/rlvr-transfer/.venv-rlvr, NOT this project's
# .venv. vLLM pins torch 2.10 while this project needs 2.11+cu128 for the host
# driver, so the two cannot share an environment. Keeping the model behind HTTP
# is what lets them coexist: src/http_matcher.py only speaks /chat/completions.
#
# Usage:  bash scripts/serve_teacher.sh [PORT] [TP]
# Then:   uv run python scripts/05_recommend_demo.py "<query>" \
#             --backend http --llm-base-url http://127.0.0.1:8000/v1 \
#             --llm-model Qwen/Qwen3-32B

set -euo pipefail

PORT="${1:-8000}"
TP="${2:-2}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"

export HF_HOME=/data/huangyanyu/hf_cache
export HF_HUB_OFFLINE=1          # weights are local; a HEAD request to the Hub must not gate startup
export VLLM_WORKER_MULTIPROC_METHOD=spawn

VENV=/data/huangyanyu/rlvr-transfer/.venv-rlvr

if [ ! -x "$VENV/bin/vllm" ]; then
  echo "vLLM not found at $VENV/bin/vllm" >&2
  exit 1
fi

# Refuse to start on top of a job that is already using the GPUs. An index build
# holds ~70 GB per card; launching here would OOM both processes.
BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
if [ "$BUSY" -gt 0 ]; then
  echo "Refusing to start: $BUSY GPU(s) already in use. Wait for the running job." >&2
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >&2
  exit 1
fi

echo "Serving $MODEL on port $PORT with tensor-parallel-size $TP"
exec "$VENV/bin/vllm" serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --served-model-name "$MODEL"
