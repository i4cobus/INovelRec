#!/usr/bin/env bash
# GRPO on the SFT reranker, with a rule-verifiable constraint reward.
#
# Runs from /data/huangyanyu/.venv-verl, not this project's venv — verl brings its own
# torch/vLLM stack and this project is pinned to torch 2.11+cu128 by the host driver.
# The reward function is reached over PYTHONPATH instead of being installed: every
# module on the path from src/verl_reward.py imports only the standard library at
# module level, so verl's interpreter needs nothing from here.
#
# Usage:  bash scripts/22_grpo_train.sh [N_GPUS] [EXPERIMENT_NAME]
set -euo pipefail

# 8, not 6. verl requires ppo_mini_batch_size % (n_gpus * micro_batch_per_gpu) == 0,
# and 32 % (6 * 4) is not an integer — six cards fail the assertion outright. Four
# and eight both divide; eight is chosen because every card is free, the actor shard
# drops from 16 GB to 8 GB per GPU (real headroom on a first run whose memory profile
# is still unmeasured), and a diagnostic run should return its verdict sooner.
N_GPUS="${1:-8}"
EXPERIMENT="${2:-grpo-constraint-v1}"
# A step budget rather than an epoch: one epoch over 13,468 episodes is ~210 steps,
# and the first run exists to read the monitoring curves, not to converge.
MAX_STEPS="${3:-50}"
# Consume them, so the trailing "$@" forwards only genuine hydra overrides. Left in
# place, hydra tries to parse "8" as an override and dies on LexerNoViableAltException.
shift $(( $# < 3 ? $# : 3 ))
PROJECT=/data/huangyanyu/INovelRec
VENV=/data/huangyanyu/.venv-verl

export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data/huangyanyu/hf_cache
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Ray inherits the environment and would otherwise route worker traffic through the
# host proxy, which closes connections to 127.0.0.1 — the same trap documented for
# the local vLLM endpoint.
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="$no_proxy"

# flash-attn is not installable on this host and both settings below follow from that.
#
# verl defaults the actor to flash_attention_2, and use_remove_padding routes through
# verl/utils/attention_utils.py, which imports flash_attn.bert_padding on CUDA with no
# fallback. There is no prebuilt flash-attn wheel for torch 2.9 or 2.10 in any upstream
# release, and building from source needs a CUDA 12.x toolkit to match torch's cu128 —
# this host only has nvcc 11.7 and no root.
#
# So: SDPA attention, and padding kept. The cost is real (padded tokens get computed,
# and response lengths here vary from a 74-token median to a 512-token tail), but it is
# throughput, not correctness. Flip both back the day a matching wheel exists.
ATTN="${ATTN:-sdpa}"
REMOVE_PADDING="${REMOVE_PADDING:-False}"

cd "$PROJECT"

if [ ! -f data/processed/verl/train.parquet ]; then
  echo "Missing data/processed/verl/train.parquet — run 20_build_grpo_data.py then 21_export_verl_dataset.py" >&2
  exit 1
fi

# Rollout is cheap here and training is not: measured 30.5 generations/s per A100 at
# realistic batch, so 512 generations cost ~17 GPU-seconds while the policy and
# reference forward/backward dominate the step. Group size 8 is the usual GRPO
# setting and, measured on the SFT checkpoint, already yields 0/64 degenerate groups
# — every group has reward variance, so every group contributes gradient.
"$VENV/bin/python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=data/processed/verl/train.parquet \
  data.val_files=data/processed/verl/val.parquet \
  data.train_batch_size=64 \
  data.max_prompt_length=1536 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=data/checkpoints/sft-qwen3-4b \
  +actor_rollout_ref.model.override_config.attn_implementation="$ATTN" \
  actor_rollout_ref.model.use_remove_padding="$REMOVE_PADDING" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.custom_reward_function.path="$PROJECT/src/verl_reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  trainer.logger=['console','tensorboard'] \
  trainer.project_name=inovelrec-grpo \
  trainer.experiment_name="$EXPERIMENT" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=25 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$MAX_STEPS" \
  trainer.default_local_dir="data/checkpoints/$EXPERIMENT" \
  "$@"
