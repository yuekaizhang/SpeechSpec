#!/bin/bash
# CosyVoice3 DSpark drafter (3 layers, block 8, 8192 draft vocab) —
# yuekai/cosyvoice3_llm_dspark (accept 2.98 with the vLLM repetition-penalty
# mirror, 1.9/1.9/1.7/1.2x at bs 1/8/16/64 on seed-tts test_zh).
#
# Exact command recovered from the training run's wandb metadata.
# CV3 labels MUST be regenerated with the official sampling params
# (temp 0.8 / top_p 0.95 / top_k 15 / rep_penalty 1.1) — see regen_cv3.py.
set -euo pipefail

SPECULATORS=${SPECULATORS:-/path/to/speculators}   # github.com/vllm-project/speculators
WORK=${WORK:-./cv3_dspark_work}

torchrun --nproc_per_node ${NUM_GPUS:-4} $SPECULATORS/scripts/train.py \
  --verifier-name-or-path yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF \
  --data-path $WORK/training_data_cv3_500k_regen \
  --hidden-states-path $WORK/hidden_states_cv3_500k \
  --save-path $WORK/ckpt_dspark_cv3_500k_regen \
  --speculator-type dspark \
  --block-size 8 \
  --max-anchors 3072 \
  --num-layers 3 \
  --draft-vocab-size 8192 \
  --markov-rank 256 \
  --markov-head-type vanilla \
  --enable-confidence-head \
  --confidence-head-with-markov \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --confidence-head-alpha 1.0 \
  --epochs 10 \
  --lr 3e-4 \
  --scheduler-warmup-ratio 0.02 \
  --total-seq-len 8192 \
  --on-missing raise \
  --logger wandb
