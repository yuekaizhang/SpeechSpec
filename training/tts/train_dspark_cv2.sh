#!/bin/bash
# CosyVoice2 DSpark drafter (3 layers, block 8, 8192 draft vocab) — the
# recommended CV2 deployment checkpoint (yuekai/cosyvoice2_llm_dspark,
# accept 3.26, 2.1-3.1x on seed-tts test_zh).
#
# Prereqs (see ../README.md):
#   1. regen_cv2.py — regenerate speech-token labels on-policy with the target
#      under deployment sampling params -> train_conversations_regen.jsonl
#   2. speculators' data prep produces training_data/ + hidden_states/
#      (train.py generates hidden states on the fly with --on-missing generate
#       and a --vllm-endpoint, or reuse a precomputed --hidden-states-path).
# 4xH100, DP4, ~10 epochs.
set -euo pipefail

SPECULATORS=${SPECULATORS:-/path/to/speculators}   # github.com/vllm-project/speculators
WORK=${WORK:-./cv2_dspark_work}

torchrun --nproc_per_node ${NUM_GPUS:-4} $SPECULATORS/scripts/train.py \
  --verifier-name-or-path yuekai/cosyvoice2_llm \
  --data-path $WORK/training_data_50k_regen \
  --hidden-states-path $WORK/hidden_states_50k_regen \
  --save-path $WORK/ckpt_dspark_50k_regen \
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
