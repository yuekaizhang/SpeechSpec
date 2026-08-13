#!/bin/bash
# CosyVoice2 DFlash drafter (5 layers, block 4, 8192 draft vocab) —
# yuekai/cosyvoice2_llm_dflash (accept 2.81, best at large batch: 22.4k tok/s
# at bs64 on seed-tts test_zh).
set -euo pipefail

SPECULATORS=${SPECULATORS:-/path/to/speculators}   # github.com/vllm-project/speculators
WORK=${WORK:-./cv2_dflash_work}

torchrun --nproc_per_node ${NUM_GPUS:-4} $SPECULATORS/scripts/train.py \
  --verifier-name-or-path yuekai/cosyvoice2_llm \
  --data-path $WORK/training_data_500k_regen \
  --hidden-states-path $WORK/hidden_states_500k_regen \
  --save-path $WORK/ckpt_dflash_500k_regen \
  --speculator-type dflash \
  --block-size 4 \
  --max-anchors 3072 \
  --num-layers 5 \
  --draft-vocab-size 8192 \
  --epochs 5 \
  --lr 3e-4 \
  --scheduler-warmup-ratio 0.02 \
  --total-seq-len 8192 \
  --on-missing raise \
  --logger wandb
