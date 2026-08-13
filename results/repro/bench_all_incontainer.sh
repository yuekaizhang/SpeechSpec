#!/bin/bash
# Runs INSIDE the container on the compute node: 6 benchmark_speech_llm.py runs
# in parallel (one GPU each; every run launches its own sglang server).
set -uo pipefail
ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative
cd $ROOT/SpeechSpec
PY=$ROOT/.specforge_env/bin/python
export HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
OUT=results/repro

# Triton JIT needs Python.h — provided by the uv-managed python3.12
# (ln -sf ~/.local/share/uv/python/cpython-3.12*/bin/python3.12 /usr/bin/python3.12).

run() { # gpu port name extra-args...
  local gpu=$1 port=$2 name=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu $PY benchmark_speech_llm.py \
    --port $port --output-dir $OUT/$name "$@" > $OUT/$name.log 2>&1
  echo "DONE $name rc=$?"
}

OMNI=Qwen/Qwen3-Omni-30B-A3B-Instruct
Q2A=yuekai/qwen2_audio_aishell_sft
run 0 30100 omni_baseline --target-model $OMNI --dataset librispeech &
run 1 30101 omni_eagle3   --target-model $OMNI --dataset librispeech \
    --draft-model yuekai/qwen3_omni_30b_a3b_instruct_eagle3_audio &
run 2 30102 omni_dflash   --target-model $OMNI --dataset librispeech \
    --draft-model yuekai/qwen3_omni_30b_a3b_instruct_dflash_block8 &
run 3 30103 q2a_baseline  --target-model $Q2A --dataset aishell &
run 4 30104 q2a_eagle3    --target-model $Q2A --dataset aishell \
    --draft-model yuekai/qwen2_audio_aishell_sft_eagle3 &
run 5 30105 q2a_dflash    --target-model $Q2A --dataset aishell \
    --draft-model yuekai/qwen2_audio_aishell_sft_dflash_block4 &
wait
echo "=== ALL SUMMARIES ==="
for d in omni_baseline omni_eagle3 omni_dflash q2a_baseline q2a_eagle3 q2a_dflash; do
  echo "--- $d"; cat $OUT/$d/summary.txt 2>/dev/null || echo MISSING
done
echo "REPRO_ALL_DONE"
