#!/bin/bash
# Submit the 6-run speech-LLM benchmark reproduction (1 node x 8 GPU).
set -euo pipefail
ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative
RESULTS_DIR=$ROOT/SpeechSpec/results/repro
CONTAINER=/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.0724.sqsh
mkdir -p "$RESULTS_DIR"

sbatch \
    --nodes=1 \
    --account=coreai_dlalgo_nemorl \
    --job-name=speechspec-repro \
    --partition=batch \
    --time=4:00:00 \
    --gres=gpu:8 \
    --output="$RESULTS_DIR/slurm-%j.out" \
    --error="$RESULTS_DIR/slurm-%j.err" \
    --wrap="srun --container-image=$CONTAINER --container-mounts=/lustre:/lustre --no-container-mount-home bash $RESULTS_DIR/bench_all_incontainer.sh"
