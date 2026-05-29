#!/bin/bash
#SBATCH --job-name=luna
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

source ~/.bashrc
conda activate watermark

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1

DATASET="wiki"
TEMPERATURE="0.7"
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dataset)     DATASET="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        *)             EXTRA_ARGS+=("$1"); shift ;;
    esac
done

RESULTS_ROOT="results/main_${DATASET}_T${TEMPERATURE}"

date

python -u scripts/main_experiment/run_main_chunk.py \
    --pairs \
        "LUNA en" \
        "LUNA de" \
        "LUNA ar" \
        "LUNA zh" \
        "LUNA ja" \
        "LUNA ko" \
    --dataset "${DATASET}" \
    --temperature "${TEMPERATURE}" \
    --num-samples 500 \
    --results-root "${RESULTS_ROOT}" \
    "${EXTRA_ARGS[@]}"

date
rm -rf __pycache__