#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --exclusive

# Backfill ground-truth force error for all passed chunks of job 890's completed run,
# using the same fairchem/UMA reference calculator that run used live.

source /home/michael/miniconda3/etc/profile.d/conda.sh
conda activate cascade

source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/../env_setup.sh"
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/.."
python scripts/evaluate_passed_chunk_error.py \
    --run-id 2026.09.03-22:03:35-e79143 \
    --calc-type fairchem \
    --calc-model ../1_ml-potential/uma/uma_omat_ft_mofoff_r2scan.pt \
    --n-samples-per-chunk 15 \
    --device cuda:0 \
    --log-level INFO
