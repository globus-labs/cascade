#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --exclusive

source /home/michael/miniconda3/etc/profile.d/conda.sh
conda activate cascade

source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/env_setup.sh"

python evaluate_calculator.py \
    --run-id reference-2026.08.21-19:31:10-43722e \
    --calc-type mace \
    --calc-model small \
    --device cuda:0 \
    --output "eval_mace_small_vs_uma_omat.${SLURM_JOB_ID}.xyz" \
    --log-level DEBUG > "evaluate_calculator.${SLURM_JOB_ID}.log" 2>&1
