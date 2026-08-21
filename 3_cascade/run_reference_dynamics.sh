#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --exclusive

source /home/michael/miniconda3/etc/profile.d/conda.sh
conda activate cascade

source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/env_setup.sh"

python run_reference_dynamics.py \
    --init-config-json init_config_mof_crystalline_1000K.json \
    --target-length 10000 \
    --log-interval 100 \
    --calc-type fairchem \
    --calc-model ../1_ml-potential/uma/uma_omat_ft_mofoff_r2scan.pt \
    --device cuda:0 \
    --log-level DEBUG \
    --max-workers 1 > "reference_run.${JOB_TAG}.log"
