#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --exclusive


source /home/michael/miniconda3/etc/profile.d/conda.sh
conda activate cascade

source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/env_setup.sh"
python run_cascade_academy.py \
    --max-workers 1 \
    --init-config-json init_config_mof_crystalline_1000K.json \
    --chunk-size 1000 \
    --target-length 5000 \
    --retrain-len 10000000000 \
    --retrain-fraction 1 \
    --n-sample-frames 100 \
    --n-ensemble 4 \
    --learner mace \
    --audit-task uq_threshold \
    --target-ferr 0.8 \
    --audit-random-fail-rate 0.25 \
    --calc-type fairchem \
    --calc-model ../1_ml-potential/uma/uma_omat_ft_mofoff_r2scan.pt \
    --replay-dataset ./datasets/mace-mp/sampled_1000.traj \
    --replay-batch-size 2 \
    --log-level DEBUG \
    --device-train cuda:0 \
    --device-label cuda:0 \
    --device-dyn cuda:0
