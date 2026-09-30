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
    --chunk-size 200 \
    --target-length 600 \
    --retrain-len 10000000000 \
    --retrain-fraction 1 \
    --n-sample-frames 100 \
    --n-ensemble 4 \
    --learner mace \
    --init-weights-paths /home/michael/repos/cascade/3_cascade/pretrained_mace_ensemble/member0_weights.pt,/home/michael/repos/cascade/3_cascade/pretrained_mace_ensemble/member1_weights.pt,/home/michael/repos/cascade/3_cascade/pretrained_mace_ensemble/member2_weights.pt,/home/michael/repos/cascade/3_cascade/pretrained_mace_ensemble/member3_weights.pt \
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
