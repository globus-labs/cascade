#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --exclusive
source /home/michael/miniconda3/etc/profile.d/conda.sh
conda activate cascade
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/env_setup.sh"
python scripts/benchmark_trainer.py \
    --run-id reference-2026.08.07-17:51:00-bd260f \
    --max-frames 1000 \
    --base-model small \
    --batch-sizes 2 \
    --replay-batch-size 2 \
    --num-epochs 200 \
    --patience 15 \
    --device cuda:0 \
    --replay-dataset ./datasets/mace-mp/sampled_1000.traj \
    --out-dir benchmark_trainer_out_batch_small_200

# python scripts/benchmark_trainer.py \
#     --run-id reference-2026.08.07-17:51:00-bd260f \
#     --max-frames 1000 \
#     --base-model small \
#     --batch-sizes 2 \
#     --replay-batch-size 2,4,8,16,32,64,128 \
#     --num-epochs 2 \
#     --patience 10 \
#     --device cuda:0 \
#     --replay-dataset ./datasets/mace-mp/sampled_1000.traj \
#     --out-dir benchmark_trainer_out_replay_small
