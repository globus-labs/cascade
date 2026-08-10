#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --exclusive

source /home/michael/miniconda3/etc/profile.d/conda.sh
conda activate cascade

source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/env_setup.sh"

JOB_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
nvidia-smi --query-gpu=timestamp,utilization.gpu,temperature.gpu,power.draw,memory.used \
    --format=csv -l 5 >> "gpu_mon.${JOB_TAG}.csv" &
GPU_MON_PID=$!
vmstat -SM -t 5 >> "vmstat.${JOB_TAG}.log" &
VMSTAT_PID=$!
# Postgres connection count over time - catches a session/connection leak in
# TrajectoryDB
(while true; do
    echo "$(date +%s) $(ss -tn state established '( dport = :5432 or sport = :5432 )' | wc -l)"
    sleep 5
done) >> "pg_conns.${JOB_TAG}.log" &
PG_CONN_PID=$!

trap 'kill $GPU_MON_PID $VMSTAT_PID $PG_CONN_PID 2>/dev/null' EXIT

python run_reference_dynamics.py \
    --init-config-json init_config_mof_crystalline_200_300K.json \
    --target-length 10000 \
    --log-interval 100 \
    --device cuda:0 \
    --log-level DEBUG \
    --max-workers 1 > "reference_run.${JOB_TAG}.log"

kill $GPU_MON_PID $VMSTAT_PID $PG_CONN_PID 2>/dev/null
