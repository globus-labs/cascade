#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --array=0-1
#SBATCH --job-name=diag_mace_leak

# A/B repro for the run_reference_dynamics.py memory leak, isolated from cascade's
# DB/Parsl code (see diagnose_mace_leak.py). Array version of diagnose_mace_leak_ab.sh -
# runs the two cases as separate array tasks (each with its own exclusive GPU node
# allocation) instead of in serial, so both finish in parallel:
#   task 0: mtknpt          - matches production: NPT, cell/neighbor list changes every step
#   task 1: velocity-verlet - control: fixed cell, neighbor list should stay ~constant
#
# Because these run as two independent Slurm jobs (not two Parsl tasks sharing one
# worker), this isolates the per-step MACE/torch hypothesis, not the cross-trajectory
# worker-carryover hypothesis - see project_reference_dynamics_memory_leak memory.

source /home/michael/miniconda3/etc/profile.d/conda.sh
conda activate cascade

STRUCTURE="../MOFs/data/CASCADE_Example_Files_05142026/MOF_Structures/POSCAR_MOF5_crystalline"
STEPS=2000
LOG_INTERVAL=20
JOB_TAG="${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d-%H%M%S)}_${SLURM_ARRAY_TASK_ID:-0}"

case "$SLURM_ARRAY_TASK_ID" in
    0)
        CASE_NAME="mtknpt"
        DYN_ARGS=(--dyn-cls mtknpt --temperature-K 200 --pressure-GPa 1.0 --tdamp-fs 100 --pdamp-fs 1000)
        ;;
    1)
        CASE_NAME="vv"
        DYN_ARGS=(--dyn-cls velocity-verlet --temperature-K 200)
        ;;
    *)
        echo "Unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
        exit 1
        ;;
esac

nvidia-smi --query-gpu=timestamp,utilization.gpu,temperature.gpu,power.draw,memory.used \
    --format=csv -l 5 >> "diag_gpu_mon.${CASE_NAME}.${JOB_TAG}.csv" &
GPU_MON_PID=$!
vmstat -SM -t 5 >> "diag_vmstat.${CASE_NAME}.${JOB_TAG}.log" &
VMSTAT_PID=$!
trap 'kill $GPU_MON_PID $VMSTAT_PID 2>/dev/null' EXIT

echo "=== task $SLURM_ARRAY_TASK_ID: $CASE_NAME ==="
python diagnose_mace_leak.py \
    --structure "$STRUCTURE" \
    "${DYN_ARGS[@]}" \
    --calc-model medium --device cuda:0 \
    --steps "$STEPS" --log-interval "$LOG_INTERVAL" \
    --out-csv "diag_${CASE_NAME}.${JOB_TAG}.csv" \
    > "diag_${CASE_NAME}.${JOB_TAG}.log" 2>&1

kill $GPU_MON_PID $VMSTAT_PID 2>/dev/null
echo "Done. Output: diag_${CASE_NAME}.${JOB_TAG}.csv"
