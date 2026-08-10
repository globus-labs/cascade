#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --array=0-1
#SBATCH --job-name=diag_mace_leak_fixed

# Same A/B repro as diagnose_mace_leak_array.sh, with the two candidate fixes for the
# CUDA-allocator leak (see project_reference_dynamics_memory_leak memory) applied:
#   fix 1: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True - reduces allocator
#          fragmentation from ever-changing tensor shapes (env var, no code change)
#   fix 2: --torch-empty-cache - periodically calls torch.cuda.empty_cache() so any
#          reserved-but-unused blocks get released back to the (unified) memory pool
#
#   task 0: mtknpt_fixed - same NPT settings as the original leaking run, with fixes on
#   task 1: vv_fixed      - control, with fixes on (sanity check they don't break it)
#
# Outputs are named *_fixed so they sit alongside the original diag_mtknpt.328_0.csv /
# diag_vv.328_1.csv without overwriting them - the notebook plots both side by side.

source /home/michael/miniconda3/etc/profile.d/conda.sh
conda activate cascade

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STRUCTURE="../MOFs/data/CASCADE_Example_Files_05142026/MOF_Structures/POSCAR_MOF5_crystalline"
STEPS=2000
LOG_INTERVAL=20
JOB_TAG="${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d-%H%M%S)}_${SLURM_ARRAY_TASK_ID:-0}"

case "$SLURM_ARRAY_TASK_ID" in
    0)
        CASE_NAME="mtknpt_fixed"
        DYN_ARGS=(--dyn-cls mtknpt --temperature-K 200 --pressure-GPa 1.0 --tdamp-fs 100 --pdamp-fs 1000)
        ;;
    1)
        CASE_NAME="vv_fixed"
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

echo "=== task $SLURM_ARRAY_TASK_ID: $CASE_NAME (PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF) ==="
python diagnose_mace_leak.py \
    --structure "$STRUCTURE" \
    "${DYN_ARGS[@]}" \
    --calc-model medium --device cuda:0 \
    --steps "$STEPS" --log-interval "$LOG_INTERVAL" \
    --torch-empty-cache \
    --out-csv "diag_${CASE_NAME}.${JOB_TAG}.csv" \
    > "diag_${CASE_NAME}.${JOB_TAG}.log" 2>&1

kill $GPU_MON_PID $VMSTAT_PID 2>/dev/null
echo "Done. Output: diag_${CASE_NAME}.${JOB_TAG}.csv"
