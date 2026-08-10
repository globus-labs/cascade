#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --array=0-1
#SBATCH --job-name=diag_mace_leak_ablation

# 2x2 ablation of the two candidate fixes, isolating which one actually matters:
#   fix A: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (allocator config, no code change)
#   fix B: --torch-empty-cache (periodic torch.cuda.empty_cache() every log-interval steps)
#
# The (neither) and (both) corners are already covered by prior runs, so this script
# only fills in the two missing corners:
#   task 0: A only  (expandable_segments, no empty_cache)      -> diag_mtknpt_expseg_only
#   task 1: B only  (empty_cache, no expandable_segments)      -> diag_mtknpt_emptycache_only
# (neither) = diag_mtknpt.328_0.csv from job 328
# (both)    = diag_mtknpt_fixed.330_0.csv from job 330
#
# Only the mtknpt case is run here - the velocity-verlet control never leaked under any
# combination of settings (see job 328/330), so it adds nothing to the ablation.

source /home/michael/miniconda3/etc/profile.d/conda.sh
conda activate cascade

STRUCTURE="../MOFs/data/CASCADE_Example_Files_05142026/MOF_Structures/POSCAR_MOF5_crystalline"
STEPS=2000
LOG_INTERVAL=20
JOB_TAG="${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d-%H%M%S)}_${SLURM_ARRAY_TASK_ID:-0}"

EXTRA_ARGS=()
case "$SLURM_ARRAY_TASK_ID" in
    0)
        CASE_NAME="mtknpt_expseg_only"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        ;;
    1)
        CASE_NAME="mtknpt_emptycache_only"
        EXTRA_ARGS+=(--torch-empty-cache)
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

echo "=== task $SLURM_ARRAY_TASK_ID: $CASE_NAME (PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-unset}, extra_args=${EXTRA_ARGS[*]:-none}) ==="
python diagnose_mace_leak.py \
    --structure "$STRUCTURE" \
    --dyn-cls mtknpt --temperature-K 200 --pressure-GPa 1.0 --tdamp-fs 100 --pdamp-fs 1000 \
    --calc-model medium --device cuda:0 \
    --steps "$STEPS" --log-interval "$LOG_INTERVAL" \
    "${EXTRA_ARGS[@]}" \
    --out-csv "diag_${CASE_NAME}.${JOB_TAG}.csv" \
    > "diag_${CASE_NAME}.${JOB_TAG}.log" 2>&1

kill $GPU_MON_PID $VMSTAT_PID 2>/dev/null
echo "Done. Output: diag_${CASE_NAME}.${JOB_TAG}.csv"
