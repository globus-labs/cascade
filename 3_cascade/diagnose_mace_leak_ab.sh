#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --exclusive

# A/B repro for the run_reference_dynamics.py memory leak, isolated from cascade's
# DB/Parsl code (see diagnose_mace_leak.py). Runs the same MOF structure through:
#   A. mtknpt  - matches production: NPT, cell (and neighbor list) changes every step
#   B. velocity-verlet - control: fixed cell, neighbor list should stay ~constant
# in serial, in the same process lineage would matter for cross-task carryover, but
# since these are two separate `python` invocations (not two Parsl tasks in one
# worker), this specifically isolates the per-step MACE/torch behavior (hypothesis
# #1), not the cross-trajectory worker-carryover hypothesis (#3).

source /home/michael/miniconda3/etc/profile.d/conda.sh
conda activate cascade

STRUCTURE="../MOFs/data/CASCADE_Example_Files_05142026/MOF_Structures/POSCAR_MOF5_crystalline"
STEPS=2000
LOG_INTERVAL=20
JOB_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"

nvidia-smi --query-gpu=timestamp,utilization.gpu,temperature.gpu,power.draw,memory.used \
    --format=csv -l 5 >> "diag_gpu_mon.${JOB_TAG}.csv" &
GPU_MON_PID=$!
vmstat -SM -t 5 >> "diag_vmstat.${JOB_TAG}.log" &
VMSTAT_PID=$!
trap 'kill $GPU_MON_PID $VMSTAT_PID 2>/dev/null' EXIT

echo "=== A: mtknpt (production-matching, cell/neighbor-list changes every step) ==="
python diagnose_mace_leak.py \
    --structure "$STRUCTURE" \
    --dyn-cls mtknpt --temperature-K 200 --pressure-GPa 1.0 --tdamp-fs 100 --pdamp-fs 1000 \
    --calc-model medium --device cuda:0 \
    --steps "$STEPS" --log-interval "$LOG_INTERVAL" \
    --out-csv "diag_mtknpt.${JOB_TAG}.csv" \
    > "diag_mtknpt.${JOB_TAG}.log" 2>&1

echo "=== B: velocity-verlet (control, fixed cell) ==="
python diagnose_mace_leak.py \
    --structure "$STRUCTURE" \
    --dyn-cls velocity-verlet --temperature-K 200 \
    --calc-model medium --device cuda:0 \
    --steps "$STEPS" --log-interval "$LOG_INTERVAL" \
    --out-csv "diag_vv.${JOB_TAG}.csv" \
    > "diag_vv.${JOB_TAG}.log" 2>&1

kill $GPU_MON_PID $VMSTAT_PID 2>/dev/null
echo "Done. Outputs: diag_mtknpt.${JOB_TAG}.csv, diag_vv.${JOB_TAG}.csv, diag_*mon.${JOB_TAG}.*"
