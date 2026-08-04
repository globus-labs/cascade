#strace -f -e trace=process,open,openat,close,socket,connect,accept \
#    -o trace.log \
#    --replay-dataset ./datasets/mace-mp/sampled_1000.traj \
#    --replay-downselect 10 \
#    --replay-batch-size 2 \
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/env_setup.sh"
python run_cascade_academy.py \
    --init-config-json init_config_mof_crystalline_200_300K.json \
    --chunk-size 5 \
    --target-length 10 \
    --retrain-len 10 \
    --retrain-fraction 0.5 \
    --n-sample-frames 5 \
    --accept-rate .5 \
    --learner mace \
    --n-ensemble 2 \
    --log-level DEBUG
