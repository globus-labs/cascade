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
    --replay-dataset ./datasets/mace-mp/sampled_1000.traj \
    --replay-batch-size 2 \
    --n-ensemble 4 \
    --log-level DEBUG \
    --device-train cuda:0 \
    --device-label cuda:0 \
    --device-dyn cuda:0
