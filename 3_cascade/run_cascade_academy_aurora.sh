#!/bin/bash
# Run cascade from an Aurora login node (e.g. inside screen); Parsl submits a
# debug-queue PBS job and runs the tasks there, one worker per GPU tile.
# Start the database first with pg/pg-serve.sh.
set -eo pipefail  # no -u: lmod references unset variables

cd "$(dirname "${BASH_SOURCE[0]}")"
REPO=$(cd .. && pwd)
PGROOT=${PGROOT:-$(dirname "$REPO")/cascade-pg}

module load frameworks
source "$REPO/venv/bin/activate"
export CASCADE_DB_URL=$(cat "$PGROOT/db_url")

# TODO: point at the smoke-test initial configs
python run_cascade_academy.py \
    --parsl-config aurora \
    --queue debug \
    --walltime 0:30:00 \
    --nodes 1 \
    --init-config-json smoke_test_init_config.json \
    --chunk-size 5 \
    --target-length 10 \
    --retrain-len 10 \
    --retrain-fraction 0.5 \
    --n-sample-frames 5 \
    --accept-rate .5 \
    --learner mace \
    --n-ensemble 4 \
    --num-epochs 2 \
    --log-level DEBUG \
    --device-train xpu \
    --device-label xpu \
    --device-dyn xpu
