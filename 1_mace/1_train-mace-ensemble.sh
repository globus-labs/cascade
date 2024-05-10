#!/usr/bin/env bash

for seed in {0,}; do
    echo "training replica $seed"
    time python 1_train-mace-replica.py \
    --seed $seed \
    --train-files ../0_setup/md/packmol-CH4-in-H2O\=32-seed={0..2}*/md.traj \
    |tee mlrun.log 2>&1
done
