python run_cascade_academy.py \
    --initial-structures \
        ../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp \
        ../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp \
    --chunk-size 10 \
    --target-length 10 \
    --retrain-len 100000 \
    --n-sample-frames 1 \
    --accept-rate 1.0 \
    --learner mace \
    --calc mace \
    --dyn-cls velocity-verlet \
    --dt_fs 1.0 \
    --loginterval 1