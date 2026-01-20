#strace -f -e trace=process,open,openat,close,socket,connect,accept \
#    -o trace.log \
python run_cascade_academy.py \
    --initial-structures \
        ../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp \
        ../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp \
    --chunk-size 5 \
    --target-length 15 \
    --retrain-len 10 \
    --retrain-fraction 0.75 \
    --n-sample-frames 5 \
    --accept-rate .5 \
    --learner mace \
    --calc mace \
    --dyn-cls velocity-verlet \
    --dt_fs 1.0 \
    --loginterval 1
