#PBS -A Athena
#PBS -l select=1:ncpus=128:mpiprocs=128
#PBS -l walltime=72:00:00
#PBS -q compute


proxima_path=/lcrc/project/Athena/mtynes/cascade/2_proxima/0_run-serial-proxima.py

# Load the conda environment
source activate /home/mtynes/miniconda3/envs/cascade
which python
hostname

# Load environment
module load gcc mpich
module list

nnodes=`cat $PBS_NODEFILE | sort | uniq | wc -l`
ranks_per_node=16
total_ranks=$(($ranks_per_node * $nnodes))
threads_per_rank=$((128 / ranks_per_node))

echo Running $total_ranks ranks across $nnodes nodes with $threads_per_rank threads per rank

export CASCADE_CP2K_TEMPLATE=/lcrc/project/Athena/mtynes/cascade/cascade/files
#export ASE_CP2K_COMMAND="OMP_NUM_THREADS=$threads_per_rank mpiexec -n $total_ranks -ppn $ranks_per_node --bind-to core:$threads_per_rank /lcrc/project/Athena/cp2k-mpich/exe/local/cp2k_shell.psmp"
export ASE_CP2K_COMMAND="/lcrc/project/Athena/cp2k-mpich/exe/local/cp2k_shell.ssmp"

# Run the stuff
cd $PBS_O_WORKDIR
#    --initial-data ../0_setup/md/packmol-uo2_nitrate-waters\=64-seed\=*-blyp-*/md.traj \
#    --initial-model ../1_fit-surrogate/torchani/20240726T140122-cfc18061/model.pt \
#    --initial-model ../1_fit-surrogate/torchani/runs/20240801T003125-f579a90e/model.pt \  # Round 1 model
#    --ensemble nvt \


python $proxima_path \
    --starting-strc ../0_setup/initial-geometries/si-vacancy-2x2x2.vasp \
    --seed $seed \
    --temperature $temp \
    --model-type ani \
    --min-target-frac $frac \
    --stress-tau 25 \
    --steps $steps \
    --retrain-freq $retrain_freq \
    --training-epochs $epochs \
    --online-training \
    --target-error $ferr \
    --ensemble $ens \
    --training-device cpu \
    --calculator lda \
    --n-blending-steps $blend \
    --training-max-size $max_retrain \
    --initial-volume $vol
