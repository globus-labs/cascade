#PBS -A Athena
#PBS -l select=2:ncpus=128:mpiprocs=128
#PBS -l walltime=0:30:00


# Load the conda environment
source activate /lcrc/project/Athena/cascade/env
which python

# Load environment
module load gcc openmpi
module load cp2k
module list

nnodes=`cat $PBS_NODEFILE | sort | uniq | wc -l`
ranks_per_node=16
total_ranks=$(($ranks_per_node * $nnodes))
threads_per_rank=$((128 / ranks_per_node))

echo Running $total_ranks ranks across $nnodes nodes with $threads_per_rank threads per rank

export OMP_NUM_THREADS=$threads_per_rank
export ASE_CP2K_COMMAND="mpirun -N $total_ranks -n $ranks_per_node --cpus-per-proc $threads_per_rank /lcrc/project/Athena/cp2k/exe/local/cp2k_shell.psmp"

# Run the stuff
cd $PBS_O_WORKDIR
papermill 1_relax-initial-geometry.ipynb /dev/null
