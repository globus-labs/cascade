"""Compute the forces on ML trajectories
"""
from concurrent.futures import as_completed
import inspect
from pathlib import Path
from tqdm.auto import tqdm
from glob import glob

from parsl.executors import HighThroughputExecutor
from parsl.launchers import SimpleLauncher
from parsl.providers import PBSProProvider
from parsl.config import Config
from parsl import python_app
import parsl

@python_app
def compute_reference_forces(ml_trajectory: str, 
                             method: str = 'blyp'):
    
    from cascade.calculator import make_calculator
    from time import perf_counter
    from ase.io import read
    import numpy as np
    from pathlib import Path
    import shutil
    from tempfile import TemporaryDirectory

    # Make new file in the reference directory
    traj_dir = Path(ml_trajectory).parent
    forces_file = traj_dir / 'forces.npz'

    # read in the ML trajectory
    traj = read(ml_trajectory, index=':')
    n_frames = len(traj)
    frame_shape = traj[0].positions.shape
    # save all forces in a numpy array
    # (there are some issues getting ASE to save forces in a traj file)
    all_forces = np.zeros((n_frames, *frame_shape))
    
    # save DFT logs in temporary directory
    with TemporaryDirectory(dir='cp2k-run') as tmpdir:
        calc = make_calculator(method, 
                               directory=tmpdir)
        
        # compute forces for each frame
        start = perf_counter()
        for i, atoms in enumerate(traj):
            atoms.calc = calc
            forces = atoms.get_forces()
            all_forces[i, :, :] = forces
        elapsed = perf_counter() - start
    np.savez(str(forces_file), all_forces)
    return forces_file, elapsed
        
if __name__ == '__main__':


    # handle arguments
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Equilibrate systems using CP2K. One MD run per node. Within a node, cp2k uses MPI+OMP parallelism."
    )
    parser.add_argument('--num_parallel', 
                        type=int, 
                        required=True,
                        help='Number of nodes to ask for (one MD run per node)')
    parser.add_argument('--method',
                        type=str, 
                        default='blyp', 
                        help='Level of chemical theory, passed to cascade.calculator.make_calculator')
    parser.add_argument('--ranks_per_node',
                        type=int, 
                        default=16, 
                        help='Number of MPI ranks for a cp2k calculation')
    args = parser.parse_args()

    # Make the parsl configurations
    config = Config(
    retries=1,
    executors=[
        HighThroughputExecutor(
            #enable_mpi_mode=True,
            max_workers_per_node=1,
            #cores_per_worker=1e-6,
            provider=PBSProProvider(
                    account="Athena",
                    select_options="mpiprocs=128",
                    worker_init=inspect.cleandoc(f"""
                        module reset
                        # set up conda                         
                        source activate /lcrc/project/Athena/cascade/env
                        which python
                        cd $PBS_O_WORKDIR
                        pwd

                        # Load environment
                        module load gcc mpich
                        module list

                        nnodes=`cat $PBS_NODEFILE | sort | uniq | wc -l`
                        ranks_per_node={args.ranks_per_node}
                        total_ranks=$(($ranks_per_node * $nnodes))
                        threads_per_rank=$((128 / ranks_per_node))

                        echo Running $total_ranks ranks across $nnodes nodes with $threads_per_rank threads per rank

                        export OMP_NUM_THREADS=$threads_per_rank
                        export ASE_CP2K_COMMAND="mpiexec -n $total_ranks -ppn $ranks_per_node --bind-to numa /lcrc/project/Athena/cp2k-mpich/exe/local/cp2k_shell.psmp"
                        """),
                    walltime="02:00:00",
                    queue="compute",
                    launcher=SimpleLauncher(),#SingleNodeLauncher(),
                    nodes_per_block=1,
                    #cpus_per_node=128,
                    init_blocks=args.num_parallel,
                    min_blocks=args.num_parallel,
                    max_blocks=args.num_parallel,
                ),
            ),
        ]
    )
    parsl.load(config)

    input_files = list(sorted(glob('md/packmol-CH4-in-H2O=32-seed=*finetuned/md.traj')))
    futures = []
    for input_file in input_files:
        futures.append(
            compute_reference_forces(
                input_file, 
                method=args.method)
        )
    print('running md on:')
    print(input_files)
    for future in tqdm(as_completed(futures), 
                       total=len(futures),
                       desc='Computing reference forces for trajectories'
                       ):
        exc = future.exception()
        if exc is not None: 
            print(exc)
            continue
        name, elapsed = future.result()
        print(f'Finished calculation for {name}\nIn {elapsed:0.3f}s')
