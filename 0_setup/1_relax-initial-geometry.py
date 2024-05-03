
# # Relax the Initial Geometries
# Produce a relaxed structure based on one of the initial geometries

from concurrent.futures import as_completed
import inspect
import os
from pathlib import Path
from tqdm.auto import tqdm
import time


from parsl.addresses import address_by_hostname
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher, SimpleLauncher, SingleNodeLauncher
from parsl.providers import PBSProProvider, LocalProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl import python_app
import parsl


@python_app
def relax_geometry_cp2k(initial_geometry: str, 
                        method: str = 'blyp',
                        ):
    
    from pathlib import Path
    from ase.filters import UnitCellFilter
    from ase.io.trajectory import Trajectory
    from ase.optimize import LBFGS
    from ase.io import read
    from cascade.calculator import make_calculator


    # make a directory to store the relaxation trajectory and logs
    name = f'{Path(initial_geometry).name[:-5]}-{method}'
    run_dir = Path('relax') / name
    run_dir.mkdir(exist_ok=True, parents=True)

    # Perform the Relaxation
    ## Either from the initial geometry or the latest relaxation step from the trajectory
    relax_traj = run_dir / 'relax.traj'
    if relax_traj.is_file() and relax_traj.stat().st_size > 75:
        atoms = read(str(relax_traj), index=-1)
        print('Loaded last structure')
    else:
        atoms = read(initial_geometry)


    # create directory for cp2k logs
    cp2k_dir = Path(f'cp2k-run/{name}')
    if cp2k_dir.exists():
        (cp2k_dir / 'cp2k.out').write_text('')

    atoms.calc = make_calculator(method, 
                                 directory=cp2k_dir,
                                 )
    init_vol = atoms.get_volume()
    ecf = UnitCellFilter(atoms, hydrostatic_strain=True)
    with Trajectory(str(relax_traj), mode='a') as traj:
        dyn = LBFGS(ecf, 
                    logfile=str(run_dir / 'relax.log'),
                    trajectory=traj)
        #run the dynamics timed    
        start = time.perf_counter()
        dyn.run(fmax=0.1)
        elapsed = time.perf_counter() - start
    
    final_vol = atoms.get_volume()


    Path('final-geometries').mkdir(exist_ok=True)
    atoms.write(f'final-geometries/{name}.vasp')
    return name, elapsed

if __name__ == '__main__':


    # handle arguments
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Relax geometries using CP2K. One relaxation per node. Within a node, cp2k uses MPI+OMP parallelism."
    )
    parser.add_argument('--num_parallel', 
                        type=int, 
                        required=True,
                        help='Number of nodes to ask for (one cp2k relaxation per node)')
    parser.add_argument('--method',
                        type=str, 
                        default='blyp', 
                        help='Level of chemical theory, passed to cascade.calculator.make_calculator')
    parser.add_argument('--ranks_per_node',
                        type=int, 
                        default=16, 
                        help='Number of MPI ranks for a cp2k calculation')
    parser.add_argument('--input_dir',
                        type=str, 
                        default='initial-geometries',
                        required=False)
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
                        module load gcc openmpi
                        module load cp2k
                        module list

                        nnodes=`cat $PBS_NODEFILE | sort | uniq | wc -l`
                        ranks_per_node={args.ranks_per_node}
                        total_ranks=$(($ranks_per_node * $nnodes))
                        threads_per_rank=$((128 / ranks_per_node))

                        echo Running $total_ranks ranks across $nnodes nodes with $threads_per_rank threads per rank

                        export OMP_NUM_THREADS=$threads_per_rank
                        export ASE_CP2K_COMMAND="mpirun -N $total_ranks -n $ranks_per_node --cpus-per-proc $threads_per_rank /lcrc/project/Athena/cp2k/exe/local/cp2k_shell.psmp"
                        """),
                    walltime="01:00:00",
                    queue="debug",
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

    input_dir = args.input_dir
    input_files = os.listdir(input_dir)
    futures = []
    for input_file in input_files:
        futures.append(
            relax_geometry_cp2k(Path(input_dir) / input_file, 
                                method=args.method)
        )

    for future in tqdm(as_completed(futures), 
                       total=len(futures),
                       desc='Completed relaxations'
                       ):
        exc = future.exception()
        if exc is not None: 
            print(exc)
            continue
        name, elapsed = future.result()
        print(f'Finished calculation for {name}\nIn {elapsed:0.3f}s')
