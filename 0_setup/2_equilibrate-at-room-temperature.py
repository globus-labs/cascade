<<<<<<< HEAD
# # Run NPT MD 
# Run a structure to see if it equilibrates

from cascade.calculator import make_calculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.npt import NPT
from ase.io import read
from ase import units
from pathlib import Path
import shutil

method: str = 'blyp'
temperature = 298
initial_geometry = 'final-geometries/packmol-CH4-in-H2O=32-seed=0-blyp.vasp'
steps: int = 512


# Derived
name = f'{Path(initial_geometry).name[:-5]}-npt={temperature}'
run_dir = Path('md') / name

run_dir.mkdir(exist_ok=True, parents=True)
# ## Perform the Dynamics
# Run a set number of MD steps
traj_file = run_dir / 'md.traj'
if traj_file.is_file() and traj_file.stat().st_size > 0:
    traj = read(str(traj_file), slice(None))
    start = len(traj)
    atoms = traj[-1]
    print('Loaded last structure')
else:
    atoms = read(initial_geometry)
    start = 0
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature * 2)


# Relax it
atoms.calc = make_calculator(method, 
                             directory='cp2k-run',
                            )

dyn = NPT(atoms,
          timestep=0.5 * units.fs,
          temperature_K=temperature,
          ttime=100 * units.fs,
          pfactor=0.01,
          externalstress=0,
          logfile=str(run_dir / 'md.log'),
          trajectory=str(traj_file),
          append_trajectory=True)
dyn.run(512 - start)

=======
"""Equilibrate the relaxed geometries with cp2k
"""
from concurrent.futures import as_completed
import inspect
import os
from pathlib import Path
from tqdm.auto import tqdm

from parsl.executors import HighThroughputExecutor
from parsl.launchers import SimpleLauncher
from parsl.providers import PBSProProvider
from parsl.config import Config
from parsl import python_app
import parsl

@python_app
def equilibrate_cp2k(initial_geometry: str, 
                     method: str = 'blyp', 
                     temperature: float|int = 298,
                     steps: int = 512):
    # # Run NPT MD 
    # Run a structure to see if it equilibrates

    from cascade.calculator import make_calculator
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.npt import NPT
    from ase.io import read
    from ase import units
    from pathlib import Path
    import shutil
    from tempfile import TemporaryDirectory

    # Set up MD trajectory directory and file
    # Derived
    name = f'{Path(initial_geometry).name[:-5]}-npt={temperature}'
    run_dir = Path('md') / name
    run_dir.mkdir(exist_ok=True, parents=True)
    # ## Perform the Dynamics
    # Run a set number of MD steps
    traj_file = run_dir / 'md.traj'
    if traj_file.is_file() and traj_file.stat().st_size > 0:
        traj = read(str(traj_file), slice(None))
        start = len(traj)
        atoms = traj[-1]
        print('Loaded last structure')
    else:
        atoms = read(initial_geometry)
        start = 0
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature * 2)

    # save DFT logs in temporary directory
    with TemporaryDirectory(dir='cp2k-run') as tmpdir:
        # set up DFT run directory and calculator
        atoms.calc = make_calculator(method, 
                                     directory=tmpdir,
                                     )

        # set up dynamics and run
        dyn = NPT(atoms,
                timestep=0.5 * units.fs,
                temperature_K=temperature,
                ttime=100 * units.fs,
                pfactor=0.01,
                externalstress=0,
                logfile=str(run_dir / 'md.log'),
                trajectory=str(traj_file),
                append_trajectory=True)
        dyn.run(steps - start)


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
    parser.add_argument('--input_dir',
                        type=str, 
                        default='final-geometries',
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

    input_dir = args.input_dir
    input_files = os.listdir(input_dir)
    futures = []
    for input_file in input_files:
        futures.append(
            equilibrate_cp2k(Path(input_dir) / input_file, 
                             method=args.method)
        )
    print('running md on:')
    print(input_files)
    for future in tqdm(as_completed(futures), 
                       total=len(futures),
                       desc='Completed equilibrations'
                       ):
        exc = future.exception()
        if exc is not None: 
            print(exc)
            continue
        name, elapsed = future.result()
        print(f'Finished calculation for {name}\nIn {elapsed:0.3f}s')
>>>>>>> 3810f47ac74ab3bed789098a27b222a11fd26b73
