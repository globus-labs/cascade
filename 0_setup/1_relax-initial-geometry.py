
# # Relax the Initial Geometries
# Produce a relaxed structure based on one of the initial geometries

from concurrent.futures import as_completed

from tempfile import TemporaryDirectory
from contextlib import chdir

from parsl.addresses import address_by_hostname
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher, SimpleLauncher
from parsl.providers import PBSProProvider, LocalProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl import python_app
import parsl


def relax_geometry_cp2k(initial_geometry: str, 
                        method: str = 'b3lyp',
                        persist_cp2k_logs: bool = False
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
    if persist_cp2k_logs:
        cp2k_dir = Path(f'cp2k-run/{name}')
        if cp2k_dir.exists():
            (cp2k_dir / 'cp2k.out').write_text('')
    else: 
        pass # todo add tempfiles before merge

    atoms.calc = make_calculator(method, 
                                 directory=cp2k_dir,
                                 )
    init_vol = atoms.get_volume()
    ecf = UnitCellFilter(atoms, hydrostatic_strain=True)
    with Trajectory(str(relax_traj), mode='a') as traj:
        dyn = LBFGS(ecf, 
                    logfile=str(run_dir / 'relax.log'),
                    trajectory=traj)
        dyn.run(fmax=0.1)
    final_vol = atoms.get_volume()


    Path('final-geometries').mkdir(exist_ok=True)
    atoms.write(f'final-geometries/{name}.vasp')

if __name__ == '__main__':
    # set up possible parsl configs
    parsl_configs = {}

    parsl_configs['improv'] = Config(
        retries=1,
        executors=[
            HighThroughputExecutor(
                address=address_by_hostname(),
                prefetch_capacity=0,  # Increase if you have many more tasks than workers
                start_method="fork",  # Needed to avoid interactions between MPI and os.fork
                max_workers=1,
                provider=PBSProProvider(
                    account="Athena",
                    worker_init=f"""
module reset
# set up conda
module load miniconda3
conda activate /lcrc/project/Athena/cascade/env
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
cd $PBS_O_WORKDIR""",
                    walltime="1:00:00",
                    queue="debug",
                    scheduler_options="#PBS -l filesystems=home:eagle:grand",
                    launcher=SimpleLauncher(),
                    nodes_per_block=1,
                    min_blocks=0,
                    max_blocks=1,
                ),
            ),
        ]
    )
    parsl_configs['local'] = Config(
        executors=[
            HighThroughputExecutor(
                label="htex_Local",
                worker_debug=True,
                cores_per_worker=1,
                provider=LocalProvider(
                    channel=LocalChannel(),
                    init_blocks=1,
                    max_blocks=1,
                ),
            )
        ],
        strategy=None,
    )

    # handle arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input_files', 
                        type=str,
                        nargs='+', 
                        help='VASP files holding initial geometries')
    parser.add_argument('--method',
                        type=str, 
                        default='b3lyp', 
                        help='Level of chemical theory, passed to cascade.calculator.make_calculator')
    parser.add_argument('--persist_cp2k_logs',
                        type=int, 
                        default=0, 
                        choices=[0,1],
                        help='1 to persist cp2k logs in their own directory, 0 to use a temp directory')
    parser.add_argument('--parsl_config', 
                        type=str, 
                        required=True,
                        options=list(parsl_configs.keys())
                        )
    args = parser.parse_args()

    # Make the parsl configurations

    parsl.load(parsl_configs[args.parsl_config])

    futures = []
    for input_file in args.input_files:
        futures.append(
            relax_geometry_cp2k(input_file, 
                                method=args.method, 
                                persist_cp2k_logs=args.persist_cp2k_logs)
        )
