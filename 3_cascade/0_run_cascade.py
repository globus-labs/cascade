import argparse
from pathlib import Path
import logging
import sys
from typing import Sequence
from queue import Queue, Empty
from functools import partial, update_wrapper

from colmena.models import Result
from colmena.queue import PipeQueues, ColmenaQueues
from colmena.task_server.local import LocalTaskServer
from colmena.thinker import BaseThinker, ResourceCounter, task_submitter, result_processor
import ase
from ase import Atoms
from ase.io import read, write
from ase.optimize.optimize import Dynamics
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.filters import Filter
import numpy as np
from glob import glob
from mace.calculators import mace_mp

from cascade.learning.base import BaseLearnableForcefield
from cascade.learning.mace import MACEInterface


def advance_dynamics(
    traj_dir: str | Path,
    learner: BaseLearnableForcefield,
    model_msg: bytes,
    name: str,
    steps: int,
    starting_step: int,
    write_freq: int,
    dyn_class: type[Dynamics],
    dyn_kwargs: dict[str, object],
    run_kwargs: dict[str, object] = None,
    # dyn_filter: tuple[type[Filter], dict[str, object]] | None = None,
    # callbacks: Sequence[tuple[type, int, dict[str, object]]] = (),
    # repeat: bool = False
) -> tuple[list[ase.Atoms], int]:
    """Advance an MD trajectory by n_steps
    """

    # read in starting structure
    if starting_step == 0:
        # if we are starting, just read the initial structure
        atoms_file = traj_dir/'init_strc.traj'
        chunk_i_last = 0
    else:
        # otherwise: read the trajectory from the last chunk
        chunks = sorted(a, key=lambda s: int(s.split('_')[-1]))
        chunk_last = chunks[-1]
        chunk_i_last = int(chunk_last.split('_'))
        atoms_file = traj_dir / chunk_last / 'md.traj'
    atoms = read(atoms_file, index=-1)
    
    # set up calculator
    calc = learner.make_calculator(model_msg, device='cpu')
    atoms.calc = calc
    
    # create new dir for this chunk
    chunk_i = chunk_i_last + 1
    chunk_current = traj_dir / f'chunk_{chunk_i}'
    chunk_current.mkdir(parents=True, exist_ok=True)
    
    # set up dynamics
    dyn = dyn_class(
        atoms,
        traj_file=chunk_current/'md.traj',
        **dyn_kwargs
    )

    # run dynamics
    dyn.run(steps, **run_kwargs)
    return


class Thinker(BaseThinker):

    def __init__(
        self,
        initial_frames: list[Atoms],
        queues: ColmenaQueues,
        run_dir: Path,
        advance_steps: int,
        total_steps: int,
        num_workers: int
    ):
        """thinker
        """
        super().__init__(queues, resource_counter=ResourceCounter(num_workers))
        self.run_dir = run_dir
        self.initial_frames = initial_frames
        self.advance_steps = advance_steps
        self.total_steps = total_steps
        self.n_traj = len(self.initial_frames)
        self.traj_progress = np.zeros(self.n_traj)
        self.traj_avail = Queue(self.n_traj)
        # populate the queue with all trajectories
        for i in range(self.n_traj):
            self.traj_avail.put(i)

        # set up initial directory structure with initial_frames
        self.traj_dirs = []
        for i, a in enumerate(initial_frames):
            traj_dir = self.run_dir / f'traj_{i}'
            traj_dir.mkdir(exist_ok=True)
            init_file = traj_dir/'init_strc.traj'
            write(str(init_file), a)
            self.traj_dirs.append(traj_dir)

    @task_submitter
    def submit_dynamics(self):
        while True:
            try:
                traj_id = self.traj_avail.get(block=True, timeout=1)
            except Empty:
                if self.done.is_set():
                    return
            else:
                break
        self.logger.info(f'Submitting trajecotry {traj_id} for advancement')
        self.queues.send_inputs(
            method='advance_dynamics_partial',
            input_kwargs=dict(
                traj_dir=self.traj_dirs[traj_id],
                starting_step=self.traj_progress[traj_id],
                steps=self.advance_steps,
            ),
            task_info=dict(
                traj_id=traj_id,
                steps=self.advance_steps
            )
        )
        return

    @result_processor
    def process_dynamics(self, result: Result):
        if not result.success:
            raise RuntimeError(result.failure_info.exception)  # todo: implement custom error
        
        self.rec.release(1)  # free the resource from this traj
        traj_id = result.task_info['traj_id']
        self.traj_progress[traj_id] += result.task_info['steps']
        traj_done = self.traj_progress[traj_id] >= self.total_steps
        
        if not traj_done:
            self.traj_avail.put(traj_id)
        
        if np.all(traj_progress >= self.total_steps):
            self.done.set()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--advance-steps', type=int, required=True)
    parser.add_argument('--total-steps', type=int, required=True)
    parser.add_argument(
        '--init-strc',
        type=str,
        nargs='+',
        required=True
    )
    parser.add_argument('--num-workers', type=int, default=1)
    #parser.add_argument('--model-file', type=str)

    args = parser.parse_args()

    # Prepare the output directory
    run_name = args.run_name
    run_dir = Path('runs') / f'run-{run_name}'
    is_restart = run_dir.exists()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up Queues and TaskServer
    queues = PipeQueues(keep_inputs=False)
    task_server = LocalTaskServer(
        queues=queues, 
        num_workers=args.num_workers, 
        methods=[]
    )

    # read in initial structures
    initial_frames = []
    for s in args.init_strc:
        initial_frames.append(read(s, ':'))

    # Set up Thinker
    thinker = Thinker(
        num_workers=args.num_workers,
        initial_frames=initial_frames,
        queues=queues,
        run_dir=run_dir,
        advance_steps=args.advance_steps,
        total_steps=args.total_steps
    )

    # Make a logger
    logger = logging.getLogger('main')
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(filename=run_dir / 'run.log')]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        for my_logger in [logger, logging.getLogger('cascade'), thinker.logger]:
            my_logger.addHandler(handler)
            my_logger.setLevel(logging.INFO)
    logger.info(f'Started. Running in {run_dir}. {"Restarting from previous run" if is_restart else ""}')

    # set up dynamics
    # todo: make this configurable via CLI
    dyn_class = VelocityVerlet
    dyn_kwargs = {'dt': 1 * units.fs}
    run_kwargs = {}
    # set up learner
    # todo: make this configurable via CLI
    learner = MACEInterface()
    calc = mace_mp('small', device='cpu', default_dtype="float32")
    model = calc.models[0]
    model_msg = learner.serialize_model(model)
    # wrap run function
    advance_dynamics_partial = partial(
        advance_dynamics,
        learner=learner,
        write_freq=1,  # todo: make an arg
        model_msg=model_msg,  # todo: allow this to vary with the run
        dyn_class=dyn_class,
        dyn_kwargs=dyn_kwargs,
        run_kwargs=run_kwargs,
    )
    advance_dynamics_partial = update_wrapper(advance_dynamics_partial, advance_dynamics) 
    # Start the workflow
    try:
        task_server.start()
        thinker.run()
    finally:
        queues.send_kill_signal()
        task_server.join()
    logger.info('Done!')
