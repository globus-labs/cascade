import argparse
from pathlib import Path
import logging
import sys
import os
from typing import Sequence
from queue import Queue, Empty
from functools import partial, update_wrapper
from dataclasses import dataclass

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
from cascade.utils import canonicalize


def advance_dynamics(
    atoms: Atoms,
    run_dir: Path,
    traj_i: int,
    learner: BaseLearnableForcefield,
    model_msg: bytes,
    steps: int,
    current_step: int,
    chunk_i: int,
    dyn_class: type[Dynamics],
    dyn_kwargs: dict[str, object],
    run_kwargs: dict[str, object] = None,
    # dyn_filter: tuple[type[Filter], dict[str, object]] | None = None,
    # callbacks: Sequence[tuple[type, int, dict[str, object]]] = (),
    # repeat: bool = False
) -> tuple[list[ase.Atoms], int]:
    """Advance an MD trajectory by n_steps
    """

    # set up calculator
    calc = learner.make_calculator(model_msg, device='cpu')
    atoms.calc = calc

    # create new dir for this chunk
    chunk_current = run_dir / f'traj_{traj_i}' / f'chunk_{chunk_i}'
    chunk_current.mkdir(parents=True, exist_ok=True)
    traj_file = str(chunk_current/'md.traj')

    # if chunk exists: remove
    # (if its here, it failed an audit)
    # todo: possibly keep all with an index
    if os.path.exists(traj_file):
        os.remove(traj_file)

    # set up dynamics
    dyn = dyn_class(
        atoms,
        trajectory=traj_file,
        **dyn_kwargs
    )

    # run dynamics
    dyn.run(steps, **run_kwargs)
    traj = read(traj_file, index=':')
    return traj


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
        self.atoms = initial_frames
        self.advance_steps = advance_steps
        self.total_steps = total_steps
        self.n_traj = len(self.atoms)
        self.traj_progress = np.zeros(self.n_traj)
        self.traj_avail = Queue(self.n_traj)
        # populate the queue with all trajectories
        for i in range(self.n_traj):
            self.traj_avail.put(i)

    @task_submitter()
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

        # calculate chunk index and how many steps to advance
        step_current = self.traj_progress[traj_id]
        steps = min(self.advance_steps, self.total_steps - step_current)
        chunk_i = step_current // self.advance_steps

        atoms = self.atoms[traj_id]
        # print('\n\n')
        # print(type(atoms))
        # print('\n\n')

        self.queues.send_inputs(
            topic='dynamics',
            method='advance_dynamics',
            input_kwargs=dict(
                atoms=atoms,
                steps=steps,
                traj_i=traj_id,
                current_step=step_current,
                chunk_i=chunk_i,
                run_dir=self.run_dir
            ),
            task_info=dict(
                traj_id=traj_id,
                steps=self.advance_steps
            )
        )
        return

    @result_processor(topic='dynamics')
    def process_dynamics(self, result: Result):
        """receive the results of dynamcis and launch an audit task"""

        # ensure dynamics ran successfully
        if not result.success:
            print(result.failure_info.traceback)
            raise RuntimeError(result.failure_info.exception)  # todo: implement custom error

        # free the resources
        self.rec.release(None, 1)

        # parse the results
        traj = result.value
        traj_id = result.task_info['traj_id']

        # launch audit task
        self.queues.send_inputs(
            topic='audit',
            method='audit',
            input_kwargs=dict(
                traj=traj
            ),
            task_info=dict(
                traj_id=traj_id,
                steps=result.task_info['steps']
            )
        )

    @result_processor(topic='audit')
    def process_audit(self, result: Result):

        # parse out results
        good, score, atoms = result.value
        traj_id = result.task_info['traj_id']
        steps = result.task_info['steps']

        # only advance things if the audit passes
        if not good:
            return

        self.atoms[traj_id] = atoms
        self.traj_progress[traj_id] += steps
        # check if this traj is finished
        traj_done = self.traj_progress[traj_id] >= self.total_steps
        # mark this trajectory as available
        if not traj_done:
            self.traj_avail.put(traj_id)
        # check if all trajectories are finished
        if np.all(self.traj_progress[traj_id] >= self.total_steps):
            self.done.set()

class Auditor():

    def audit(self, traj: list[Atoms]) -> tuple[bool, float, Atoms]:
        raise NotImplementedError()


@dataclass
class DummyAuditor(Auditor):

    accept_rate = 0.75

    def audit(self, traj: list[Atoms]) -> tuple[bool, float, Atoms]:
        """a stub of a real audit"""
        good = np.random.random() < self.accept_rate
        score = float(good)
        atoms = traj[0] if good else traj[-1]
        return good, score, atoms


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
    queues = PipeQueues(
        topics=['dynamics', 'audit'],
        keep_inputs=False
    )

    # read in initial structures
    initial_frames = []
    for s in args.init_strc:
        initial_frames.append(read(s, '-1'))

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


    # set up auditor
    auditor = DummyAuditor()

    # set up dynamics
    # todo: make this configurable via CLI
    dyn_class = VelocityVerlet
    dyn_kwargs = {'timestep': 1 * units.fs}
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
        #write_freq=1,  # todo: make an arg
        model_msg=model_msg,  # todo: allow this to vary with the run
        dyn_class=dyn_class,
        dyn_kwargs=dyn_kwargs,
        run_kwargs=run_kwargs,
    )
    update_wrapper(advance_dynamics_partial, advance_dynamics)
    # Start the workflow

    task_server = LocalTaskServer(
        queues=queues,
        num_workers=args.num_workers,
        methods=[advance_dynamics_partial, auditor.audit]
    )
    try:
        task_server.start()
        thinker.run()
    finally:
        queues.send_kill_signal()
        task_server.join()
    logger.info('Done!')
