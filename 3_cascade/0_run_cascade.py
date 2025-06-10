import argparse
from pathlib import Path
import logging
import sys
import os
from typing import Sequence
from queue import Queue, Empty
from functools import partial, update_wrapper
from dataclasses import dataclass
from random import sample
from typing import Callable

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
from ase.db import connect
from ase.calculators.calculator import Calculator
import numpy as np
from glob import glob
from mace.calculators import mace_mp

from cascade.learning.base import BaseLearnableForcefield
from cascade.learning.mace import MACEInterface
from cascade.utils import canonicalize


def advance_dynamics(
    atoms: Atoms,
    db_path: str,
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
    # chunk_current = run_dir / f'traj_{traj_i}' / f'chunk_{chunk_i:d}'
    # chunk_current.mkdir(parents=True, exist_ok=True)
    # traj_file = str(chunk_current/'md.traj')
    # # if chunk exists: remove
    # # (if its here, it failed an audit)
    # # todo: possibly keep all with an index
    # if os.path.exists(traj_file):
    #     os.remove(traj_file)

    with connect(db_path) as db:

        # delete existing rows for this chunk (we are rerunning)
        rows = db.select(chunk=chunk_i, traj=traj_i)
        ids = [r.id for r in rows]
        if ids is not None:
            db.delete(ids)

        # set up writer
        def write_to_db():
            db.write(
                atoms=atoms,
                traj=traj_i,
                chunk=chunk_i,
            )

        # set up dynamics
        dyn = dyn_class(
            atoms,
            # trajectory=traj_file,
            **dyn_kwargs
        )
        
        dyn.attach(write_to_db)
        # run dynamics
        dyn.run(steps, **run_kwargs)
        
        traj = db.select(traj=traj_i, chunk=chunk_i)
        traj = [row.toatoms() for row in traj]
        #traj = read(traj_file, index=':')
    return traj


def select_training_frames(
    atoms: list[Atoms],
    n_frames: int
) -> list[Atoms]:
    """Return a frame or frames for training"""
    return sample(atoms, n_frames)


def label_training_frame(
    atoms: Atoms,
    calc_factory: Callable[[], Calculator]
) -> Atoms:
    calc = calc_factory()
    atoms.calc = calc
    atoms.calculate()
    atoms = canonicalize(atoms)
    return atoms

class Thinker(BaseThinker):

    def __init__(
        self,
        initial_frames: list[Atoms],
        queues: ColmenaQueues,
        run_dir: Path,
        db_path: Path | str,
        advance_steps: int,
        total_steps: int,
        num_workers: int,
        training_frames_per_chunk: int
    ):
        """thinker
        """
        super().__init__(queues, resource_counter=ResourceCounter(num_workers))
        self.run_dir = run_dir
        self.atoms = initial_frames
        self.advance_steps = advance_steps
        self.total_steps = total_steps
        self.frames_per_chunk = training_frames_per_chunk
        self.db_path = str(db_path)
        self.n_traj = len(self.atoms)
        self.traj_progress = np.zeros(self.n_traj)
        self.to_advance = Queue(self.n_traj)
        self.to_select = Queue(self.n_traj)
        self.to_label = Queue()
        # populate the queue with all trajectories
        for i in range(self.n_traj):
            self.to_advance.put(i)

    @task_submitter()
    def submit_dynamics(self):
        while True:
            try:
                traj_id = self.to_advance.get(block=True, timeout=1)
            except Empty:
                if self.done.is_set():
                    return
            else:
                break
        self.logger.info(f'Submitting trajecotry {traj_id} for advancement')

        # calculate chunk index and how many steps to advance
        step_current = self.traj_progress[traj_id]
        steps = min(self.advance_steps, self.total_steps - step_current)
        chunk_i = int(step_current // self.advance_steps)

        atoms = self.atoms[traj_id]

        self.queues.send_inputs(
            topic='dynamics',
            method='advance_dynamics',
            input_kwargs=dict(
                atoms=atoms,
                steps=steps,
                traj_i=traj_id,
                current_step=step_current,
                chunk_i=chunk_i,
                db_path=self.db_path
            ),
            task_info=dict(
                traj_id=traj_id,
                steps=self.advance_steps
            )
        )
        return

    @task_submitter()
    def submit_data_selection(self):
        while True:
            try:
                traj_id = self.to_select.get(block=True, timeout=1)
            except Empty:
                if self.done.is_set():
                    return
            else:
                break
        self.logger.info(f'Submitting trajecotry {traj_id} for data selection')

        step_current = self.traj_progress[traj_id]
        chunk_i = int(step_current // self.advance_steps)
        chunk_current = run_dir / f'traj_{traj_id}' / f'chunk_{chunk_i:d}' / 'md.traj'
        atoms = read(str(chunk_current), index=':')

        self.queues.send_inputs(
            topic='frame_selection',
            method='select_training_frames',
            input_kwargs=dict(
                atoms=atoms,
                n_frames=self.frames_per_chunk,

            ),
            task_info=dict(
                traj_id=traj_id,
            )
        )
        return

    @result_processor(topic='dynamics')
    def process_dynamics(self, result: Result):
        """receive the results of dynamcis and launch an audit task"""

        # ensure dynamics ran successfully
        traj_id = result.task_info['traj_id']
        self.logger.info(f'Processing result from traj {traj_id}')
        if not result.success:
            self.logger.error('Task failed with traceback:\n' + result.failure_info.traceback)
            raise RuntimeError(result.failure_info.exception)

        # free the resources
        self.logger.info('Freeing resourcs used in dynamics')
        self.rec.release(None, 1)

        # parse the results
        traj = result.value
        # launch audit task
        self.logger.info(f'Launching audit task for traj {traj_id}')
        self.logger.info(f'\n\nn_frames: {len(traj)}\n\n')
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

        if not result.success:
            self.logger.error('Task failed with traceback:\n' + result.failure_info.traceback)
            raise RuntimeError(result.failure_info.exception)

        # parse out results
        good, score, atoms = result.value
        traj_id = result.task_info['traj_id']
        steps = result.task_info['steps']

        # only advance things if the audit passes
        if not good:
            self.logger.info(f'Audit for {traj_id} failed, reverting trajectory to previous state')
            self.to_select.put(traj_id)
            return

        self.logger.info(f'Audit for {traj_id} passed, updating trajectory state')
        self.atoms[traj_id] = atoms
        self.traj_progress[traj_id] += steps
        self.logger.info(f'Traj {traj_id} has completed {self.traj_progress[traj_id]} steps')
        # check if this traj is finished
        traj_done = self.traj_progress[traj_id] >= self.total_steps
        # mark this trajectory as available
        if not traj_done:
            self.logger.info(f'Traj {traj_id} not finished, marking as available')
            self.to_advance.put(traj_id)
        else:
            self.logger.info(f'Traj {traj_id} finished')
        # check if all trajectories are finished
        if np.all(self.traj_progress[traj_id] >= self.total_steps):
            self.logger.info(f'All trajectories finished, setting thinker to done')
            self.done.set()

    @result_processor(topic='frame_selection')
    def process_frame_selection(self, result: Result):

        if not result.success:
            self.logger.error('Task failed with traceback:\n' + result.failure_info.traceback)
            raise RuntimeError(result.failure_info.exception)

        self.rec.release(None, 1)
        atoms = result.value
        self.to_advance.put(traj_id) 
        # todo: wait until the model has been updated to do this. 
        # maybe put these in a do-not-advance list and then clear that when 
        # the model updates?
        for a in atoms:
            self.to_label.put(a)

    @task_submitter()
    def submit_labeling(self):
        """label a training example"""
        while True:
            try:
                atoms = self.to_label.get(block=True, timeout=1)
            except Empty:
                if self.done.is_set():
                    return
            else:
                break

        self.queues.send_inputs(
            topic='label',
            method='label_training_frame',
            input_kwargs=dict(
                atoms=atoms
            )
        )

    @result_processor(topic='label')
    def store_labeled_data(self, result: Result):
        if not result.success:
            self.logger.error('Task failed with traceback:\n' + result.failure_info.traceback)
            raise RuntimeError(result.failure_info.exception)

        self.rec.release(None, 1)
        atoms = result.value
        with connect(self.db_path) as db:
            db.write(atoms=atoms, training=True)

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

    # set up ase db
    db_path = run_dir / 'database.db'
    if os.path.exists(db_path):
        os.remove(db_path)
    # (it will be created on first access)

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
        db_path=db_path,
        advance_steps=args.advance_steps,
        total_steps=args.total_steps,
        training_frames_per_chunk=1,
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
        # write_freq=1,  # todo: make an arg
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
