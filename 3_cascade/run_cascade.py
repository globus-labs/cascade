import argparse
from pathlib import Path
import logging
import sys
import os
from queue import Queue, Empty
from functools import partial, update_wrapper
from dataclasses import dataclass
from random import sample
from typing import Callable
from threading import Event

from colmena.models import Result
from colmena.queue import PipeQueues, ColmenaQueues
from colmena.task_server.local import LocalTaskServer
from colmena.thinker import (
    BaseThinker, ResourceCounter, 
    task_submitter, result_processor,
    event_responder
)
import ase
from ase import Atoms
from ase.io import read
from ase.optimize.optimize import Dynamics
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.db import connect
from ase.calculators.calculator import Calculator
import numpy as np
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
    chunk_i: int,
    dyn_class: type[Dynamics],
    dyn_kwargs: dict[str, object],
    run_kwargs: dict[str, object] = {},
    # dyn_filter: tuple[type[Filter], dict[str, object]] | None = None,
    # callbacks: Sequence[tuple[type, int, disct[str, object]]] = (),
    # repeat: bool = False
) -> tuple[list[ase.Atoms], int]:
    """Advance an MD trajectory by steps

    :atoms: starting atoms to advance
    :db_path: ASE databse path to save the steps
    :traj_i: index of this trajectory in the cascade run
    :learner: used to construct the ML forcefield
    :model_msg: bytes representation of the ML forcefield weights
    :steps: how many steps to advance by
    :chunk_i: the index of this chunk in the trajectory
    :dyn_class: constructor for the dynamics to run this trajectory
    :dyn_kwargs: passed to dyn_class
    :run_kwargs: passed to dyn_class().run
    """

    # set up calculator
    calc = learner.make_calculator(model_msg, device='cpu')
    atoms.calc = calc

    with connect(db_path) as db:

        # delete existing rows for this chunk (we are rerunning)
        rows = db.select(chunk=chunk_i, traj=traj_i)
        ids = [r.id for r in rows]
        if ids is not None:
            db.delete(ids)

        # set up writer
        def write_to_db():
            # needs to be 64 bit for db read
            f = atoms.calc.results['forces']
            atoms.calc.results['forces'] = f.astype(np.float64)
            db.write(
                atoms=atoms,
                traj=traj_i,
                chunk=chunk_i,
            )

        # set up dynamics
        dyn = dyn_class(
            atoms,
            **dyn_kwargs
        )

        dyn.attach(write_to_db)
        # run dynamics
        dyn.run(steps, **run_kwargs)

        traj = db.select(traj=traj_i, chunk=chunk_i)
        traj = [row.toatoms() for row in traj]
    return traj


def select_training_frames(
    atoms: list[Atoms],
    n_frames: int
) -> list[Atoms]:
    """Return a frame or frames for training
    RIGHT NOW IS JUST A STUB
    """
    return sample(atoms, n_frames)


def label_training_frame(
    atoms: Atoms,
    calc_factory: Callable[[], Calculator]
) -> Atoms:
    """Assign a assign a label (forces) to a training frame (atoms)
    by running a calculator.

    :atoms: the training frame to label
    :calc_factory: a callable that creates a calculator
    """
    calc = calc_factory()
    atoms.calc = calc
    atoms.get_forces()
    atoms = canonicalize(atoms)
    return atoms


def train_model(learner: BaseLearnableForcefield) -> bytes:
    """Train a new model and return its bytes representaiton
    CURRENTLY JUST A STUB
    """
    calc = mace_mp('small', device='cpu', default_dtype="float32")
    model = calc.models[0]
    model_msg = learner.serialize_model(model)
    return model_msg


class Thinker(BaseThinker):

    def __init__(
        self,
        initial_frames: list[Atoms],
        queues: ColmenaQueues,
        run_dir: Path,
        db_path: Path | str,
        advance_steps: int,
        total_steps: int,
        n_workers: int,
        sample_frames: int,
        start_train_frac: float,
        learner: BaseLearnableForcefield,
        model_msg: bytes,
    ):
        """Cascade Thinker

        Runs cascade by advancing trajectories to total_steps, by advance_steps
        at a time. After each advance, the trajectories are audited. If they fail
        the audit, frames are sampled, labeled, and the model is updated

        initial_frames: initial conditions for md trajectories
        queues: for colmena
        run_dir: where to store logs and ase database
        db_path: what to call the ase database
        advance_steps: how many steps to advance md by at a time
        total_steps: how long md steps should be
        n_workers: for the ResourceCounter
        sample_frames: how many frames to sample from a chunk
        start_train_frac: start training when this fraction of trajectoreies 
            have been sampled from
        """
        super().__init__(
            queues,
            resource_counter=ResourceCounter(n_workers, task_types=['sim', 'train']))
        self.run_dir = run_dir
        self.atoms = initial_frames
        self.advance_steps = advance_steps
        self.total_steps = total_steps
        self.sample_frames = sample_frames
        self.n_workers = n_workers
        self.db_path = str(db_path)
        self.start_train_frac = start_train_frac
        self.n_traj = len(self.atoms)
        self.traj_progress = np.zeros(self.n_traj)
        self.to_advance = Queue(self.n_traj)
        self.to_select = Queue(self.n_traj)
        self.to_label = Queue()

        # populate the queue with all trajectories
        for i in range(self.n_traj):
            self.to_advance.put(i)
        self.sampled_from = set()  # which trajectories have new data in the training set
        self.start_training = Event()
        # start with all sim workers
        self.rec.reallocate(None, 'sim', self.n_workers)
        self.learner = learner
        self.model_msg = model_msg

        # keep track of how many finished
        self.done_ctr = 0

    @task_submitter(task_type='sim')
    def submit_dynamics(self):
        """Submit tasks to advance dynamics"""
        while True:
            # spin wait if training is started
            if self.start_training.is_set():
                continue
            try:
                traj_id = self.to_advance.get(block=True, timeout=1)
            except Empty:
                if self.done.is_set():
                    return
            else:
                break
        # calculate chunk index and how many steps to advance
        step_current = self.traj_progress[traj_id]
        steps = min(self.advance_steps, self.total_steps - step_current)
        chunk_i = int(step_current // self.advance_steps)

        self.logger.info(
            (f'Submitting trajecotry {traj_id} for advancement, '
             f'current step: {step_current} '
             f'steps to run: {steps} '
             f'{chunk_i=}'))

        atoms = self.atoms[traj_id]

        self.queues.send_inputs(
            topic='dynamics',
            method='advance_dynamics',
            input_kwargs=dict(
                atoms=atoms,
                steps=steps,
                traj_i=traj_id,
                chunk_i=chunk_i,
                db_path=self.db_path,
                model_msg=self.model_msg
            ),
            task_info=dict(
                traj_id=traj_id,
                steps=self.advance_steps
            )
        )
        return

    @result_processor(topic='dynamics')
    def process_dynamics(self, result: Result):
        """Process the results of dynamcis and launch an audit on this trajectory"""

        # ensure dynamics ran successfully
        traj_id = result.task_info['traj_id']
        self.logger.info(f'Processing result from traj {traj_id}')
        if not result.success:
            self.logger.error('Task failed with traceback:\n' + result.failure_info.traceback)
            raise RuntimeError(result.failure_info.exception)

        # free the resources
        self.logger.info('Freeing resourcs used in dynamics')
        self.rec.release('sim', 1)

        # parse the results
        traj = result.value
        # launch audit task
        self.logger.info(
            (
                f'Launching audit task for traj {traj_id} with '
                f'{len(traj)} frames'
            )
        )
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
        """If audit passes: put in advance queue, else put in selection queue"""

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
            self.logger.info(f'Adding {traj_id} to selection queue')
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
            self.done_ctr += 1
            self.logger.info(f'Traj {traj_id} not finished, marking as available')
            self.to_advance.put(traj_id)
        else:
            self.logger.info(f'Traj {traj_id} finished')
        # check if all trajectories are finished
        if np.all(self.traj_progress[traj_id] >= self.total_steps):
            self.logger.info('All trajectories finished, setting thinker to done')
            self.done.set()

    @task_submitter(task_type='sim')
    def submit_data_selection(self):
        """Submit a trajectory chunk with a failed-audit for data selection"""
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
        with connect(self.db_path) as db:
            # make sure theres really only one
            rows = db.select(traj=traj_id, chunk=chunk_i)
        atoms = [r.toatoms() for r in rows]

        self.queues.send_inputs(
            topic='frame_selection',
            method='select_training_frames',
            input_kwargs=dict(
                atoms=atoms,
                n_frames=self.sample_frames,
            ),
            task_info=dict(
                traj_id=traj_id,
            )
        )
        return

    @result_processor(topic='frame_selection')
    def process_frame_selection(self, result: Result):
        """Put selected frames into the labeling queue"""
        if not result.success:
            self.logger.error('Task failed with traceback:\n' + result.failure_info.traceback)
            raise RuntimeError(result.failure_info.exception)

        self.rec.release('sim', 1)
        atoms = result.value
        traj_id = result.task_info['traj_id']
        # todo: wait until the model has been updated to do this.
        # maybe put these in a do-not-advance list and then clear that when
        # the model updates?
        # indicate that we are updating the model from this trajectory
        self.logger.info(f'Marking {traj_id} as "sampled from"')
        self.sampled_from.add(result.task_info['traj_id'])

        # add sampled atoms to labeling queue
        self.logger.info(f'Adding {len(atoms)} frames from {traj_id} to labeling queue')
        for a in atoms:
            self.to_label.put(a)

    @task_submitter(task_type='sim')
    def submit_labeling(self):
        """Submit frames from labeling queue to be labeled"""
        while True:
            try:
                atoms = self.to_label.get(block=True, timeout=1)
            except Empty:
                if self.done.is_set():
                    return
            else:
                break
        self.logger.info('Submitting one frame to be labeled')
        self.queues.send_inputs(
            topic='label',
            method='label_training_frame',
            input_kwargs=dict(
                atoms=atoms
            )
        )

    @result_processor(topic='label')
    def store_labeled_data(self, result: Result):
        """Store labeled frames and check/decide if it is time to update model"""
        if not result.success:
            self.logger.error('Task failed with traceback:\n' + result.failure_info.traceback)
            raise RuntimeError(result.failure_info.exception)

        self.rec.release('sim', 1)
        atoms = result.value
        self.logger.info('Labeled one frame')
        with connect(self.db_path) as db:
            db.write(atoms=atoms, training=True)

        # check how many trajs have been sampled from
        n_sampled = len(self.sampled_from)
        frac_sampled = n_sampled / (self.n_traj - self.done_ctr)  # is this threadsafe???
        logger.info(f'{n_sampled=}, {frac_sampled=}, {self.start_train_frac=}')

        # if we've sampled from enough trajectories and weve labeled everything
        # start training
        # todo: better logic for this, this won't work at scale
        if frac_sampled >= self.start_train_frac and self.to_label.empty():
            logger.info('Finished labeling frames, starting training')
            self.start_training.set()

    @event_responder(event_name='start_training', reallocate_resources=True,
                     gather_from="sim", gather_to="train", disperse_to="sim", max_slots=1)
    def train_models(self):
        """STUB: Submit models to be retrained"""
        self.logger.info('Training model')
        # put the sampled trajectories back into the advancing queue
        for traj_id in self.sampled_from:
            self.to_advance.put(traj_id)

        # launch the training task
        self.queues.send_inputs(
            topic='train',
            method='train_model',
            input_kwargs=dict(
                learner=self.learner
            ),
        )

    @result_processor(topic='train')
    def process_model_result(self, result: Result):
        """store the new model and release resources"""
        # update the model
        # todo: does this need a lock or anything?
        self.model_msg = result.value

        # todo: think carefully about concurrency for this
        self.sampled_from = set()
        self.logger.info('Done training model')
        self.rec.release('train', 1)  # todo: is this right?


class Auditor():
    """Auditor base class"""
    def audit(self, traj: list[Atoms]) -> tuple[bool, float, Atoms]:
        raise NotImplementedError()


@dataclass
class DummyAuditor(Auditor):
    """STUB: Randomly pass or fail with accept_rate"""

    accept_rate = 0.75

    def audit(self, traj: list[Atoms]) -> tuple[bool, float, Atoms]:
        """a stub of a real audit"""
        good = np.random.random() < self.accept_rate
        score = float(good)
        atoms = traj[0] if good else traj[-1]
        return good, score, atoms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Cascade',
        description="""Launch a cascade job. Trajectories will be advanced under an ML surrogate 
        in chunks of size <advance-steps> until they are all of length <total-steps>. 
        When chunks from <start-train-frac> of the trajectories are marked as untrusted, the model will be updated.
        """
    )
    parser.add_argument('--run-name', type=str, required=True, help='Name that appears in the run directory')
    parser.add_argument('--advance-steps', type=int, required=True, help='How many steps to advance a trajectory at a time')
    parser.add_argument('--total-steps', type=int, required=True, help='How long a trajectory should be')
    parser.add_argument(
        '--init-strc',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to starting structures for molecular dynamics. One trajectory will be run for each starting structure'
    )
    parser.add_argument('--start-train-frac', type=float, default=1., help='What fraction of trajectories have to have untrusted segments before training begins')
    parser.add_argument('--num-workers', type=int, default=8, help='How many workers')
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
        topics=[
            'dynamics',
            'audit',
            'frame_selection',
            'label',
            'train',
        ],
        keep_inputs=False
    )

    # read in initial structures
    initial_frames = []
    for s in args.init_strc:
        initial_frames.append(read(s, '-1'))

    # set up learner
    # todo: make this configurable via CLI
    learner = MACEInterface()
    calc = mace_mp('small', device='cpu', default_dtype="float32")
    model = calc.models[0]
    model_msg = learner.serialize_model(model)

    # Set up Thinker
    thinker = Thinker(
        n_workers=args.num_workers,
        initial_frames=initial_frames,
        queues=queues,
        run_dir=run_dir,
        db_path=db_path,
        advance_steps=args.advance_steps,
        total_steps=args.total_steps,
        sample_frames=1,
        start_train_frac=args.start_train_frac,
        learner=learner,
        model_msg=model_msg
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

    # wrap run function
    advance_dynamics_partial = partial(
        advance_dynamics,
        learner=learner,
        # write_freq=1,  # todo: make an arg
        dyn_class=dyn_class,
        dyn_kwargs=dyn_kwargs,
        run_kwargs=run_kwargs,
    )
    update_wrapper(advance_dynamics_partial, advance_dynamics)

    # set up reference calc
    calc_factory = partial(mace_mp, 'medium', device='cpu', default_type='float32')
    label_partial = partial(label_training_frame, calc_factory=calc_factory)
    update_wrapper(label_partial, label_training_frame)
    # Start the workflow
    task_server = LocalTaskServer(
        queues=queues,
        num_workers=args.num_workers,
        methods=[
            advance_dynamics_partial,
            auditor.audit,
            select_training_frames,
            label_partial
        ]
    )
    try:
        task_server.start()
        thinker.run()
    finally:
        queues.send_kill_signal()
        task_server.join()
    logger.info('Done!')
