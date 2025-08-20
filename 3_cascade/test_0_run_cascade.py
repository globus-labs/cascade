from pytest import fixture
import logging

from ase import Atoms, units
from ase.io import read
from ase.md.verlet import VelocityVerlet
from mace.calculators import mace_mp

from colmena.models import Result
from colmena.queue import PipeQueues, ColmenaQueues
from colmena.exceptions import TimeoutException


from cascade.learning.base import BaseLearnableForcefield
from cascade.learning.mace import MACEInterface

from run_cascade import (
    Thinker,
    advance_dynamics
)


@fixture()
def atoms() -> Atoms:
    return read('../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp', index=-1)


@fixture
def queues():
    yield PipeQueues(
        topics=[
            'dynamics',
            'audit',
            'frame_selection',
            'label',
            'train',
        ],
        keep_inputs=False
    )

learner = MACEInterface()

@fixture
def model_msg():
    calc = mace_mp('small', device='cpu', default_dtype="float32")
    model = calc.models[0]
    model_msg = learner.serialize_model(model)
    return model_msg


def _pull_tasks(queues: ColmenaQueues) -> list[tuple[str, Result]]:
    """Pull all tasks available on the queue"""
    tasks = []
    while True:
        try:
            task = queues.get_task(timeout=1)
        except TimeoutException:
            break
        tasks.append(task)
    return tasks


@fixture()
def thinker(queues, initial_frames):

    thinker = Thinker(
        queues=queues,
        initial_frames=initial_frames,
        n_workers=1,
        run_dir='pytest',
        db_path='pytest/database.db',
        advance_steps=10,
        total_steps=30,
        sample_frames=1,
        start_train_frac=1,

    )
    # Make a logger
    logger = logging.getLogger('main')
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(filename=run_dir / 'run.log')]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        for my_logger in [logger, logging.getLogger('cascade'), thinker.logger]:
            my_logger.addHandler(handler)
            my_logger.setLevel(logging.INFO)

    with thinker:
        thinker.start()
        yield thinker
        thinker.done.set()
        queues._all_complete.set()  # Act like all tasks have finished
        queues._active_tasks.clear()
    thinker.join()


def test_advance_dynamics(
    atoms,
    model_msg,
    learner=learner,
    db_path='test.db',
    traj_i=0,
    steps=10,
    chunk_i=0,
    dyn_class=VelocityVerlet,
    dyn_kwargs={'timestep': 1*units.fs}
    ):

    traj = advance_dynamics(
        atoms=atoms,
        db_path=db_path,
        traj_i=traj_i,
        learner=learner,
        model_msg=model_msg,
        steps=steps,
        chunk_i=chunk_i,
        dyn_class=dyn_class,
        dyn_kwargs=dyn_kwargs)

    print(len(traj))
    assert len(traj) == steps+1, f'should advance for {steps} steps'
