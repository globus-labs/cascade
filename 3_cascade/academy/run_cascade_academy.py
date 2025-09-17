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

import asyncio
from concurrent.futures import ThreadPoolExecutor
from academy.agent import Agent, action
from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager
from academy.handle import Handle

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
