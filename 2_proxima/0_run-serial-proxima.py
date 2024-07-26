"""Run an experiment with a serial form of proxima"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import pickle as pkl
import logging
import json
import sys

import numpy as np
from ase.io import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units, io
from ase.db import connect
from ase.md import MDLogger
from ase.md.nptberendsen import NPTBerendsen

from cascade.proxima import SerialLearningCalculator
from cascade.calculator import make_calculator
from cascade.learning.torchani import TorchANI
from cascade.learning.torchani.build import make_output_nets, make_aev_computer
from cascade.utils import canonicalize

main_logger = logging.getLogger('main')

if __name__ == "__main__":
    # Get the user configuration
    parser = ArgumentParser()

    parser.add_argument('--file', default=None, help='Path to JSON file with options. Will overwrite any provided by command line.')

    group = parser.add_argument_group(title="Dynamics Problem", description="Define the physics problem being modeled")
    group.add_argument('--starting-strc', help='Path to the starting structure')
    group.add_argument('--temperature', type=float, help='Temperature of the dynamics. Units: K')
    group.add_argument('--timestep', type=float, default=1, help='Timestep length. Units: fs')
    group.add_argument('--calculator', default='blyp', help='Name of the method to use for the target function')
    group.add_argument('--steps', type=int, default=128, help='Number of dynamics steps to run')
    group.add_argument('--seed', type=int, default=1, help='Random seed used to start dynamics')

    group = parser.add_argument_group(title="Learner Details", description="Configure the surrogate model")
    group.add_argument('--initial-model', help='Path to initial model in message format. Code will generate a network with default settings if none provided')
    group.add_argument('--initial-data', nargs='*', default=(), help='Path to data files (e.g., ASE .traj and .db) containing initial training data')
    group.add_argument('--ensemble-size', type=int, default=2, help='Number of models to train on different data segments')
    group.add_argument('--online-training', action='store_true', help='Whether to restart training from the same weights each time')
    group.add_argument('--training-epochs', type=int, default=32, help='Number of epochs per training event')
    group.add_argument('--training-batch-size', type=int, default=32, help='Which device to use for training models')
    group.add_argument('--training-max-size', type=int, default=None, help='Maximum training set size to use when updating models')
    group.add_argument('--training-recency-bias', type=float, default=1., help='Factor by which to favor recent data when reducing training set size')
    group.add_argument('--training-device', default='cuda', help='Which device to use for training models')

    group = parser.add_argument_group(title='Proxima', description="Settings for learning on the fly")
    group.add_argument('--target-error', type=float, default=0.3, help='Target maximum force error to achieve with controller')
    group.add_argument('--error-history', type=int, default=8, help='Number of past timesteps to use to inform controller')
    group.add_argument('--retrain-freq', type=int, default=8, help='How frequently to retrain the evaluator')
    group.add_argument('--min-target-frac', type=float, default=0.05, help='Minimum fraction of evaluations to use DFT')

    args = parser.parse_args()
    if args.file is not None:
        with open(args.file) as fp:
            args.__dict__.update(json.load(fp))

    # Read in the starting structure
    strc_path = Path(args.starting_strc)
    strc_name = strc_path.with_suffix('').name

    # Create the run directory
    _skip_keys = ('steps', 'training-device', 'file')
    params_hash = sha256(json.dumps([(k, v) for k, v in args.__dict__.items() if k not in _skip_keys]).encode()).hexdigest()[-8:]
    run_name = f'{strc_name}-temp={args.temperature}-method={args.calculator}-{params_hash}'
    run_dir = Path('runs') / run_name
    run_dir.mkdir(exist_ok=True, parents=True)

    with (run_dir / 'params.json').open('w') as fp:
        json.dump(args.__dict__, fp, indent=2)

    # Create the logging
    handlers = [logging.FileHandler(run_dir / 'runtime.log'),
                logging.StreamHandler(sys.stdout)]
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)

    for logger in [main_logger, logging.getLogger('cascade')]:
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    main_logger.info(f'Starting a run in {run_dir}')

    # Pre-populate training database, if not yet created
    db_path = run_dir / 'proxima.db'
    if not db_path.exists():
        with connect(db_path) as db:
            for initial_data in args.initial_data:
                main_logger.info(f'Adding data from {initial_data} to database')
                for frame in io.iread(initial_data):
                    db.write(frame)

    with connect(db_path) as db:
        main_logger.info(f'Training database has {len(db)} entries.')

    # Set up the models
    learner = TorchANI()

    atoms = io.read(strc_path)
    if args.initial_model is None:
        species = list(set(atoms.symbols))
        aev = make_aev_computer(species)

        models = [(aev, make_output_nets(species, aev), dict((s, 0.) for s in species))
                  for i in range(args.ensemble_size)]
        logger.info('Created new default TorchANI networks')
    else:
        model = Path(args.initial_model).read_bytes()
        models = [model] * args.ensemble_size
        logger.info(f'Loaded a model from {args.initial_model}')

    # Get either the last step from the traj, or thermalize the starting structure
    traj_path = run_dir / 'md.traj'
    if traj_path.exists() and len(Trajectory(traj_path)) > 0:
        atoms = io.read(traj_path, -1)
        main_logger.info(f'Loaded last structure from {traj_path}')
        start_frame = len(io.read(traj_path, ':'))
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=np.random.RandomState(args.seed))
        main_logger.info(f'Initialized velocities to T={args.temperature}')
        start_frame = 0

    # Set up the proxima calculator
    calc_dir = Path('cp2k-run') / params_hash
    calc_dir.mkdir(parents=True, exist_ok=True)
    target_calc = make_calculator(args.calculator, directory=str(calc_dir))

    learning_calc = SerialLearningCalculator(
        target_calc=target_calc,
        learner=learner,
        models=models,
        train_kwargs={
            'num_epochs': args.training_epochs,
            'batch_size': args.training_batch_size,
            'reset_weights': not args.online_training,
            'device': args.training_device},  # Configuration for the training routines
        train_freq=args.retrain_freq,
        train_max_size=args.training_max_size,
        train_recency_bias=args.training_recency_bias,
        target_ferr=args.target_error,
        history_length=args.error_history,
        min_target_fraction=args.min_target_frac,
        db_path=db_path,
    )

    state_path = run_dir / 'proxima-state.pkl'
    if state_path.exists():
        with state_path.open('rb') as fp:
            learning_calc.set_state(pkl.load(fp))
        main_logger.info(f'Read model state from {state_path}')


    # Make functions which will track and store state of proxima
    def _save_state():
        with state_path.open('wb') as fs:
            pkl.dump(learning_calc.get_state(), fs)


    proxima_log_path = run_dir / 'proxima-log.json'
    start_time = perf_counter()


    def _log_proxima():
        global start_time
        step_time = perf_counter() - start_time
        start_time = perf_counter()
        with open(run_dir / 'proxima-log.json', 'a') as fp:
            last_uncer, last_error = learning_calc.error_history[-1]
            print(json.dumps({
                'step_time': step_time,
                'energy': float(atoms.get_potential_energy()),
                'maximum_force': float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
                'stress': atoms.get_stress().astype(float).tolist(),
                'temperature': atoms.get_temperature(),
                'volume': atoms.get_volume(),
                'used_surrogate': bool(learning_calc.used_surrogate),
                'proxima_alpha': learning_calc.alpha,
                'proxima_threshold': learning_calc.threshold,
                'last_uncer': float(last_uncer),
                'last_error': float(last_error),
            }), file=fp)


    def _write_to_traj():
        with Trajectory(traj_path, mode='a') as traj:
            can_atoms = canonicalize(atoms)
            traj.write(can_atoms)

    # Prepare the dynamics
    md_log_path = run_dir / 'md.log'
    atoms.calc = learning_calc
    npt = NPTBerendsen(atoms,
                       timestep=args.timestep * units.fs,
                       temperature_K=args.temperature,
                       pressure_au=0,
                       compressibility_au=5e-5 / units.bar)  # Close to the compressibility of water
    md_logger = MDLogger(np, atoms, str(md_log_path), stress=True)
    npt.attach(_log_proxima)
    npt.attach(_save_state)
    npt.attach(md_logger)
    npt.attach(_write_to_traj)

    # Run dynamics
    with md_logger:
        npt.run(args.steps - start_frame)
