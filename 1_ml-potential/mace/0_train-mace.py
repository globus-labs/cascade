"""Training script where we vary different parameters of MACE"""
import json
import logging
import sys
from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from platform import node
from time import perf_counter

from mace.calculators import mace_mp
import numpy as np
import torch
from ase import Atoms
from ase.io import read, iread

from cascade.learning.finetuning import MultiHeadConfig
from cascade.learning.mace import MACEInterface

if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()

    group = parser.add_argument_group(title='Training Data', description='Setting which control how the training data are generated')
    group.add_argument('--train-files', nargs='+', help='Trajectories used for training')
    group.add_argument('--test-files', nargs='+', help='Trajectories used for testing')
    group.add_argument('--val-frac', default=0.1, help='What fraction of the trajectory to use for validation. '
                                                       'We use the last frames of the trajectory for validation')

    group = parser.add_argument_group(title='Optimization Setting', description='Parameters of the optimization algorithm')
    group.add_argument('--num-epochs', default=32, help='Number of training epochs', type=int)
    group.add_argument('--batch-size', default=64, help='Number of records per batch', type=int)
    group.add_argument('--learning-rate', default=1e-4, help='Learning rate for optimizer', type=float)
    group.add_argument('--loss-weights', default=(1, 1, 0.1), help='Weights for the energy, forces, and stress', nargs=3, type=float)

    group = parser.add_argument_group(title='Model Architecture', description='Parameters describing the CHGNet model')
    group.add_argument('--reset-weights', action='store_true', help='Whether to reset the pre-trained weights')
    group.add_argument('--freeze-layers', default=None, type=int, help='Number of layers to freeze')
    group.add_argument('--model-size', default='small', help='Which pretrained model to use')

    group = parser.add_argument_group(title='Replay Settings', description='Settings about replaying original training data during finetuning')
    group.add_argument('--replay-dataset', default=None, help='Path to an ASE database containing data to be retrained')
    group.add_argument('--replay-downselect', default=None, type=int, help='Maximum number of entries to use from replay dataset')
    group.add_argument('--replay-frequency', default=1, type=int, help='How frequency to replay')
    group.add_argument('--replay-lr-reduction', default=1, type=float, help='Factor by which to reduce LR during replay')
    group.add_argument('--replay-batch-size', default=None, type=int, help='Batch size used during replay')
    args = parser.parse_args()

    # Load the training data
    train_atoms: list[Atoms] = []
    valid_atoms: list[Atoms] = []
    train_hasher = sha256()
    rng = np.random.RandomState(1)
    for file in args.train_files:
        my_atoms = read(file, slice(None))

        # Shuffle the data if they are not from a traj file
        if not file.endswith('.traj'):
            rng.shuffle(my_atoms)

        # Hash dataset for reproducibility
        for atoms in my_atoms:
            train_hasher.update(atoms.positions.tobytes())
            train_hasher.update(atoms.cell.tobytes())

        # Split off the validation set
        valid_start = int(len(my_atoms) * args.val_frac)
        valid_atoms.extend(my_atoms[-valid_start:])
        train_atoms.extend(my_atoms[:-valid_start])
    train_hash = train_hasher.hexdigest()[-8:]

    # Load the test data
    test_hasher = sha256()
    test_atoms: list[Atoms] = []
    test_source: list[int] = []
    for i, file in enumerate(args.test_files):
        for atoms in iread(file):
            test_hasher.update(atoms.positions.tobytes())
            test_hasher.update(atoms.cell.tobytes())
            test_atoms.append(atoms)
            test_source.append(i)
    test_hash = test_hasher.hexdigest()[-8:]

    # Make a run directory
    params = args.__dict__.copy()
    params.update({
        'train_hash': train_hash,
        'train_size': len(train_atoms),
        'valid_size': len(valid_atoms),
        'test_hash': test_hash,
        'test_size': len(test_atoms),
        'cuda_version': torch.version.cuda,
        'device_name': torch.cuda.get_device_name(0),
        'host': node()
    })

    run_hash = sha256(json.dumps(params).encode()).hexdigest()[-8:]
    start_time = datetime.now().isoformat(timespec="seconds").replace(":", "").replace("-", "")
    run_dir = Path('runs') / f'train-{train_hash}' / f'{start_time}-{run_hash}'
    run_dir.mkdir(parents=True)

    # Make a logger
    logger = logging.getLogger('main')
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(filename=run_dir / 'run.log')]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        for my_logger in [logger, logging.getLogger('cascade')]:
            my_logger.addHandler(handler)
            my_logger.setLevel(logging.INFO)
    logger.info(f'Started. Running in {run_dir}')
    with open(run_dir / 'params.json', 'w') as fp:
        json.dump(params, fp, indent=2)

    # Assemble the model
    model = mace_mp(args.model_size, device='cpu').models[0]
    logger.info('Loaded model and moved it to RAM for now')

    # Create the replay, if desired
    if args.replay_dataset is not None:
        replay = MultiHeadConfig(
            original_dataset=read(args.replay_dataset, slice(None)),
            num_downselect=args.replay_downselect,
            epoch_frequency=args.replay_frequency,
            lr_reduction=args.replay_lr_reduction,
            batch_size=args.replay_batch_size,
        )
    else:
        replay = None

    # Train using the wrapper function from Cascade
    mace = MACEInterface()
    loss_e, loss_f, loss_s = args.loss_weights
    train_time = perf_counter()
    model_msg, train_log = mace.train(
        model,
        train_data=train_atoms,
        valid_data=valid_atoms,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        force_weight=loss_f / loss_e,
        stress_weight=loss_s / loss_e,
        reset_weights=args.reset_weights,
        num_freeze=args.freeze_layers,
        replay=replay,
        device='cuda'
    )
    train_time = perf_counter() - train_time
    logger.info('Training complete. Running inference')

    # Run inference on the validation dataset
    performance = {'train_time': train_time / args.num_epochs}
    for name, atoms in [('train', train_atoms), ('valid', valid_atoms), ('test', test_atoms)]:
        eval_time = perf_counter()
        pred_e, pred_f, pred_s = mace.evaluate(model_msg, atoms, batch_size=args.batch_size, device='cuda')
        eval_time = perf_counter() - eval_time
        logger.info(f'Computing performance on {name} set')

        true_n = np.array([len(a) for a in atoms])
        true_e = np.array([a.get_potential_energy() for a in atoms])
        true_f = [a.get_forces() for a in atoms]
        true_s = np.array([a.get_stress(False) for a in atoms])

        pred_e_pa = pred_e / true_n
        true_e_pa = true_e / true_n

        performance[name] = {
            'size': len(atoms),
            'eval_time': eval_time / len(atoms),
            'energy_mae': float(np.abs(pred_e_pa - true_e_pa).mean()),
            'force_mean_error': float(np.mean([np.linalg.norm(t - p, axis=1).mean() for t, p in zip(true_f, pred_f)])),
            'force_max_error': float(np.mean([np.linalg.norm(t - p, axis=1).max() for t, p in zip(true_f, pred_f)])),
            'stress_mae': float(np.abs(true_s.flatten() - pred_s.flatten()).mean())
        }
        np.savez_compressed(run_dir / f'{name}_true.npz', source=test_source, count=true_n, energy=true_e, forces=np.concatenate(true_f, axis=0), stress=true_s)
        np.savez_compressed(run_dir / f'{name}_pred.npz', source=test_source, count=true_n, energy=pred_e, forces=np.concatenate(pred_f, axis=0), stress=pred_s)
    for key, value in train_log.iloc[-1].to_dict().items():
        performance[f'last_epoch.{key}'] = value

    # Store the model, predictions, and ground truth
    logger.info('Saving model and results')
    (run_dir / 'model.pt').write_bytes(model_msg)
    train_log.to_csv(run_dir / 'log.csv', index=False)
    with open(run_dir / 'performance.json', 'w') as fp:
        json.dump(performance, fp, indent=2)
