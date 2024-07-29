"""Training script where we vary different parameters of ANI"""
import json
import logging
import sys
from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from platform import node
from time import perf_counter

import numpy as np
import torch
from ase import Atoms
from ase.io import read, iread

from cascade.learning.torchani import build, TorchANI

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
    group.add_argument('--loss-weights', default=(1, 10, 100), help='Weights for the energy, forces, and stress', nargs=3, type=float)
    group.add_argument('--scale-energies', action='store_true', help='Whether to scale the weights to match the mean/std of training data.')

    group = parser.add_argument_group(title='Model Architecture', description='Parameters describing the ANI model')
    group.add_argument('--hidden-layers', default=2, help='Number of hidden layers in the output network', type=int)
    group.add_argument('--hidden-units', default=128, help='Number of units in the first hidden layer of output networks', type=int)
    group.add_argument('--hidden-decay', default=0.8, help='Decrease in number of units between hidden layers', type=float)
    group.add_argument('--radial-cutoff', default=5.2, help='Maximum distance of radial terms', type=float)
    group.add_argument('--angular-cutoff', default=3.5, help='Maximum distance of angular terms', type=float)
    group.add_argument('--radial-eta', default=16, help='Width parameter of radial terms', type=float)
    group.add_argument('--angular-eta', default=8, help='Width parameter of angular terms', type=float)
    group.add_argument('--angular-zeta', default=8, help='Angular width parameter of angular terms', type=float)
    group.add_argument('--num-radial-terms', default=16, help='Number of radial terms', type=int)
    group.add_argument('--num-angular-dist-terms', default=4, help='Number of radial steps of angular terms', type=int)
    group.add_argument('--num-angular-angl-terms', default=8, help='Number of radial steps of angular terms', type=int)
    args = parser.parse_args()

    # Load the training data
    train_atoms: list[Atoms] = []
    valid_atoms: list[Atoms] = []
    train_hasher = sha256()
    for file in args.train_files:
        my_atoms = read(file, slice(None))
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
    run_dir = Path('runs') / f'{datetime.now().isoformat()}-{run_hash}'
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
    species = set()
    for atoms in train_atoms:
        species.update(atoms.symbols)
    species = sorted(species)
    atom_energies = dict((s, 0) for s in species)  # Guess zero to start
    logger.info(f'Found {len(species)} species: {", ".join(species)}')

    aev_computer = build.make_aev_computer(
        radial_cutoff=args.radial_cutoff,
        angular_cutoff=args.angular_cutoff,
        radial_eta=args.radial_eta,
        angular_eta=args.angular_eta,
        zeta=args.angular_zeta,
        num_radial_terms=args.num_radial_terms,
        num_angular_dist_terms=args.num_angular_dist_terms,
        num_angular_angl_terms=args.num_angular_angl_terms,
        species=species
    )
    aev_computer.to('cuda')
    aev_length = aev_computer.aev_length
    logger.info(f'Made an AEV computer which produces {aev_length} features')

    nn = build.make_output_nets(species, aev_computer, args.hidden_units, args.hidden_layers, args.hidden_decay)

    model = torch.nn.Sequential(aev_computer, nn).to('cuda')
    total_params = sum(np.prod(p.size()) for p in model.parameters(recurse=True))
    logger.info(f'Made the ANI model with {total_params} parameters. Training')

    # Train using the wrapper function from Cascade
    ani = TorchANI()
    loss_e, loss_f, loss_s = args.loss_weights
    train_time = perf_counter()
    model_msg, train_log = ani.train(
        (aev_computer, nn, atom_energies),
        train_data=train_atoms,
        valid_data=valid_atoms,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        force_weight=loss_f / loss_e,
        stress_weight=loss_s / loss_e,
        reset_weights=True,
        scale_energies=args.scale_energies,
        device='cuda'
    )
    train_time = perf_counter() - train_time
    logger.info('Training complete. Running inference')

    # Run inference on the validation dataset
    eval_time = perf_counter()
    pred_e, pred_f, pred_s = ani.evaluate(model_msg, test_atoms, batch_size=args.batch_size, device='cuda')
    eval_time = perf_counter() - eval_time
    logger.info('Inference complete. Computing performance')

    true_n = np.array([len(a) for a in test_atoms])
    true_e = np.array([a.get_potential_energy() for a in test_atoms])
    true_f = [a.get_forces() for a in test_atoms]
    true_s = np.array([a.get_stress(False) for a in test_atoms])

    pred_e_pa = pred_e / true_n
    true_e_pa = true_e / true_n

    performance = {
        'train_time': train_time / args.num_epochs,
        'eval_time': eval_time / len(test_atoms),
        'energy_mae': float(np.abs(pred_e_pa - true_e_pa).mean()),
        'force_mean_error': float(np.mean([np.linalg.norm(t - p, axis=1).mean() for t, p in zip(true_f, pred_f)])),
        'force_max_error': float(np.mean([np.linalg.norm(t - p, axis=1).max() for t, p in zip(true_f, pred_f)])),
        'stress_mae': float(np.abs(true_s.flatten() - pred_s.flatten()).mean())
    }

    # Store the model, predictions, and ground truth
    logger.info('Saving model and results')
    torch.save(model_msg, run_dir / 'model.pt')
    train_log.to_csv(run_dir / 'log.csv', index=False)
    with open(run_dir / 'performance.json', 'w') as fp:
        json.dump(performance, fp, indent=2)

    np.savez_compressed(run_dir / 'test_true.npz', source=test_source, count=true_n, energy=true_e, forces=np.concatenate(true_f, axis=0), stress=true_s)
    np.savez_compressed(run_dir / 'test_pred.npz', source=test_source, count=true_n, energy=pred_e, forces=np.concatenate(pred_f, axis=0), stress=pred_s)
