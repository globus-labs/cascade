"""Training script where we vary different parameters of MACE"""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import itertools
import logging
import sys

import torch
import ignite
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import iread
from ignite.contrib.handlers import MLflowLogger
from ignite.engine import Engine, Events
from ignite.metrics import Loss
from mace.data import AtomicData
from mace.data.utils import config_from_atoms, compute_average_E0s
from mace.modules import WeightedHuberEnergyForcesStressLoss
from mace.tools.torch_geometric.dataloader import DataLoader
from mace.calculators.foundations_models import mace_mp

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
    group.add_argument('--batch-size', default=4, help='Number of records per batch', type=int)
    group.add_argument('--unfreeze-layers', nargs='+', default=('readouts',),
                       help='Which parts of the model to optimize during training')
    group.add_argument('--learning-rate', default=1e-4, help='Learning rate for optimizer', type=float)
    group.add_argument('--loss-weights', default=(1, 10, 100), help='Weights for the energy, forces, and stress', nargs=3, type=float)
    group.add_argument('--loss-huber', default=0.01, help='Huber delta parameter used in loss functions', type=float)

    group = parser.add_argument_group(title='Model Architecture', description='Parameters describing the MACE model')
    group.add_argument('--model-type', default='mace_mp', help='Name of the foundation model to use')
    group.add_argument('--model-size', default='small', help='Size of the model to create')
    group.add_argument('--model-dtype', default='float32', help='Precision to use for model weights')
    group.add_argument('--seed', default=0, help='Random seed for bootstrap sampling', type=int)

    args = parser.parse_args()

    # Make a logger
    logger = logging.getLogger('main')
    handlers = [logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info('Started')

    # Load the model
    if args.model_type == 'mace_mp':
        calc = mace_mp(args.model_size, device='cuda', default_dtype=args.model_dtype)
        model = calc.models[0]
    else:
        raise ValueError(f'Unknown model type: {args.model_type}')
    logger.info('Loaded model')


    # Make the training loaders
    def _prepare_atoms(my_atoms: Atoms):
        """MACE expects the training outputs to be stored in `info` and `arrays`"""
        my_atoms.info = {
            'energy': my_atoms.get_potential_energy(),
            'stress': my_atoms.get_stress()
        }
        my_atoms.arrays.update({
            'forces': my_atoms.get_forces(),
            'positions': my_atoms.positions,
        })


    configs = defaultdict(list)
    for file in args.train_files:
        # Load all the configurations
        my_configs = []
        for atoms in iread(file):
            _prepare_atoms(atoms)
            my_configs.append(config_from_atoms(atoms))

        # Split off the validation set
        valid_start = int(len(my_configs) * args.val_frac)
        configs['train'].extend(my_configs[:-valid_start])
        configs['valid'].extend(my_configs[-valid_start:])
        logger.info(f'Loaded {len(my_configs)} from {file}. Stored {valid_start} as validation entries')

    # bootstrap sample the training configs
    rng = np.random.RandomState(args.seed)
    configs['train'] = rng.choice(configs['train'], size=(len(configs['train']),), replace=True)
    logger.info(f'Bootstrap resampled configs with seed {args.seed}')

    loaders = dict()
    for key, my_configs in configs.items():
        loader = DataLoader(
            [AtomicData.from_config(c, z_table=calc.z_table, cutoff=calc.r_max) for c in my_configs],
            batch_size=args.batch_size,
            shuffle=key == 'train',
            drop_last=key == 'train',
        )
        logger.info(f'Made {key} data loader')
        loaders[key] = loader

    # Update the atomic energies using the data from all trajectories
    atomic_energies_dict = compute_average_E0s(list(itertools.chain(*configs.values())), calc.z_table)
    atomic_energies = np.array([atomic_energies_dict[z] for z in calc.z_table.zs])

    with torch.no_grad():
        ae = model.atomic_energies_fn.atomic_energies
        model.atomic_energies_fn.atomic_energies = torch.from_numpy(atomic_energies).to(ae.dtype).to(ae.device)
    logger.info('Updated atomic energies')

    # Update the shift of the energy scale
    error = []
    for file in args.train_files:
        for atoms in iread(file):
            dft = atoms.get_potential_energy() / len(atoms)
            ml = calc.get_potential_energy(atoms) / len(atoms)
            error.append(ml - dft)
    model.scale_shift.shift -= np.mean(error)
    logger.info(f'Shifted model energy scale by {np.mean(error):.3f} eV/atom')

    # Log the parameters for this run
    mlflow_logger = MLflowLogger()
    mlflow_logger.log_params({
        **args.__dict__.copy(),
        'model': model.__class__.__name__,
        "pytorch version": torch.__version__,
        "ignite version": ignite.__version__,
        "cuda version": torch.version.cuda,
        "device name": torch.cuda.get_device_name(0)
    })


    # Get an initial measurement of model accuracy
    def _evaluate_model(engine: Engine | None = None) -> pd.DataFrame:
        """Evaluate current state of the model against all structures available for training"""
        output = []
        for file in args.train_files:
            for atoms in iread(file):
                output.append({
                    'dft': atoms.get_potential_energy() / len(atoms),
                    'dft-forces': atoms.get_forces(),
                    'ml': calc.get_potential_energy(atoms) / len(atoms),
                    'ml-forces': calc.get_forces(atoms),
                })
        output = pd.DataFrame(output)
        output['rmse'] = [np.power(x - y, 2).mean() for x, y in
                          zip(output['dft-forces'], output['ml-forces'])]
        eng_mae = (output['dft'] - output['ml']).abs().mean()
        logger.info(f'Evaluated model performance on {len(output)} configurations. '
                    f'Force RMSE: {output["rmse"].mean() * 1000:.2f} meV/Ang - Energy MAE: {eng_mae * 1000:.3f} meV/atom')

        if engine is not None:
            mlflow_logger.log_metrics({
                'energy-mae': eng_mae,
                'force-rmse': output["rmse"].mean()
            }, step=engine.state.epoch)
        return output


    output = _evaluate_model()

    # Prepare optimizer
    if args.unfreeze_layers == ['all']:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for unfreeze in args.unfreeze_layers:
            submodel = getattr(model, unfreeze)
            for p in submodel.parameters():
                p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = WeightedHuberEnergyForcesStressLoss(
        energy_weight=args.loss_weights[0],
        forces_weight=args.loss_weights[1],
        stress_weight=args.loss_weights[2],
        huber_delta=args.loss_huber,
    )


    # Prepare the training engine
    def train_step(engine, batch):
        """Borrowed from the training step used inside MACE"""
        model.train()
        opt.zero_grad()
        batch = batch.to('cuda')
        y = model(
            batch,
            training=True,
            compute_force=True,
            compute_virials=False,
            compute_stress=True,
        )
        loss = criterion(pred=y, ref=batch)
        loss.backward()
        opt.step()
        return loss.item()


    val_metrics = {
        "loss": Loss(criterion)
    }
    trainer = Engine(train_step)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, _evaluate_model)

    logger.info('Started training')
    trainer.run(loaders['train'], max_epochs=args.num_epochs)
    logger.info('Finished training')

    # Store the model performance
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        output = _evaluate_model()
        out_path = tmp / 'output.csv.gz'
        output.to_csv(out_path, index=False)
        mlflow_logger.log_artifact(out_path)
        logger.info('Logged the output data')

        # Store the model
        torch.save(calc, tmp / 'model.pt')
        mlflow_logger.log_artifact(tmp / 'model.pt')

    # store the model for ensembling
    ensemble_dir = Path('ensemble')
    ensemble_dir.mkdir(exist_ok=True)
    model = calc.models[0]
    torch.save(model, ensemble_dir / f'model_{args.seed}.pt')
