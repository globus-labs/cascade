"""Interface to the higher-order equivariant neural networks
of `Batatia et al. <https://arxiv.org/abs/2206.07697>`_"""

import logging
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import ase
import torch
import numpy as np
import pandas as pd
from ase import Atoms, data
from ase.calculators.calculator import Calculator
from ignite.engine import Engine, Events
from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.modules import WeightedHuberEnergyForcesStressLoss, ScaleShiftMACE
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric.dataloader import DataLoader
from mace.tools.scripts_utils import extract_config_mace_model
from mace.calculators import MACECalculator

from cascade.learning.base import BaseLearnableForcefield, State
from cascade.learning.finetuning import MultiHeadConfig
from cascade.learning.utils import estimate_atomic_energies

logger = logging.getLogger(__name__)

MACEState = ScaleShiftMACE
"""Just the model, which we require being the MACE which includes scale shifting logic"""


def _update_offset_factors(model: ScaleShiftMACE, train_data: list[Atoms], train_loader: DataLoader, device: str):
    """Update the atomic energies and scale offset layers of a model

    Args:
        model: Model to be adjusted
        train_data: Training dataset
        train_loader: Loader built using the training set
        device: Device on which to perform inference
    """
    # Update the atomic energies using the data from all trajectories
    z_table = AtomicNumberTable(model.atomic_numbers.cpu().numpy().tolist())
    new_ae = model.atomic_energies_fn.atomic_energies.cpu().numpy()
    atomic_energies_dict = estimate_atomic_energies(train_data)
    for s, e in atomic_energies_dict.items():
        new_ae[z_table.zs.index(data.atomic_numbers[s])] = e
    with torch.no_grad():
        old_ae = model.atomic_energies_fn.atomic_energies
        model.atomic_energies_fn.atomic_energies = torch.from_numpy(new_ae).to(old_ae.dtype).to(old_ae.device)

    # Update the shift of the energy scale
    errors = []
    for batch in train_loader:
        batch = batch.to(device)
        num_atoms = batch.ptr[1:] - batch.ptr[:-1]  # Use the offsets to compute the number of atoms per inference
        ml = model(
            batch,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
        )
        error = (ml["energy"] - batch["energy"]) / num_atoms
        errors.extend(error.cpu().detach().numpy().tolist())
    model.scale_shift.shift -= np.mean(errors)


# TODO (wardlt): Use https://github.com/ACEsuit/mace/pull/830 when merged
def freeze_layers(model: torch.nn.Module, n: int = 4) -> None:
    """
    Freezes the first `n` layers of a model. If `n` is negative, freezes the last `|n|` layers.
    Args:
        model (torch.nn.Module): The model.
        n (int): The number of layers to freeze.
    """
    layers = list(model.children())
    num_layers = len(layers)

    logging.info(f"Total layers in model: {num_layers}")

    if abs(n) > num_layers:
        logging.warning(
            f"Requested {n} layers, but model only has {num_layers}. Adjusting `n` to fit the model."
        )
        n = num_layers if n > 0 else -num_layers

    frozen_layers = layers[:n] if n > 0 else layers[n:]

    logging.info(f"Freezing {len(frozen_layers)} layers.")

    for layer in frozen_layers:
        for param in layer.parameters():
            param.requires_grad = False


def atoms_to_loader(atoms: list[Atoms], batch_size: int, z_table: AtomicNumberTable, r_max: float, **kwargs):
    """
    Make a data loader from a list of ASE atoms objects

    Args:
        atoms: Atoms from which to create the loader
        batch_size: Batch size for the loader
        z_table: Map between atom ID in mace and periodic table
        r_max: Cutoff distance
    """

    def _prepare_atoms(my_atoms: Atoms):
        """MACE expects the training outputs to be stored in `info` and `arrays`"""
        # Start with a copy of positions, which should be available always
        my_atoms.arrays.update({
            'positions': my_atoms.positions,
        })

        if my_atoms.calc is None:
            return my_atoms  # No calc, no results

        # Now make an info dictionary if one doesn't exist yet
        if my_atoms.info is None:
            my_atoms.info = {}

        # Copy over all property data which exists
        if 'energy' in my_atoms.calc.results:
            my_atoms.info['energy'] = my_atoms.get_potential_energy()

        if 'stress' in my_atoms.calc.results:
            my_atoms.info['stress'] = my_atoms.get_stress()

        if 'forces' in my_atoms.calc.results:
            my_atoms.arrays['forces'] = my_atoms.get_forces()

        return my_atoms

    atoms = [config_from_atoms(_prepare_atoms(a)) for a in atoms]
    return DataLoader(
        [AtomicData.from_config(c, z_table=z_table, cutoff=r_max) for c in atoms],
        batch_size=batch_size,
        **kwargs
    )


class MACEInterface(BaseLearnableForcefield[MACEState]):
    """Interface to the `MACE library <https://github.com/ACEsuit/mace>`_"""

    def create_extra_heads(self, model: ScaleShiftMACE, num_heads: int) -> list[ScaleShiftMACE]:
        """Create multiple instances of a ScaleShiftMACE model that share some of the same layers

        The new models will share the node embedding, interaction, and product layers;
        but will have separate atomic energy, readout, and scale_shift layers.

        Args:
            model: Model to be replicated
            num_heads: Number of replicas to create
        Returns:
            Additional copies of the model with the same internal layers
        """

        _shared_layers = ('node_embedding', 'interactions', 'products')

        output = []
        for _ in range(num_heads):
            # Make a deep copy of the model
            new_model = self.get_model(self.serialize_model(model))

            # Copy over the shared layers
            for layer in _shared_layers:
                setattr(new_model, layer, getattr(model, layer))
            output.append(new_model)
        return output

    def evaluate(self,
                 model_msg: bytes | State,
                 atoms: list[ase.Atoms],
                 batch_size: int = 64,
                 device: str = 'cpu') -> (np.ndarray, list[np.ndarray], np.ndarray):
        # Ready the models and the data
        model = self.get_model(model_msg)
        r_max = model.r_max.item()
        z_table = AtomicNumberTable(model.atomic_numbers.cpu().numpy().tolist())

        model.to(device)
        loader = atoms_to_loader(atoms, batch_size=batch_size, z_table=z_table, r_max=r_max, shuffle=False, drop_last=False)

        # Compile results
        energies = []
        forces = []
        stresses = []
        model.to(device)
        model.eval()
        for batch in loader:
            batch = batch.to(device)
            y = model(
                batch,
                training=False,
                compute_force=True,
                compute_virials=False,
                compute_stress=True,
            )
            energies.extend(y['energy'].cpu().detach().numpy())
            stresses.extend(y['stress'].cpu().detach().numpy())
            forces_numpy = y['forces'].cpu().detach().numpy()
            for i, j in zip(batch.ptr, batch.ptr[1:]):
                forces.append(forces_numpy[i:j, :])
        return np.array(energies), forces, np.array(stresses)

    def train(self,
              model_msg: bytes | State,
              train_data: list[Atoms],
              valid_data: list[Atoms],
              num_epochs: int,
              device: str = 'cpu',
              batch_size: int = 32,
              learning_rate: float = 1e-3,
              huber_deltas: tuple[float, float, float] = (0.5, 1, 1),
              force_weight: float = 10,
              stress_weight: float = 100,
              reset_weights: bool = False,
              patience: int | None = None,
              num_freeze: int | None = None,
              replay: MultiHeadConfig | None = None
              ) -> tuple[bytes, pd.DataFrame]:
        """Train a model

        Args:
            model_msg: Model to be retrained
            train_data: Structures used for training
            valid_data: Structures used for validation
            num_epochs: Number of training epochs
            device: Device (e.g., 'cuda', 'cpu') used for training
            batch_size: Batch size during training
            learning_rate: Initial learning rate for optimizer
            huber_deltas: Delta parameters for the loss functions for energy and force
            force_weight: Amount of weight to use for the force part of the loss function
            stress_weight: Amount of weight to use for the stress part of the loss function
            reset_weights: Whether to reset the weights before training
            patience: Halt training after validation error increases for these many epochs
            num_freeze: Number of layers to freeze. Starts from the top of the model (node embedding)
                See: `Radova et al. <https://arxiv.org/html/2502.15582v1>`_
            replay: Settings for replaying an initial training set
        Returns:
            - model: Retrained model
            - history: Training history
        """

        # Load the model
        model = self.get_model(model_msg)
        r_max = model.r_max.item()
        z_table = AtomicNumberTable(model.atomic_numbers.cpu().numpy().tolist())

        # Reset weights if desired
        if reset_weights:
            config = extract_config_mace_model(model)
            model = ScaleShiftMACE(**config)
        model.to(device)

        # Unpin weights
        for p in model.parameters():
            p.requires_grad = True

        # Freeze desired layers
        if num_freeze is not None:
            freeze_layers(model, num_freeze)

        # Convert the training data from ASE -> MACE Configs
        train_loader = atoms_to_loader(train_data, batch_size, z_table, r_max, shuffle=True, drop_last=True)
        valid_loader = atoms_to_loader(valid_data, batch_size, z_table, r_max, shuffle=False, drop_last=True)

        # Update the atomic energies for the current dataset
        _update_offset_factors(model, train_data, train_loader, device)

        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = WeightedHuberEnergyForcesStressLoss(
            energy_weight=1,
            forces_weight=force_weight,
            stress_weight=stress_weight,
            huber_delta=huber_deltas[0],
        )

        # Prepare the training engine
        train_losses = []

        def get_loss_stats(b, y):
            """Compute the losses"""
            na = b.ptr[1:] - b.ptr[:-1]
            return {
                'energy_mae': torch.mean(torch.abs(b['energy'] - y['energy']) / na).item(),
                'force_rmse': torch.sqrt(torch.square(b['forces'] - y['forces']).mean()).item(),
                'stress_rmse': torch.sqrt(torch.square(b['stress'] - y['stress']).mean()).item()
            }

        def train_step(engine, batch):
            """Borrowed from the training step used inside MACE"""
            model.train()
            opt.zero_grad()
            batch = batch.to(device)
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

            # Get the training stats
            detailed_loss = get_loss_stats(batch, y)
            detailed_loss['epoch'] = engine.state.epoch - 1
            detailed_loss['total_loss'] = loss.item()
            train_losses.append(detailed_loss)
            return loss.item()

        trainer = Engine(train_step)

        # Make the validation step
        valid_losses = []
        patience_status = {'best_loss': np.inf, 'patience': patience}

        @trainer.on(Events.EPOCH_COMPLETED)
        def validation_process(engine: Engine):
            model.eval()
            logger.info(f'Started validation for epoch {engine.state.epoch - 1}')

            for batch in valid_loader:
                batch.to(device)
                y = model(
                    batch,
                    training=False,
                    compute_force=True,
                    compute_virials=False,
                    compute_stress=True,
                )
                loss = criterion(pred=y, ref=batch)
                detailed_loss = get_loss_stats(batch, y)
                detailed_loss['epoch'] = engine.state.epoch - 1
                detailed_loss['total_loss'] = loss.item()
                valid_losses.append(detailed_loss)
            logger.info(f'Completed validation for epoch {engine.state.epoch - 1}')

        # Add multi-head replay, if desired
        if replay is not None:
            # Downselect data, if desired
            #  TODO (wardlt): Use FPS to pick samples, as in https://arxiv.org/abs/2412.02877
            if replay.num_downselect == 0:
                replay_data = replay.original_dataset.copy()
                random.shuffle(replay_data)
                replay_data = replay_data[:replay.num_downselect]
            else:
                replay_data = replay.original_dataset

            # Create and re-scale replay
            replay_model = self.create_extra_heads(model, 1)[0]
            replay_model.to(device)
            replay_loader = atoms_to_loader(replay_data, replay.batch_size or batch_size,
                                            z_table, r_max, shuffle=True, drop_last=True)
            _update_offset_factors(replay_model, replay_data, replay_loader, device)

            # Make the replay loss
            replay_opt = torch.optim.Adam(replay_model.parameters(), lr=learning_rate // replay.lr_reduction)

            @trainer.on(Events.EPOCH_COMPLETED(every=replay.epoch_frequency))
            def replay_process(engine: Engine):
                replay_model.train()
                epoch = engine.state.epoch - 1
                logger.info(f'Started replay for epoch {engine.state.epoch - 1}')

                for batch in replay_loader:
                    batch.to(device)
                    y = replay_model(
                        batch,
                        training=True,
                        compute_force=True,
                        compute_virials=False,
                        compute_stress=True,
                    )
                    loss = criterion(pred=y, ref=batch)
                    loss.backward()
                    replay_opt.step()

                    detailed_loss = dict((f'{k}_replay', v) for k, v in get_loss_stats(batch, y).items())
                    detailed_loss['epoch'] = engine.state.epoch - 1
                    detailed_loss['total_loss_replay'] = loss.item()
                    valid_losses.append(detailed_loss)

            # Add early stopping if desired
            if patience_status['patience'] is not None:
                cur_loss = np.mean([x['total_loss'] for x in valid_losses if x['epoch'] == engine.state.epoch - 1])
                if cur_loss < patience_status['best_loss']:
                    patience_status['best_loss'] = cur_loss
                    patience_status['patience'] = patience
                else:
                    patience_status['patience'] -= 1

                if patience_status['patience'] < 0:
                    engine.terminate()
                    logger.info('Early stopping criterion met')

        logger.info('Started training')
        trainer.run(train_loader, max_epochs=num_epochs)
        logger.info('Finished training')
        model.cpu()  # Force it off the GPU

        # Compile the loss
        train_losses = pd.DataFrame(train_losses).groupby('epoch').mean().reset_index()
        valid_losses = pd.DataFrame(valid_losses).groupby('epoch').mean().reset_index()
        log = train_losses.merge(valid_losses, on='epoch', suffixes=('_train', '_valid'))
        return self.serialize_model(model), log

    def make_calculator(self, model_msg: bytes | State, device: str) -> Calculator:
        # MACE calculator loads the model from disk, so let's write to disk
        with TemporaryDirectory(self.scratch_dir, prefix='mace_') as tmp:
            model_path = Path(tmp) / 'model.pt'
            model_path.write_bytes(self.serialize_model(model_msg))

            return MACECalculator(model_paths=[model_path], device=device, compile_mode=None)
