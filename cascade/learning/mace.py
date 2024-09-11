"""Interface to the higher-order equivariant neural networks
of `Batatia et al. <https://arxiv.org/abs/2206.07697>`_"""

import logging
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
from cascade.learning.utils import estimate_atomic_energies

logger = logging.getLogger(__name__)

MACEState = ScaleShiftMACE
"""Just the model, which we require being the MACE which includes scale shifting logic"""


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
        my_atoms.info = {
            'energy': my_atoms.get_potential_energy(),
            'stress': my_atoms.get_stress()
        }
        my_atoms.arrays.update({
            'forces': my_atoms.get_forces(),
            'positions': my_atoms.positions,
        })
        return my_atoms

    atoms = [config_from_atoms(_prepare_atoms(a)) for a in atoms]
    return DataLoader(
        [AtomicData.from_config(c, z_table=z_table, cutoff=r_max) for c in atoms],
        batch_size=batch_size,
        **kwargs
    )


class MACEInterface(BaseLearnableForcefield[MACEState]):
    """Interface to the `MACE library <https://github.com/ACEsuit/mace>`_"""

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
        for batch in loader:
            y = model(
                batch,
                training=False,
                compute_force=True,
                compute_virials=False,
                compute_stress=True,
            )
            energies.extend(y['energy'].cpu().detach().numpy())
            stresses.append(y['stress'].cpu().detach().numpy())
            forces_numpy = y['forces'].cpu().detach().numpy()
            for i, j in zip(batch.ptr, batch.ptr[1:]):
                forces.append(forces_numpy[i:j, :])
        return np.array(energies), forces, np.array(stresses)[0, :, :, :]

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
              **kwargs) -> tuple[bytes, pd.DataFrame]:

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

        # Convert the training data from ASE -> MACE Configs
        train_loader = atoms_to_loader(train_data, batch_size, z_table, r_max, shuffle=True, drop_last=True)
        valid_loader = atoms_to_loader(valid_data, batch_size, z_table, r_max, shuffle=False, drop_last=True)

        # Update the atomic energies using the data from all trajectories
        new_ae = model.atomic_energies_fn.atomic_energies.cpu().numpy()
        atomic_energies_dict = estimate_atomic_energies(train_data)
        for s, e in atomic_energies_dict.items():
            new_ae[z_table.zs.index(data.atomic_numbers[s])] = e

        with torch.no_grad():
            old_ae = model.atomic_energies_fn.atomic_energies
            model.atomic_energies_fn.atomic_energies = torch.from_numpy(new_ae).to(old_ae.dtype).to(old_ae.device)
        logger.info('Updated atomic energies')

        # Update the shift of the energy scale
        errors = []
        for _, batch in zip(range(4), train_loader):  # Use only the first 4 batches, for computational efficiency
            num_atoms = batch.ptr[1:] - batch.ptr[:-1]  # Taken from loss function, still don't understand it
            batch = batch.to(device)
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
        logger.info(f'Shifted model energy scale by {np.mean(errors):.3f} eV/atom')

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
            na = batch.ptr[1:] - batch.ptr[:-1]
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

        @trainer.on(Events.EPOCH_COMPLETED)
        def validation_process(engine):
            model.eval()

            for batch in valid_loader:
                batch.to(device)
                y = model(
                    batch,
                    training=True,
                    compute_force=True,
                    compute_virials=False,
                    compute_stress=True,
                )
                loss = criterion(pred=y, ref=batch)
                detailed_loss = get_loss_stats(batch, y)
                detailed_loss['epoch'] = engine.state.epoch - 1
                detailed_loss['total_loss'] = loss.item()
                valid_losses.append(detailed_loss)

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
