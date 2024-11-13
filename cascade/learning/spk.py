"""Utilities for using models based on SchNet"""
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import List, Dict
from pathlib import Path
import os

from ase.calculators.calculator import Calculator
from more_itertools import batched
from schnetpack.data import AtomsLoader, ASEAtomsData

from schnetpack import transform as trn
import schnetpack as spk
from torch import optim
import pandas as pd
import numpy as np
import torch
import ase

from .base import BaseLearnableForcefield, State


def ase_to_spkdata(atoms: List[ase.Atoms], path: Path) -> ASEAtomsData:
    """Add a list of Atoms objects to a SchNetPack database

    Args:
        atoms: List of Atoms objects
        path: Path to the database file
    Returns:
        A link to the database
    """

    _props = ['energy', 'forces', 'stress']
    if Path(path).exists():
        raise ValueError('Path already exists')
    db = ASEAtomsData(str(path))

    # Get the properties as dictionaries
    prop_lst = []
    for a in atoms:
        props = {}
        # If we have the property, store it
        if a.calc is not None:
            calc = a.calc
            for k in _props:
                if k in calc.results:
                    props[k] = np.atleast_1d(calc.results[k])
        else:
            # If not, store a placeholder
            props.update(dict((k, np.atleast_1d([])) for k in ['energy', 'forces', 'stress']))
        prop_lst.append(props)
    db.add_systems(prop_lst, atoms)
    return db


class SchnetPackInterface(BaseLearnableForcefield):
    """Forcefield based on the SchNetPack implementation of SchNet"""

    def __init__(self, scratch_dir: Path | None = None, timeout: float = None):
        """

        Args:
            scratch_dir: Directory in which to cache converted data
            timeout: Maximum training time
        """
        super().__init__(scratch_dir)
        self.timeout = timeout

    def evaluate(self,
                 model_msg: bytes | State,
                 atoms: list[ase.Atoms],
                 batch_size: int = 64,
                 device: str = 'cpu') -> (np.ndarray, list[np.ndarray], np.ndarray):
        # Get the message
        model_msg = self.get_model(model_msg)

        # Iterate over chunks, coverting as we go
        converter = spk.interfaces.AtomsConverter(
            neighbor_list=trn.MatScipyNeighborList(cutoff=5.0), dtype=torch.float32, device=device
        )
        energies = []
        forces = []
        stresses = []
        for batch in batched(atoms, batch_size):
            # Push the batch to the device
            inputs = converter(list(batch))
            pred = model_msg(inputs)

            # Extract data
            energies.extend(pred['energy'].detach().cpu().numpy().tolist())
            batch_f = pred['forces'].detach().cpu().numpy()
            forces.extend(np.array_split(batch_f, np.cumsum([len(a) for a in batch]))[:-1])
            print(pred['stress'])
            stresses.append(pred['stress'].detach().cpu().numpy())

        return np.array(energies), forces, np.concatenate(stresses)

    def train(self,
              model_msg: bytes | State,
              train_data: list[ase.Atoms],
              valid_data: list[ase.Atoms],
              num_epochs: int,
              device: str = 'cpu',
              batch_size: int = 32,
              learning_rate: float = 1e-3,
              huber_deltas: tuple[float, float, float] = (0.5, 1, 1),
              force_weight: float = 10,
              stress_weight: float = 100,
              reset_weights: bool = False,
              **kwargs) -> tuple[bytes, pd.DataFrame]:

        # Make sure the models are converted to Torch models
        model_msg = self.get_model(model_msg)

        # If desired, re-initialize weights
        if reset_weights:
            for module in model_msg.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

        # Start the training process
        with TemporaryDirectory(dir=self.scratch_dir, prefix='spk') as td:
            # Save the data to an ASE Atoms database
            train_file = Path(td) / 'train_data.db'
            train_db = ase_to_spkdata(train_data, train_file)
            train_loader = AtomsLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=8,
                                       pin_memory=device != "cpu")

            valid_file = Path(td) / 'valid_data.db'
            valid_db = ase_to_spkdata(train_data, valid_file)
            valid_loader = AtomsLoader(valid_db, batch_size=batch_size, num_workers=8, pin_memory=device != "cpu")

            # Make the trainer
            opt = optim.Adam(model_msg.parameters(), lr=learning_rate)

            # tradeoff
            rho_tradeoff = 0.9

            # loss function
            if huber_deltas is None:
                # Use mean-squared loss
                def loss(batch, result):
                    # compute the mean squared error on the energies
                    diff_energy = batch['energy'] - result['energy']
                    err_sq_energy = torch.mean(diff_energy ** 2)

                    # compute the mean squared error on the forces
                    diff_forces = batch['forces'] - result['forces']
                    err_sq_forces = torch.mean(diff_forces ** 2)

                    # build the combined loss function
                    err_sq = rho_tradeoff * err_sq_energy + (1 - rho_tradeoff) * err_sq_forces

                    return err_sq
            else:
                # Use huber loss
                delta_energy, delta_force = huber_deltas

                def loss(batch: Dict[str, torch.Tensor], result):
                    # compute the mean squared error on the energies per atom
                    n_atoms = batch['_atom_mask'].sum(axis=1)
                    err_sq_energy = torch.nn.functional.huber_loss(batch['energy'] / n_atoms,
                                                                   result['energy'].float() / n_atoms,
                                                                   delta=delta_energy)

                    # compute the mean squared error on the forces
                    err_sq_forces = torch.nn.functional.huber_loss(batch['forces'], result['forces'], delta=delta_force)

                    # build the combined loss function
                    err_sq = rho_tradeoff * err_sq_energy + (1 - rho_tradeoff) * err_sq_forces

                    return err_sq

            metrics = [
                spk.metrics.MeanAbsoluteError('energy'),
                spk.metrics.MeanAbsoluteError('forces')
            ]

            hooks = [
                trn.CSVHook(log_path=td, metrics=metrics),
            ]

            trainer = trn.Trainer(
                model_path=td,
                model=model_msg,
                hooks=hooks,
                loss_fn=loss,
                optimizer=opt,
                train_loader=train_loader,
                validation_loader=valid_loader,
                checkpoint_interval=num_epochs + 1  # Turns off checkpointing
            )

            trainer.train(device, n_epochs=num_epochs)

            # Load in the best model
            model_msg = torch.load(os.path.join(td, 'best_model'), map_location='cpu')

            # Load in the training results
            train_results = pd.read_csv(os.path.join(td, 'log.csv'))

            return self.serialize_model(model_msg), train_results

    def make_calculator(self, model_msg: bytes | State, device: str) -> Calculator:
        # Write model to disk
        with NamedTemporaryFile(suffix='.pt') as tf:
            tf.close()
            tf_path = Path(tf.name)
            tf_path.write_bytes(self.serialize_model(model_msg))

            return spk.interfaces.SpkCalculator(
                model_file=str(tf_path),
                neighbor_list=spk.transform.SkinNeighborList(
                    cutoff_skin=2.0,
                    neighbor_list=spk.transform.ASENeighborList(cutoff=5.)
                ),
                energy_unit='eV',
                stress_key='stress',
                device=device
            )
