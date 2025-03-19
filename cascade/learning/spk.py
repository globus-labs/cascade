"""Utilities for using models based on SchNet"""
from tempfile import TemporaryDirectory, NamedTemporaryFile
from pathlib import Path

from ase.calculators.calculator import Calculator
from schnetpack.data import ASEAtomsData
from schnetpack import transform as trn
from more_itertools import batched
from pytorch_lightning.loggers import CSVLogger
from ase import data
import pytorch_lightning as pl
import schnetpack as spk
import torchmetrics
import pandas as pd
import numpy as np
import torch
import ase
from schnetpack.model import NeuralNetworkPotential

from .base import BaseLearnableForcefield, State
from .utils import estimate_atomic_energies


class SchnetPackInterface(BaseLearnableForcefield[NeuralNetworkPotential]):
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
        model = self.get_model(model_msg)
        model.to(device)

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
            pred = model(inputs)

            # Extract data
            energies.extend(pred['energy'].detach().cpu().numpy().tolist())
            batch_f = pred['forces'].detach().cpu().numpy()
            forces.extend(np.array_split(batch_f, np.cumsum([len(a) for a in batch]))[:-1])
            stresses.append(pred['stress'].detach().cpu().numpy())

        return np.array(energies), forces, np.concatenate(stresses)

    def train(self,
              model_msg: bytes | spk.model.NeuralNetworkPotential,
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
        model = self.get_model(model_msg)

        # If desired, re-initialize weights
        if reset_weights:
            for module in model.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

        # Update the atomic energies using the data from all trajectories
        atomrefs = np.zeros((100,), dtype=np.float32)
        atomic_energies_dict = estimate_atomic_energies(train_data)
        for s, e in atomic_energies_dict.items():
            atomrefs[data.atomic_numbers[s]] = e

        # Access the cutoff distance from the representation layer
        cutoff = model.representation.cutoff_fn.cutoff.cpu().numpy().item()
        for post in model.postprocessors:
            if isinstance(post, trn.AddOffsets):
                post.atomref = torch.from_numpy(atomrefs)

        # Start the training process
        with TemporaryDirectory(dir=self.scratch_dir, prefix='spk') as td:
            td = Path(td)
            # Save the data to a single ASE database, manually setting the splits
            db_file = td / 'train_data.db'
            property_dict = {'energy': 'eV', 'forces': 'eV/Ang', 'stress': 'eV/Ang/Ang/Ang'}
            train_dataset = ASEAtomsData.create(str(db_file),
                                                distance_unit='Ang',
                                                property_unit_dict=property_dict)
            for atoms_list in [train_data, valid_data]:
                for atoms in atoms_list:
                    train_dataset.add_system(atoms,
                                             energy=np.array(atoms.get_potential_energy())[None],
                                             forces=atoms.get_forces(),
                                             stress=atoms.get_stress(voigt=False)[None, :, :])

            dataset = spk.data.AtomsDataModule(
                str(db_file),
                batch_size=batch_size,
                distance_unit='Ang',
                property_units=property_dict,
                num_train=len(train_dataset),
                num_val=len(valid_data),
                transforms=[
                    trn.ASENeighborList(cutoff=cutoff),
                    trn.CastTo32()
                ],
                num_workers=1,
            )
            dataset.train_idx = np.arange(0, len(train_data)).tolist()
            dataset.val_idx = np.arange(len(train_data), len(train_data) + len(valid_data)).tolist()
            dataset.test_idx = np.array([]).tolist()

            # Make the loss function
            output_energy = spk.task.ModelOutput(
                name='energy',
                loss_fn=torch.nn.MSELoss(),
                loss_weight=1,
                metrics={
                    "MAE": torchmetrics.MeanAbsoluteError()
                }
            )
            output_forces = spk.task.ModelOutput(
                name='forces',
                loss_fn=torch.nn.MSELoss(),
                loss_weight=force_weight,
                metrics={
                    "MAE": torchmetrics.MeanAbsoluteError()
                }
            )
            output_stress = spk.task.ModelOutput(
                name='stress',
                loss_fn=torch.nn.MSELoss(),
                loss_weight=stress_weight,
                metrics={
                    "MAE": torchmetrics.MeanAbsoluteError()
                }
            )

            # Make the trainer
            task = spk.task.AtomisticTask(
                model=model,
                outputs=[output_energy, output_forces, output_stress],
                optimizer_cls=torch.optim.AdamW,
                optimizer_args={"lr": learning_rate}
            )

            csv_writer = CSVLogger(td)
            model_path = td / "best_inference_model"
            callbacks = [
                spk.train.ModelCheckpoint(
                    model_path=str(model_path),
                    save_top_k=1,
                    monitor="val_loss"
                ),
            ]

            trainer = pl.Trainer(
                callbacks=callbacks,
                logger=csv_writer,
                default_root_dir=td,
                max_epochs=num_epochs,
                enable_progress_bar=False,
                accelerator=device,
            )
            trainer.fit(task, dataset)

            # Load in the best model
            model = torch.load(model_path, map_location='cpu')

            # Load in the training results
            train_results = pd.read_csv(next(td.rglob('metrics.csv')))

            return self.serialize_model(model), train_results

    def make_calculator(self, model_msg: bytes | NeuralNetworkPotential, device: str) -> Calculator:
        # Write model to disk
        with NamedTemporaryFile(suffix='.pt') as tf:
            tf.close()
            tf_path = Path(tf.name)
            tf_path.write_bytes(self.serialize_model(model_msg))

            model = self.get_model(model_msg)
            cutoff = model.representation.cutoff_fn.cutoff.cpu().numpy().item()
            return spk.interfaces.SpkCalculator(
                model_file=str(tf_path),
                neighbor_list=spk.transform.SkinNeighborList(
                    cutoff_skin=2.0,
                    neighbor_list=spk.transform.ASENeighborList(cutoff=cutoff)
                ),
                energy_unit='eV',
                stress_key='stress',
                device=device
            )
