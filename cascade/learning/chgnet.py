"""Training based on the `"ChargeNet" neural network <https://doi.org/10.1038/s42256-023-00716-3>`_"""
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import ase
import pandas as pd
from ase import units
from ase.calculators.calculator import Calculator
from chgnet.data.dataset import StructureData, get_loader
from chgnet.model import CHGNet, CHGNetCalculator
from chgnet.trainer import Trainer
from pymatgen.io.ase import AseAtomsAdaptor

from .base import BaseLearnableForcefield, State


def make_chgnet_dataset(atoms: list[ase.Atoms]) -> StructureData:
    """Make a dataset ready for use by CHGNet for training

    Args:
        atoms: List of atoms with computed properties
    Returns:
        Training dataset including forces, energies, and stresses
    """

    structures = []
    energies = []
    forces = []
    stresses = []
    for a in atoms:
        structures.append(AseAtomsAdaptor.get_structure(a))

        # Retrieve the properties from the ASE calculator
        for p in ['energy', 'forces', 'stress']:
            if p not in a.calc.results:
                raise ValueError(f'Atoms is missing property: {p}')

        energies.append(a.get_potential_energy() / len(a))  # CHGNet uses atomic energies
        forces.append(a.get_forces())
        stresses.append(a.get_stress(voigt=False) / units.GPa * -10)  # CHGNet expects training data in kBar and using VASP's sign convention

    return StructureData(
        structures=structures,
        energies=energies,
        forces=forces,
        stresses=stresses,
    )


class CHGNetInterface(BaseLearnableForcefield[CHGNet]):
    """Interface to training and running CHGnet forcefields"""

    def evaluate(self,
                 model_msg: bytes | CHGNet,
                 atoms: list[ase.Atoms],
                 batch_size: int = 64,
                 device: str = 'cpu') -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        model = self.get_model(model_msg)
        model.to(device)

        # Convert structures to Pymatgen
        structures = [AseAtomsAdaptor.get_structure(a) for a in atoms]

        # Run everything
        model.to(device)
        preds = model.predict_structure(structures, task='efs', batch_size=batch_size)
        model.to('cpu')

        # Transpose into Numpy arrayes
        energies = np.array([r['e'] for r in preds])
        if model.is_intensive:
            atom_counts = np.array([len(a) for a in atoms])
            energies *= atom_counts
        forces = [r['f'][:len(a), :] for a, r in zip(atoms, preds)]
        stress = np.array([r['s'] for r in preds]) * units.GPa
        return energies, forces, stress

    def train(self,
              model_msg: bytes | CHGNet,
              train_data: list[ase.Atoms],
              valid_data: list[ase.Atoms],
              num_epochs: int,
              device: str = 'cpu',
              batch_size: int = 32,
              learning_rate: float = 1e-3,
              huber_deltas: tuple[float, float, float] = (0.1, 0.1, 0.1),
              force_weight: float = 1,
              stress_weight: float = 0.1,
              reset_weights: bool = False,
              **kwargs) -> tuple[bytes, pd.DataFrame]:
        model = self.get_model(model_msg)

        # Reset weights, if needed
        if reset_weights:
            def init_params(m):
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()

            model.apply(init_params)

        with TemporaryDirectory(prefix='chgnet_') as tmpdir:
            tmpdir = Path(tmpdir)
            with open(tmpdir / 'chgnet.stdout', 'w') as fp, redirect_stdout(fp):
                # Make the data loaders
                train_dataset = make_chgnet_dataset(train_data)
                valid_dataset = make_chgnet_dataset(valid_data)
                train_loader = get_loader(train_dataset, batch_size=batch_size)
                valid_loader = get_loader(valid_dataset, batch_size=batch_size)

                # Fit the atomic reference energies
                model.composition_model.fit(
                    train_dataset.structures,
                    train_dataset.energies
                )

                # Run the training
                trainer = Trainer(
                    model=model,
                    targets='efs',
                    criterion="Huber",
                    force_loss_ratio=force_weight,
                    stress_loss_ratio=stress_weight,
                    epochs=num_epochs,
                    learning_rate=learning_rate,
                    use_device=device,
                    print_freq=num_epochs + 1
                )
                trainer.train(train_loader, valid_loader, train_composition_model=True, save_dir=str(tmpdir))
                model.to('cpu')

                # Store the results
                best_model = trainer.get_best_model()

        log = {}
        for key_1, history_1 in trainer.training_history.items():
            for key_2, history in history_1.items():
                if len(history) != num_epochs:
                    continue
                log[f'{key_1}_{key_2}'] = history
        log = pd.DataFrame(log)

        return self.serialize_model(best_model), log

    def make_calculator(self, model_msg: bytes | State, device: str) -> Calculator:
        model = self.get_model(model_msg)
        return CHGNetCalculator(model=model, use_device=device)
