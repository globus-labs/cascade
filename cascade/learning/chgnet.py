"""Training based on the `"ChargeNet" neural network <https://doi.org/10.1038/s42256-023-00716-3>`_"""
import numpy as np
import ase
from ase import units
from ase.calculators.calculator import Calculator
from chgnet.model import CHGNet, CHGNetCalculator
from pymatgen.io.ase import AseAtomsAdaptor

from .base import BaseLearnableForcefield, State


class CHGNetInterface(BaseLearnableForcefield[CHGNet]):
    """Interface to training and running CHGnet forcefields"""

    def evaluate(self,
                 model_msg: bytes | CHGNet,
                 atoms: list[ase.Atoms],
                 batch_size: int = 64,
                 device: str = 'cpu') -> (np.ndarray, list[np.ndarray], np.ndarray):
        model = self.get_model(model_msg)

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

    def make_calculator(self, model_msg: bytes | State, device: str) -> Calculator:
        model = self.get_model(model_msg)
        return CHGNetCalculator(model=model, use_device=device)
