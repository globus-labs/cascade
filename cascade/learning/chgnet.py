"""Training based on the `"ChargeNet" neural network <https://doi.org/10.1038/s42256-023-00716-3>`_"""
import ase
import numpy as np

from chgnet.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor

from .base import BaseLearnableForcefield


class CHGNetInterface(BaseLearnableForcefield[CHGNet]):
    """Interface to training and running CHGnet forcefields"""

    def evaluate(self,
                 model_msg: bytes | CHGNet,
                 atoms: list[ase.Atoms],
                 batch_size: int = 64,
                 device: str = 'cpu') -> (np.ndarray, list[np.ndarray], np.ndarray):
        model = model_msg
        if isinstance(model_msg, bytes):
            model = self.get_model(model_msg)

        # Convert structures to Pymatgen
        structures = [AseAtomsAdaptor.get_structure(a) for a in atoms]

        # Run everything
        model.to(device)
        preds = model.predict_structure(structures, task='efs', batch_size=batch_size)
        model.to('cpu')

        # Transpose into Numpy arrayes
        energies = np.array([r['e'] for r in preds])
        forces = [r['f'][:len(a), :] for a, r in zip(atoms, preds)]
        stress = np.array([r['s'] for r in preds])
        return energies, forces, stress
