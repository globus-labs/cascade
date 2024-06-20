"""Interface and glue code for to models built using `TorchANI <https://github.com/aiqm/torchani>_"""
import ase
import numpy as np
from torch.utils.data import DataLoader

from torchani import AEVComputer, ANIModel
from torchani.aev import SpeciesAEV
from torchani.data import collate_fn
import torch

from cascade.learning.base import BaseLearnableForcefield

__all__ = ['TorchANI']

ANIModelContents = tuple[AEVComputer, ANIModel, dict[str, float]]
"""Contents of the serialized form of a model:

1. Compute for atomic environments
2. The model which maps environments to energies
3. Ordered dict of chemical symbol to atomic energies (all Py3 dicts are ordered)
"""

my_collate_dict = {
    'species': -1,
    'coordinates': 0.0,
    'forces': 0.0,
    'energies': 0.0,
    'cells': 0.0
}


def ase_to_ani(atoms: ase.Atoms, species: list[str]) -> dict[str, torch.Tensor]:
    """Make an ANI-format dictionary from an ASE Atoms object

    Args:
        atoms: Atoms object to be converted
        species: List of species used to determine index given chemical symbol
    Returns:
        Atoms object in a tensor format
    """

    # An entry _must_ have the species and coordinates
    output = {
        'species': torch.from_numpy(np.array(([species.index(s) for s in atoms.symbols]))),
        'coordinates': torch.from_numpy(atoms.positions).float(),
        'cells': torch.from_numpy(atoms.cell.array).float()
    }

    if atoms.calc is not None:
        if 'energy' in atoms.calc.results:
            output['energies'] = torch.from_numpy(np.atleast_1d(atoms.get_potential_energy())).float()
        if 'forces' in atoms.calc.results:
            output['forces'] = torch.from_numpy(atoms.get_forces()).float()
    return output


def forward_batch(batch: dict[str, torch.Tensor],
                  aev_computer: AEVComputer,
                  nn: ANIModel,
                  atom_energies: np.ndarray,
                  pbc: torch.Tensor,
                  device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
    """Run the forward step on a batch of entries

    Args:
        batch: Batch from the data loader
        aev_computer: Atomic environment computer
        nn: Model which maps atomic environments to energies
        atom_energies: Array holding the reference energy for each species
        pbc: Periodic boundary conditions used by all members of the batch
        device: Device on which to run computations
    Returns:
        - Energies for each member
        - Forces for each member
    """
    # Move the data to the device
    batch_z = batch['species'].to(device)
    batch_x = batch['coordinates'].float().to(device).requires_grad_(True)
    batch_c = batch['cells'].float().to(device)

    # Compute the energy offset per member (run on the CPU because it's fast)
    batch_o = atom_energies[batch['species'].numpy()].sum(axis=1)
    batch_o = torch.from_numpy(batch_o).to(device)

    # Compute the AEVs individually because TorchANI assumes all entries have the same cell size
    batch_a = []
    for row_z, row_x, row_c in zip(batch_z, batch_x, batch_c):
        row_z = torch.unsqueeze(row_z, 0)
        row_x = torch.unsqueeze(row_x, 0)
        batch_a.append(aev_computer((row_z, row_x), row_c, pbc).aevs)
    batch_a = torch.concat(batch_a)
    batch_a = SpeciesAEV(batch_z, batch_a)

    # Get the energies for each member
    _, batch_e_pred = nn(batch_a)
    batch_e_pred = batch_e_pred + batch_o
    batch_f_pred = -torch.autograd.grad(batch_e_pred.sum(), batch_x, create_graph=True)[0]
    return batch_e_pred, batch_f_pred


class TorchANI(BaseLearnableForcefield[ANIModelContents]):
    """Interface to the high-dimensional neural networks implemented by `TorchANI <https://github.com/aiqm/torchani>`_"""

    def evaluate(self,
                 model_msg: bytes | ANIModelContents,
                 atoms: list[ase.Atoms],
                 batch_size: int = 64,
                 device: str = 'cpu') -> (np.ndarray, list[np.ndarray]):

        # TODO (wardlt): Put model in "eval" mode, skip making the graph when computing gradients in `forward_batch` <- performance optimizations

        # Unpack the model
        if isinstance(model_msg, bytes):
            model_msg = self.get_model(model_msg)
        aev_computer, model, atomic_energies = model_msg
        model.to(device)
        aev_computer.to(device)

        # Convert data into PyTorch format
        species = list(atomic_energies.keys())
        atoms_ani = [ase_to_ani(a, species) for a in atoms]

        # Unpack the reference energies as a float32 array
        ref_energies = np.array([atomic_energies[s] for s in species]).astype(np.float32)

        # Build the data loader
        loader = DataLoader(atoms_ani,
                            collate_fn=lambda x: collate_fn(x, my_collate_dict),
                            batch_size=batch_size,
                            shuffle=False)

        # Run inference on all data
        energies = []
        forces = []
        pbc = torch.from_numpy(np.ones((3,), bool)).to(device)  # TODO (don't hard code to 3D)
        for batch in loader:
            batch_e_pred, batch_f_pred = forward_batch(batch, aev_computer, model, ref_energies, pbc, device)
            energies.extend(batch_e_pred.detach().cpu().numpy())  # Energies are the same regardless of size of input

            # The shape of the force array differs depending on size
            batch_n = (batch['species'] >= 0).sum(dim=1).cpu().numpy()  # Number of real atoms per batch
            for entry_f, entry_n in zip(batch_f_pred.detach().cpu().numpy(), batch_n):
                forces.append(entry_f[:entry_n, :])

        return np.array(energies), list(forces)
