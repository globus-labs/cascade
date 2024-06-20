"""Tools used by many types of surrogates"""

from ase import Atoms
import numpy as np


def estimate_atomic_energies(data: list[Atoms]) -> dict[str, float]:
    """Estimate the energy offset

    Args:
        data: List of atomic configurations that include energy
    Returns:
        List of energies per atom type
    """

    # Determine the available species
    species = set()
    for atoms in data:
        species.update(atoms.symbols)
    species = sorted(species)

    x = np.zeros((len(data), len(species)))
    for i, a in enumerate(data):
        symbols = a.symbols
        for j, s in enumerate(species):
            x[i, j] = symbols.count(s)
    y = [a.get_potential_energy() for a in data]
    b = np.linalg.lstsq(x, y, rcond=None)[0]

    return dict(zip(species, b))
