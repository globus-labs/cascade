"""Utilities and data models used when finetuning a model"""
from typing import Collection, Iterable, Any
from dataclasses import dataclass, field

from ase import Atoms


def filter_by_elements(atoms_gen: Iterable[Atoms], allowed_elems: Collection[str]) -> Iterable[Atoms]:
    """Process a stream of entries to only include those with allowed elements

    Args:
        atoms_gen: Stream of Atoms structures to be filtered
        allowed_elems: List of elements which are allowed in the dataset
    Yields:
        Atoms from the stream which contain only the desired elements
    """

    allowed_elems = set(allowed_elems)
    for atoms in atoms_gen:
        elems = set(atoms.get_chemical_symbols())
        if any(e not in allowed_elems for e in elems):
            continue
        yield atoms


# TODO (wardlt): Build towards more advanced methods, like
@dataclass
class MultiHeadConfig:
    """Configuration used to define replay training"""

    # Defining the training data
    original_dataset: list[Atoms] = ...
    """Path to dataset containing the original training samples

    Must be in a form readable by ASE.
    """
    num_downselect: int | None = None
    """Number of points from the dataset to use for training each training round"""

    # Defining the training procedure
    epoch_frequency: int = 1
    """How often to retrain using the original dataset"""
    lr_reduction: float = 1
    """Factor by which to reduce the learning rate during replay"""
    batch_size: int | None = None
    """Batch size to use during replay"""

    learner_options: dict[str, Any] = field(default_factory=dict)
    """Options specific to a certain learner"""
