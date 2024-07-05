"""Interface definitions"""
from io import BytesIO
from pathlib import Path
from typing import Generic, TypeVar

import ase
from ase.calculators.calculator import Calculator
import numpy as np
import pandas as pd
import torch.nn

from cascade.calculator import EnsembleCalculator

# TODO (wardlt): Break the hard-wire to PyTorch, maybe. I don't have a model yet which uses something else
State = TypeVar('State')
"""Generic type for the state of a certain model"""


class BaseLearnableForcefield(Generic[State]):
    """Interface for learning and evaluating a forcefield

    Using a Learnable Forcefield
    ----------------------------


    The learnable forcefield class defines a reduced interface to a surrogate model
    that computes the energies and forces of a system of atoms.
    The interfaces are designed to be simple to facilitate integration within a workflow
    and operate on serializable Python types to allow the workflow to run across distributed nodes.

    The key functions for use in workflows are

    - :meth:`evaluate` to predict the energies and forces of a series of atomic structures
    - :meth:`train` to update the machine learning model given a new set of structures
    - :meth:`make_make_calculator` to produce an ASE Calculator suitable for use in running dynamics

    The first argument to each of these functions is a "State" that is the components of a machine learning model
    in their original or serialized form.
    Each implementation varies in what defines the "State" of a model, but all share
    the :meth:`serialize_model` function to produce a byte-string version of the State
    before sending it to a remote compute node.

    Implementing a Learnable Forcefield
    -----------------------------------

    Implementations must define the :meth:`evaluate` and :meth:`train` functions,
    which provide an imperfect but sufficient interface for training the model.

    The functions must take either the serialized version of the model as a byte string
    or the unserialized version for workflows run on a single node.
    Use the :meth:`get_model` function to deserialize a byte string.

    Express the type used to express the state of your model as a Python type specification
    that as passed as a generic argument to the class.
    For example,

    .. code: python

        State = list[float]

        class MyClass(BaseLearnableForcefield[State]):

            def evaluate(...

    Add any arguments to the :meth:`train` function as appropriate for updating the weights
    of a machine learning model, but do not add any for creating a new architecture.
    Cascade workflows are designed to use a fixed architecture rather than
    perform hyperparameter optimization.
    Create utility functions for defining the architecture in a separate module.
    """

    def __init__(self, scratch_dir: Path | None = None):
        """

        Args:
            scratch_dir: Path used to store temporary files
        """
        self.scratch_dir = scratch_dir

    def serialize_model(self, state: State) -> bytes:
        """Serialize the state of a model into a byte string

        Args:
            state: Model state
        Returns:
            Form ready for transmission to a compute node
        """
        b = BytesIO()
        torch.save(state, b)
        return b.getvalue()

    def get_model(self, model_msg: bytes) -> State:
        """Load a model from the provided message and place on the CPU memory

        Args:
            model_msg: Model message
        Returns:
            The model ready for use in a function
        """
        return torch.load(BytesIO(model_msg), map_location='cpu')

    def evaluate(self,
                 model_msg: bytes | State,
                 atoms: list[ase.Atoms],
                 batch_size: int = 64,
                 device: str = 'cpu') -> (np.ndarray, list[np.ndarray]):
        """Run inference for a series of structures

        Args:
            model_msg: Model to evaluate
            atoms: List of structures to evaluate
            batch_size: Number of molecules to evaluate per batch
            device: Device on which to run the computation
        Returns:
            - Energies for each inference. (N,) array of floats, where N is the number of structures
            - Forces for each inference. List of N arrays of (n, 3), where n is the number of atoms in each structure
        """
        raise NotImplementedError()

    def train(self,
              model_msg: bytes | State,
              train_data: list[ase.Atoms],
              valid_data: list[ase.Atoms],
              num_epochs: int,
              device: str = 'cpu',
              batch_size: int = 32,
              learning_rate: float = 1e-3,
              huber_deltas: tuple[float, float] = (0.5, 1),
              force_weight: float = 0.9,
              reset_weights: bool = False,
              **kwargs) -> tuple[bytes, pd.DataFrame]:
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
            force_weight: Amount of weight to use for the energy part of the loss function
            reset_weights: Whether to reset the weights before training
        Returns:
            - model: Retrained model
            - history: Training history
        """
        raise NotImplementedError()

    def make_calculator(self, model_msg: bytes | State, device: str) -> Calculator:
        """Make an ASE calculator form of the provided model

        Args:
            model_msg: Serialized form of the model
            device: Device on which to run computations
        Returns:
            Model turned into a calculator
        """
        raise NotImplementedError()

    def make_ensemble_calculator(self, model_msgs: list[bytes | State], device: str) -> EnsembleCalculator:
        raise NotImplementedError()
