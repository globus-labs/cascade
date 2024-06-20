"""Interface and glue code for to models built using `TorchANI <https://github.com/aiqm/torchani>_"""
from typing import Any

import ase
import numpy as np

from torchani import AEVComputer, ANIModel
import torch

from cascade.learning.base import BaseLearnableForcefield, ModelMsgType

ANIModelContents = tuple[AEVComputer, ANIModel, dict[str, tuple]]
"""Contents of the serialized form of a model:

1. Compute for atomic environments
2. The model which maps environments to energies
3. Ordered dict of chemical symbol to atomic energi
"""


class TorchANI(BaseLearnableForcefield[ANIModelContents]):
    """Interface to the high-dimensional neural networks implemented by `TorchANI <https://github.com/aiqm/torchani>`_"""

    def evaluate(self,
                 model_msg: ModelMsgType,
                 atoms: list[ase.Atoms],
                 batch_size: int = 64,
                 device: str = 'cpu') -> (list[float], list[np.ndarray]):

        # Unpack the model
        aev_computer, model, atomic_energies = self.get_model(model_msg)
