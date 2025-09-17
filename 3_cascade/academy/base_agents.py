import asyncio

from academy.agent import Agent, action
from ase import Atoms

from cascade.learning.base import BaseLearnableForcefield


class Auditor(Agent):
    """Auditor base class"""
    async def audit(self, traj: list[Atoms]) -> tuple[bool, float, Atoms]:
        raise NotImplementedError()


class TrainingSampler(Agent):

    @action
    async def sample_frames(
        self,
        atoms: list[Atoms],
        n_frames: int
    ) -> list[Atoms]:
        raise NotImplementedError()


class TrainingLabeler(Agent):

    @action
    def label_frames(
        self,
        frame: list[Atoms]
    ) -> list[Atoms]:
        raise NotImplementedError()


class ModelTrainer(Agent):

    @action
    def train_model(
        self,
        learner: BaseLearnableForcefield
    ) -> bytes:
        raise NotImplementedError()
