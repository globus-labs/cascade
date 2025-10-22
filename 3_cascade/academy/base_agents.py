import asyncio
from __future__ import annotations
from queue import Queue

from academy.agent import Agent, action, loop
from academy.handle import Handle
from ase import Atoms

from cascade.learning.base import BaseLearnableForcefield
from model import AuditStatus, TrajectoryChunk, Trajectory


class Database(Agent):
    """"""

    def __init__(
        self,
        trainer: Handle[ModelTrainer],
    ):
        """initialize connection/client"""
        pass

    @action
    async def write_chunk(
        self,
        chunk: TrajectoryChunk
    ):
        raise NotImplementedError()

    @action
    async def mark_sampled(self, traj_id):
        """To keep track of what has been sampled from for training"""
        raise NotImplementedError

    @action
    async def write_training_frame(
        self,
        frame: Atoms
    ):
        raise NotImplementedError()

    @loop
    async def periodic_retrain(self, shutdown: asyncio.Event):
        """Decide when to retrain, start retrain actions

        Monitors DB for accumulation of training data
        Calls the trainer when it is time
        """
        pass


class DynamicsEngine(Agent):
    """Dynamics engine base class"""

    def __init__(
        self,
        auditor: Handle[Auditor]
    ):
        # initialize a queue
        raise NotImplementedError()

    @loop
    async def advance_trajectory(
        self,
        shutdown: asyncio.Event
    ) -> None:
        """Read from a queue, run dynamics simulations, call auditor (should it call directly?)"""
        while not shutdown.is_set():
            pass
        raise NotImplementedError()

    @action
    async def submit(
        self,
        traj: Trajectory
    )-> None:
        """Push atoms onto the queue"""
        raise NotImplementedError()

    @action
    async def update_weights(
        self,
        model_version: int,
        model_msg: bytes
    ) -> None:
        """Not really sure if we should do this way, but I think facilitates
        training concurrently"""
        raise NotImplementedError()


class Auditor(Agent):
    """Auditor base class"""

    def __init__(
        self,
        sampler: Handle["TrainingSampler"],
        dynamics_engine: Handle["DynamicsEngine"]
    ):
        raise NotImplementedError()

    @loop
    async def audit(
        self,
        shutdown: asyncio.Event
    ) -> None:
        """Pull trajectories off the queue and audit"""
        raise NotImplementedError()

    @action
    async def submit(self, traj: Trajectory) -> None:
        """Push a trajectory onto the queue"""
        raise NotImplementedError()


class TrainingSampler(Agent):

    @action
    async def submit(
        self,
        traj: Trajectory,
    ) -> None:
        """Push onto the queue"""
        raise NotImplementedError()

    @loop
    async def sample(
        self, 
        shutdown: asyncio.Event
    ) -> None:
        """Pull off of queue and sample"""
        raise NotImplementedError()


class TrainingLabeler(Agent):
    """drawing on the MOFA Validator design"""

    def submit(self, frame: Atoms | list[Atoms]):
        """Add frames to the labeling queue"""
        pass

    @loop
    def label_frames(
        self,
        shutdown: asyncio.Event
    ):
        """pull off of the queue and label

        should the training data just stay in memory?
        """
        raise NotImplementedError()


class ModelTrainer(Agent):

    @action
    def train_model(
        self,
        learner: BaseLearnableForcefield
    ) -> bytes:
        raise NotImplementedError()