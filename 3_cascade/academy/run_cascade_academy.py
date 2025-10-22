from academy.exchange import LocalExchangeFactory
from academy.manager import Manager
from concurrent.futures import ThreadPoolExecutor

from agents import (
    DummyDatabase,
    DynamicsEngine,
    DummyAuditor,
    DummySampler,
    DummyLabeler,
    DummyTrainer
)

from model import Trajectory

from ase.io import read


# args
init_strc = []
target_length = 100
retrain_len = 10
n_sample_frames = 1
accept_rate = 1.
learner = None
init_weights = None
dyn_cls = None
dyn_kws = {}
run_kws = {}

if __name__ == '__main__':
    
    # intial conditions, trajectories
    initial_frames = []
    trajectories = []
    for i, s in enumerate(init_strc):
        initial_frames.append(read(s, '-1'))
        trajectories.append(
            Trajectory(
                chunks=[],
                target_length=target_length,
                traj_id=i,
            )
        )

    # initialize manager, exhcange
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:

        # register all agents with manager
        db_reg = manager.register_agent(DummyDatabase)
        trainer_reg = manager.register_agent(DummyTrainer)
        labeler_reg = manager.register_agent(DummyLabeler)
        sampler_reg = manager.register_agent(DummySampler)
        auditor_reg = manager.register_agent(DummyAuditor)
        dynamics_reg = manager.register_agent(DynamicsEngine)

        # get handles to all agents
        db_handle = manager.get_handle(db_reg)
        trainer_handle = manager.get_handle(trainer_reg)
        labeler_handle = manager.get_handle(labeler_reg)
        sampler_handle = manager.get_handle(sampler_reg)
        auditor_handle = manager.get_handle(auditor_reg)
        dynamics_handle = manager.get_handle(dynamics_reg)

        # launch all agents
        manager.launch(
            db_handle,
            trajectories,
            retrain_len,
            trainer_handle,
            dynamics_handle
        )

        manager.launch(
            trainer_handle,
            learner
        )

        manager.launch(
            sampler_handle,
            n_sample_frames,
            sampler_handle
        )

        manager.launch(
            auditor_handle,
            accept_rate,
            sampler_handle,
            dynamics_handle,
            db_handle
        )

        manager.launch(
            dynamics_handle,
            auditor_handle,
            learner,
            init_weights,
            dyn_cls,
            dyn_kws,
            run_kws
        )
