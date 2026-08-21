from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator

from cascade.agents.agents import Trainer
from cascade.agents.config import TrainerConfig
from cascade.agents.db_orm import TrajectoryDB


@pytest.fixture
def temp_db_url(tmpdir) -> str:
    db_path = Path(tmpdir) / "test.db"
    return f"sqlite:///{db_path}"


@pytest.fixture
def traj_db(temp_db_url) -> TrajectoryDB:
    db = TrajectoryDB(temp_db_url)
    db.create_tables()
    return db


@pytest.fixture
def populated_run(traj_db) -> str:
    """Initialize a trajectory and add 10 labeled training frames marked for round 0."""
    run_id = "run"
    traj_db.initialize_trajectory(run_id, 0, 100, molecule('H2O'))
    for i in range(10):
        frame_id = traj_db.write_frame(run_id, 0, 0, 0, i, molecule('H2O'))
        atoms = molecule('H2O')
        atoms.calc = SinglePointCalculator(atoms, energy=float(i))
        traj_db.add_training_frame(
            run_id=run_id,
            trajectory_frame_id=frame_id,
            model_version_sampled_from=0,
            traj_id=0, chunk_id=0, attempt_index=0,
            atoms_labeled=atoms,
        )
    traj_db.mark_training_frames_for_round(run_id, training_round=0)
    return run_id


def _make_trainer(temp_db_url, run_id, weights, training_task, bootstrap_fraction=1.0):
    config = TrainerConfig(
        run_id=run_id,
        db_url=temp_db_url,
        weights=weights,
        training_task=training_task,
        training_args=(),
        training_kws={},
        learner=None,
        executor=ThreadPoolExecutor(len(weights)),
        bootstrap_fraction=bootstrap_fraction,
    )
    return Trainer(config)


@pytest.mark.asyncio
async def test_bootstrap_and_parallel_submit(temp_db_url, traj_db, populated_run):
    """Each ensemble member gets its own weights and its own bootstrapped
    training set, submitted and gathered in parallel, with results logged
    under the correct member_index."""
    calls = []

    def fake_training_task(learner, weights, train_data, valid_data, train_kws, replay=None):
        calls.append({
            'weights': weights,
            'train_data': list(train_data),
            'valid_data': list(valid_data),
        })
        return weights + b'-trained', pd.DataFrame({'epoch': [0], 'loss': [0.1]})

    agent = _make_trainer(temp_db_url, populated_run, [b'w0', b'w1', b'w2'], fake_training_task)
    await agent.agent_on_startup()

    new_weights = await agent.train_model(training_round=0)

    assert new_weights == [b'w0-trained', b'w1-trained', b'w2-trained']
    assert len(calls) == 3
    assert {c['weights'] for c in calls} == {b'w0', b'w1', b'w2'}

    # valid_data is shared across members, not bootstrapped
    valid_lens = {len(c['valid_data']) for c in calls}
    assert len(valid_lens) == 1

    # train_data is bootstrapped (with replacement) to the configured fraction of the training split
    n_train_expected = len(calls[0]['train_data'])
    assert all(len(c['train_data']) == n_train_expected for c in calls)

    logs = traj_db.get_training_logs(populated_run)
    assert sorted(logs['member_index']) == [0, 1, 2]
    assert (logs['training_round'] == 0).all()
