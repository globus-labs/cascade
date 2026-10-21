"""Auditor storing frames that an audit task labeled"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator

from cascade.agents.agents import Auditor
from cascade.agents.config import AuditorConfig
from cascade.agents.db_orm import TrajectoryDB
from cascade.agents.task import max_force_error
from cascade.model import AuditResult, AuditStatus, Chunk, ChunkEventType, TrainingFrame

RUN_ID = 'test_run'


def _with_forces(atoms, forces):
    atoms = atoms.copy()
    atoms.calc = SinglePointCalculator(atoms, energy=0., forces=forces)
    return atoms


def test_auditor_stores_audit_labeled_frames(tmpdir):
    db_url = f"sqlite:///{Path(tmpdir) / 'test.db'}"
    traj_db = TrajectoryDB(db_url)
    traj_db.create_tables()

    water = molecule('H2O')
    water.cell = [4.] * 3
    water.pbc = True
    traj_db.initialize_trajectory(run_id=RUN_ID, traj_id=0, target_length=100, init_atoms=water)
    traj_db.add_chunk_attempt(run_id=RUN_ID, traj_id=0, chunk_id=0, model_version=0, n_frames=1, attempt_index=0)
    frame_id = traj_db.write_frame(run_id=RUN_ID, traj_id=0, chunk_id=0, attempt_index=0, frame_index=0, atoms=water)

    predicted = _with_forces(water, np.zeros((3, 3)))
    frame = TrainingFrame(atoms=predicted, model_version=0, traj_id=0, chunk_id=0, attempt_index=0,
                          frame_id=frame_id, n_sampled_frames=1)
    frame.atoms_labeled = _with_forces(water, np.full((3, 3), 0.1))
    chunk = Chunk(atoms=[predicted], frame_ids=[frame_id], traj_id=0, chunk_id=0, attempt_ix=0, model_version=0)

    def stub_audit(chunk, **kws):
        return AuditResult(status=AuditStatus.PASSED, score=0., reason='last_frame_error', labeled_frames=[frame])

    config = AuditorConfig(run_id=RUN_ID, db_url=db_url, audit_task=stub_audit, audit_kws={},
                           executor=ThreadPoolExecutor(1), error_fn=max_force_error)
    auditor = Auditor(sampler=None, config=config)
    auditor.logger = logging.getLogger('test')
    auditor._traj_db = traj_db

    async def audit_twice():
        await auditor.audit(chunk)
        await auditor.audit(chunk)  # re-storing the same frame must not duplicate it

    # asyncio.run instead of @pytest.mark.asyncio: pytest-asyncio isn't installed in the venv
    asyncio.run(audit_twice())

    assert traj_db.count_training_frames(RUN_ID) == 1
    # the audit records no labeling events, so the Labeler's per-chunk bookkeeping is untouched
    for event in (ChunkEventType.STARTED_LABELING, ChunkEventType.FINISHED_LABELING):
        assert not traj_db.has_chunk_event(run_id=RUN_ID, traj_id=0, chunk_id=0, attempt_index=0, event_type=event)
