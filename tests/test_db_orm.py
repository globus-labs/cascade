"""Tests for the TrajectoryDB ORM layer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.db import connect
from ase.build import molecule

from cascade.agents.db_orm import TrajectoryDB, DBTrajectory, DBTrajectoryChunk, DBTrainingFrame
from cascade.model import AuditStatus


@pytest.fixture
def temp_db_url(tmpdir) -> str:
    """Create a temporary SQLite database for testing."""
    db_path = Path(tmpdir) / "test.db"
    return f"sqlite:///{db_path}"


@pytest.fixture
def traj_db(temp_db_url) -> TrajectoryDB:
    """Create a TrajectoryDB instance with a temporary database."""
    db = TrajectoryDB(temp_db_url)
    db.create_tables()
    return db


@pytest.fixture
def ase_db(temp_db_url) -> connect:
    """Create an ASE database connection."""
    return connect(temp_db_url.replace("sqlite:///", ""))


@pytest.fixture
def example_atoms() -> Atoms:
    """Create example atoms for testing."""
    water = molecule('H2O')
    water.cell = [4.] * 3
    water.pbc = True
    return water


class TestTrajectoryDB:
    """Test suite for TrajectoryDB class."""

    def test_initialize_trajectory(self, traj_db, example_atoms):
        """Test initializing a trajectory."""
        run_id = "test_run"
        traj_id = 0
        target_length = 100

        traj_db.initialize_trajectory(
            run_id=run_id,
            traj_id=traj_id,
            target_length=target_length,
            init_atoms=example_atoms
        )

        # Query to verify using get_trajectory
        db_traj = traj_db.get_trajectory(run_id, traj_id)
        assert db_traj is not None
        assert db_traj['run_id'] == run_id
        assert db_traj['traj_id'] == traj_id
        assert db_traj['target_length'] == target_length
        assert db_traj['chunks_completed'] == 0
        assert db_traj['init_atoms_json'] is not None
        assert 'positions' in db_traj['init_atoms_json']
        assert 'numbers' in db_traj['init_atoms_json']
    
    def test_get_initial_atoms(self, traj_db, example_atoms):
        """Test getting initial atoms for a trajectory."""
        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=0,
            target_length=100,
            init_atoms=example_atoms
        )
        
        # Get initial atoms
        init_atoms = traj_db.get_initial_atoms("test_run", 0)
        assert init_atoms is not None
        assert len(init_atoms) == len(example_atoms)
        assert np.allclose(init_atoms.get_positions(), example_atoms.get_positions())
        assert np.allclose(init_atoms.get_atomic_numbers(), example_atoms.get_atomic_numbers())
        assert np.allclose(init_atoms.cell, example_atoms.cell)
        assert np.allclose(init_atoms.pbc, example_atoms.pbc)

    def test_initialize_trajectory_idempotent(self, traj_db, example_atoms):
        """Test that initializing the same trajectory twice returns the same object."""
        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=0,
            target_length=100,
            init_atoms=example_atoms
        )

        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=0,
            target_length=100,
            init_atoms=example_atoms
        )

        # Verify only one trajectory exists
        traj = traj_db.get_trajectory("test_run", 0)
        assert traj is not None

    def test_add_chunk_attempt(self, traj_db, example_atoms):
        """Test adding a chunk attempt."""
        # First initialize trajectory
        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=0,
            target_length=100,
            init_atoms=example_atoms
        )

        # Add a chunk attempt
        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            model_version=0,
            n_frames=10,
            audit_status=AuditStatus.PENDING
        )

        # Query to verify
        chunk = traj_db.get_chunk_attempt("test_run", 0, 0, 0)
        assert chunk is not None
        assert chunk['run_id'] == "test_run"
        assert chunk['traj_id'] == 0
        assert chunk['chunk_id'] == 0
        assert chunk['model_version'] == 0
        assert chunk['n_frames'] == 10
        assert chunk['audit_status'] == AuditStatus.PENDING
        assert chunk['attempt_index'] == 0

    def test_add_chunk_attempt_multiple_attempts(self, traj_db, example_atoms):
        """Test adding multiple attempts for the same chunk."""
        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=0,
            target_length=100,
            init_atoms=example_atoms
        )

        # Add first attempt
        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            model_version=0,
            n_frames=10,
            audit_status=AuditStatus.FAILED
        )

        # Add second attempt
        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            model_version=1,
            n_frames=10,
            audit_status=AuditStatus.PENDING
        )

        # Query to verify
        chunk1 = traj_db.get_chunk_attempt("test_run", 0, 0, 0)
        chunk2 = traj_db.get_chunk_attempt("test_run", 0, 0, 1)
        assert chunk1 is not None
        assert chunk2 is not None
        assert chunk1['attempt_index'] == 0
        assert chunk2['attempt_index'] == 1

    def test_add_chunk_attempt_manual_attempt_index(self, traj_db, example_atoms):
        """Test adding chunk with manual attempt index."""
        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=0,
            target_length=100,
            init_atoms=example_atoms
        )

        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            model_version=0,
            n_frames=10,
            audit_status=AuditStatus.PENDING,
            attempt_index=5
        )

        # Query to verify
        chunk = traj_db.get_chunk_attempt("test_run", 0, 0, 5)
        assert chunk is not None
        assert chunk['attempt_index'] == 5

    def test_update_chunk_audit_status(self, traj_db, example_atoms):
        """Test updating chunk audit status."""
        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=0,
            target_length=100,
            init_atoms=example_atoms
        )

        chunk = traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            model_version=0,
            n_frames=10,
            audit_status=AuditStatus.PENDING
        )

        traj_db.update_chunk_audit_status(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            attempt_index=0,
            audit_status=AuditStatus.PASSED
        )

        # Verify the update
        updated_chunk = traj_db.get_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            attempt_index=0
        )

        assert updated_chunk['audit_status'] == AuditStatus.PASSED

    def test_update_chunk_audit_status_increments_completed(self, traj_db, example_atoms):
        """Test that updating to PASSED increments chunks_completed."""
        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=0,
            target_length=100,
            init_atoms=example_atoms
        )

        # Add and pass chunk 0
        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            model_version=0,
            n_frames=10,
            audit_status=AuditStatus.PENDING
        )

        traj_db.update_chunk_audit_status(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            attempt_index=0,
            audit_status=AuditStatus.PASSED
        )

        traj = traj_db.get_trajectory("test_run", 0)
        assert traj['chunks_completed'] == 1

    def test_get_passed_chunks(self, traj_db, example_atoms):
        """Test getting all passed chunks."""
        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=0,
            target_length=100,
            init_atoms=example_atoms
        )

        # Add some chunks with different statuses
        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            model_version=0,
            n_frames=10,
            audit_status=AuditStatus.PENDING
        )

        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=1,
            model_version=0,
            n_frames=10,
            audit_status=AuditStatus.PENDING
        )

        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=2,
            model_version=0,
            n_frames=10,
            audit_status=AuditStatus.PENDING
        )

        # Pass chunks 0 and 2
        traj_db.update_chunk_audit_status(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            attempt_index=0,
            audit_status=AuditStatus.PASSED
        )

        traj_db.update_chunk_audit_status(
            run_id="test_run",
            traj_id=0,
            chunk_id=2,
            attempt_index=0,
            audit_status=AuditStatus.PASSED
        )

        passed_chunks = traj_db.get_passed_chunks("test_run", 0)

        # Extract chunk_id values
        chunk_ids = [chunk['chunk_id'] for chunk in passed_chunks]
        assert len(chunk_ids) == 2
        assert set(chunk_ids) == {0, 2}

    def test_get_trajectory_chunks_atoms(self, traj_db, example_atoms, ase_db):
        """Test getting all atoms from passed chunks."""
        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=0,
            target_length=100,
            init_atoms=example_atoms
        )

        # Add and pass some chunks with frames in ASE DB
        for chunk_id in range(3):
            attempt_idx = traj_db.get_next_attempt_index("test_run", 0, chunk_id)
            
            # Write some frames to ASE DB
            for frame_idx in range(5):
                atoms = example_atoms.copy()
                atoms.positions += 0.1 * frame_idx
                ase_db.write(atoms, 
                            run_id="test_run",
                            traj_id=0,
                            chunk_id=chunk_id,
                            attempt_index=attempt_idx)

            traj_db.add_chunk_attempt(
                run_id="test_run",
                traj_id=0,
                chunk_id=chunk_id,
                model_version=0,
                n_frames=5,
                audit_status=AuditStatus.PENDING,
                attempt_index=attempt_idx
            )

            traj_db.update_chunk_audit_status(
                run_id="test_run",
                traj_id=0,
                chunk_id=chunk_id,
                attempt_index=attempt_idx,
                audit_status=AuditStatus.PASSED
            )

        # Get all atoms
        all_atoms = traj_db.get_trajectory_chunks_atoms("test_run", 0, ase_db)

        assert len(all_atoms) == 15  # 3 chunks * 5 frames

    def test_get_latest_chunk_attempt(self, traj_db, example_atoms):
        """Test getting the latest chunk attempt."""
        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=0,
            target_length=100,
            init_atoms=example_atoms
        )

        # Add multiple attempts for same chunk
        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            model_version=0,
            n_frames=10,
            audit_status=AuditStatus.FAILED
        )

        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            model_version=1,
            n_frames=10,
            audit_status=AuditStatus.PENDING
        )

        latest = traj_db.get_latest_chunk_attempt("test_run", 0, 0)

        assert latest is not None
        assert latest['attempt_index'] == 1
        assert latest['model_version'] == 1
        assert latest['audit_status'] == AuditStatus.PENDING

    def test_get_latest_passed_chunk(self, traj_db, example_atoms):
        """Test getting the latest passed chunk."""
        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=0,
            target_length=100,
            init_atoms=example_atoms
        )

        # Add chunks and pass some
        for chunk_id in range(5):
            traj_db.add_chunk_attempt(
                run_id="test_run",
                traj_id=0,
                chunk_id=chunk_id,
                model_version=0,
                n_frames=10,
                audit_status=AuditStatus.PENDING
            )

            if chunk_id in [1, 3, 4]:
                traj_db.update_chunk_audit_status(
                    run_id="test_run",
                    traj_id=0,
                    chunk_id=chunk_id,
                    attempt_index=0,
                    audit_status=AuditStatus.PASSED
                )

        latest_passed = traj_db.get_latest_passed_chunk("test_run", 0)

        assert latest_passed is not None
        assert latest_passed['chunk_id'] == 4

    def test_get_latest_chunk_id(self, traj_db, example_atoms):
        """Test getting the latest chunk ID."""
        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=0,
            target_length=100,
            init_atoms=example_atoms
        )

        # Add chunks in different order
        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=2,
            model_version=0,
            n_frames=10,
            audit_status=AuditStatus.PENDING
        )

        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            model_version=0,
            n_frames=10,
            audit_status=AuditStatus.PENDING
        )

        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=1,
            model_version=0,
            n_frames=10,
            audit_status=AuditStatus.PENDING
        )

        latest_id = traj_db.get_latest_chunk_id("test_run", 0)

        assert latest_id == 2

    def test_is_trajectory_done(self, traj_db, example_atoms):
        """Test checking if trajectory is done."""
        target_length = 50
        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=0,
            target_length=target_length,
            init_atoms=example_atoms
        )

        # Not done yet
        assert not traj_db.is_trajectory_done("test_run", 0)

        # Add chunks with total frames less than target
        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            model_version=0,
            n_frames=20,
            audit_status=AuditStatus.PENDING
        )
        traj_db.update_chunk_audit_status(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            attempt_index=0,
            audit_status=AuditStatus.PASSED
        )

        assert not traj_db.is_trajectory_done("test_run", 0)

        # Add more chunks to exceed target
        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=1,
            model_version=0,
            n_frames=20,
            audit_status=AuditStatus.PENDING
        )
        traj_db.update_chunk_audit_status(
            run_id="test_run",
            traj_id=0,
            chunk_id=1,
            attempt_index=0,
            audit_status=AuditStatus.PASSED
        )

        traj_db.add_chunk_attempt(
            run_id="test_run",
            traj_id=0,
            chunk_id=2,
            model_version=0,
            n_frames=20,
            audit_status=AuditStatus.PENDING
        )
        traj_db.update_chunk_audit_status(
            run_id="test_run",
            traj_id=0,
            chunk_id=2,
            attempt_index=0,
            audit_status=AuditStatus.PASSED
        )

        assert traj_db.is_trajectory_done("test_run", 0)

    def test_add_training_frame(self, traj_db):
        """Test adding a training frame."""
        traj_db.add_training_frame(
            run_id="test_run",
            ase_db_id=42,
            model_version_sampled_from=5
        )

        # Verify by counting
        count = traj_db.count_training_frames("test_run")
        assert count == 1

    def test_add_training_frame_idempotent(self, traj_db):
        """Test that adding the same training frame twice returns the same object."""
        traj_db.add_training_frame(
            run_id="test_run",
            ase_db_id=42,
            model_version_sampled_from=5
        )

        traj_db.add_training_frame(
            run_id="test_run",
            ase_db_id=42,
            model_version_sampled_from=5
        )

        # Should still only have one frame
        count = traj_db.count_training_frames("test_run")
        assert count == 1

    def test_count_training_frames(self, traj_db):
        """Test counting training frames."""
        count = traj_db.count_training_frames("test_run")
        assert count == 0

        # Add some training frames
        traj_db.add_training_frame("test_run", 1, 0)
        traj_db.add_training_frame("test_run", 2, 0)
        traj_db.add_training_frame("test_run", 3, 1)

        count = traj_db.count_training_frames("test_run")
        assert count == 3

    def test_get_sampled_traj_ids(self, traj_db, ase_db, example_atoms):
        """Test getting sampled trajectory IDs from training frames."""
        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=0,
            target_length=100,
            init_atoms=example_atoms
        )

        traj_db.initialize_trajectory(
            run_id="test_run",
            traj_id=1,
            target_length=100,
            init_atoms=example_atoms
        )

        # Add some frames to ASE DB
        atoms1 = example_atoms.copy()
        atoms2 = example_atoms.copy()
        atoms3 = example_atoms.copy()

        ase_db.write(atoms1, run_id="test_run", traj_id=0, chunk_id=0, attempt_index=0)
        ase_db.write(atoms2, run_id="test_run", traj_id=0, chunk_id=0, attempt_index=0)
        ase_db.write(atoms3, run_id="test_run", traj_id=1, chunk_id=0, attempt_index=0)

        # Add training frames
        traj_db.add_training_frame("test_run", 1, 0)
        traj_db.add_training_frame("test_run", 2, 0)
        traj_db.add_training_frame("test_run", 3, 0)

        sampled_ids = traj_db.get_sampled_traj_ids("test_run", ase_db)

        assert sampled_ids == {0, 1}
    
    def test_count_active_trajs_with_samples(self, traj_db, ase_db, example_atoms):
        """Test counting active trajectories with samples."""
        # Initialize 3 trajectories
        traj_db.initialize_trajectory("test_run", 0, 100, example_atoms)
        traj_db.initialize_trajectory("test_run", 1, 100, example_atoms)
        traj_db.initialize_trajectory("test_run", 2, 100, example_atoms)
        
        # Add some atoms to ASE DB
        atoms0 = Atoms(positions=[[0, 0, 0], [1, 1, 1]], numbers=[1, 1])
        atoms1 = Atoms(positions=[[2, 2, 2], [3, 3, 3]], numbers=[2, 2])
        
        id0 = ase_db.write(atoms0, run_id="test_run", traj_id=0, chunk_id=0, attempt_index=0)
        id1 = ase_db.write(atoms1, run_id="test_run", traj_id=1, chunk_id=0, attempt_index=0)
        
        # Initially, no samples
        total_active, with_samples = traj_db.count_active_trajs_with_samples("test_run", ase_db)
        assert total_active == 3
        assert with_samples == 0
        
        # Add training frames from trajectories 0 and 1
        traj_db.add_training_frame("test_run", id0, 0)
        traj_db.add_training_frame("test_run", id1, 0)
        
        total_active, with_samples = traj_db.count_active_trajs_with_samples("test_run", ase_db)
        assert total_active == 3
        assert with_samples == 2
        
        # Mark trajectory 2 as done by completing target_length frames
        traj_db.add_chunk_attempt("test_run", 2, 0, 0, 100, AuditStatus.PENDING)
        traj_db.update_chunk_audit_status("test_run", 2, 0, 0, AuditStatus.PASSED)
        
        # Now only 2 active trajectories
        total_active, with_samples = traj_db.count_active_trajs_with_samples("test_run", ase_db)
        assert total_active == 2
        assert with_samples == 2
        
        # Mark trajectory 1 as done
        traj_db.add_chunk_attempt("test_run", 1, 1, 0, 100, AuditStatus.PENDING)
        traj_db.update_chunk_audit_status("test_run", 1, 1, 0, AuditStatus.PASSED)
        
        # Now only 1 active trajectory (trajectory 0)
        total_active, with_samples = traj_db.count_active_trajs_with_samples("test_run", ase_db)
        assert total_active == 1
        assert with_samples == 1
    
    def test_list_runs(self, traj_db, example_atoms):
        """Test listing all runs in the database."""
        # Create multiple runs
        traj_db.initialize_trajectory("run1", 0, 100, example_atoms)
        traj_db.initialize_trajectory("run1", 1, 100, example_atoms)
        traj_db.initialize_trajectory("run2", 0, 100, example_atoms)
        
        runs = traj_db.list_runs()
        
        # Should have 2 runs
        assert len(runs) == 2
        
        # Check run1
        run1 = next(r for r in runs if r['run_id'] == 'run1')
        assert run1['n_trajectories'] == 2
        assert run1['n_done_trajectories'] == 0
        assert run1['first_created'] is not None
        assert run1['last_updated'] is not None
        
        # Check run2
        run2 = next(r for r in runs if r['run_id'] == 'run2')
        assert run2['n_trajectories'] == 1
        assert run2['n_done_trajectories'] == 0
    
    def test_list_runs_empty(self, traj_db):
        """Test listing runs when database is empty."""
        runs = traj_db.list_runs()
        assert runs == []
    
    def test_list_run_summary(self, traj_db, example_atoms):
        """Test getting run summary statistics."""
        run_id = "test_run"
        
        # Create trajectories
        traj_db.initialize_trajectory(run_id, 0, 100, example_atoms)
        traj_db.initialize_trajectory(run_id, 1, 100, example_atoms)
        
        # Add chunks with different statuses
        traj_db.add_chunk_attempt(run_id, 0, 0, 0, 10, AuditStatus.PENDING)
        traj_db.add_chunk_attempt(run_id, 0, 0, 0, 10, AuditStatus.FAILED, attempt_index=1)  # Second attempt
        traj_db.add_chunk_attempt(run_id, 0, 1, 0, 10, AuditStatus.PASSED)
        traj_db.add_chunk_attempt(run_id, 1, 0, 0, 10, AuditStatus.PENDING)
        
        # Add training frames
        traj_db.add_training_frame(run_id, 123, 0)
        traj_db.add_training_frame(run_id, 456, 0)
        
        # Mark chunk 1 as passed for trajectory 0
        traj_db.update_chunk_audit_status(run_id, 0, 1, 0, AuditStatus.PASSED)
        
        # To mark trajectory as done, we need to pass enough chunks to reach target_length
        # Since target_length is 100 and each chunk has 10 frames, we need 10 passed chunks
        # Let's just check that the trajectory is not done yet (since we only have 1 passed chunk)
        
        summary = traj_db.list_run_summary(run_id)
        
        assert summary is not None
        assert summary['run_id'] == run_id
        assert summary['n_trajectories'] == 2
        assert summary['n_done'] == 0  # Not done yet, only 1 chunk passed
        assert summary['n_active'] == 2
        assert summary['total_chunks'] == 4
        assert summary['total_passed_chunks'] == 1
        assert summary['total_failed_chunks'] == 1
        assert summary['total_pending_chunks'] == 2
        assert summary['total_training_frames'] == 2
        assert summary['first_created'] is not None
        assert summary['last_updated'] is not None
    
    def test_list_run_summary_nonexistent(self, traj_db):
        """Test getting summary for non-existent run."""
        summary = traj_db.list_run_summary("nonexistent")
        assert summary is None
    
    def test_list_trajectory_summary(self, traj_db, example_atoms):
        """Test getting trajectory summary with chunk breakdown."""
        run_id = "test_run"
        traj_id = 0
        
        traj_db.initialize_trajectory(run_id, traj_id, 100, example_atoms)
        
        # Add multiple attempts for chunk 0
        traj_db.add_chunk_attempt(run_id, traj_id, 0, 0, 10, AuditStatus.PENDING)
        traj_db.add_chunk_attempt(run_id, traj_id, 0, 0, 10, AuditStatus.FAILED, attempt_index=1)  # attempt 1
        traj_db.add_chunk_attempt(run_id, traj_id, 0, 0, 10, AuditStatus.PASSED, attempt_index=2)  # attempt 2
        
        # Add chunk 1
        traj_db.add_chunk_attempt(run_id, traj_id, 1, 0, 10, AuditStatus.PENDING)
        
        # Mark chunk 0, attempt 2 as passed
        traj_db.update_chunk_audit_status(run_id, traj_id, 0, 2, AuditStatus.PASSED)
        
        summary = traj_db.list_trajectory_summary(run_id, traj_id)
        
        assert summary is not None
        assert summary['run_id'] == run_id
        assert summary['traj_id'] == traj_id
        assert summary['target_length'] == 100
        assert summary['chunks_completed'] == 1  # chunk 0 passed, so chunks_completed = 0 + 1 = 1
        assert summary['done'] == False  # Not done, only 10 frames (need 100)
        assert summary['n_chunk_attempts'] == 4
        assert summary['n_unique_chunks'] == 2
        
        # Check chunk breakdown
        assert 0 in summary['chunk_breakdown']
        assert 1 in summary['chunk_breakdown']
        
        chunk0 = summary['chunk_breakdown'][0]
        assert chunk0['n_attempts'] == 3
        assert chunk0['latest_status'] == 'PASSED'
        assert chunk0['latest_attempt_index'] == 2
        
        chunk1 = summary['chunk_breakdown'][1]
        assert chunk1['n_attempts'] == 1
        assert chunk1['latest_status'] == 'PENDING'
        assert chunk1['latest_attempt_index'] == 0
        
        # Check status counts
        assert summary['status_counts']['PENDING'] == 2
        assert summary['status_counts']['PASSED'] == 1
        assert summary['status_counts']['FAILED'] == 1
    
    def test_atoms_serialization_roundtrip(self, example_atoms):
        """Test that atoms can be serialized and deserialized correctly"""
        from cascade.agents.db_orm import atoms_to_dict, dict_to_atoms
        
        # Serialize
        atoms_dict = atoms_to_dict(example_atoms)
        
        # Deserialize
        restored_atoms = dict_to_atoms(atoms_dict)
        
        # Check positions
        assert np.allclose(restored_atoms.get_positions(), example_atoms.get_positions())
        
        # Check atomic numbers
        assert np.array_equal(restored_atoms.get_atomic_numbers(), example_atoms.get_atomic_numbers())
        
        # Check cell
        assert np.allclose(restored_atoms.cell, example_atoms.cell)
        
        # Check pbc
        assert np.array_equal(restored_atoms.pbc, example_atoms.pbc)
    
    def test_atoms_serialization_no_cell_no_pbc(self):
        """Test serialization with no cell or pbc"""
        from cascade.agents.db_orm import atoms_to_dict, dict_to_atoms
        from ase import Atoms
        
        # Create atoms without cell/pbc
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        
        # Serialize
        atoms_dict = atoms_to_dict(atoms)
        
        # Deserialize
        restored_atoms = dict_to_atoms(atoms_dict)
        
        assert np.allclose(restored_atoms.get_positions(), atoms.get_positions())
        assert np.array_equal(restored_atoms.get_atomic_numbers(), atoms.get_atomic_numbers())
        assert restored_atoms.cell is None or np.allclose(restored_atoms.cell, 0)
        assert restored_atoms.pbc is None or np.allclose(restored_atoms.pbc, False)
    
    def test_calc_results_serialization_roundtrip(self):
        """Test that calculator results can be serialized and deserialized"""
        from cascade.agents.db_orm import calc_results_to_dict, dict_to_calc_results
        from ase import Atoms
        
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        
        # Create calculator results
        calc_results = {
            'energy': -1.5,
            'forces': np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3], [0.0, 0.0, 0.0]], dtype=np.float64),
            'stress': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64),  # Voigt notation
        }
        
        # Serialize
        results_dict = calc_results_to_dict(calc_results)
        
        # Check types
        assert isinstance(results_dict['energy'], float)
        assert isinstance(results_dict['forces'], list)
        assert isinstance(results_dict['stress'], list)
        
        # Deserialize
        restored_results = dict_to_calc_results(results_dict)
        
        # Check energy
        assert restored_results['energy'] == calc_results['energy']
        assert isinstance(restored_results['energy'], (int, float))
        
        # Check forces
        assert np.allclose(restored_results['forces'], calc_results['forces'])
        assert isinstance(restored_results['forces'], np.ndarray)
        assert restored_results['forces'].dtype == np.float64
        
        # Check stress
        assert np.allclose(restored_results['stress'], calc_results['stress'])
        assert isinstance(restored_results['stress'], np.ndarray)
        assert restored_results['stress'].dtype == np.float64
    
    def test_write_frame_and_retrieve(self, traj_db, example_atoms):
        """Test writing a frame and retrieving it"""
        from ase.calculators.singlepoint import SinglePointCalculator
        
        # Add calculator with results
        forces = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3], [0.0, 0.0, 0.0]], dtype=np.float64)
        example_atoms.calc = SinglePointCalculator(example_atoms, energy=-1.5, forces=forces)
        
        # Write frame
        success = traj_db.write_frame(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            attempt_index=0,
            frame_index=0,
            atoms=example_atoms
        )
        assert success is True
        
        # Retrieve frame
        retrieved_atoms = traj_db.get_frame_by_id(1)  # ID should be 1 for first frame
        
        assert retrieved_atoms is not None
        assert len(retrieved_atoms) == len(example_atoms)
        assert np.allclose(retrieved_atoms.get_positions(), example_atoms.get_positions())
        assert np.array_equal(retrieved_atoms.get_atomic_numbers(), example_atoms.get_atomic_numbers())
        
        # Check calculator results
        assert retrieved_atoms.calc is not None
        assert 'energy' in retrieved_atoms.calc.results
        assert 'forces' in retrieved_atoms.calc.results
        assert retrieved_atoms.calc.results['energy'] == -1.5
        assert np.allclose(retrieved_atoms.calc.results['forces'], forces)
    
    def test_write_frame_no_calculator(self, traj_db, example_atoms):
        """Test writing a frame without a calculator"""
        # Write frame without calculator
        success = traj_db.write_frame(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            attempt_index=0,
            frame_index=0,
            atoms=example_atoms
        )
        assert success is True
        
        # Retrieve frame
        retrieved_atoms = traj_db.get_frame_by_id(1)
        
        assert retrieved_atoms is not None
        assert len(retrieved_atoms) == len(example_atoms)
        # Calculator should be None or have no results
        if retrieved_atoms.calc:
            assert not retrieved_atoms.calc.results or len(retrieved_atoms.calc.results) == 0
    
    def test_get_frames_multiple(self, traj_db, example_atoms):
        """Test writing and retrieving multiple frames"""
        from ase.calculators.singlepoint import SinglePointCalculator
        from cascade.agents.db_orm import dict_to_atoms, dict_to_calc_results
        
        # Write multiple frames
        for frame_idx in range(5):
            atoms = example_atoms.copy()
            atoms.positions += 0.1 * frame_idx
            atoms.calc = SinglePointCalculator(atoms, energy=-1.5 - frame_idx * 0.1)
            
            success = traj_db.write_frame(
                run_id="test_run",
                traj_id=0,
                chunk_id=0,
                attempt_index=0,
                frame_index=frame_idx,
                atoms=atoms
            )
            assert success is True
        
        # Retrieve all frames
        frames = traj_db.get_frames(
            run_id="test_run",
            traj_id=0,
            chunk_id=0,
            attempt_index=0
        )
        
        assert len(frames) == 5
        
        # Check ordering
        for i, frame_data in enumerate(frames):
            assert frame_data['frame_index'] == i
            assert 'atoms_data' in frame_data
            assert 'calc_results' in frame_data
            
            # Check that we can reconstruct atoms
            atoms = dict_to_atoms(frame_data['atoms_data'])
            assert np.allclose(atoms.get_positions()[0], example_atoms.get_positions()[0] + 0.1 * i)
            
            if frame_data['calc_results']:
                results = dict_to_calc_results(frame_data['calc_results'])
                assert 'energy' in results
                assert np.isclose(results['energy'], -1.5 - i * 0.1)
    
    def test_atoms_with_stress(self, traj_db):
        """Test serialization with stress tensor"""
        from cascade.agents.db_orm import atoms_to_dict, dict_to_atoms, calc_results_to_dict, dict_to_calc_results
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator
        
        # Create periodic structure with stress
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], 
                     cell=[10, 10, 10], pbc=[True, True, True])
        
        # Add stress in Voigt notation
        stress_voigt = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
        atoms.calc = SinglePointCalculator(atoms, energy=-1.5, stress=stress_voigt)
        
        # Serialize
        atoms_dict = atoms_to_dict(atoms)
        results_dict = calc_results_to_dict(atoms.calc.results)
        
        # Deserialize
        restored_atoms = dict_to_atoms(atoms_dict)
        restored_results = dict_to_calc_results(results_dict)
        
        # Check stress
        assert 'stress' in restored_results
        assert np.allclose(restored_results['stress'], stress_voigt)
        
        # Write to database and retrieve
        success = traj_db.write_frame("test_run", 0, 0, 0, 0, atoms)
        assert success is True
        
        retrieved = traj_db.get_frame_by_id(1)
        assert retrieved is not None
        assert retrieved.calc is not None
        assert 'stress' in retrieved.calc.results
        assert np.allclose(retrieved.calc.results['stress'], stress_voigt)
    
    def test_atoms_data_types(self, example_atoms):
        """Test that data types are preserved correctly"""
        from cascade.agents.db_orm import atoms_to_dict, dict_to_atoms
        
        atoms_dict = atoms_to_dict(example_atoms)
        
        # Check that positions are lists of floats
        assert isinstance(atoms_dict['positions'], list)
        assert all(isinstance(row, list) for row in atoms_dict['positions'])
        assert all(isinstance(val, (int, float)) for row in atoms_dict['positions'] for val in row)
        
        # Check that numbers are list of ints
        assert isinstance(atoms_dict['numbers'], list)
        assert all(isinstance(n, int) for n in atoms_dict['numbers'])
        
        # Deserialize and check types
        restored = dict_to_atoms(atoms_dict)
        assert isinstance(restored.get_positions(), np.ndarray)
        assert isinstance(restored.get_atomic_numbers(), np.ndarray)
        assert restored.get_positions().dtype == np.float64
        assert restored.get_atomic_numbers().dtype == np.int32 or restored.get_atomic_numbers().dtype == np.int64
    
    def test_mace_calculator_serialization(self, traj_db, example_atoms):
        """Test serializing/deserializing results from MACE calculator
        
        MACE calculators produce non-standard properties like 'node_energy' that
        SinglePointCalculator doesn't support. This test ensures we filter those out.
        """
        pytest.importorskip("mace.calculators")
        from mace.calculators import mace_mp
        from cascade.agents.db_orm import atoms_to_dict, dict_to_atoms, calc_results_to_dict, dict_to_calc_results
        from ase.calculators.singlepoint import SinglePointCalculator
        
        # Create atoms with MACE calculator
        atoms = example_atoms.copy()
        mace_calc = mace_mp('small', device='cpu', default_dtype="float32")
        atoms.calc = mace_calc
        
        # Run calculation to populate results
        atoms.get_forces()
        atoms.get_potential_energy()
        
        # Verify MACE produces non-standard properties
        assert 'node_energy' in atoms.calc.results or 'node_energies' in atoms.calc.results or len([k for k in atoms.calc.results.keys() if 'node' in k.lower()]) > 0
        
        # Serialize - should filter out non-standard properties
        atoms_dict = atoms_to_dict(atoms)
        results_dict = calc_results_to_dict(atoms.calc.results)
        
        # Verify non-standard properties are filtered out
        assert 'node_energy' not in results_dict
        assert 'node_energies' not in results_dict
        
        # Verify standard properties are preserved
        assert 'energy' in results_dict
        assert 'forces' in results_dict
        
        # Deserialize and create SinglePointCalculator - should not raise error
        restored_atoms = dict_to_atoms(atoms_dict)
        restored_results = dict_to_calc_results(results_dict)
        
        # This should work without AssertionError about 'node_energy'
        restored_atoms.calc = SinglePointCalculator(restored_atoms, **restored_results)
        
        # Verify standard properties are correct
        assert np.isclose(restored_atoms.get_potential_energy(), atoms.get_potential_energy(), rtol=1e-5)
        assert np.allclose(restored_atoms.get_forces(), atoms.get_forces(), rtol=1e-5)
        
        # Test writing to database and retrieving
        success = traj_db.write_frame("test_run", 0, 0, 0, 0, atoms)
        assert success is True
        
        retrieved = traj_db.get_frame_by_id(1)
        assert retrieved is not None
        assert retrieved.calc is not None
        assert 'energy' in retrieved.calc.results
        assert 'forces' in retrieved.calc.results
        # Non-standard properties should not be present
        assert 'node_energy' not in retrieved.calc.results

    
    def test_list_trajectory_summary_nonexistent(self, traj_db):
        """Test getting summary for non-existent trajectory."""
        summary = traj_db.list_trajectory_summary("test_run", 999)
        assert summary is None
    
    def test_list_trajectories_in_run(self, traj_db, example_atoms):
        """Test listing all trajectories in a run."""
        run_id = "test_run"
        
        # Create multiple trajectories
        traj_db.initialize_trajectory(run_id, 0, 100, example_atoms)
        traj_db.initialize_trajectory(run_id, 2, 200, example_atoms)  # Skip 1 to test ordering
        traj_db.initialize_trajectory(run_id, 1, 150, example_atoms)
        
        # Add chunk for trajectory 2 with 100 frames (target_length is 200)
        traj_db.add_chunk_attempt(run_id, 2, 0, 0, 100, AuditStatus.PENDING)
        traj_db.update_chunk_audit_status(run_id, 2, 0, 0, AuditStatus.PASSED)
        
        # Add another chunk with 100 frames to reach target_length of 200
        traj_db.add_chunk_attempt(run_id, 2, 1, 0, 100, AuditStatus.PENDING)
        traj_db.update_chunk_audit_status(run_id, 2, 1, 0, AuditStatus.PASSED)
        
        trajectories = traj_db.list_trajectories_in_run(run_id)
        
        # Should be sorted by traj_id
        assert len(trajectories) == 3
        assert trajectories[0]['traj_id'] == 0
        assert trajectories[1]['traj_id'] == 1
        assert trajectories[2]['traj_id'] == 2
        
        # Check trajectory 0
        assert trajectories[0]['target_length'] == 100
        assert trajectories[0]['chunks_completed'] == 0
        assert trajectories[0]['done'] == False
        
        # Check trajectory 2 (done - has 200 frames from 2 chunks of 100 each)
        assert trajectories[2]['target_length'] == 200
        assert trajectories[2]['chunks_completed'] == 2  # chunks 0 and 1 passed
        assert trajectories[2]['done'] == True
        
        # All should have timestamps
        for traj in trajectories:
            assert traj['created_at'] is not None
            assert traj['updated_at'] is not None
    
    def test_list_trajectories_in_run_empty(self, traj_db):
        """Test listing trajectories in non-existent run."""
        trajectories = traj_db.list_trajectories_in_run("nonexistent")
        assert trajectories == []

