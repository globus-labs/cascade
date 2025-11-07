# Trajectory ORM Usage Guide

This document explains how to use the ORM layer for tracking trajectories and trajectory chunks in PostgreSQL.

## Overview

The ORM layer (`cascade/agents/db_orm.py`) provides two SQLAlchemy models:
- `DBTrajectory`: Tracks trajectory metadata (run_id, traj_id, target_length, chunks_completed, init_atoms)
- `DBTrajectoryChunk`: Tracks chunk metadata (run_id, traj_id, chunk_id, attempt_index, model_version, audit_status, n_frames)

## Basic Usage

All agents inherit `_traj_db` property from `CascadeAgent` which provides access to the `TrajectoryDB` manager.

### Initializing a Trajectory

```python
# In your agent initialization or trajectory creation logic
db_traj = self._traj_db.initialize_trajectory(
    run_id=self.config.run_id,
    traj_id=traj_id,
    target_length=target_length,
    init_atoms=initial_atoms
)
```

### Recording a Chunk Attempt

When generating a new chunk, record it in the database:

```python
# After generating a chunk
attempt_index = self._traj_db.get_next_attempt_index(
    run_id=self.config.run_id,
    traj_id=chunk.traj_id,
    chunk_id=chunk.chunk_id
)

db_chunk = self._traj_db.add_chunk_attempt(
    run_id=self.config.run_id,
    traj_id=chunk.traj_id,
    chunk_id=chunk.chunk_id,
    model_version=chunk.model_version,
    n_frames=len(chunk.atoms),
    audit_status=AuditStatus.PENDING,
    attempt_index=attempt_index
)
```

### Updating Audit Status

When a chunk passes or fails audit:

```python
self._traj_db.update_chunk_audit_status(
    run_id=self.config.run_id,
    traj_id=chunk.traj_id,
    chunk_id=chunk.chunk_id,
    attempt_index=attempt_index,
    audit_status=AuditStatus.PASSED  # or AuditStatus.FAILED
)
```

### Querying Passed Chunks

To get all passed chunks for a trajectory:

```python
chunks = self._traj_db.get_passed_chunks(
    run_id=self.config.run_id,
    traj_id=traj_id
)

# Get all atoms from passed chunks
all_atoms = self._traj_db.get_trajectory_chunks_atoms(
    run_id=self.config.run_id,
    traj_id=traj_id,
    ase_db=self._db  # ASE database connection
)
```

### Getting Trajectory Information

```python
traj = self._traj_db.get_trajectory(
    run_id=self.config.run_id,
    traj_id=traj_id
)

if traj:
    print(f"Trajectory {traj_id}: {traj.chunks_completed} chunks completed")
```

## Integration Points

The ORM is designed to work alongside the existing ASE database:
- **ASE DB (`_db`)**: Stores individual frames with metadata (run_id, traj_id, chunk_id)
- **ORM (`_traj_db`)**: Stores trajectory and chunk metadata, tracks attempts, audit status, etc.

Both use the same PostgreSQL database URL, so they can query each other's data seamlessly.

