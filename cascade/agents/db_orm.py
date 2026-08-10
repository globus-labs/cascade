"""ORM layer for tracking trajectories and trajectory chunks in PostgreSQL.

This module provides SQLAlchemy models and utilities for persisting trajectory
and chunk metadata, and storing trajectory frames as serialized Atoms objects.
"""
from __future__ import annotations

import contextlib
import gc
import json
import logging
from datetime import datetime
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from ase import Atoms

if TYPE_CHECKING:
    # Only import ORM classes for type checking, not at runtime
    pass  # ORM classes are defined in this module

logger = logging.getLogger(__name__)
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    Enum as SQLEnum,
    DateTime,
    Float,
    ForeignKey,
    func,
    JSON,
    LargeBinary,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from cascade.model import AuditStatus, TrajectoryStatus, ChunkEventType

Base = declarative_base()


class DBTrajectory(Base):
    """ORM model for trajectory metadata"""
    __tablename__ = 'trajectories'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    traj_id = Column(Integer, nullable=False)
    target_length = Column(Integer, nullable=False)
    chunks_completed = Column(Integer, default=0, nullable=False)
    status = Column(SQLEnum(TrajectoryStatus), nullable=False, default=TrajectoryStatus.RUNNING)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship to chunks
    chunks = relationship('DBTrajectoryChunk', back_populates='trajectory', order_by='DBTrajectoryChunk.chunk_id')

    # Store initial atoms as JSON for easy reconstruction
    init_atoms_json = Column(JSON, nullable=False)

    __table_args__ = (
        UniqueConstraint('run_id', 'traj_id', name='uq_trajectory_run_traj'),
    )

    def __repr__(self):
        return (
            f"<DBTrajectory(run_id={self.run_id}, traj_id={self.traj_id}, "
            f"status={self.status.name}, chunks_completed={self.chunks_completed})>"
        )


class DBTrajectoryChunk(Base):
    """ORM model for trajectory chunk metadata"""
    __tablename__ = 'trajectory_chunks'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    trajectory_id = Column(Integer, ForeignKey('trajectories.id'), nullable=False)
    traj_id = Column(Integer, nullable=False, index=True)
    chunk_id = Column(Integer, nullable=False, index=True)
    attempt_index = Column(Integer, nullable=False, default=0)
    model_version = Column(Integer, nullable=False)
    audit_status = Column(SQLEnum(AuditStatus), nullable=False, default=AuditStatus.PENDING)
    audit_reason = Column(String, nullable=True)
    """Which mechanism produced audit_status, e.g. 'threshold', 'burn_in', 'random_fail'; null while PENDING"""
    n_frames = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship back to trajectory
    trajectory = relationship('DBTrajectory', back_populates='chunks')

    __table_args__ = (
        UniqueConstraint('run_id', 'traj_id', 'chunk_id', 'attempt_index', name='uq_chunk_run_traj_chunk_attempt'),
    )

    def __repr__(self):
        return f"<DBTrajectoryChunk(run_id={self.run_id}, traj_id={self.traj_id}, chunk_id={self.chunk_id}, attempt={self.attempt_index}, status={self.audit_status})>"


class DBTrajectoryFrame(Base):
    """ORM model for storing trajectory frame Atoms objects as BLOBs"""
    __tablename__ = 'trajectory_frames'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    traj_id = Column(Integer, nullable=False, index=True)
    chunk_id = Column(Integer, nullable=False, index=True)
    attempt_index = Column(Integer, nullable=False, index=True)
    frame_index = Column(Integer, nullable=False)  # 0-based index within the chunk
    atoms_blob = Column(LargeBinary, nullable=False)  # Serialized Atoms object (JSON format as bytes)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('run_id', 'traj_id', 'chunk_id', 'attempt_index', 'frame_index', name='uq_frame_run_traj_chunk_attempt_index'),
    )

    def __repr__(self):
        return f"<DBTrajectoryFrame(run_id={self.run_id}, traj_id={self.traj_id}, chunk_id={self.chunk_id}, attempt_index={self.attempt_index}, frame_index={self.frame_index})>"


class DBTrainingFrame(Base):
    """ORM model for training frames"""
    __tablename__ = 'training_frames'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    trajectory_frame_id = Column(Integer, ForeignKey('trajectory_frames.id'), nullable=False, index=True)
    model_version_sampled_from = Column(Integer, nullable=False)
    # Denormalized chunk info for faster queries
    traj_id = Column(Integer, nullable=False, index=True)
    chunk_id = Column(Integer, nullable=False, index=True)
    attempt_index = Column(Integer, nullable=False)
    # Training round tracking
    training_round = Column(Integer, nullable=True, index=True)
    atoms_labeled_blob = Column(LargeBinary, nullable=True)
    # Calibration inputs for adaptive audit thresholding (see Controller agent)
    calibration_uq = Column(Float, nullable=True)
    """UQ scalar recorded at sample time (e.g. atoms.info[uq_field])"""
    calibration_error = Column(Float, nullable=True)
    """Observed error vs. DFT label, computed by Labeler via error_fn"""
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('run_id', 'trajectory_frame_id', name='uq_training_frame_run_frame'),
    )

    def __repr__(self):
        return f"<DBTrainingFrame(run_id={self.run_id}, trajectory_frame_id={self.trajectory_frame_id}, traj_id={self.traj_id}, chunk_id={self.chunk_id}, attempt_index={self.attempt_index}, training_round={self.training_round})>"


class DBControllerLog(Base):
    """ORM model for the Controller's threshold/alpha history"""
    __tablename__ = 'controller_log'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    traj_id = Column(Integer, nullable=True, index=True)
    """Trajectory this calibration is specific to; null when the threshold is shared across all trajectories"""
    model_version = Column(Integer, nullable=False)
    """model_version_sampled_from of the calibration window this entry was fit from"""
    threshold = Column(Float, nullable=False)
    alpha = Column(Float, nullable=False)
    mean_error = Column(Float, nullable=False)
    n_observations = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<DBControllerLog(run_id={self.run_id}, model_version={self.model_version}, threshold={self.threshold}, alpha={self.alpha}, n_observations={self.n_observations})>"


class DBChunkEvent(Base):
    """ORM model for chunk-level events"""
    __tablename__ = 'chunk_events'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    traj_id = Column(Integer, nullable=False, index=True)
    chunk_id = Column(Integer, nullable=False, index=True)
    attempt_index = Column(Integer, nullable=False, index=True)
    frame_id = Column(Integer, ForeignKey('trajectory_frames.id'), nullable=True, index=True)
    event_type = Column(SQLEnum(ChunkEventType), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        # Composite index for efficient latest event queries
        Index('idx_chunk_events_latest', 'run_id', 'traj_id', 'chunk_id', 'attempt_index', 'created_at'),
    )

    def __repr__(self):
        return f"<DBChunkEvent(run_id={self.run_id}, traj_id={self.traj_id}, chunk_id={self.chunk_id}, attempt_index={self.attempt_index}, event_type={self.event_type.name}, frame_id={self.frame_id})>"


class DBTrainingEvent(Base):
    """ORM model for training-level events"""
    __tablename__ = 'training_events'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    event_type = Column(SQLEnum(ChunkEventType), nullable=False, index=True)
    training_round = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    def __repr__(self):
        return f"<DBTrainingEvent(run_id={self.run_id}, event_type={self.event_type.name}, training_round={self.training_round})>"


class DBTrainingLog(Base):
    """ORM model for per-round training loss history"""
    __tablename__ = 'training_logs'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    training_round = Column(Integer, nullable=False, index=True)
    member_index = Column(Integer, nullable=False, default=0)
    log_json = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('run_id', 'training_round', 'member_index', name='uq_training_log_run_round_member'),
    )

    def __repr__(self):
        return f"<DBTrainingLog(run_id={self.run_id}, training_round={self.training_round})>"


class TrajectoryDB:
    """Wrapper for the database representations of trajectories and chunks"""
    
    def __init__(self, db_url: str, logger: Optional[logging.Logger] = None):
        """Initialize the trajectory database manager
        
        Args:
            db_url: PostgreSQL connection URL (e.g., 'postgresql://user:pass@host:port/dbname')
            logger: Optional logger for tracking engine creation
        """
        self.db_url = db_url
        self._logger = logger or logging.getLogger(__name__)
        
        # Create engine with default settings
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self._logger.info(f"Created TrajectoryDB engine (id={id(self.engine)})")
        
    def create_tables(self):
        """Create all tables if they don't exist"""
        Base.metadata.create_all(self.engine)
    
    @contextlib.contextmanager
    def session(self):
        """Context manager for database sessions"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            # Force garbage collection after session close to release any held references
            gc.collect()
    
    @staticmethod
    def _serialize_atoms(atoms: Atoms) -> bytes:
        """Serialize an Atoms object to bytes using cascade.utils
        
        Args:
            atoms: Atoms object to serialize
            
        Returns:
            Serialized Atoms as bytes (JSON format)
        """
        from cascade.utils import canonicalize, write_to_string
        canonical_atoms = canonicalize(atoms)
        atoms_str = write_to_string(canonical_atoms, fmt='extxyz')
        return atoms_str.encode('utf-8')
    
    @staticmethod
    def _deserialize_atoms(data: bytes) -> Atoms:
        """Deserialize bytes to an Atoms object using cascade.utils
        
        Args:
            data: Serialized Atoms as bytes (JSON format)
            
        Returns:
            Deserialized Atoms object
        """
        from cascade.utils import read_from_string
        atoms_str = data.decode('utf-8')
        return read_from_string(atoms_str, fmt='extxyz')
    
    def write_frame(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        attempt_index: int,
        frame_index: int,
        atoms: Atoms
    ) -> int:
        """Write a trajectory frame to the database
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            attempt_index: Attempt index
            frame_index: Frame index within the chunk (0-based)
            atoms: Atoms object to store
            
        Returns:
            ID of the created frame record
        """
        with self.session() as sess:
            # Serialize atoms
            atoms_blob = self._serialize_atoms(atoms)
            
            # Check if frame already exists
            existing = sess.query(DBTrajectoryFrame).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index,
                frame_index=frame_index
            ).first()
            
            if existing:
                # Update existing frame
                existing.atoms_blob = atoms_blob
                sess.flush()
                return existing.id
            
            # Create new frame
            db_frame = DBTrajectoryFrame(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index,
                frame_index=frame_index,
                atoms_blob=atoms_blob
            )
            sess.add(db_frame)
            sess.flush()
            sess.refresh(db_frame)
            return db_frame.id
    
    def _set_trajectory_status(
        self,
        sess,
        traj: DBTrajectory,
        status: TrajectoryStatus,
    ) -> None:
        """Internal helper to update trajectory status."""
        previous_status = traj.status
        if previous_status != status:
            traj.status = status
            sess.flush()

    def mark_trajectory_status(
        self,
        run_id: str,
        traj_id: int,
        status: TrajectoryStatus,
    ) -> bool:
        """Set the lifecycle status for a trajectory."""
        with self.session() as sess:
            traj = sess.query(DBTrajectory).filter_by(
                run_id=run_id,
                traj_id=traj_id,
            ).first()
            if not traj:
                logger.warning(
                    "Attempted to update status for missing trajectory %s/%s",
                    run_id,
                    traj_id,
                )
                return False
            self._set_trajectory_status(sess, traj, status)
            return True

    def mark_trajectory_running(self, run_id: str, traj_id: int) -> bool:
        """Mark a trajectory as actively running."""
        return self.mark_trajectory_status(run_id, traj_id, TrajectoryStatus.RUNNING)

    def mark_trajectory_failed(self, run_id: str, traj_id: int) -> bool:
        """Mark a trajectory as failed."""
        return self.mark_trajectory_status(run_id, traj_id, TrajectoryStatus.FAILED)

    def mark_trajectory_completed(self, run_id: str, traj_id: int) -> bool:
        """Mark a trajectory as completed."""
        return self.mark_trajectory_status(run_id, traj_id, TrajectoryStatus.COMPLETED)
    
    def initialize_trajectory(
        self,
        run_id: str,
        traj_id: int,
        target_length: int,
        init_atoms: Atoms
    ) -> bool:
        """Initialize a new trajectory in the database
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            target_length: Target length of the trajectory
            init_atoms: Initial atoms structure
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session() as sess:
                # Check if trajectory already exists
                existing = sess.query(DBTrajectory).filter_by(
                    run_id=run_id,
                    traj_id=traj_id
                ).first()
                
                if existing:
                    # Idempotent - trajectory already exists
                    return True
                
                # Convert atoms to JSON-serializable format
                init_atoms_json = {
                    'positions': init_atoms.get_positions().tolist(),
                    'numbers': init_atoms.get_atomic_numbers().tolist(),
                    'cell': init_atoms.get_cell().tolist() if init_atoms.cell is not None else None,
                    'pbc': init_atoms.get_pbc().tolist() if init_atoms.pbc is not None else None,
                }
                
                db_traj = DBTrajectory(
                    run_id=run_id,
                    traj_id=traj_id,
                    target_length=target_length,
                    chunks_completed=0,
                    init_atoms_json=init_atoms_json,
                    status=TrajectoryStatus.RUNNING,
                )
                sess.add(db_traj)
                self._set_trajectory_status(sess, db_traj, TrajectoryStatus.RUNNING)
                return True
        except Exception as e:
            logger.error(f"Failed to initialize trajectory {traj_id} for run {run_id}: {e}")
            return False
    
    def add_chunk_attempt(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        model_version: int,
        n_frames: int,
        audit_status: AuditStatus = AuditStatus.PENDING,
        attempt_index: Optional[int] = None
    ) -> bool:
        """Add a new chunk attempt to the database
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            model_version: Model version used to generate the chunk
            n_frames: Number of frames in the chunk
            audit_status: Audit status of the chunk
            attempt_index: Attempt index (auto-incremented if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session() as sess:
                # Get the trajectory
                db_traj = sess.query(DBTrajectory).filter_by(
                    run_id=run_id,
                    traj_id=traj_id
                ).first()
                
                if not db_traj:
                    logger.error(f"Trajectory {traj_id} not found for run {run_id}")
                    return False
                
                # Determine attempt index if not provided
                if attempt_index is None:
                    existing_attempts = sess.query(DBTrajectoryChunk).filter_by(
                        run_id=run_id,
                        traj_id=traj_id,
                        chunk_id=chunk_id
                    ).count()
                    attempt_index = existing_attempts
                
                # Check if this exact attempt already exists
                existing = sess.query(DBTrajectoryChunk).filter_by(
                    run_id=run_id,
                    traj_id=traj_id,
                    chunk_id=chunk_id,
                    attempt_index=attempt_index
                ).first()
                
                if existing:
                    # Update existing attempt
                    existing.model_version = model_version
                    existing.audit_status = audit_status
                    existing.n_frames = n_frames
                    sess.flush()
                    return True
                
                # Create new chunk attempt
                db_chunk = DBTrajectoryChunk(
                    run_id=run_id,
                    trajectory_id=db_traj.id,
                    traj_id=traj_id,
                    chunk_id=chunk_id,
                    attempt_index=attempt_index,
                    model_version=model_version,
                    audit_status=audit_status,
                    n_frames=n_frames
                )
                sess.add(db_chunk)

                # Ensure trajectory marked running if new chunk attempt created
                self._set_trajectory_status(sess, db_traj, TrajectoryStatus.RUNNING)
                return True
        except Exception as e:
            logger.error(f"Failed to add chunk attempt for traj {traj_id}, chunk {chunk_id}, attempt {attempt_index}: {e}")
            return False
    
    def update_chunk_audit_done_status(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        attempt_index: int,
        audit_status: AuditStatus,
        audit_reason: str | None = None,
    ):
        """Update the audit status of a chunk and mark trajectory as done if it is complete

        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            attempt_index: Attempt index
            audit_status: New audit status
            audit_reason: Which mechanism produced audit_status (see AuditResult.reason)
        """
        with self.session() as sess:
            chunk = sess.query(DBTrajectoryChunk).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index
            ).first()

            if not chunk:
                raise ValueError(
                    f"Chunk not found: run_id={run_id}, traj_id={traj_id}, "
                    f"chunk_id={chunk_id}, attempt_index={attempt_index}"
                )

            chunk.audit_status = audit_status
            chunk.audit_reason = audit_reason
            
            # If chunk passed, increment chunks_completed on trajectory
            if audit_status == AuditStatus.PASSED:
                # Flush to ensure the updated status is visible in queries
                sess.flush()
                # Check if this is the latest passed chunk
                latest_passed = sess.query(func.max(DBTrajectoryChunk.chunk_id)).filter_by(
                    run_id=run_id,
                    traj_id=traj_id,
                    audit_status=AuditStatus.PASSED
                ).scalar()
                
                if latest_passed is not None:
                    traj = sess.query(DBTrajectory).filter_by(
                        run_id=run_id,
                        traj_id=traj_id
                    ).first()
                    if traj:
                        traj.chunks_completed = latest_passed + 1
                        
                        # Check if trajectory is done
                        passed_chunks = sess.query(DBTrajectoryChunk).filter_by(
                            run_id=run_id,
                            traj_id=traj_id,
                            audit_status=AuditStatus.PASSED
                        ).all()
                        total_frames = sum(chunk.n_frames for chunk in passed_chunks)
                        new_status = (
                            TrajectoryStatus.COMPLETED
                            if total_frames >= traj.target_length
                            else TrajectoryStatus.RUNNING
                        )
                        self._set_trajectory_status(sess, traj, new_status)
    
    def get_latest_passed_chunk(
        self,
        run_id: str,
        traj_id: int
    ) -> Optional[dict]:
        """Get the latest (highest chunk_id) passed chunk for a trajectory
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            
        Returns:
            Dict with chunk metadata for the latest passed chunk, or None if no chunks passed.
            Contains: chunk_id, attempt_index, model_version, audit_status, n_frames
        """
        with self.session() as sess:
            # Get the chunk with highest chunk_id that has PASSED status
            chunk = sess.query(DBTrajectoryChunk).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                audit_status=AuditStatus.PASSED
            ).order_by(DBTrajectoryChunk.chunk_id.desc()).first()
            
            if not chunk:
                return None
            
            # Return scalar values to avoid detached instance issues
            return {
                'chunk_id': chunk.chunk_id,
                'attempt_index': chunk.attempt_index,
                'model_version': chunk.model_version,
                'audit_status': chunk.audit_status,
                'n_frames': chunk.n_frames
            }
    
    def get_passed_chunks(
        self,
        run_id: str,
        traj_id: int
    ) -> list[dict]:
        """Get all passed chunks for a trajectory, ordered by chunk_id
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            
        Returns:
            List of dicts with chunk metadata for passed chunks
        """
        with self.session() as sess:
            chunks = sess.query(DBTrajectoryChunk).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                audit_status=AuditStatus.PASSED
            ).order_by(DBTrajectoryChunk.chunk_id).all()
            
            # Return scalar values to avoid detached instance issues
            return [
                {
                    'run_id': chunk.run_id,
                    'traj_id': chunk.traj_id,
                    'chunk_id': chunk.chunk_id,
                    'attempt_index': chunk.attempt_index,
                    'model_version': chunk.model_version,
                    'audit_status': chunk.audit_status,
                    'n_frames': chunk.n_frames
                }
                for chunk in chunks
            ]
    
    def get_trajectory_atoms(
        self,
        run_id: str,
        traj_id: int
    ) -> list[Atoms]:
        """Get all atoms from passed chunks for a trajectory
        
        This queries the trajectory_frames table for all frames matching the passed chunks
        and returns them as a list of Atoms objects. When reconstructing the full
        trajectory, the first frame of each chunk (except chunk 0) is skipped to
        avoid duplicates, since the first frame of chunk N is the same as the
        last frame of chunk N-1.
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            
        Returns:
            List of Atoms objects from all passed chunks, in order, with duplicates removed
        """
        chunks = self.get_passed_chunks(run_id, traj_id)
        
        all_atoms = []
        with self.session() as sess:
            for i, chunk in enumerate(chunks):
                # Query frames for this chunk and attempt
                frames = sess.query(DBTrajectoryFrame).filter_by(
                    run_id=run_id,
                    traj_id=traj_id,
                    chunk_id=chunk['chunk_id'],
                    attempt_index=chunk['attempt_index']
                ).order_by(DBTrajectoryFrame.frame_index).all()
                
                # Skip first frame of all chunks except chunk 0
                # (first frame is duplicate of previous chunk's last frame)
                if i > 0:
                    frames = frames[1:]
                
                # Deserialize directly from ORM objects
                for frame in frames:
                    all_atoms.append(self._deserialize_atoms(frame.atoms_blob))
        
        # Force garbage collection after deserializing large binary data
        gc.collect()
        return all_atoms
    
    def get_trajectory(
        self,
        run_id: str,
        traj_id: int
    ) -> Optional[dict]:
        """Get trajectory by run_id and traj_id
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            
        Returns:
            Dict with trajectory metadata or None if not found
        """
        with self.session() as sess:
            traj = sess.query(DBTrajectory).filter_by(
                run_id=run_id,
                traj_id=traj_id
            ).first()
            
            if not traj:
                return None
            
            return {
                'run_id': traj.run_id,
                'traj_id': traj.traj_id,
                'target_length': traj.target_length,
                'chunks_completed': traj.chunks_completed,
                'status': traj.status,
                'init_atoms_json': traj.init_atoms_json,
                'created_at': traj.created_at,
                'updated_at': traj.updated_at
            }
    
    def get_next_attempt_index(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int
    ) -> int:
        """Get the next attempt index for a chunk
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            
        Returns:
            Next attempt index (0-based)
        """
        with self.session() as sess:
            count = sess.query(DBTrajectoryChunk).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id
            ).count()
            return count
    
    def get_chunk_attempt(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        attempt_index: int
    ) -> Optional[dict]:
        """Get a specific chunk attempt
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            attempt_index: Attempt index
            
        Returns:
            Dict with chunk metadata or None if not found
        """
        with self.session() as sess:
            chunk = sess.query(DBTrajectoryChunk).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index
            ).first()
            
            if not chunk:
                return None
            
            return {
                'run_id': chunk.run_id,
                'traj_id': chunk.traj_id,
                'chunk_id': chunk.chunk_id,
                'attempt_index': chunk.attempt_index,
                'model_version': chunk.model_version,
                'audit_status': chunk.audit_status,
                'n_frames': chunk.n_frames
            }
    
    def get_latest_chunk_attempt(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int
    ) -> Optional[dict]:
        """Get the latest (most recent) chunk attempt for a given chunk
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            
        Returns:
            Dict with chunk metadata or None if not found
        """
        with self.session() as sess:
            # Get the latest attempt by ordering by attempt_index descending
            chunk = sess.query(DBTrajectoryChunk).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id
            ).order_by(DBTrajectoryChunk.attempt_index.desc()).first()
            
            if not chunk:
                return None
                
            # Return scalar values to avoid detached instance issues
            return {
                'attempt_index': chunk.attempt_index,
                'model_version': chunk.model_version,
                'audit_status': chunk.audit_status,
                'n_frames': chunk.n_frames
            }
    
    def get_latest_chunk_attempt_atoms(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int
    ) -> list[Atoms]:
        """Get the atoms for the latest attempt of a chunk
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
        
        Returns:
            List of Atoms objects for the latest chunk attempt (empty if none found)
        """
        latest_attempt = self.get_latest_chunk_attempt(run_id, traj_id, chunk_id)
        if not latest_attempt:
            return []
        
        attempt_index = latest_attempt['attempt_index']
        
        with self.session() as sess:
            frames = sess.query(DBTrajectoryFrame).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index
            ).order_by(DBTrajectoryFrame.frame_index).all()
            
            if not frames:
                return []
            
            # Deserialize directly from ORM objects
            atoms_list = [self._deserialize_atoms(frame.atoms_blob) for frame in frames]
        
        # Force garbage collection after deserializing large binary data
        gc.collect()
        return atoms_list
    
    def is_trajectory_done(
        self,
        run_id: str,
        traj_id: int
    ) -> bool:
        """Check if a trajectory is complete (has reached target length with passed chunks)
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            
        Returns:
            True if trajectory is done, False otherwise
        """
        with self.session() as sess:
            traj = sess.query(DBTrajectory).filter_by(
                run_id=run_id,
                traj_id=traj_id
            ).first()
            
            if not traj:
                return False
            
            return traj.status == TrajectoryStatus.COMPLETED

    def get_trajectory_status(
        self,
        run_id: str,
        traj_id: int,
    ) -> Optional[TrajectoryStatus]:
        """Return the lifecycle status for a trajectory, if it exists."""
        with self.session() as sess:
            traj = sess.query(DBTrajectory).filter_by(
                run_id=run_id,
                traj_id=traj_id,
            ).first()
            if not traj:
                return None
            return traj.status
    
    def get_first_frame_from_chunk(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        attempt_index: int
    ) -> Optional[Atoms]:
        """Get the first frame from a chunk
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            attempt_index: Attempt index for this chunk
            
        Returns:
            Atoms object for the first frame, or None if no frames found
        """
        with self.session() as sess:
            frame = sess.query(DBTrajectoryFrame).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index,
                frame_index=0
            ).first()
            
            if not frame:
                return None
            
            # Deserialize directly from ORM object
            atoms = self._deserialize_atoms(frame.atoms_blob)
        
        # Force garbage collection after deserializing large binary data
        gc.collect()
        return atoms
    
    def get_last_frame_from_chunk(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        attempt_index: int
    ) -> Optional[Atoms]:
        """Get the last frame from a chunk
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            attempt_index: Attempt index for this chunk
            
        Returns:
            Atoms object for the last frame, or None if no frames found
        """
        with self.session() as sess:
            # Get the frame with the highest frame_index for this chunk/attempt
            frame = sess.query(DBTrajectoryFrame).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index
            ).order_by(DBTrajectoryFrame.frame_index.desc()).first()
            
            if not frame:
                return None
            
            # Deserialize directly from ORM object
            atoms = self._deserialize_atoms(frame.atoms_blob)
        
        # Force garbage collection after deserializing large binary data
        gc.collect()
        return atoms
    
    def get_chunk_frame_ids(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        attempt_index: int
    ) -> list[int]:
        """Get frame IDs for a chunk, ordered by frame_index
        
        Uses with_entities to only select the ID column, avoiding loading full ORM objects.
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            attempt_index: Attempt index for this chunk
            
        Returns:
            List of frame IDs ordered by frame_index
        """
        with self.session() as sess:
            # Query only the ID and frame_index columns, then extract IDs in order
            frame_rows = sess.query(
                DBTrajectoryFrame.id,
                DBTrajectoryFrame.frame_index
            ).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index
            ).order_by(DBTrajectoryFrame.frame_index).all()
            return [row[0] for row in frame_rows]
    
    def get_initial_trajectory_frame(
        self,
        run_id: str,
        traj_id: int
    ) -> Optional[Atoms]:
        """Get the initial frame of a trajectory
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            
        Returns:
            Atoms object for the initial frame, or None if trajectory not found
        """
        with self.session() as sess:
            traj = sess.query(DBTrajectory).filter_by(
                run_id=run_id,
                traj_id=traj_id
            ).first()
            
            if not traj or not traj.init_atoms_json:
                return None
            
            # Reconstruct atoms from JSON
            init_atoms_json = traj.init_atoms_json
            atoms = Atoms(
                numbers=init_atoms_json['numbers'],
                positions=init_atoms_json['positions']
            )
            
            if init_atoms_json.get('cell') is not None:
                atoms.set_cell(init_atoms_json['cell'])
            
            if init_atoms_json.get('pbc') is not None:
                atoms.set_pbc(init_atoms_json['pbc'])
        
        # Force garbage collection after reconstructing atoms
        gc.collect()
        return atoms
    
    def get_current_training_round(self, run_id: str) -> int:
        """Get the current training round number for a run
        
        Returns the maximum training_round in the database, or 0 if none exists.
        This represents the most recent training round that has frames.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Current training round number (0-based)
        """
        with self.session() as sess:
            max_round = sess.query(func.max(DBTrainingFrame.training_round)).filter_by(
                run_id=run_id
            ).scalar()
            return max_round if max_round is not None else 0
    
    
    def add_training_frame(
        self,
        run_id: str,
        trajectory_frame_id: int,
        model_version_sampled_from: int,
        traj_id: int,
        chunk_id: int,
        attempt_index: int,
        atoms_labeled: Atoms,
        calibration_uq: Optional[float] = None,
        calibration_error: Optional[float] = None,
    ) -> DBTrainingFrame:
        """Add a training frame to the database

        Args:
            run_id: Run identifier
            trajectory_frame_id: ID of the frame in the trajectory_frames table
            model_version_sampled_from: Model version that generated this frame
            traj_id: Trajectory identifier (denormalized)
            chunk_id: Chunk identifier (denormalized)
            attempt_index: Attempt index (denormalized)
            atoms_labeled: Labeled atoms with energy/forces to store
            calibration_uq: UQ scalar recorded at sample time, for Controller calibration
            calibration_error: Observed error vs. the DFT label, for Controller calibration

        Returns:
            DBTrainingFrame instance
        """
        with self.session() as sess:
            existing = sess.query(DBTrainingFrame).filter_by(
                run_id=run_id,
                trajectory_frame_id=trajectory_frame_id
            ).first()

            if existing:
                return existing

            db_training_frame = DBTrainingFrame(
                run_id=run_id,
                trajectory_frame_id=trajectory_frame_id,
                model_version_sampled_from=model_version_sampled_from,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index,
                training_round=None,
                atoms_labeled_blob=self._serialize_atoms(atoms_labeled),
                calibration_uq=calibration_uq,
                calibration_error=calibration_error,
            )
            sess.add(db_training_frame)
            sess.flush()
            sess.refresh(db_training_frame)
            return db_training_frame
    
    def get_training_frames(
        self,
        run_id: str,
        training_round: int,
    ) -> list[Atoms]:
        """Get labeled training frames for a specific training round.

        Args:
            run_id: Run identifier
            training_round: Round whose frames should be returned

        Returns:
            List of labeled Atoms objects
        """
        with self.session() as sess:
            training_frames = sess.query(DBTrainingFrame).filter_by(
                run_id=run_id,
                training_round=training_round,
            ).all()

            if not training_frames:
                return []

            atoms_list = [
                self._deserialize_atoms(tf.atoms_labeled_blob)
                for tf in training_frames
                if tf.atoms_labeled_blob is not None
            ]

        gc.collect()
        return atoms_list
    
    def count_training_frames(self, run_id: str) -> int:
        """Count the number of training frames for a run
        
        Args:
            run_id: Run identifier
            
        Returns:
            Number of training frames
        """
        with self.session() as sess:
            return sess.query(DBTrainingFrame).filter_by(
                run_id=run_id
            ).count()
    
    def mark_training_frames_for_round(self, run_id: str, training_round: int) -> int:
        """Mark all unmarked training frames with a training round number
        
        Args:
            run_id: Run identifier
            training_round: Training round number to assign
            
        Returns:
            Number of frames marked
        """
        with self.session() as sess:
            # Update all frames that don't have a training_round yet
            updated = sess.query(DBTrainingFrame).filter_by(
                run_id=run_id
            ).filter(
                DBTrainingFrame.training_round.is_(None)
            ).update(
                {'training_round': training_round},
                synchronize_session=False
            )
            return updated
    
    def get_sampled_traj_ids(
        self,
        run_id: str
    ) -> set[int]:
        """Get unique trajectory IDs that have training frames sampled from them
        
        Args:
            run_id: Run identifier
            
        Returns:
            Set of unique trajectory IDs from sampled frames
        """
        with self.session() as sess:
            training_frames = sess.query(DBTrainingFrame).filter_by(
                run_id=run_id
            ).all()
            
            if not training_frames:
                return set()
            
            # Extract unique trajectory IDs from denormalized data
            unique_traj_ids = {tf.traj_id for tf in training_frames}
            
            return unique_traj_ids
    
    def get_latest_chunk_id(
        self,
        run_id: str,
        traj_id: int
    ) -> Optional[int]:
        """Get the latest chunk ID for a trajectory
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            
        Returns:
            Latest chunk ID or None if no chunks exist
        """
        with self.session() as sess:
            # Get the maximum chunk_id for this trajectory
            max_chunk_id = sess.query(func.max(DBTrajectoryChunk.chunk_id)).filter_by(
                run_id=run_id,
                traj_id=traj_id
            ).scalar()
            
            return max_chunk_id
    
    def get_latest_passed_chunk(
        self,
        run_id: str,
        traj_id: int
    ) -> Optional[dict]:
        """Get the latest passed chunk for a trajectory
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            
        Returns:
            Dict with chunk metadata or None if no passed chunks exist
        """
        with self.session() as sess:
            chunk = sess.query(DBTrajectoryChunk).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                audit_status=AuditStatus.PASSED
            ).order_by(DBTrajectoryChunk.chunk_id.desc()).first()
            
            if not chunk:
                return None
            
            return {
                'chunk_id': chunk.chunk_id,
                'attempt_index': chunk.attempt_index,
                'model_version': chunk.model_version,
                'audit_status': chunk.audit_status,
                'n_frames': chunk.n_frames
            }
    
    def get_initial_atoms(
        self,
        run_id: str,
        traj_id: int
    ) -> Optional[Atoms]:
        """Get the initial atoms for a trajectory
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            
        Returns:
            Atoms object for initial conditions or None if not found
        """
        traj = self.get_trajectory(run_id, traj_id)
        if not traj or not traj['init_atoms_json']:
            return None
        
        # Reconstruct Atoms from JSON
        atoms_json = traj['init_atoms_json']
        atoms = Atoms(
            positions=np.array(atoms_json['positions']),
            numbers=np.array(atoms_json['numbers'])
        )
        
        if atoms_json.get('cell') is not None:
            atoms.cell = np.array(atoms_json['cell'])
        
        if atoms_json.get('pbc') is not None:
            atoms.pbc = np.array(atoms_json['pbc'])
        
        # Force garbage collection after reconstructing atoms
        gc.collect()
        return atoms
    
    def record_chunk_event(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        attempt_index: int,
        event_type: ChunkEventType,
        frame_id: Optional[int] = None
    ) -> None:
        """Record a chunk-level event
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            attempt_index: Attempt index
            event_type: Type of event
            frame_id: Optional frame ID for frame-level events
        """
        with self.session() as sess:
            db_event = DBChunkEvent(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index,
                frame_id=frame_id,
                event_type=event_type
            )
            sess.add(db_event)
    
    def record_training_event(
        self,
        run_id: str,
        event_type: ChunkEventType,
        training_round: int
    ) -> None:
        """Record a training-level event
        
        Args:
            run_id: Run identifier
            event_type: Type of event (STARTED_TRAINING or FINISHED_TRAINING)
            training_round: Training round number
        """
        with self.session() as sess:
            db_event = DBTrainingEvent(
                run_id=run_id,
                event_type=event_type,
                training_round=training_round
            )
            sess.add(db_event)
    
    def write_training_log(self, run_id: str, training_round: int, log: pd.DataFrame, member_index: int = 0) -> None:
        """Persist per-epoch training metrics for a completed training round.

        Args:
            run_id: Run identifier
            training_round: Training round number
            log: DataFrame returned by MACEInterface.train, one row per epoch
            member_index: Which ensemble member this log belongs to (0 if not ensembling)
        """
        with self.session() as sess:
            sess.add(DBTrainingLog(
                run_id=run_id,
                training_round=training_round,
                member_index=member_index,
                # NaN (e.g. from replay columns that only populate every few epochs) is not
                # valid JSON and Postgres' json/jsonb columns reject it outright. `to_dict`
                # leaves NaN as-is, but `to_json` correctly renders it as `null`, so round-trip
                # through that instead.
                log_json=json.loads(log.to_json(orient='records')),
            ))

    def get_training_logs(self, run_id: str) -> pd.DataFrame:
        """Return all training loss history for a run as a single DataFrame.

        Each row is one epoch from one training round/member. ``training_round``
        and ``member_index`` columns are prepended so callers can group or filter by them.

        Args:
            run_id: Run identifier

        Returns:
            DataFrame with columns [training_round, member_index, epoch, <metric columns>],
            or an empty DataFrame if no logs exist yet.
        """
        with self.session() as sess:
            rows = (
                sess.query(DBTrainingLog)
                .filter_by(run_id=run_id)
                .order_by(DBTrainingLog.training_round)
                .all()
            )
            frames = []
            for row in rows:
                df = pd.DataFrame(row.log_json)
                df.insert(0, 'training_round', row.training_round)
                df.insert(1, 'member_index', row.member_index)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_controller_observations(
        self,
        run_id: str,
        burn_in_model_versions: int,
        limit: int,
        traj_id: int | None = None,
    ) -> tuple[int, list[tuple[float, float]]]:
        """Return recent (uq, error) pairs for the Controller's threshold calibration.

        Only frames from the newest model version present among qualifying, labeled
        frames are returned, so a calibration window is never blended across model
        versions (each model version has its own UQ/error relationship). "Newest" is
        inferred from the data itself (MAX(model_version_sampled_from)) rather than
        pushed in from another agent, since sampling/labeling happens concurrently
        across trajectories and label completion order does not track generation order.

        Args:
            run_id: Run identifier
            burn_in_model_versions: Ignore frames sampled below this model version
            limit: Max number of most-recent observations to return
            traj_id: If set, restrict to this trajectory's own observations only
                (per-trajectory calibration); if None, pool across all trajectories.

        Returns:
            (model_version, observations): the model version the window was drawn
            from, and (calibration_uq, calibration_error) pairs, newest first. Both
            are empty/None if no qualifying frames exist yet.
        """
        with self.session() as sess:
            base_filters = [
                DBTrainingFrame.run_id == run_id,
                DBTrainingFrame.calibration_error.isnot(None),
            ]
            if traj_id is not None:
                base_filters.append(DBTrainingFrame.traj_id == traj_id)

            latest_version = (
                sess.query(func.max(DBTrainingFrame.model_version_sampled_from))
                .filter(
                    *base_filters,
                    DBTrainingFrame.model_version_sampled_from >= burn_in_model_versions,
                )
                .scalar()
            )
            if latest_version is None:
                return None, []

            rows = (
                sess.query(DBTrainingFrame.calibration_uq, DBTrainingFrame.calibration_error)
                .filter(
                    *base_filters,
                    DBTrainingFrame.model_version_sampled_from == latest_version,
                )
                .order_by(DBTrainingFrame.created_at.desc())
                .limit(limit)
                .all()
            )
            return latest_version, [(uq, err) for uq, err in rows]

    def write_controller_log(
        self,
        run_id: str,
        model_version: int,
        threshold: float,
        alpha: float,
        mean_error: float,
        n_observations: int,
        traj_id: int | None = None,
    ) -> None:
        """Persist one Controller calibration event for later analysis.

        Args:
            run_id: Run identifier
            model_version: model_version_sampled_from of the calibration window
            threshold: Newly calibrated audit threshold
            alpha: Newly fit alpha (error / UQ ratio)
            mean_error: Mean observed error over the calibration window
            n_observations: Number of observations the calibration window contained
            traj_id: Trajectory this calibration is specific to, or None if shared
                across all trajectories
        """
        with self.session() as sess:
            sess.add(DBControllerLog(
                run_id=run_id,
                traj_id=traj_id,
                model_version=model_version,
                threshold=threshold,
                alpha=alpha,
                mean_error=mean_error,
                n_observations=n_observations,
            ))

    def get_controller_log(self, run_id: str) -> pd.DataFrame:
        """Return the full calibration history for a run as a DataFrame.

        Args:
            run_id: Run identifier

        Returns:
            DataFrame with columns [traj_id, model_version, threshold, alpha,
            mean_error, n_observations, created_at], ordered by created_at, or an
            empty DataFrame if none exist. traj_id is None for entries logged while
            the threshold was shared across all trajectories.
        """
        with self.session() as sess:
            rows = (
                sess.query(DBControllerLog)
                .filter_by(run_id=run_id)
                .order_by(DBControllerLog.created_at)
                .all()
            )
            return pd.DataFrame([
                {
                    'traj_id': r.traj_id,
                    'model_version': r.model_version,
                    'threshold': r.threshold,
                    'alpha': r.alpha,
                    'mean_error': r.mean_error,
                    'n_observations': r.n_observations,
                    'created_at': r.created_at,
                }
                for r in rows
            ])

    def has_chunk_event(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        attempt_index: int,
        event_type: ChunkEventType
    ) -> bool:
        """Check if a chunk event exists (for idempotent checks)
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            attempt_index: Attempt index
            event_type: Type of event to check
            
        Returns:
            True if event exists, False otherwise
        """
        with self.session() as sess:
            count = sess.query(DBChunkEvent).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index,
                event_type=event_type
            ).count()
            return count > 0
    
    def get_latest_event_for_chunk(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        attempt_index: int
    ) -> Optional[ChunkEventType]:
        """Get the most recent event type for a chunk
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            attempt_index: Attempt index
            
        Returns:
            Most recent event type, or None if no events exist
        """
        with self.session() as sess:
            event = sess.query(DBChunkEvent).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index
            ).order_by(DBChunkEvent.created_at.desc()).first()
            
            if not event:
                return None
            return event.event_type
    
    def get_latest_event_per_trajectory(self, run_id: str) -> list[dict]:
        """For each trajectory, the latest chunk event (by created_at).

        One row per traj_id. Returns list of dicts with traj_id, event_type,
        chunk_id, attempt_index.
        """
        with self.session() as sess:
            subq = (
                sess.query(
                    DBChunkEvent.traj_id,
                    DBChunkEvent.event_type,
                    DBChunkEvent.chunk_id,
                    DBChunkEvent.attempt_index,
                    func.row_number()
                    .over(
                        partition_by=DBChunkEvent.traj_id,
                        order_by=DBChunkEvent.created_at.desc(),
                    )
                    .label("rn"),
                )
                .filter_by(run_id=run_id)
                .subquery()
            )
            rows = (
                sess.query(
                    subq.c.traj_id,
                    subq.c.event_type,
                    subq.c.chunk_id,
                    subq.c.attempt_index,
                )
                .filter(subq.c.rn == 1)
                .all()
            )
            return [
                {
                    "traj_id": t,
                    "event_type": e,
                    "chunk_id": c,
                    "attempt_index": a,
                }
                for t, e, c, a in rows
            ]

    def get_trajs_with_latest_event(
        self,
        run_id: str,
        event_type: ChunkEventType
    ) -> list[dict]:
        """Trajectories whose latest chunk event (by created_at) matches the given type.

        Returns list of dicts with traj_id, chunk_id, attempt_index (at most one per trajectory).
        """
        rows = self.get_latest_event_per_trajectory(run_id)
        return [
            {"traj_id": r["traj_id"], "chunk_id": r["chunk_id"], "attempt_index": r["attempt_index"]}
            for r in rows
            if r["event_type"] == event_type
        ]
    
    def list_chunk_events(
        self,
        run_id: str,
        traj_id: Optional[int] = None,
        chunk_id: Optional[int] = None,
        limit: Optional[int] = None
    ) -> list[dict]:
        """List chunk events for a run, optionally filtered by trajectory or chunk.
        
        Args:
            run_id: Run identifier
            traj_id: Optional trajectory ID to filter by
            chunk_id: Optional chunk ID to filter by
            limit: Optional maximum number of events to return
            
        Returns:
            List of dicts with run_id, traj_id, chunk_id, attempt_index, event_type,
            frame_id, created_at. event_type is the enum name (e.g. 'STARTED_LABELING').
            Ordered by traj_id, chunk_id, attempt_index, created_at.
        """
        with self.session() as sess:
            q = sess.query(DBChunkEvent).filter_by(run_id=run_id)
            if traj_id is not None:
                q = q.filter_by(traj_id=traj_id)
            if chunk_id is not None:
                q = q.filter_by(chunk_id=chunk_id)
            q = q.order_by(
                DBChunkEvent.traj_id,
                DBChunkEvent.chunk_id,
                DBChunkEvent.attempt_index,
                DBChunkEvent.created_at
            )
            if limit is not None:
                q = q.limit(limit)
            events = q.all()
            
            result = []
            for e in events:
                event_type_str = e.event_type.name if hasattr(e.event_type, 'name') else str(e.event_type)
                result.append({
                    'run_id': e.run_id,
                    'traj_id': e.traj_id,
                    'chunk_id': e.chunk_id,
                    'attempt_index': e.attempt_index,
                    'event_type': event_type_str,
                    'frame_id': e.frame_id,
                    'created_at': e.created_at
                })
            return result

    def count_chunk_events_by_type(self, run_id: str) -> dict[str, int]:
        """Count chunk events grouped by event type for a run.

        Args:
            run_id: Run identifier

        Returns:
            Dict mapping event type name to count, e.g. {'STARTED_LABELING': 10, ...}
        """
        with self.session() as sess:
            rows = (
                sess.query(DBChunkEvent.event_type, func.count().label("count"))
                .filter_by(run_id=run_id)
                .group_by(DBChunkEvent.event_type)
                .all()
            )
            return {event_type.name: count for event_type, count in rows}

    def count_active_trajs_with_labeling(
        self,
        run_id: str
    ) -> tuple[int, int]:
        """Count active trajectories and those with labeling since last training.
        
        Only FINISHED_LABELING events after the most recent FINISHED_TRAINING are
        counted. If there is no previous training, all FINISHED_LABELING events count.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Tuple of (total_active_trajectories, active_trajectories_with_labeling)
        """
        with self.session() as sess:
            # Last training completion time; None if we haven't trained yet
            last_train = (
                sess.query(DBTrainingEvent.created_at)
                .filter_by(
                    run_id=run_id,
                    event_type=ChunkEventType.FINISHED_TRAINING
                )
                .order_by(DBTrainingEvent.created_at.desc())
                .limit(1)
                .scalar()
            )
            
            # Get all trajectories for this run
            all_trajectories = sess.query(DBTrajectory).filter_by(
                run_id=run_id
            ).all()
            
            # Unique traj_ids with FINISHED_LABELING since last training
            q = sess.query(DBChunkEvent.traj_id).filter_by(
                run_id=run_id,
                event_type=ChunkEventType.FINISHED_LABELING
            )
            if last_train is not None:
                q = q.filter(DBChunkEvent.created_at > last_train)
            labeled_traj_ids = {r[0] for r in q.distinct().all()}
            
            # Extract trajectory info while session is active
            trajectory_info = [
                {'traj_id': traj.traj_id, 'status': traj.status}
                for traj in all_trajectories
            ]
        
        # Now check which trajectories are active and have labeling
        active_count = 0
        active_with_labeling_count = 0
        
        for traj_info in trajectory_info:
            # Check if trajectory is active (not done)
            if traj_info['status'] == TrajectoryStatus.RUNNING:
                active_count += 1
                # Check if this trajectory has been labeled
                if traj_info['traj_id'] in labeled_traj_ids:
                    active_with_labeling_count += 1
        
        return (active_count, active_with_labeling_count)
    
    def get_last_training_completion_time(
        self,
        run_id: str
    ) -> Optional[datetime]:
        """Get the timestamp of the most recent FINISHED_TRAINING event
        
        Args:
            run_id: Run identifier
            
        Returns:
            Timestamp of most recent training completion, or None if no training events
        """
        with self.session() as sess:
            event = sess.query(DBTrainingEvent).filter_by(
                run_id=run_id,
                event_type=ChunkEventType.FINISHED_TRAINING
            ).order_by(DBTrainingEvent.created_at.desc()).first()
            
            if not event:
                return None
            return event.created_at
    
    def count_labeled_frames_for_chunk(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        attempt_index: int
    ) -> int:
        """Count FINISHED_LABELING_FRAME events for a chunk
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            attempt_index: Attempt index
            
        Returns:
            Number of frames labeled for this chunk
        """
        with self.session() as sess:
            count = sess.query(DBChunkEvent).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index,
                event_type=ChunkEventType.FINISHED_LABELING_FRAME
            ).count()
            return count
    
    def list_runs(self) -> list[dict]:
        """List all unique runs in the database with metadata
        
        Returns:
            List of dicts with run metadata, sorted by first_created (newest first).
            Each dict contains:
                - run_id: Run identifier
                - first_created: Earliest trajectory creation time in this run
                - last_updated: Latest trajectory update time in this run
                - n_trajectories: Total number of trajectories
                - n_done_trajectories: Number of completed trajectories
        """
        with self.session() as sess:
            # Query distinct run_ids and aggregate statistics
            runs = sess.query(
                DBTrajectory.run_id,
                func.min(DBTrajectory.created_at).label('first_created'),
                func.max(DBTrajectory.updated_at).label('last_updated'),
                func.count(DBTrajectory.id).label('n_trajectories'),
                func.sum(func.cast(DBTrajectory.status == TrajectoryStatus.COMPLETED, Integer)).label('n_done_trajectories')
            ).group_by(DBTrajectory.run_id).all()
            
            result = []
            for run in runs:
                result.append({
                    'run_id': run.run_id,
                    'first_created': run.first_created,
                    'last_updated': run.last_updated,
                    'n_trajectories': run.n_trajectories,
                    'n_done_trajectories': run.n_done_trajectories or 0
                })
            
            # Sort by first_created, newest first
            result.sort(key=lambda x: x['first_created'] or '', reverse=True)
            return result
    
    def list_run_summary(self, run_id: str) -> Optional[dict]:
        """Get summary statistics for a specific run
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dict with run summary statistics or None if run doesn't exist.
            Contains:
                - run_id: Run identifier
                - n_trajectories: Total number of trajectories
                - n_done: Number of completed trajectories
                - n_active: Number of active (not done) trajectories
                - total_chunks: Total number of chunk records (all attempts)
                - total_passed_chunks: Count of chunks with audit_status=PASSED
                - total_failed_chunks: Count of chunks with audit_status=FAILED
                - total_pending_chunks: Count of chunks with audit_status=PENDING
                - total_training_frames: Count of training frames
                - first_created: Earliest trajectory creation time
                - last_updated: Latest trajectory update time
        """
        with self.session() as sess:
            # Check if run exists
            run_exists = sess.query(DBTrajectory).filter_by(run_id=run_id).first()
            if not run_exists:
                return None
            
            # Get trajectory statistics
            traj_stats = sess.query(
                func.count(DBTrajectory.id).label('n_trajectories'),
                func.sum(func.cast(DBTrajectory.status == TrajectoryStatus.COMPLETED, Integer)).label('n_done'),
                func.min(DBTrajectory.created_at).label('first_created'),
                func.max(DBTrajectory.updated_at).label('last_updated')
            ).filter_by(run_id=run_id).first()
            
            # Get chunk statistics
            chunk_stats = sess.query(
                func.count(DBTrajectoryChunk.id).label('total_chunks'),
                func.sum(func.cast(DBTrajectoryChunk.audit_status == AuditStatus.PASSED, Integer)).label('total_passed'),
                func.sum(func.cast(DBTrajectoryChunk.audit_status == AuditStatus.FAILED, Integer)).label('total_failed'),
                func.sum(func.cast(DBTrajectoryChunk.audit_status == AuditStatus.PENDING, Integer)).label('total_pending')
            ).filter_by(run_id=run_id).first()
            
            # Get training frame count
            training_frame_count = sess.query(func.count(DBTrainingFrame.id)).filter_by(
                run_id=run_id
            ).scalar() or 0
            
            return {
                'run_id': run_id,
                'n_trajectories': traj_stats.n_trajectories or 0,
                'n_done': traj_stats.n_done or 0,
                'n_active': (traj_stats.n_trajectories or 0) - (traj_stats.n_done or 0),
                'total_chunks': chunk_stats.total_chunks or 0,
                'total_passed_chunks': chunk_stats.total_passed or 0,
                'total_failed_chunks': chunk_stats.total_failed or 0,
                'total_pending_chunks': chunk_stats.total_pending or 0,
                'total_training_frames': training_frame_count,
                'first_created': traj_stats.first_created,
                'last_updated': traj_stats.last_updated
            }
    
    def list_trajectory_summary(self, run_id: str, traj_id: int) -> Optional[dict]:
        """Get detailed statistics for a specific trajectory
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            
        Returns:
            Dict with trajectory summary statistics or None if trajectory doesn't exist.
            Contains:
                - run_id, traj_id, target_length, chunks_completed, done
                - created_at, updated_at
                - n_chunk_attempts: Total number of chunk attempt records
                - n_unique_chunks: Number of unique chunk_id values
                - chunk_breakdown: Dict mapping chunk_id to:
                    - n_attempts: Number of attempts for this chunk
                    - latest_status: Audit status of the latest attempt
                    - latest_attempt_index: The attempt_index of the latest attempt
                - status_counts: Dict with counts of PENDING, PASSED, FAILED attempts
        """
        with self.session() as sess:
            # Get trajectory
            traj = sess.query(DBTrajectory).filter_by(
                run_id=run_id,
                traj_id=traj_id
            ).first()
            
            if not traj:
                return None
            
            # Extract trajectory data while session is active
            traj_data = {
                'run_id': traj.run_id,
                'traj_id': traj.traj_id,
                'target_length': traj.target_length,
                'chunks_completed': traj.chunks_completed,
                'created_at': traj.created_at,
                'updated_at': traj.updated_at
            }
            
            # Get all chunks for this trajectory
            chunks = sess.query(DBTrajectoryChunk).filter_by(
                run_id=run_id,
                traj_id=traj_id
            ).order_by(DBTrajectoryChunk.chunk_id, DBTrajectoryChunk.attempt_index).all()
            
            # Extract chunk data while session is active
            chunk_data = []
            for chunk in chunks:
                chunk_data.append({
                    'chunk_id': chunk.chunk_id,
                    'attempt_index': chunk.attempt_index,
                    'audit_status': chunk.audit_status
                })
        
        # Process chunk data outside session
        n_chunk_attempts = len(chunk_data)
        unique_chunk_ids = set(c['chunk_id'] for c in chunk_data)
        n_unique_chunks = len(unique_chunk_ids)
        
        # Build chunk breakdown
        chunk_breakdown = {}
        for chunk_id in unique_chunk_ids:
            chunk_attempts = [c for c in chunk_data if c['chunk_id'] == chunk_id]
            # Latest attempt is the one with highest attempt_index
            latest = max(chunk_attempts, key=lambda x: x['attempt_index'])
            # Convert AuditStatus enum to string name
            latest_status = latest['audit_status']
            if hasattr(latest_status, 'name'):
                latest_status_str = latest_status.name
            else:
                latest_status_str = str(latest_status)
            chunk_breakdown[chunk_id] = {
                'n_attempts': len(chunk_attempts),
                'latest_status': latest_status_str,
                'latest_attempt_index': latest['attempt_index']
            }
        
        # Count statuses
        status_counts = {
            'PENDING': sum(1 for c in chunk_data if c['audit_status'] == AuditStatus.PENDING),
            'PASSED': sum(1 for c in chunk_data if c['audit_status'] == AuditStatus.PASSED),
            'FAILED': sum(1 for c in chunk_data if c['audit_status'] == AuditStatus.FAILED)
        }
        
        return {
            **traj_data,
            'n_chunk_attempts': n_chunk_attempts,
            'n_unique_chunks': n_unique_chunks,
            'chunk_breakdown': chunk_breakdown,
            'status_counts': status_counts
        }
    
    def list_trajectory_attempts(self, run_id: str, traj_id: int) -> list[dict]:
        """List all chunk attempts for a specific trajectory
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            
        Returns:
            List of dicts, one per attempt, ordered by chunk_id then attempt_index.
            Each dict contains:
                - chunk_id: Chunk identifier
                - attempt_index: Attempt number for this chunk
                - n_frames: Number of frames in this attempt
                - audit_status: Audit status (PENDING, PASSED, or FAILED)
                - model_version: Model version used for this attempt
                - created_at: When this attempt was created
                - updated_at: When this attempt was last updated
        """
        with self.session() as sess:
            # Get trajectory to verify it exists
            traj = sess.query(DBTrajectory).filter_by(
                run_id=run_id,
                traj_id=traj_id
            ).first()
            
            if not traj:
                return []
            
            # Get all attempts for this trajectory
            # Each row in DBTrajectoryChunk is a separate attempt
            attempts = sess.query(DBTrajectoryChunk).filter_by(
                run_id=run_id,
                traj_id=traj_id
            ).order_by(DBTrajectoryChunk.chunk_id, DBTrajectoryChunk.attempt_index).all()
            
            # Extract data while session is active
            result = []
            for attempt in attempts:
                # Convert AuditStatus enum to string name
                status = attempt.audit_status
                if hasattr(status, 'name'):
                    status_str = status.name
                else:
                    status_str = str(status)
                
                result.append({
                    'chunk_id': attempt.chunk_id,
                    'attempt_index': attempt.attempt_index,
                    'n_frames': attempt.n_frames,
                    'audit_status': status_str,
                    'model_version': attempt.model_version,
                    'created_at': attempt.created_at,
                    'updated_at': attempt.updated_at
                })
            
            return result
    
    def list_trajectories_in_run(self, run_id: str) -> list[dict]:
        """List all trajectories in a run with basic info
        
        Args:
            run_id: Run identifier
            
        Returns:
            List of dicts with trajectory metadata, sorted by traj_id.
            Each dict contains:
                - traj_id, target_length, chunks_completed, status, done
                - created_at, updated_at
        """
        with self.session() as sess:
            trajectories = sess.query(DBTrajectory).filter_by(
                run_id=run_id
            ).order_by(DBTrajectory.traj_id).all()
            
            result = []
            for traj in trajectories:
                result.append({
                    'traj_id': traj.traj_id,
                    'target_length': traj.target_length,
                    'chunks_completed': traj.chunks_completed,
                    'status': traj.status,
                    'created_at': traj.created_at,
                    'updated_at': traj.updated_at
                })
            
            return result

