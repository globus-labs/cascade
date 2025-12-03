"""ORM layer for tracking trajectories and trajectory chunks in PostgreSQL.

This module provides SQLAlchemy models and utilities for persisting trajectory
and chunk metadata, and storing trajectory frames as serialized Atoms objects.
"""
from __future__ import annotations

import contextlib
import gc
import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
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
    ForeignKey,
    func,
    JSON,
    LargeBinary,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from cascade.model import AuditStatus, TrajectoryStatus

Base = declarative_base()


class DBTrajectory(Base):
    """ORM model for trajectory metadata"""
    __tablename__ = 'trajectories'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    traj_id = Column(Integer, nullable=False)
    target_length = Column(Integer, nullable=False)
    chunks_completed = Column(Integer, default=0, nullable=False)
    done = Column(Boolean, default=False, nullable=False)  # legacy flag
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
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('run_id', 'trajectory_frame_id', name='uq_training_frame_run_frame'),
    )

    def __repr__(self):
        return f"<DBTrainingFrame(run_id={self.run_id}, trajectory_frame_id={self.trajectory_frame_id}, traj_id={self.traj_id}, chunk_id={self.chunk_id}, attempt_index={self.attempt_index}, training_round={self.training_round})>"


class TrajectoryDB:
    """Wrapper for the database representations of trajectories and chunks"""
    
    # Track pool sizes across instances for growth detection
    _pool_size_history: dict[int, list[int]] = {}  # engine_id -> list of sizes
    
    def __init__(self, db_url: str, logger: Optional[logging.Logger] = None, use_null_pool: bool = False):
        """Initialize the trajectory database manager
        
        Args:
            db_url: PostgreSQL connection URL (e.g., 'postgresql://user:pass@host:port/dbname')
            logger: Optional logger for tracking engine creation
            use_null_pool: If True, use NullPool (no connection pooling).
                          Use for short-lived instances like worker processes.
        """
        self.db_url = db_url
        self._logger = logger or logging.getLogger(__name__)
        
        if use_null_pool:
            # Use NullPool for short-lived instances - no pooling, connections closed immediately
            from sqlalchemy.pool import NullPool
            poolclass = NullPool
            pool_kwargs = {}
        else:
            # Use very restrictive pool for long-lived agent instances
            from sqlalchemy.pool import QueuePool
            poolclass = QueuePool
            pool_kwargs = {
                'pool_size': 1,  # Only 1 connection in pool
                'max_overflow': 1,  # Allow 1 overflow = max 2 connections total
                'pool_recycle': 300,  # Recycle connections after 5 minutes
                'pool_reset_on_return': 'commit',  # Reset connections on return
            }
        
        # Create engine and session factory
        self.engine = create_engine(
            db_url, 
            echo=False, 
            pool_pre_ping=True,
            poolclass=poolclass,
            **pool_kwargs
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self._logger.info(f"Created TrajectoryDB engine (id={id(self.engine)}) with poolclass={poolclass.__name__}")
        
    def create_tables(self):
        """Create all tables if they don't exist"""
        Base.metadata.create_all(self.engine)
    
    def log_pool_stats(self, logger: logging.Logger) -> None:
        """Log SQLAlchemy connection pool statistics for debugging
        
        Args:
            logger: Logger instance to use for logging
        """
        try:
            pool = self.engine.pool
            engine_id = id(self.engine)
            
            # Get available stats (not all pools have all attributes)
            current_size = pool.size()
            stats = {
                'size': current_size,
                'checked_in': pool.checkedin(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
            }
            # Only add invalid if the attribute exists
            if hasattr(pool, 'invalid'):
                stats['invalid'] = pool.invalid()
            
            # Track pool size history for growth detection
            if engine_id not in TrajectoryDB._pool_size_history:
                TrajectoryDB._pool_size_history[engine_id] = []
            history = TrajectoryDB._pool_size_history[engine_id]
            history.append(current_size)
            # Keep only last 10 measurements
            if len(history) > 10:
                history.pop(0)
            
            # Check for growth trends
            if len(history) >= 3:
                recent_growth = history[-1] - history[-3]
                if recent_growth > 5:
                    logger.warning(
                        f"Pool size growing rapidly: {history[-3]} -> {current_size} "
                        f"(+{recent_growth} in last 3 checks). Possible connection leak!"
                    )
            
            # Alert on high pool size
            if current_size > 20:
                logger.warning(
                    f"Pool size is high ({current_size}). Consider checking for connection leaks."
                )
            
            # Alert on many checked out connections
            checked_out = stats['checked_out']
            if checked_out > 10:
                logger.warning(
                    f"Many connections checked out ({checked_out}). "
                    "Sessions may not be properly closed."
                )
            
            logger.info(f"SQLAlchemy pool stats: {', '.join(f'{k}={v}' for k, v in stats.items())}")
        except Exception as e:
            logger.warning(f"Failed to get pool stats: {e}")
    
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
        atoms_str = write_to_string(canonical_atoms, fmt='json')
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
        return read_from_string(atoms_str, fmt='json')
    
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
    
    def _sync_trajectory_done_flag(self, traj: DBTrajectory) -> None:
        """Keep legacy boolean in sync with enum-based trajectory status."""
        traj.done = traj.status == TrajectoryStatus.COMPLETED

    def _set_trajectory_status(
        self,
        sess,
        traj: DBTrajectory,
        status: TrajectoryStatus,
    ) -> None:
        """Internal helper to update trajectory status and maintain derived fields."""
        previous_status = traj.status
        previous_done = traj.done
        if previous_status != status:
            traj.status = status
        self._sync_trajectory_done_flag(traj)
        if previous_status != traj.status or previous_done != traj.done:
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
                    done=False,
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
    
    def update_chunk_audit_status(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        attempt_index: int,
        audit_status: AuditStatus
    ):
        """Update the audit status of a chunk
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            attempt_index: Attempt index
            audit_status: New audit status
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
    
    def add_training_frame(
        self,
        run_id: str,
        trajectory_frame_id: int,
        model_version_sampled_from: int,
        traj_id: int,
        chunk_id: int,
        attempt_index: int
    ) -> DBTrainingFrame:
        """Add a training frame to the database
        
        Args:
            run_id: Run identifier
            trajectory_frame_id: ID of the frame in the trajectory_frames table
            model_version_sampled_from: Model version that generated this frame
            traj_id: Trajectory identifier (denormalized)
            chunk_id: Chunk identifier (denormalized)
            attempt_index: Attempt index (denormalized)
            
        Returns:
            DBTrainingFrame instance
        """
        with self.session() as sess:
            # Check if training frame already exists
            existing = sess.query(DBTrainingFrame).filter_by(
                run_id=run_id,
                trajectory_frame_id=trajectory_frame_id
            ).first()
            
            if existing:
                return existing
            
            # Create new training frame entry
            db_training_frame = DBTrainingFrame(
                run_id=run_id,
                trajectory_frame_id=trajectory_frame_id,
                model_version_sampled_from=model_version_sampled_from,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index,
                training_round=None  # Will be set when used in training
            )
            sess.add(db_training_frame)
            sess.flush()
            sess.refresh(db_training_frame)
            return db_training_frame
    
    def get_training_frames(
        self,
        run_id: str
    ) -> list[Atoms]:
        """Get all training frames for a run
        
        Args:
            run_id: Run identifier
            
        Returns:
            List of Atoms objects from all training frames
        """
        with self.session() as sess:
            training_frames = sess.query(DBTrainingFrame).filter_by(
                run_id=run_id
            ).filter(
                DBTrainingFrame.training_round.is_(None)
            ).all()
            
            if not training_frames:
                return []
            
            # Get trajectory frame IDs
            frame_ids = [tf.trajectory_frame_id for tf in training_frames]
            
            # Deserialize directly from ORM objects
            atoms_list = []
            for frame_id in frame_ids:
                frame = sess.query(DBTrajectoryFrame).filter_by(id=frame_id).first()
                if frame:
                    atoms_list.append(self._deserialize_atoms(frame.atoms_blob))
        
        # Force garbage collection after deserializing large binary data
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
    
    def get_chunks_from_training_round(
        self,
        run_id: str,
        training_round: int
    ) -> list[dict]:
        """Get unique chunks that generated frames used in a training round
        
        Args:
            run_id: Run identifier
            training_round: Training round number
            
        Returns:
            List of dicts with unique (traj_id, chunk_id, attempt_index) tuples.
            Each dict contains:
                - traj_id: Trajectory identifier
                - chunk_id: Chunk identifier
                - attempt_index: Attempt index
        """
        with self.session() as sess:
            # Get all frames from this training round
            frames = sess.query(DBTrainingFrame).filter_by(
                run_id=run_id,
                training_round=training_round
            ).all()
            
            # Extract unique (traj_id, chunk_id, attempt_index) tuples
            unique_chunks = {}
            for frame in frames:
                key = (frame.traj_id, frame.chunk_id, frame.attempt_index)
                if key not in unique_chunks:
                    unique_chunks[key] = {
                        'traj_id': frame.traj_id,
                        'chunk_id': frame.chunk_id,
                        'attempt_index': frame.attempt_index
                    }
            
            return list(unique_chunks.values())
    
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
    
    def count_active_trajs_with_samples(
        self,
        run_id: str
    ) -> tuple[int, int]:
        """Count active trajectories and those with sampled training frames
        
        Args:
            run_id: Run identifier
            
        Returns:
            Tuple of (total_active_trajectories, active_trajectories_with_samples)
        """
        with self.session() as sess:
            # Get all trajectories for this run
            all_trajectories = sess.query(DBTrajectory).filter_by(
                run_id=run_id
            ).all()
            
            # Get training frames that haven't been consumed by a training round yet
            training_frames = sess.query(DBTrainingFrame).filter_by(
                run_id=run_id
            ).filter(
                DBTrainingFrame.training_round.is_(None)
            ).all()
            
            # Extract trajectory info while session is active
            trajectory_info = [
                {'traj_id': traj.traj_id, 'status': traj.status}
                for traj in all_trajectories
            ]
            
            # Get unique trajectory IDs from sampled frames (denormalized data)
            sampled_traj_ids = {tf.traj_id for tf in training_frames}
        
        # Now check which trajectories are active and have samples
        active_count = 0
        active_with_samples_count = 0
        
        for traj_info in trajectory_info:
            # Check if trajectory is active (not done)
            if traj_info['status'] == TrajectoryStatus.RUNNING:
                active_count += 1
                # Check if this trajectory has been sampled from
                if traj_info['traj_id'] in sampled_traj_ids:
                    active_with_samples_count += 1
        
        return (active_count, active_with_samples_count)
    
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
                func.sum(func.cast(DBTrajectory.done, Integer)).label('n_done_trajectories')
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
                func.sum(func.cast(DBTrajectory.done, Integer)).label('n_done'),
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
                'done': traj.done,
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
                    'done': traj.done,
                    'created_at': traj.created_at,
                    'updated_at': traj.updated_at
                })
            
            return result

