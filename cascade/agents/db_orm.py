"""ORM layer for tracking trajectories and trajectory chunks in PostgreSQL.

This module provides SQLAlchemy models and utilities for persisting trajectory
and chunk metadata alongside the ASE database for individual frames.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
import ase
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
    Text,
    JSON,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship, sessionmaker, Session, declarative_base
from cascade.model import AuditStatus

Base = declarative_base()


class DBTrajectory(Base):
    """ORM model for trajectory metadata"""
    __tablename__ = 'trajectories'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    traj_id = Column(Integer, nullable=False)
    target_length = Column(Integer, nullable=False)
    chunks_completed = Column(Integer, default=0, nullable=False)
    done = Column(Boolean, default=False, nullable=False)
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
        return f"<DBTrajectory(run_id={self.run_id}, traj_id={self.traj_id}, chunks_completed={self.chunks_completed})>"


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


class DBModelVersion(Base):
    """ORM model for tracking model versions and their file locations"""
    __tablename__ = 'model_versions'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    version = Column(Integer, nullable=False)
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('run_id', 'version', name='uq_model_version_run_version'),
    )

    def __repr__(self):
        return f"<DBModelVersion(run_id={self.run_id}, version={self.version}, file_path={self.file_path})>"


class DBTrainingFrame(Base):
    """ORM model for training frames"""
    __tablename__ = 'training_frames'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    ase_db_id = Column(Integer, nullable=False, index=True)
    model_version_sampled_from = Column(Integer, nullable=False)
    # Denormalized chunk info for faster queries
    traj_id = Column(Integer, nullable=False, index=True)
    chunk_id = Column(Integer, nullable=False, index=True)
    attempt_index = Column(Integer, nullable=False)
    # Training round tracking
    training_round = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('run_id', 'ase_db_id', name='uq_training_frame_run_ase'),
    )

    def __repr__(self):
        return f"<DBTrainingFrame(run_id={self.run_id}, ase_db_id={self.ase_db_id}, traj_id={self.traj_id}, chunk_id={self.chunk_id}, attempt_index={self.attempt_index}, training_round={self.training_round})>"


class TrajectoryDB:
    """Manager for trajectory and chunk persistence using SQLAlchemy ORM"""
    
    def __init__(self, db_url: str):
        """Initialize the trajectory database manager
        
        Args:
            db_url: PostgreSQL connection URL (e.g., 'postgresql://user:pass@host:port/dbname')
        """
        self.db_url = db_url
        # Create engine and session factory
        self.engine = create_engine(db_url, echo=False, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        
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
                    init_atoms_json=init_atoms_json
                )
                sess.add(db_traj)
                sess.flush()
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
                sess.flush()
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
                        traj.chunks_completed = latest_passed + 1  # +1 because chunk_id is 0-indexed
                        
                        # Check if trajectory is done
                        passed_chunks = sess.query(DBTrajectoryChunk).filter_by(
                            run_id=run_id,
                            traj_id=traj_id,
                            audit_status=AuditStatus.PASSED
                        ).all()
                        total_frames = sum(chunk.n_frames for chunk in passed_chunks)
                        if total_frames >= traj.target_length:
                            traj.done = True
    
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
    
    def get_trajectory_chunks_atoms(
        self,
        run_id: str,
        traj_id: int,
        ase_db: ase.database.connect
    ) -> list[Atoms]:
        """Get all atoms from passed chunks for a trajectory
        
        This queries the ASE database for all frames matching the passed chunks
        and returns them as a list of Atoms objects. When reconstructing the full
        trajectory, the first frame of each chunk (except chunk 0) is skipped to
        avoid duplicates, since the first frame of chunk N is the same as the
        last frame of chunk N-1.
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            ase_db: ASE database connection to query frames from
            
        Returns:
            List of Atoms objects from all passed chunks, in order, with duplicates removed
        """
        chunks = self.get_passed_chunks(run_id, traj_id)
        
        all_atoms = []
        for i, chunk in enumerate(chunks):
            # Query ASE DB for all frames in this chunk and attempt
            # Passed chunks are the specific attempt that passed, so we query by attempt_index
            frames = list(ase_db.select(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk['chunk_id'],
                attempt_index=chunk['attempt_index']
            ))
            # Sort by some order if needed (e.g., id or custom key)
            frames.sort(key=lambda row: row.id)
            
            # Skip first frame of all chunks except the first one
            # (first frame is duplicate of previous chunk's last frame)
            if i > 0:
                frames = frames[1:]  # Skip first frame
            
            all_atoms.extend([row.toatoms() for row in frames])
        
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
            
            # Return the stored done flag
            return traj.done
    
    def get_first_frame_from_chunk(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        attempt_index: int,
        ase_db: ase.database.connect
    ) -> Optional[Atoms]:
        """Get the first frame from a chunk
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            attempt_index: Attempt index for this chunk
            ase_db: ASE database connection to query frames from
            
        Returns:
            Atoms object for the first frame, or None if no frames found
        """
        # Query ASE DB for all frames in this chunk and attempt
        frames = list(ase_db.select(
            run_id=run_id,
            traj_id=traj_id,
            chunk_id=chunk_id,
            attempt_index=attempt_index
        ))
        
        if not frames:
            return None
        
        # Sort by id to ensure proper ordering
        frames.sort(key=lambda row: row.id)
        
        # Return the first frame
        return frames[0].toatoms()
    
    def get_last_frame_from_chunk(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        attempt_index: int,
        ase_db: ase.database.connect
    ) -> Optional[Atoms]:
        """Get the last frame from a chunk
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            attempt_index: Attempt index for this chunk
            ase_db: ASE database connection to query frames from
            
        Returns:
            Atoms object for the last frame, or None if no frames found
        """
        # Query ASE DB for all frames in this chunk and attempt
        frames = list(ase_db.select(
            run_id=run_id,
            traj_id=traj_id,
            chunk_id=chunk_id,
            attempt_index=attempt_index
        ))
        
        if not frames:
            return None
        
        # Sort by id to ensure proper ordering
        frames.sort(key=lambda row: row.id)
        
        # Return the last frame
        return frames[-1].toatoms()
    
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
            
            return atoms
    
    def add_training_frame(
        self,
        run_id: str,
        ase_db_id: int,
        model_version_sampled_from: int,
        traj_id: int,
        chunk_id: int,
        attempt_index: int
    ) -> DBTrainingFrame:
        """Add a training frame to the database
        
        Args:
            run_id: Run identifier
            ase_db_id: ID of the frame in the ASE database
            model_version_sampled_from: Model version that generated this frame
            traj_id: Trajectory identifier (denormalized from ASE DB)
            chunk_id: Chunk identifier (denormalized from ASE DB)
            attempt_index: Attempt index (denormalized from ASE DB)
            
        Returns:
            DBTrainingFrame instance
        """
        with self.session() as sess:
            # Check if training frame already exists
            existing = sess.query(DBTrainingFrame).filter_by(
                run_id=run_id,
                ase_db_id=ase_db_id
            ).first()
            
            if existing:
                return existing
            
            # Create new training frame entry
            db_training_frame = DBTrainingFrame(
                run_id=run_id,
                ase_db_id=ase_db_id,
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
        run_id: str,
        ase_db: ase.database.connect
    ) -> list[Atoms]:
        """Get all training frames for a run
        
        Args:
            run_id: Run identifier
            ase_db: ASE database connection to query frames from
            
        Returns:
            List of Atoms objects from all training frames
        """
        with self.session() as sess:
            training_frames = sess.query(DBTrainingFrame).filter_by(
                run_id=run_id
            ).all()
            
            if not training_frames:
                return []
            
            # Get actual Atoms objects from ASE DB
            atoms_list = []
            for tf in training_frames:
                try:
                    row = ase_db.get(tf.ase_db_id)
                    atoms_list.append(row.toatoms())
                except KeyError:
                    # Frame might have been deleted from ASE DB
                    continue
            
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
        run_id: str,
        ase_db
    ) -> set[int]:
        """Get unique trajectory IDs that have training frames sampled from them
        
        Args:
            run_id: Run identifier
            ase_db: ASE database connection to query frames from
            
        Returns:
            Set of unique trajectory IDs from sampled frames
        """
        with self.session() as sess:
            training_frames = sess.query(DBTrainingFrame).filter_by(
                run_id=run_id
            ).all()
            
            if not training_frames:
                return set()
            
            # Extract ASE DB IDs while session is still active
            ase_db_ids = [tf.ase_db_id for tf in training_frames]
            
            # Get unique trajectory IDs from ASE DB
            unique_traj_ids = set()
            for ase_db_id in ase_db_ids:
                try:
                    row = ase_db.get(ase_db_id)
                    traj_id = row.get('traj_id')
                    if traj_id is not None:
                        unique_traj_ids.add(traj_id)
                except KeyError:
                    # Frame might have been deleted from ASE DB
                    continue
            
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
        
        return atoms
    
    def count_active_trajs_with_samples(
        self,
        run_id: str,
        ase_db: ase.database.connect
    ) -> tuple[int, int]:
        """Count active trajectories and those with sampled training frames
        
        Args:
            run_id: Run identifier
            ase_db: ASE database connection to query frames from
            
        Returns:
            Tuple of (total_active_trajectories, active_trajectories_with_samples)
        """
        # Get training frames and trajectories outside the session to avoid deadlocks
        with self.session() as sess:
            # Get all trajectories for this run
            all_trajectories = sess.query(DBTrajectory).filter_by(
                run_id=run_id
            ).all()
            
            # Get all training frames for this run
            training_frames = sess.query(DBTrainingFrame).filter_by(
                run_id=run_id
            ).all()
            
            # Extract ASE DB IDs while session is active
            ase_db_ids = [tf.ase_db_id for tf in training_frames]
            
            # Extract trajectory info while session is active to avoid DetachedInstanceError
            trajectory_info = [
                {'traj_id': traj.traj_id, 'target_length': traj.target_length, 'done': traj.done}
                for traj in all_trajectories
            ]
        
        # Get unique trajectory IDs from sampled frames (outside session to avoid deadlock)
        sampled_traj_ids = set()
        for ase_db_id in ase_db_ids:
            try:
                row = ase_db.get(ase_db_id)
                traj_id = row.get('traj_id')
                if traj_id is not None:
                    sampled_traj_ids.add(traj_id)
            except KeyError:
                # Frame might have been deleted from ASE DB
                continue
        
        # Now check which trajectories are active and have samples
        active_count = 0
        active_with_samples_count = 0
        
        for traj_info in trajectory_info:
            # Check if trajectory is active (not done)
            if not traj_info['done']:
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
    
    def save_model_weights(
        self,
        run_id: str,
        weights: bytes,
        weights_dir: str
    ) -> int:
        """Save model weights to disk and record in database
        
        This method uses an atomic write pattern:
        1. Write to temporary file
        2. Verify file was written correctly
        3. Atomically rename to final location
        4. Write to database (transaction ensures atomicity)
        
        Args:
            run_id: Run identifier
            weights: Model weights as bytes
            weights_dir: Base directory for storing weights
            
        Returns:
            Version number of the saved model
            
        Raises:
            IOError: If file write/verification fails
            Exception: If database write fails
        """
        import tempfile
        import os
        import hashlib
        
        # Get next version number
        with self.session() as sess:
            max_version = sess.query(func.max(DBModelVersion.version)).filter_by(
                run_id=run_id
            ).scalar()
            next_version = (max_version or 0) + 1
        
        # Prepare final path (weights_dir already points to the run's weights directory)
        os.makedirs(weights_dir, exist_ok=True)
        final_path = os.path.join(weights_dir, f"model_v{next_version}.pth")
        
        # Write to temporary file
        temp_fd, temp_path = tempfile.mkstemp(dir=weights_dir, suffix='.tmp')
        try:
            with os.fdopen(temp_fd, 'wb') as f:
                f.write(weights)
                f.flush()
                os.fsync(f.fileno())  # Ensure written to disk
            
            # Verify file was written correctly
            if not os.path.exists(temp_path):
                raise IOError(f"Temporary file {temp_path} does not exist after write")
            
            file_size = os.path.getsize(temp_path)
            if file_size != len(weights):
                raise IOError(f"File size mismatch: expected {len(weights)}, got {file_size}")
            
            # Verify checksum
            with open(temp_path, 'rb') as f:
                checksum = hashlib.md5(f.read()).hexdigest()
            expected_checksum = hashlib.md5(weights).hexdigest()
            if checksum != expected_checksum:
                raise IOError(f"Checksum mismatch for {temp_path}")
            
            # Atomic rename (atomic on Unix, near-atomic on Windows)
            os.rename(temp_path, final_path)
            
            # Verify final file exists
            if not os.path.exists(final_path):
                raise IOError(f"Final file {final_path} does not exist after rename")
            
            # Write to database (transaction ensures atomicity)
            with self.session() as sess:
                db_version = DBModelVersion(
                    run_id=run_id,
                    version=next_version,
                    file_path=final_path
                )
                sess.add(db_version)
                sess.flush()  # Ensure it's committed
            
            logger.info(f"Saved model weights version {next_version} to {final_path}")
            return next_version
            
        except Exception as e:
            # Cleanup on failure
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            if os.path.exists(final_path):
                try:
                    os.remove(final_path)
                except:
                    pass
            logger.error(f"Failed to save model weights: {e}")
            raise
    
    def get_latest_model_version(
        self,
        run_id: str
    ) -> Optional[dict]:
        """Get the latest model version for a run
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dict with 'version' and 'file_path', or None if no models exist
        """
        with self.session() as sess:
            version = sess.query(DBModelVersion).filter_by(
                run_id=run_id
            ).order_by(DBModelVersion.version.desc()).first()
            
            if not version:
                return None
            
            return {
                'version': version.version,
                'file_path': version.file_path
            }
    
    def list_trajectories_in_run(self, run_id: str) -> list[dict]:
        """List all trajectories in a run with basic info
        
        Args:
            run_id: Run identifier
            
        Returns:
            List of dicts with trajectory metadata, sorted by traj_id.
            Each dict contains:
                - traj_id, target_length, chunks_completed, done
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
                    'done': traj.done,
                    'created_at': traj.created_at,
                    'updated_at': traj.updated_at
                })
            
            return result

