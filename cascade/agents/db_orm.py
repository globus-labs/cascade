"""ORM layer for tracking trajectories and trajectory chunks in PostgreSQL.

This module provides SQLAlchemy models and utilities for persisting trajectory
and chunk metadata alongside the ASE database for individual frames.
"""
from __future__ import annotations

import contextlib
from typing import Optional

import numpy as np
import ase
from ase import Atoms
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Enum as SQLEnum,
    DateTime,
    ForeignKey,
    func,
    Text,
    JSON,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship, sessionmaker, Session, declarative_base
from cascade.model import AuditStatus, TrajectoryChunk, Trajectory

Base = declarative_base()


class DBTrajectory(Base):
    """ORM model for trajectory metadata"""
    __tablename__ = 'trajectories'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    traj_id = Column(Integer, nullable=False)
    target_length = Column(Integer, nullable=False)
    chunks_completed = Column(Integer, default=0, nullable=False)
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


class DBTrainingFrame(Base):
    """ORM model for training frames"""
    __tablename__ = 'training_frames'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    ase_db_id = Column(Integer, nullable=False, index=True)
    model_version_sampled_from = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('run_id', 'ase_db_id', name='uq_training_frame_run_ase'),
    )

    def __repr__(self):
        return f"<DBTrainingFrame(run_id={self.run_id}, ase_db_id={self.ase_db_id}, model_version={self.model_version_sampled_from})>"


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
    ) -> DBTrajectory:
        """Initialize a new trajectory in the database
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            target_length: Target length of the trajectory
            init_atoms: Initial atoms structure
            
        Returns:
            DBTrajectory instance
        """
        with self.session() as sess:
            # Check if trajectory already exists
            existing = sess.query(DBTrajectory).filter_by(
                run_id=run_id,
                traj_id=traj_id
            ).first()
            
            if existing:
                return existing
            
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
            sess.refresh(db_traj)
            return db_traj
    
    def add_chunk_attempt(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        model_version: int,
        n_frames: int,
        audit_status: AuditStatus = AuditStatus.PENDING,
        attempt_index: Optional[int] = None
    ) -> DBTrajectoryChunk:
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
            DBTrajectoryChunk instance
        """
        with self.session() as sess:
            # Get the trajectory
            db_traj = sess.query(DBTrajectory).filter_by(
                run_id=run_id,
                traj_id=traj_id
            ).first()
            
            if not db_traj:
                raise ValueError(f"Trajectory {traj_id} not found for run {run_id}")
            
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
                sess.refresh(existing)
                return existing
            
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
            sess.refresh(db_chunk)
            return db_chunk
    
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
        and returns them as a list of Atoms objects.
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            ase_db: ASE database connection to query frames from
            
        Returns:
            List of Atoms objects from all passed chunks, in order
        """
        chunks = self.get_passed_chunks(run_id, traj_id)
        
        all_atoms = []
        for chunk in chunks:
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
            
            # Get all passed chunks and calculate total frames
            passed_chunks = sess.query(DBTrajectoryChunk).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                audit_status=AuditStatus.PASSED
            ).all()
            
            total_frames = sum(chunk.n_frames for chunk in passed_chunks)
            
            # Trajectory is done if we have at least target_length frames
            # and the latest chunk passed (ensured by update_chunk_audit_status)
            return total_frames >= traj.target_length
    
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
    
    def add_training_frame(
        self,
        run_id: str,
        ase_db_id: int,
        model_version_sampled_from: int
    ) -> DBTrainingFrame:
        """Add a training frame to the database
        
        Args:
            run_id: Run identifier
            ase_db_id: ID of the frame in the ASE database
            model_version_sampled_from: Model version that generated this frame
            
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
                model_version_sampled_from=model_version_sampled_from
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

