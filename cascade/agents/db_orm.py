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
from ase.calculators.singlepoint import SinglePointCalculator

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
    inspect,
    Index,
)
from sqlalchemy.orm import relationship, sessionmaker, Session, declarative_base
from cascade.model import AuditStatus, TrajectoryChunk, Trajectory

Base = declarative_base()

# Standard ASE calculator properties supported by SinglePointCalculator
ASE_STANDARD_PROPERTIES = {'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom', 'magmoms', 'free_energy'}


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


class DBFrame(Base):
    """ORM model for individual atomic frames"""
    __tablename__ = 'frames'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    traj_id = Column(Integer, nullable=False, index=True)
    chunk_id = Column(Integer, nullable=False, index=True)
    attempt_index = Column(Integer, nullable=False)
    frame_index = Column(Integer, nullable=False)
    
    # Atomic structure (positions, numbers, cell, pbc)
    atoms_data = Column(JSON, nullable=False)
    
    # Calculator results (forces, energy, stress, etc.)
    calc_results = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_frame_lookup', 'run_id', 'traj_id', 'chunk_id', 'attempt_index', 'frame_index'),
    )
    
    def __repr__(self):
        return f"<DBFrame(run_id={self.run_id}, traj_id={self.traj_id}, chunk_id={self.chunk_id}, attempt={self.attempt_index}, frame={self.frame_index})>"


class DBTrainingFrame(Base):
    """ORM model for training frames"""
    __tablename__ = 'training_frames'

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    frame_id = Column(Integer, ForeignKey('frames.id'), nullable=False)
    model_version_sampled_from = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('run_id', 'frame_id', name='uq_training_frame_run_frame'),
    )

    def __repr__(self):
        return f"<DBTrainingFrame(run_id={self.run_id}, frame_id={self.frame_id}, model_version={self.model_version_sampled_from})>"


# Serialization helpers
def atoms_to_dict(atoms: Atoms) -> dict:
    """Serialize Atoms to dictionary"""
    return {
        'positions': atoms.get_positions().tolist(),
        'numbers': atoms.get_atomic_numbers().tolist(),
        'cell': atoms.get_cell().tolist() if atoms.cell is not None else None,
        'pbc': atoms.get_pbc().tolist() if atoms.pbc is not None else None,
    }

def dict_to_atoms(data: dict) -> Atoms:
    """Deserialize Atoms from dictionary"""
    atoms = Atoms(
        positions=np.array(data['positions']),
        numbers=data['numbers'],
        cell=np.array(data['cell']) if data['cell'] is not None else None,
        pbc=np.array(data['pbc']) if data['pbc'] is not None else None,
    )
    return atoms

def calc_results_to_dict(calc_results: dict) -> dict:
    """Serialize calculator results - only standard ASE properties
    
    Filters out non-standard properties (like MACE's 'node_energy') that
    SinglePointCalculator doesn't support.
    """
    results = {}
    for key, value in calc_results.items():
        # Only store standard ASE properties
        if key not in ASE_STANDARD_PROPERTIES:
            continue
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()
        elif isinstance(value, (int, float)):
            results[key] = float(value)
    return results

def dict_to_calc_results(data: dict) -> dict:
    """Deserialize calculator results"""
    results = {}
    for key, value in data.items():
        if isinstance(value, list):
            results[key] = np.array(value)
        else:
            results[key] = value
    return results


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
        self._add_missing_columns()
    
    def _add_missing_columns(self):
        """Add any missing columns to existing tables (for schema migrations)"""
        from sqlalchemy import text
        
        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()
        
        # Check if 'done' column exists in trajectories table
        if 'trajectories' in table_names:
            columns = [col['name'] for col in inspector.get_columns('trajectories')]
            if 'done' not in columns:
                with self.engine.connect() as conn:
                    # Add the 'done' column with default False
                    conn.execute(text("ALTER TABLE trajectories ADD COLUMN done BOOLEAN NOT NULL DEFAULT FALSE"))
                    conn.commit()
                    logger.info("Added 'done' column to trajectories table")
        
        # Migrate training_frames table from ase_db_id to frame_id
        if 'training_frames' in table_names:
            columns = [col['name'] for col in inspector.get_columns('training_frames')]
            if 'ase_db_id' in columns and 'frame_id' not in columns:
                with self.engine.connect() as conn:
                    # For prototyping: clear existing training_frames data since we can't map ase_db_id to frame_id
                    result = conn.execute(text("SELECT COUNT(*) FROM training_frames"))
                    count = result.scalar()
                    if count > 0:
                        logger.warning(f"Clearing {count} existing training_frames records during schema migration")
                        conn.execute(text("DELETE FROM training_frames"))
                    
                    # Drop old constraint if it exists
                    try:
                        conn.execute(text("ALTER TABLE training_frames DROP CONSTRAINT IF EXISTS uq_training_frame_run_ase"))
                        conn.commit()
                    except Exception:
                        pass  # Constraint might not exist or have different name
                    
                    # Drop old column
                    conn.execute(text("ALTER TABLE training_frames DROP COLUMN ase_db_id"))
                    
                    # Add new column with NOT NULL
                    conn.execute(text("ALTER TABLE training_frames ADD COLUMN frame_id INTEGER NOT NULL"))
                    
                    # Add foreign key constraint
                    conn.execute(text("ALTER TABLE training_frames ADD CONSTRAINT fk_training_frames_frame_id FOREIGN KEY (frame_id) REFERENCES frames(id)"))
                    
                    # Add new unique constraint
                    conn.execute(text("ALTER TABLE training_frames ADD CONSTRAINT uq_training_frame_run_frame UNIQUE (run_id, frame_id)"))
                    
                    conn.commit()
                    logger.info("Migrated training_frames table from ase_db_id to frame_id")
    
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
        traj_id: int
    ) -> list[Atoms]:
        """Get all atoms from passed chunks for a trajectory
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            
        Returns:
            List of Atoms objects from all passed chunks, in order
        """
        chunks = self.get_passed_chunks(run_id, traj_id)
        
        all_atoms = []
        for chunk in chunks:
            frames = self.get_frames(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk['chunk_id'],
                attempt_index=chunk['attempt_index']
            )
            
            for frame_data in frames:
                atoms = dict_to_atoms(frame_data['atoms_data'])
                if frame_data['calc_results']:
                    results = dict_to_calc_results(frame_data['calc_results'])
                    atoms.calc = SinglePointCalculator(atoms, **results)
                all_atoms.append(atoms)
        
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
            frame = sess.query(DBFrame).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index
            ).order_by(DBFrame.frame_index.desc()).first()
            
            if not frame:
                return None
            
            atoms = dict_to_atoms(frame.atoms_data)
            if frame.calc_results:
                results = dict_to_calc_results(frame.calc_results)
                atoms.calc = SinglePointCalculator(atoms, **results)
            return atoms
    
    def write_frame(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        attempt_index: int,
        frame_index: int,
        atoms: Atoms
    ) -> bool:
        """Write a single frame to the database
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            attempt_index: Attempt index
            frame_index: Frame index within the chunk
            atoms: Atoms object with calculator attached
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session() as sess:
                frame = DBFrame(
                    run_id=run_id,
                    traj_id=traj_id,
                    chunk_id=chunk_id,
                    attempt_index=attempt_index,
                    frame_index=frame_index,
                    atoms_data=atoms_to_dict(atoms),
                    calc_results=calc_results_to_dict(atoms.calc.results) if (atoms.calc and hasattr(atoms.calc, 'results') and atoms.calc.results) else None
                )
                sess.add(frame)
                sess.flush()
                return True
        except Exception as e:
            logger.error(f"Failed to write frame: {e}")
            return False

    def get_frames(
        self,
        run_id: str,
        traj_id: int,
        chunk_id: int,
        attempt_index: int
    ) -> list[dict]:
        """Get all frames for a chunk attempt
        
        Args:
            run_id: Run identifier
            traj_id: Trajectory identifier
            chunk_id: Chunk identifier
            attempt_index: Attempt index
            
        Returns:
            List of dicts with frame data (id, frame_index, atoms_data, calc_results)
        """
        with self.session() as sess:
            frames = sess.query(DBFrame).filter_by(
                run_id=run_id,
                traj_id=traj_id,
                chunk_id=chunk_id,
                attempt_index=attempt_index
            ).order_by(DBFrame.frame_index).all()
            
            return [
                {
                    'id': frame.id,
                    'frame_index': frame.frame_index,
                    'atoms_data': frame.atoms_data,
                    'calc_results': frame.calc_results
                }
                for frame in frames
            ]

    def get_frame_by_id(self, frame_id: int) -> Optional[Atoms]:
        """Get a single frame by ID and return as Atoms object
        
        Args:
            frame_id: Frame ID
            
        Returns:
            Atoms object with calculator attached, or None if not found
        """
        with self.session() as sess:
            frame = sess.query(DBFrame).filter_by(id=frame_id).first()
            if not frame:
                return None
            
            atoms = dict_to_atoms(frame.atoms_data)
            if frame.calc_results:
                results = dict_to_calc_results(frame.calc_results)
                atoms.calc = SinglePointCalculator(atoms, **results)
            return atoms
    
    def add_training_frame(
        self,
        run_id: str,
        frame_id: int,
        model_version_sampled_from: int
    ) -> bool:
        """Add a training frame to the database
        
        Args:
            run_id: Run identifier
            frame_id: ID of the frame in the frames table
            model_version_sampled_from: Model version that generated this frame
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session() as sess:
                # Check if training frame already exists
                existing = sess.query(DBTrainingFrame).filter_by(
                    run_id=run_id,
                    frame_id=frame_id
                ).first()
                
                if existing:
                    return True
                
                # Create new training frame entry
                db_training_frame = DBTrainingFrame(
                    run_id=run_id,
                    frame_id=frame_id,
                    model_version_sampled_from=model_version_sampled_from
                )
                sess.add(db_training_frame)
                sess.flush()
                return True
        except Exception as e:
            logger.error(f"Failed to add training frame: {e}")
            return False
    
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
            ).all()
            
            if not training_frames:
                return []
            
            # Get actual Atoms objects from frames table
            atoms_list = []
            for tf in training_frames:
                frame = sess.query(DBFrame).filter_by(id=tf.frame_id).first()
                if frame:
                    atoms = dict_to_atoms(frame.atoms_data)
                    if frame.calc_results:
                        results = dict_to_calc_results(frame.calc_results)
                        atoms.calc = SinglePointCalculator(atoms, **results)
                    atoms_list.append(atoms)
            
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
            
            # Get unique trajectory IDs from frames table via join
            frames = sess.query(DBFrame.traj_id).join(
                DBTrainingFrame,
                DBFrame.id == DBTrainingFrame.frame_id
            ).filter(
                DBTrainingFrame.run_id == run_id
            ).distinct().all()
            
            return {f[0] for f in frames}
    
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
        run_id: str
    ) -> tuple[int, int]:
        """Count active trajectories and those with sampled training frames
        
        Args:
            run_id: Run identifier
            
        Returns:
            Tuple of (total_active_trajectories, active_trajectories_with_samples)
        """
        with self.session() as sess:
            # Get all trajectories for this run that are not done
            active_trajectories = sess.query(DBTrajectory).filter_by(
                run_id=run_id,
                done=False
            ).all()
            
            total_active = len(active_trajectories)
            
            if total_active == 0:
                return (0, 0)
            
            # Get unique trajectory IDs that have training frames
            sampled_traj_ids = sess.query(DBFrame.traj_id).join(
                DBTrainingFrame,
                DBFrame.id == DBTrainingFrame.frame_id
            ).filter(
                DBTrainingFrame.run_id == run_id
            ).distinct().all()
            
            sampled_traj_id_set = {t[0] for t in sampled_traj_ids}
            
            # Count how many active trajectories have been sampled from
            active_with_samples = sum(
                1 for traj in active_trajectories
                if traj.traj_id in sampled_traj_id_set
            )
            
            return (total_active, active_with_samples)
    
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

