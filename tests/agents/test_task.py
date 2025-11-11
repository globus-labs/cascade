from __future__ import annotations

from dataclasses import dataclass

from cascade.agents.task import random_audit
from cascade.model import AuditResult, ChunkSpec


@dataclass
class _DummyRNG:
    """Deterministic RNG stub returning a queued sequence of values."""

    values: tuple[float, ...]
    index: int = 0

    def random(self) -> float:
        value = self.values[self.index]
        self.index += 1
        return value


def test_random_audit_passed() -> None:
    chunk_spec = ChunkSpec(traj_id=7, chunk_id=3)
    rng = _DummyRNG((0.2, 0.8))

    result = random_audit(
        chunk_atoms=[],
        chunk_spec=chunk_spec,
        attempt_index=4,
        rng=rng,
        accept_prob=0.5,
    )

    assert isinstance(result, AuditResult)
    assert result.passed is True
    assert result.score == 0.8
    assert result.traj_id == chunk_spec.traj_id
    assert result.chunk_id == chunk_spec.chunk_id
    assert result.attempt_index == 4


def test_random_audit_failed() -> None:
    chunk_spec = ChunkSpec(traj_id=2, chunk_id=9)
    rng = _DummyRNG((0.9,))

    result = random_audit(
        chunk_atoms=[],
        chunk_spec=chunk_spec,
        attempt_index=5,
        rng=rng,
        accept_prob=0.5,
    )

    assert isinstance(result, AuditResult)
    assert result.passed is False
    assert result.score == 0.0
    assert result.traj_id == chunk_spec.traj_id
    assert result.chunk_id == chunk_spec.chunk_id
    assert result.attempt_index == 5

