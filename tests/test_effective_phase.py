"""Tests for the ``is_primary`` / ``effective_phase`` helpers."""

from __future__ import annotations

from kube_autotuner.models import (
    BenchmarkConfig,
    NodePair,
    TrialResult,
    effective_phase,
    is_primary,
)


def _make(
    *,
    phase: str | None = None,
    parent_trial_id: str | None = None,
) -> TrialResult:
    return TrialResult(
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
        phase=phase,  # ty: ignore[invalid-argument-type]
        parent_trial_id=parent_trial_id,
    )


def test_is_primary_treats_legacy_rows_as_primary() -> None:
    assert is_primary(_make()) is True


def test_is_primary_rejects_verification_rows() -> None:
    assert is_primary(_make(phase="verification")) is False


def test_is_primary_accepts_sobol_and_bayesian() -> None:
    assert is_primary(_make(phase="sobol")) is True
    assert is_primary(_make(phase="bayesian")) is True


def test_effective_phase_uses_stored_phase_when_present() -> None:
    # index and n_sobol are ignored when the field is set.
    assert effective_phase(_make(phase="bayesian"), index=0, n_sobol=50) == "bayesian"
    assert effective_phase(_make(phase="sobol"), index=99, n_sobol=5) == "sobol"


def test_effective_phase_infers_from_index_for_legacy_rows() -> None:
    legacy = _make()  # phase is None
    assert effective_phase(legacy, index=0, n_sobol=5) == "sobol"
    assert effective_phase(legacy, index=4, n_sobol=5) == "sobol"
    assert effective_phase(legacy, index=5, n_sobol=5) == "bayesian"
    assert effective_phase(legacy, index=42, n_sobol=5) == "bayesian"
