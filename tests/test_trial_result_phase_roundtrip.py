"""Tests for ``TrialResult.phase`` / ``parent_trial_id`` serialization."""

from __future__ import annotations

from kube_autotuner.models import BenchmarkConfig, NodePair, TrialResult


def _make(
    *,
    phase: str | None = None,
    parent_trial_id: str | None = None,
) -> TrialResult:
    """Build a minimal :class:`TrialResult` for round-trip tests.

    Returns:
        A fresh record with ``phase`` / ``parent_trial_id`` set from
        the keyword arguments (both default ``None``).
    """
    return TrialResult(
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values={"net.core.rmem_max": 1024},
        config=BenchmarkConfig(),
        phase=phase,  # ty: ignore[invalid-argument-type]
        parent_trial_id=parent_trial_id,
    )


def test_phase_and_parent_trial_id_roundtrip_explicit() -> None:
    primary = _make(phase="bayesian")
    revived = TrialResult.model_validate_json(primary.model_dump_json())
    assert revived.phase == "bayesian"
    assert revived.parent_trial_id is None

    verification = _make(phase="verification", parent_trial_id=primary.trial_id)
    revived_v = TrialResult.model_validate_json(verification.model_dump_json())
    assert revived_v.phase == "verification"
    assert revived_v.parent_trial_id == primary.trial_id


def test_legacy_jsonl_without_fields_defaults_to_none() -> None:
    """Pre-feature JSONL rows have no phase/parent_trial_id keys."""
    # Construct a payload, strip the new fields, and re-validate.
    primary = _make(phase="sobol")
    payload = primary.model_dump()
    payload.pop("phase", None)
    payload.pop("parent_trial_id", None)
    revived = TrialResult.model_validate(payload)
    assert revived.phase is None
    assert revived.parent_trial_id is None


def test_verification_row_carries_parent_id() -> None:
    parent = _make(phase="bayesian")
    child = _make(phase="verification", parent_trial_id=parent.trial_id)
    assert child.parent_trial_id == parent.trial_id
    assert child.phase == "verification"
