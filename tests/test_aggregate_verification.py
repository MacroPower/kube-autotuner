"""Tests for :func:`kube_autotuner.scoring.aggregate_verification`."""

from __future__ import annotations

from datetime import UTC, datetime
import math

import pytest

from kube_autotuner.experiment import ObjectivesSection
from kube_autotuner.models import (
    BenchmarkConfig,
    BenchmarkResult,
    NodePair,
    TrialResult,
)
from kube_autotuner.scoring import (
    METRIC_TO_DF_COLUMN,
    aggregate_verification,
    score_rows,
)


def _trial(
    bps: float,
    *,
    phase: str = "bayesian",
    parent_trial_id: str | None = None,
) -> TrialResult:
    """Build a one-iteration primary or verification :class:`TrialResult`.

    Returns:
        A trial record carrying a single TCP iperf3 record with the
        requested throughput, byte total, and retransmit count.
    """
    return TrialResult(
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values={"net.core.rmem_max": int(bps)},
        config=BenchmarkConfig(),
        results=[
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=bps,
                bytes_sent=int(bps),
                cpu_utilization_percent=10.0,
                cpu_server_percent=10.0,
                retransmits=1,
                iteration=0,
                client_node="a",
            ),
        ],
        phase=phase,  # ty: ignore[invalid-argument-type]
        parent_trial_id=parent_trial_id,
    )


def test_primary_only_input_keeps_scoring_identical() -> None:
    a = _trial(1e9)
    b = _trial(2e9)
    agg = aggregate_verification([a, b])
    assert {str(row["trial_id"]) for row in agg} == {a.trial_id, b.trial_id}
    scores = score_rows(
        agg,
        ObjectivesSection().pareto,
        ObjectivesSection().recommendation_weights,
    )
    # Higher throughput wins under default objectives.
    score_by_id = {str(r["trial_id"]): s for r, s in zip(agg, scores, strict=True)}
    assert score_by_id[b.trial_id] > score_by_id[a.trial_id]


def test_adding_verification_samples_changes_means_and_rank() -> None:
    # Primary A reports 1.5e9; primary B reports 1.0e9. With A alone, A wins.
    primary_a = _trial(1.5e9)
    primary_b = _trial(1.0e9)
    # Two verification samples on A both land at 0.9e9. New mean for A is
    # (1.5 + 0.9 + 0.9)/3 = 1.1e9 vs B's 1.0e9 -- A still leads, but by
    # a narrower margin than before.
    v1 = _trial(0.9e9, phase="verification", parent_trial_id=primary_a.trial_id)
    v2 = _trial(0.9e9, phase="verification", parent_trial_id=primary_a.trial_id)

    agg_primary = aggregate_verification([primary_a, primary_b])
    agg_combined = aggregate_verification([primary_a, primary_b, v1, v2])
    col = METRIC_TO_DF_COLUMN["throughput"]

    def _by_id(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
        return {str(r["trial_id"]): float(r[col]) for r in rows}

    primary_means = _by_id(agg_primary)
    combined_means = _by_id(agg_combined)
    assert primary_means[primary_a.trial_id] == pytest.approx(1.5e9)
    assert combined_means[primary_a.trial_id] == pytest.approx(
        (1.5e9 + 0.9e9 + 0.9e9) / 3,
    )
    # B has no verification samples, so its combined mean is unchanged.
    assert combined_means[primary_b.trial_id] == pytest.approx(
        primary_means[primary_b.trial_id],
    )


def test_sem_zero_when_n_equals_one() -> None:
    agg = aggregate_verification([_trial(1e9)])
    row = agg[0]
    assert row["sample_count"] == 1
    sem_col = f"{METRIC_TO_DF_COLUMN['throughput']}_sem"
    assert float(row[sem_col]) == pytest.approx(0.0)  # type: ignore[arg-type]


def test_sem_positive_with_multiple_samples() -> None:
    parent = _trial(1.0e9)
    child = _trial(0.8e9, phase="verification", parent_trial_id=parent.trial_id)
    agg = aggregate_verification([parent, child])
    row = next(r for r in agg if r["trial_id"] == parent.trial_id)
    assert row["sample_count"] == 2
    sem_col = f"{METRIC_TO_DF_COLUMN['throughput']}_sem"
    sem = float(row[sem_col])  # type: ignore[arg-type]
    assert sem > 0
    assert not math.isnan(sem)


def test_orphaned_verification_row_forms_its_own_group() -> None:
    orphan = _trial(1e9, phase="verification", parent_trial_id="ghost")
    agg = aggregate_verification([orphan])
    assert len(agg) == 1
    assert str(agg[0]["trial_id"]) == "ghost"
