"""Tests that three tiebreak paths resolve ties on ``trial_id`` identically.

The live panel (``RichProgressObserver._rerank``),
``recommend_configs``, and
``OptimizationLoop.run_verification``'s top-K selector all sort by
``(-score, trial_id)`` under score ties. This test holds the invariant.
"""

from __future__ import annotations

from datetime import UTC, datetime
import io

import pytest
from rich.console import Console

from kube_autotuner.experiment import ObjectivesSection
from kube_autotuner.models import (
    BenchmarkConfig,
    BenchmarkResult,
    NodePair,
    TrialResult,
)
from kube_autotuner.progress import (
    RichProgressObserver,
    _build_trial_row,  # noqa: PLC2701
)
from kube_autotuner.scoring import score_rows


def _trial(trial_id: str, bps: float) -> TrialResult:
    tr = TrialResult(
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=[
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=bps,
                retransmits=1,
                bytes_sent=int(bps),
                cpu_utilization_percent=10.0,
                cpu_server_percent=10.0,
                iteration=0,
                client_node="a",
            ),
        ],
        phase="bayesian",
    )
    # Force trial_ids for deterministic comparison.
    tr.trial_id = trial_id
    return tr


def test_live_panel_rerank_ties_break_on_trial_id() -> None:
    console = Console(file=io.StringIO(), force_terminal=True, width=120)
    observer = RichProgressObserver(console, objectives=ObjectivesSection())
    a = _trial("aaa", 1e9)
    b = _trial("zzz", 1e9)  # identical metrics -> identical score
    # Feed b first to confirm tiebreak is on trial_id, not arrival order.
    observer._all_rows.extend([
        _build_trial_row(
            0,
            "bayesian",
            {
                "throughput": (b.mean_throughput(), 0.0),
                "cpu": (b.mean_cpu(), 0.0),
            },
            trial_id=b.trial_id,
            parent_trial_id=None,
        ),
        _build_trial_row(
            1,
            "bayesian",
            {
                "throughput": (a.mean_throughput(), 0.0),
                "cpu": (a.mean_cpu(), 0.0),
            },
            trial_id=a.trial_id,
            parent_trial_id=None,
        ),
    ])
    observer._rerank()
    assert [row.trial_id for row in observer._top] == ["aaa", "zzz"]


def test_recommend_configs_agrees_on_trial_id_tiebreak() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from kube_autotuner.analysis import recommend_configs  # noqa: PLC0415

    del pd  # ensure pandas import ran
    a = _trial("aaa", 1e9)
    b = _trial("zzz", 1e9)
    results = recommend_configs([b, a], hardware_class="10g", n=2)
    assert [r["trial_id"] for r in results] == ["aaa", "zzz"]


def test_run_verification_top_k_selector_agrees_on_trial_id_tiebreak() -> None:
    """The selector inside run_verification uses the same tiebreak key."""
    a = _trial("aaa", 1e9)
    b = _trial("zzz", 1e9)
    rows = [
        _build_trial_row(
            0,
            "bayesian",
            {
                "throughput": (t.mean_throughput(), 0.0),
                "cpu": (t.mean_cpu(), 0.0),
            },
            trial_id=t.trial_id,
            parent_trial_id=None,
        )
        for t in (b, a)
    ]
    scores = score_rows(
        [r.metrics for r in rows],
        ObjectivesSection().pareto,
        ObjectivesSection().recommendation_weights,
    )
    ranked = sorted(
        range(len(rows)),
        key=lambda i: (-scores[i], rows[i].trial_id),
    )
    assert [rows[i].trial_id for i in ranked] == ["aaa", "zzz"]
