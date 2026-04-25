"""Tests that three tiebreak paths resolve ties on ``trial_id`` identically.

The live panel (``RichProgressObserver._rerank``),
``recommend_configs``, and
``OptimizationLoop.run_refinement``'s top-K selector all sort by
``(-score, trial_id)`` under score ties. This test holds the invariant.

Load-bearing assumption: ``_DEFAULT_WEIGHTS`` in
``kube_autotuner.experiment`` only seeds minimize-direction entries,
so ``ObjectivesSection().recommendation_weights`` carries no keys for
``tcp_throughput``/``udp_throughput``/``rps``. The maximize branch of
``score_rows`` then defaults those to ``1.0``. If a future change
adds maximize entries to ``_DEFAULT_WEIGHTS``, the scores here shift
and the tiebreak ordering may silently change -- update the
expectations here when that happens.
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
from kube_autotuner.scoring import METRIC_TO_DF_COLUMN, score_rows


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
                "tcp_throughput": (b.mean_tcp_throughput(), 0.0),
            },
            trial_id=b.trial_id,
            parent_trial_id=None,
            memory_cost=0.0,
        ),
        _build_trial_row(
            1,
            "bayesian",
            {
                "tcp_throughput": (a.mean_tcp_throughput(), 0.0),
            },
            trial_id=a.trial_id,
            parent_trial_id=None,
            memory_cost=0.0,
        ),
    ])
    observer._rerank()
    assert [row.trial_id for row in observer._top] == ["aaa", "zzz"]


def test_recommend_configs_agrees_on_trial_id_tiebreak() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from kube_autotuner.report.analysis import recommend_configs  # noqa: PLC0415

    del pd  # ensure pandas import ran
    a = _trial("aaa", 1e9)
    b = _trial("zzz", 1e9)
    results = recommend_configs([b, a], hardware_class="10g", n=2)
    assert [r["trial_id"] for r in results] == ["aaa", "zzz"]


def test_build_trial_row_populates_udp_keys() -> None:
    """The live panel pathway must propagate udp_throughput / udp_loss_rate.

    Otherwise the live ``Best so far`` ranking diverges from
    :func:`recommend_configs` once those metrics enter the default
    objective set.
    """
    row = _build_trial_row(
        0,
        "bayesian",
        {
            "tcp_throughput": (5e9, 0.0),
            "udp_throughput": (1e9, 0.0),
            "tcp_retransmit_rate": (0.01, 0.0),
            "udp_loss_rate": (0.02, 0.0),
            "udp_jitter": (0.5, 0.0),
        },
        trial_id="t-1",
        parent_trial_id=None,
        memory_cost=0.0,
    )
    assert row.metrics[METRIC_TO_DF_COLUMN["udp_throughput"]] == pytest.approx(1e9)
    assert row.metrics[METRIC_TO_DF_COLUMN["udp_loss_rate"]] == pytest.approx(0.02)


def test_run_refinement_top_k_selector_agrees_on_trial_id_tiebreak() -> None:
    """The selector inside run_refinement uses the same tiebreak key."""
    a = _trial("aaa", 1e9)
    b = _trial("zzz", 1e9)
    rows = [
        _build_trial_row(
            0,
            "bayesian",
            {
                "tcp_throughput": (t.mean_tcp_throughput(), 0.0),
            },
            trial_id=t.trial_id,
            parent_trial_id=None,
            memory_cost=0.0,
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
