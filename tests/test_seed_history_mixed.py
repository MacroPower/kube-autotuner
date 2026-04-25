"""Tests for :meth:`RichProgressObserver.seed_history` across mixed priors.

Primary rows interleaved with refinement rows must preserve file
order in ``_all_rows``, keep each primary's stored phase label, and
have ``_top`` match the aggregated ``score_rows`` ordering.
"""

from __future__ import annotations

from datetime import UTC, datetime
import io

import pytest
from rich.console import Console

pytest.importorskip("ax")

from kube_autotuner.experiment import ObjectivesSection
from kube_autotuner.models import (
    BenchmarkConfig,
    BenchmarkResult,
    NodePair,
    TrialResult,
)
from kube_autotuner.progress import RichProgressObserver
from kube_autotuner.scoring import aggregate_by_parent, score_rows


def _trial(
    trial_id: str,
    bps: float,
    *,
    phase: str | None = None,
    parent_trial_id: str | None = None,
) -> TrialResult:
    tr = TrialResult(
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values={"net.core.rmem_max": int(bps)},
        config=BenchmarkConfig(),
        results=[
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=bps,
                retransmits=5,
                bytes_sent=int(bps),
                iteration=0,
                client_node="a",
            ),
        ],
        phase=phase,  # ty: ignore[invalid-argument-type]
        parent_trial_id=parent_trial_id,
    )
    tr.trial_id = trial_id
    return tr


def _capture_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, width=120)


def test_seed_history_preserves_file_order_and_phase_labels() -> None:
    observer = RichProgressObserver(
        _capture_console(),
        objectives=ObjectivesSection(),
    )
    # Sobol budget = 2. Primary-only subsequence: [p0, p1, p2].
    p0 = _trial("p0", 1.0e9, phase="sobol")
    p1 = _trial("p1", 1.1e9, phase="sobol")
    v0 = _trial("v0", 0.9e9, phase="refinement", parent_trial_id="p0")
    p2 = _trial("p2", 1.2e9, phase="bayesian")
    v1 = _trial("v1", 1.0e9, phase="refinement", parent_trial_id="p0")

    prior = [p0, p1, v0, p2, v1]
    observer.seed_history(prior, n_sobol=2)

    # File order preserved.
    assert [row.trial_id for row in observer._all_rows] == [
        "p0",
        "p1",
        "v0",
        "p2",
        "v1",
    ]
    # Primary phase labels: p0, p1 are sobol; p2 is bayesian.
    phase_by_id = {row.trial_id: row.phase for row in observer._all_rows}
    assert phase_by_id["p0"] == "sobol"
    assert phase_by_id["p1"] == "sobol"
    assert phase_by_id["p2"] == "bayesian"
    # Refinement rows keep their phase.
    assert phase_by_id["v0"] == "refinement"
    assert phase_by_id["v1"] == "refinement"


def test_seed_history_top_matches_aggregate_score_rows() -> None:
    observer = RichProgressObserver(
        _capture_console(),
        objectives=ObjectivesSection(),
    )
    # Three primaries; one has two refinement samples pulling it down.
    parent = _trial("loser", 1.5e9, phase="bayesian")
    mid = _trial("mid", 1.2e9, phase="bayesian")
    winner = _trial("winner", 1.3e9, phase="bayesian")
    v1 = _trial("v1", 0.5e9, phase="refinement", parent_trial_id=parent.trial_id)
    v2 = _trial("v2", 0.5e9, phase="refinement", parent_trial_id=parent.trial_id)

    prior = [parent, mid, winner, v1, v2]
    observer.seed_history(prior, n_sobol=3)

    # Expected top = aggregated rows scored then sorted by
    # (-score, trial_id) ascending.
    agg = aggregate_by_parent(prior)
    scores = score_rows(
        agg,
        ObjectivesSection().pareto,
        ObjectivesSection().recommendation_weights,
    )
    expected_order = sorted(
        range(len(agg)),
        key=lambda i: (-scores[i], str(agg[i]["trial_id"])),
    )
    expected_top = [str(agg[i]["trial_id"]) for i in expected_order[:5]]

    observed_top = [row.trial_id for row in observer._top]
    assert observed_top == expected_top
