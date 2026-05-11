"""Regression test: refined parent must rank below an un-refined-but-better one.

The original failure mode in the post-hoc verification flow was that a
parent that ranked top-K on its single primary observation could
regress toward the mean once verification samples accumulated, ending
up below parents that were never re-sampled and still carried their
optimistic single-sample scores. The new refinement loop re-picks
top-K every round from the *combined* population, so that pathology
disappears -- after refinement, a parent whose mean drops below
another (un-refined or refined) parent's score is reflected in the
final ranking.

This test pins the invariant directly: with a primary A reporting an
optimistic 12 Gbps and a primary B reporting 10 Gbps, repeatedly
sampling A at 8 Gbps must eventually rank B above A under
``aggregate_by_parent`` + ``score_rows``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kube_autotuner.experiment import ObjectivesSection
from kube_autotuner.models import (
    BenchmarkConfig,
    BenchmarkResult,
    NodePair,
    TrialResult,
)
from kube_autotuner.scoring import aggregate_by_parent, score_rows


def _trial(
    bps: float,
    *,
    trial_id: str | None = None,
    phase: str = "bayesian",
    parent_trial_id: str | None = None,
    refinement_round: int | None = None,
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
                bytes_sent=int(bps),
                retransmits=1,
                iteration=0,
                client_node="a",
            ),
        ],
        phase=phase,  # ty: ignore[invalid-argument-type]
        parent_trial_id=parent_trial_id,
        refinement_round=refinement_round,
    )
    if trial_id is not None:
        tr.trial_id = trial_id
    return tr


def test_un_refined_parent_cannot_outrank_refined_and_regressed_one() -> None:
    """A is optimistic primary; refining drops its mean below B's."""
    # Primary A: single optimistic 12 Gbps reading.
    a = _trial(12e9, trial_id="parent-a")
    # Primary B: genuine 10 Gbps; never refined.
    b = _trial(10e9, trial_id="parent-b")
    # 3 refinement samples on A, all 8 Gbps.
    refinements = [
        _trial(
            8e9,
            phase="refinement",
            parent_trial_id=a.trial_id,
            refinement_round=r,
        )
        for r in (1, 2, 3)
    ]

    population = [a, b, *refinements]
    rows = aggregate_by_parent(population)

    objectives = ObjectivesSection()
    scores = score_rows(
        rows,
        objectives.pareto,
        objectives.recommendation_weights,
        sems=rows,
        tolerances=objectives.tolerances,
    )
    score_by_id = {str(r["trial_id"]): s for r, s in zip(rows, scores, strict=True)}
    # A's combined mean is (12 + 8 + 8 + 8) / 4 = 9 Gbps; B is 10 Gbps.
    # B must outrank A.
    assert score_by_id[b.trial_id] > score_by_id[a.trial_id]
