"""Integration-style tests for :meth:`OptimizationLoop.run_refinement`.

Uses the same mocked benchmark / setter pattern as ``test_optimizer.py``:
the Ax client is real (the test is gated on the ``optimize`` dep
group) but the runner, lease, and sysctl setter are all swapped for
mocks so no cluster is touched.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("ax")

from kube_autotuner.experiment import ObjectivesSection
from kube_autotuner.models import (
    BenchmarkConfig,
    BenchmarkResult,
    IterationResults,
    LatencyResult,
    NodePair,
    is_primary,
)
from kube_autotuner.optimizer import OptimizationLoop
from kube_autotuner.sysctl.params import PARAM_SPACE
from kube_autotuner.trial_log import TrialLog


def _mock_snapshot(param_names: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in param_names:
        values[name] = "6.1.0-talos" if name == "kernel.osrelease" else "212992"
    return values


def _make_results() -> list[BenchmarkResult]:
    return [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=9e9,
            retransmits=5,
            bytes_sent=10**9,
            client_node="kmain07",
            iteration=0,
        ),
    ]


def _make_latency_results() -> list[LatencyResult]:
    return [
        LatencyResult(
            timestamp=datetime.now(UTC),
            workload="saturation",
            client_node="kmain07",
            iteration=0,
            rps=1000.0,
        ),
        LatencyResult(
            timestamp=datetime.now(UTC),
            workload="fixed_qps",
            client_node="kmain07",
            iteration=0,
            rps=1000.0,
            latency_p50=0.001,
            latency_p90=0.005,
            latency_p99=0.010,
        ),
    ]


@patch("kube_autotuner.optimizer.NodeLease")
@patch("kube_autotuner.optimizer.BenchmarkRunner")
@patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
def test_run_refinement_emits_phase_parent_rows_and_no_new_complete_trial_calls(
    mock_setter_cls: MagicMock,
    mock_runner_cls: MagicMock,
    mock_lease_cls: MagicMock,  # noqa: ARG001 - patched out to avoid real K8s calls
    tmp_path,
) -> None:
    """4 primary trials + 2 rounds * 2 top-K = 4 refinement rows."""
    mock_setter = MagicMock()
    mock_setter.snapshot.side_effect = _mock_snapshot
    mock_setter_cls.return_value = mock_setter

    mock_runner = MagicMock()
    mock_runner.run.return_value = IterationResults(
        bench=_make_results(),
        latency=_make_latency_results(),
    )
    mock_runner_cls.return_value = mock_runner

    node_pair = NodePair(
        source="kmain07",
        target="kmain08",
        hardware_class="10g",
        namespace="default",
    )
    output = tmp_path / "results"

    # Use n_sobol == n_trials so the test does not depend on Ax's
    # Sobol->Bayesian transition criteria (identical-metric mocks can
    # stall the transition). The refinement pass is independent of
    # which Ax phase produced the parent.
    loop = OptimizationLoop(
        node_pair=node_pair,
        config=BenchmarkConfig(iterations=1),
        param_space=PARAM_SPACE,
        output=output,
        n_trials=4,
        n_sobol=4,
        objectives=ObjectivesSection(),
    )
    # Wrap complete_trial with a counter that still forwards to Ax.
    call_counter: dict[str, int] = {"n": 0}
    real_complete_trial = loop.client.complete_trial

    def counting_complete_trial(*args: object, **kwargs: object) -> object:
        call_counter["n"] += 1
        return real_complete_trial(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    loop.client.complete_trial = counting_complete_trial  # ty: ignore[invalid-assignment]
    loop.run()
    primary_complete_count = call_counter["n"]

    created = loop.run_refinement(top_k=2, rounds=2)
    # 2 rounds * top-K 2 = 4 refinement rows.
    assert len(created) == 4
    assert all(tr.phase == "refinement" for tr in created)
    primary_ids = {tr.trial_id for tr in loop._completed if is_primary(tr)}
    assert all(tr.parent_trial_id in primary_ids for tr in created)
    # Each created row carries a 1-indexed refinement_round.
    assert {tr.refinement_round for tr in created} == {1, 2}
    # Dataset got 4 primary + 4 refinement rows.
    persisted = TrialLog.load(output)
    assert len(persisted) == 4 + 4
    # Refinement does not call complete_trial -- Ax never sees the samples.
    assert call_counter["n"] == primary_complete_count


@patch("kube_autotuner.optimizer.NodeLease")
@patch("kube_autotuner.optimizer.BenchmarkRunner")
@patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
def test_run_refinement_skips_already_done_round(
    mock_setter_cls: MagicMock,
    mock_runner_cls: MagicMock,
    mock_lease_cls: MagicMock,  # noqa: ARG001 - patched out to avoid real K8s calls
    tmp_path,
) -> None:
    """``completed_by_round`` skips parents already sampled in a given round."""
    mock_setter = MagicMock()
    mock_setter.snapshot.side_effect = _mock_snapshot
    mock_setter_cls.return_value = mock_setter

    mock_runner = MagicMock()
    mock_runner.run.return_value = IterationResults(
        bench=_make_results(),
        latency=_make_latency_results(),
    )
    mock_runner_cls.return_value = mock_runner

    loop = OptimizationLoop(
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        config=BenchmarkConfig(iterations=1),
        param_space=PARAM_SPACE,
        output=tmp_path / "out",
        n_trials=4,
        n_sobol=4,
        objectives=ObjectivesSection(),
    )
    loop.run()

    # Pick a parent that will be in the round-1 top-K and pretend it
    # was already sampled. Score the primaries the same way
    # run_refinement does (aggregate_by_parent + score_rows over the
    # population), then take the top entry.
    from kube_autotuner.scoring import (  # noqa: PLC0415
        aggregate_by_parent,
        config_memory_cost,
        score_rows,
    )

    primaries = [tr for tr in loop._completed if is_primary(tr)]
    agg = aggregate_by_parent(primaries)
    parent_by_id = {p.trial_id: p for p in primaries}
    usable = [r for r in agg if str(r["trial_id"]) in parent_by_id]
    parents_aligned = [parent_by_id[str(r["trial_id"])] for r in usable]
    costs = [config_memory_cost(p.sysctl_values, PARAM_SPACE) for p in parents_aligned]
    scores = score_rows(
        usable,
        ObjectivesSection().pareto,
        ObjectivesSection().recommendation_weights,
        memory_costs=costs,
        memory_cost_weight=ObjectivesSection().memory_cost_weight,
    )
    ranked = sorted(
        range(len(parents_aligned)),
        key=lambda i: (-scores[i], parents_aligned[i].trial_id),
    )
    first_parent_id = parents_aligned[ranked[0]].trial_id

    created = loop.run_refinement(
        top_k=2,
        rounds=2,
        completed_by_round={1: {first_parent_id}},
    )
    # Round 1: 1 parent already done -> 1 new sample.
    # Round 2: 2 parents -> 2 new samples.
    # Total: 3 new samples.
    assert len(created) == 3
    counts: dict[str, int] = {}
    for tr in created:
        assert tr.parent_trial_id is not None
        counts[tr.parent_trial_id] = counts.get(tr.parent_trial_id, 0) + 1
    # The pre-completed parent gets one fewer sample because it was
    # skipped in round 1.
    assert counts.get(first_parent_id, 0) == 1
