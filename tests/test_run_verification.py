"""Integration-style test for :meth:`OptimizationLoop.run_verification`.

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
    TrialLog,
    is_primary,
)
from kube_autotuner.optimizer import OptimizationLoop
from kube_autotuner.sysctl.params import PARAM_SPACE


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
def test_run_verification_emits_phase_parent_rows_and_no_new_complete_trial_calls(
    mock_setter_cls: MagicMock,
    mock_runner_cls: MagicMock,
    mock_lease_cls: MagicMock,  # noqa: ARG001 - patched out to avoid real K8s calls
    tmp_path,
) -> None:
    """6 primary trials + 2 verification per top-2 = 4 verification rows."""
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
    output = tmp_path / "results.jsonl"

    # Use n_sobol == n_trials so the test does not depend on Ax's
    # Sobol->Bayesian transition criteria (identical-metric mocks can
    # stall the transition). The verification pass is independent of
    # which Ax phase produced the parent.
    loop = OptimizationLoop(
        node_pair=node_pair,
        config=BenchmarkConfig(duration=5, iterations=1),
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

    created = loop.run_verification(top_k=2, repeats=2)
    # 2 configs * 2 repeats = 4 verification rows.
    assert len(created) == 4
    assert all(tr.phase == "verification" for tr in created)
    primary_ids = {tr.trial_id for tr in loop._completed if is_primary(tr)}
    assert all(tr.parent_trial_id in primary_ids for tr in created)
    # JSONL got 4 primary + 4 verification rows.
    persisted = TrialLog.load(output)
    assert len(persisted) == 4 + 4
    # Verification does not call complete_trial -- Ax never sees the repeats.
    assert call_counter["n"] == primary_complete_count


@patch("kube_autotuner.optimizer.NodeLease")
@patch("kube_autotuner.optimizer.BenchmarkRunner")
@patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
def test_run_verification_skips_done_work(
    mock_setter_cls: MagicMock,
    mock_runner_cls: MagicMock,
    mock_lease_cls: MagicMock,  # noqa: ARG001 - patched out to avoid real K8s calls
    tmp_path,
) -> None:
    """``already_done_by_parent`` trims the per-parent repeat count."""
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
        config=BenchmarkConfig(duration=5, iterations=1),
        param_space=PARAM_SPACE,
        output=tmp_path / "out.jsonl",
        n_trials=4,
        n_sobol=4,
        objectives=ObjectivesSection(),
    )
    loop.run()
    # Rank the primaries the same way run_verification will; feed one
    # of the top-2 parents as already-done.
    from kube_autotuner.optimizer import _compute_metrics  # noqa: PLC0415, PLC2701
    from kube_autotuner.progress import _build_trial_row  # noqa: PLC0415, PLC2701
    from kube_autotuner.scoring import score_rows  # noqa: PLC0415

    rows = [
        _build_trial_row(
            0,
            tr.phase or "bayesian",
            _compute_metrics(tr),
            trial_id=tr.trial_id,
            parent_trial_id=None,
        )
        for tr in loop._completed
        if is_primary(tr)
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
    first_parent_id = rows[ranked[0]].trial_id

    created = loop.run_verification(
        top_k=2,
        repeats=2,
        already_done_by_parent={first_parent_id: 1},
    )
    # Parent one: 2-1=1 remaining; parent two: 2 remaining. Total=3.
    assert len(created) == 3
    counts: dict[str, int] = {}
    for tr in created:
        assert tr.parent_trial_id is not None
        counts[tr.parent_trial_id] = counts.get(tr.parent_trial_id, 0) + 1
    assert counts[first_parent_id] == 1
