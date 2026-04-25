"""Happy-path total budget: ``top_k * rounds`` refinement samples created."""

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
def test_total_budget_matches_top_k_times_rounds(
    mock_setter_cls: MagicMock,
    mock_runner_cls: MagicMock,
    mock_lease_cls: MagicMock,  # noqa: ARG001 - patched out to avoid real K8s calls
    tmp_path,
) -> None:
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

    created = loop.run_refinement(top_k=3, rounds=4)
    # Strict equality on the happy path: enough primaries (4) for top_k=3.
    assert len(created) == 3 * 4
    assert all(tr.phase == "refinement" for tr in created)
    assert all(tr.refinement_round is not None for tr in created)
    assert {tr.refinement_round for tr in created} == {1, 2, 3, 4}
