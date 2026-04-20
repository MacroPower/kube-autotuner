"""Integration tests for BenchmarkRunner against a real Talos cluster."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from kube_autotuner.benchmark.runner import BenchmarkRunner
from kube_autotuner.models import BenchmarkConfig, NodePair, TrialLog, TrialResult

if TYPE_CHECKING:
    from kube_autotuner.k8s.client import Kubectl

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(180),
]


def _make_runner(
    kubectl: Kubectl, node_names: dict[str, str], namespace: str
) -> BenchmarkRunner:
    node_pair = NodePair(
        source=node_names["source"],
        target=node_names["target"],
        hardware_class="1g",
        namespace=namespace,
        ip_family_policy="SingleStack",
    )
    config = BenchmarkConfig(
        duration=5, omit=0, iterations=1, parallel=1, modes=["tcp"]
    )
    return BenchmarkRunner(node_pair, config, kubectl=kubectl)


def test_setup_server(
    kubectl: Kubectl, node_names: dict[str, str], test_namespace: str
) -> None:
    runner = _make_runner(kubectl, node_names, test_namespace)
    try:
        runner.setup_server()

        obj = kubectl.get_json("deployment", runner._server_name, test_namespace)
        assert obj is not None
        assert obj["kind"] == "Deployment"

        svc = kubectl.get_json("service", runner._server_name, test_namespace)
        assert svc is not None
    finally:
        runner.cleanup()


def test_cleanup_removes_resources(
    kubectl: Kubectl, node_names: dict[str, str], test_namespace: str
) -> None:
    runner = _make_runner(kubectl, node_names, test_namespace)
    runner.setup_server()
    runner.cleanup()

    assert kubectl.get_json("deployment", runner._server_name, test_namespace) is None
    assert kubectl.get_json("service", runner._server_name, test_namespace) is None


def test_full_run_records_results(
    kubectl: Kubectl, node_names: dict[str, str], test_namespace: str
) -> None:
    """Run benchmark end-to-end and record results to JSONL, mirroring real usage."""
    runner = _make_runner(kubectl, node_names, test_namespace)
    try:
        runner.setup_server()
        results = runner.run()
    finally:
        runner.cleanup()

    # Record to JSONL exactly as the CLI does.
    output = Path(__file__).resolve().parent.parent.parent / "integration-results.jsonl"
    trial = TrialResult(
        node_pair=runner.node_pair,
        sysctl_values={},
        config=runner.config,
        results=results,
    )
    TrialLog.append(output, trial)

    # Verify the persisted data round-trips correctly.
    loaded = TrialLog.load(output)
    assert len(loaded) >= 1

    t = loaded[-1]
    assert len(t.results) >= 1
    assert t.results[0].bits_per_second > 0
    assert t.results[0].mode == "tcp"
    assert t.node_pair.source == node_names["source"]
    assert t.node_pair.target == node_names["target"]
