"""Integration tests for BenchmarkRunner against a real Talos cluster."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from kubernetes.client.exceptions import ApiException
import pytest

from kube_autotuner.benchmark.runner import BenchmarkRunner
from kube_autotuner.models import BenchmarkConfig, NodePair, TrialLog, TrialResult

if TYPE_CHECKING:
    from kube_autotuner.k8s.client import K8sClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(180),
]


def _make_runner(
    k8s_client: K8sClient, node_names: dict[str, str], namespace: str
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
    return BenchmarkRunner(node_pair, config, client=k8s_client)


def _deployment_exists(k8s_client: K8sClient, name: str, namespace: str) -> bool:
    try:
        k8s_client.apps_v1.read_namespaced_deployment(name, namespace)
    except ApiException as e:
        if e.status == 404:
            return False
        raise
    return True


def _service_exists(k8s_client: K8sClient, name: str, namespace: str) -> bool:
    try:
        k8s_client.core_v1.read_namespaced_service(name, namespace)
    except ApiException as e:
        if e.status == 404:
            return False
        raise
    return True


def test_setup_server(
    k8s_client: K8sClient, node_names: dict[str, str], test_namespace: str
) -> None:
    runner = _make_runner(k8s_client, node_names, test_namespace)
    try:
        runner.setup_server()
        assert _deployment_exists(k8s_client, runner._server_name, test_namespace)
        assert _service_exists(k8s_client, runner._server_name, test_namespace)
    finally:
        runner.cleanup()


def test_cleanup_removes_resources(
    k8s_client: K8sClient, node_names: dict[str, str], test_namespace: str
) -> None:
    runner = _make_runner(k8s_client, node_names, test_namespace)
    runner.setup_server()
    runner.cleanup()

    assert not _deployment_exists(k8s_client, runner._server_name, test_namespace)
    assert not _service_exists(k8s_client, runner._server_name, test_namespace)


def test_full_run_records_results(
    k8s_client: K8sClient, node_names: dict[str, str], test_namespace: str
) -> None:
    """Run benchmark end-to-end and record results to JSONL, mirroring real usage."""
    runner = _make_runner(k8s_client, node_names, test_namespace)
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
        results=results.bench,
        latency_results=results.latency,
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
    # Both fortio sub-stages fire per iteration.
    assert len(t.latency_results) >= 2
    workloads = {lr.workload for lr in t.latency_results}
    assert workloads == {"saturation", "fixed_qps"}
