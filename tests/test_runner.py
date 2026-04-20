"""Unit tests for :class:`kube_autotuner.benchmark.runner.BenchmarkRunner`."""

from __future__ import annotations

import json
import shutil
from unittest.mock import MagicMock

import pytest
import yaml

from kube_autotuner.benchmark.runner import CLIENT_LABEL, BenchmarkRunner
from kube_autotuner.experiment import IperfArgs, IperfSection, Patch, PatchTarget
from kube_autotuner.models import BenchmarkConfig, NodePair


def _fake_iperf_json(bps: float, remote_total: float = 12.3) -> str:
    return json.dumps({
        "start": {"timestamp": {"timesecs": 1700000000}},
        "end": {
            "sum_sent": {"bits_per_second": bps, "retransmits": 0},
            "cpu_utilization_percent": {
                "host_total": 5.0,
                "remote_total": remote_total,
            },
        },
    })


def _make_client(logs_by_job: dict[str, str]) -> MagicMock:
    client = MagicMock()
    client.apply.return_value = None
    client.wait.return_value = None
    client.rollout_status.return_value = None
    client.delete.return_value = None
    client.delete_by_label.return_value = None

    def _logs(_kind, name, _ns):
        return logs_by_job[name]

    client.logs.side_effect = _logs
    client.top_pod_containers.return_value = []
    return client


def test_single_client_single_iteration():
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])

    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(9e9)}
    client = _make_client(logs)

    runner = BenchmarkRunner(node_pair, config, client=client)
    runner.setup_server()
    results = runner.run()

    assert len(results) == 1
    r = results[0]
    assert r.bits_per_second == pytest.approx(9e9)
    assert r.client_node == "kmain07"
    assert r.iteration == 0
    assert r.cpu_server_percent == pytest.approx(12.3)


def test_multi_client_concurrent_launch():
    node_pair = NodePair(
        source="kmain07",
        target="kmain08",
        hardware_class="10g",
        extra_sources=["kmain09"],
    )
    config = BenchmarkConfig(duration=1, iterations=2, modes=["tcp"])

    logs = {
        "iperf3-client-kmain07-p5201": _fake_iperf_json(4e9),
        "iperf3-client-kmain09-p5202": _fake_iperf_json(5e9),
    }
    client = _make_client(logs)

    runner = BenchmarkRunner(node_pair, config, client=client)
    runner.setup_server()
    results = runner.run()

    # 2 clients * 2 iterations.
    assert len(results) == 4
    iterations = sorted({(r.client_node, r.iteration) for r in results})
    assert iterations == [
        ("kmain07", 0),
        ("kmain07", 1),
        ("kmain09", 0),
        ("kmain09", 1),
    ]

    # Both ports used (one client Job per port).
    applied_yamls = [c.args[0] for c in client.apply.call_args_list]
    assert any("iperf3-client-kmain07-p5201" in y for y in applied_yamls)
    assert any("iperf3-client-kmain09-p5202" in y for y in applied_yamls)

    # Server built with both ports.
    server_yaml = applied_yamls[0]
    assert "iperf3-server-5201" in server_yaml
    assert "iperf3-server-5202" in server_yaml


def test_first_exception_triggers_label_cleanup():
    node_pair = NodePair(
        source="kmain07",
        target="kmain08",
        hardware_class="10g",
        extra_sources=["kmain09"],
    )
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])

    client = _make_client({})

    def _logs(_kind, name, _ns):
        if name == "iperf3-client-kmain07-p5201":
            msg = "client failed"
            raise RuntimeError(msg)
        return _fake_iperf_json(1e9)

    client.logs.side_effect = _logs

    runner = BenchmarkRunner(node_pair, config, client=client)
    with pytest.raises(RuntimeError, match="client failed"):
        runner.run()

    # Label-based cleanup must have been invoked at least once for client label.
    label_delete_calls = [
        c for c in client.delete_by_label.call_args_list if CLIENT_LABEL in c.args
    ]
    assert label_delete_calls, "expected label-based client cleanup on failure"


def test_cleanup_removes_client_jobs_by_label():
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])
    client = _make_client({})

    runner = BenchmarkRunner(node_pair, config, client=client)
    runner.cleanup()

    labels_deleted = [
        (c.args[0], c.args[1]) for c in client.delete_by_label.call_args_list
    ]
    assert ("job", CLIENT_LABEL) in labels_deleted
    assert ("deployment", "app.kubernetes.io/name=iperf3-server") in labels_deleted
    assert ("service", "app.kubernetes.io/name=iperf3-server") in labels_deleted


def test_extra_args_threaded_into_applied_yaml():
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])
    client = _make_client({"iperf3-client-kmain07-p5201": _fake_iperf_json(1e9)})
    iperf_args = IperfSection(
        client=IperfArgs(extra_args=["-Z"]),
        server=IperfArgs(extra_args=["--forceflush"]),
    )
    runner = BenchmarkRunner(
        node_pair,
        config,
        client=client,
        iperf_args=iperf_args,
    )
    runner.setup_server()
    runner.run()

    applied = [c.args[0] for c in client.apply.call_args_list]
    assert any("--forceflush" in y for y in applied), "server extra_args missing"
    assert any("-Z" in y for y in applied), "client extra_args missing"


@pytest.mark.skipif(
    shutil.which("kustomize") is None,
    reason="kustomize binary required on PATH",
)
def test_patches_applied_to_server_yaml():
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])
    client = _make_client({"iperf3-client-kmain07-p5201": _fake_iperf_json(1e9)})
    patches = [
        Patch(
            target=PatchTarget(kind="Deployment"),
            patch={"spec": {"replicas": 3}},
        ),
    ]
    runner = BenchmarkRunner(
        node_pair,
        config,
        client=client,
        patches=patches,
    )
    runner.setup_server()

    server_yaml = client.apply.call_args_list[0].args[0]
    docs = list(yaml.safe_load_all(server_yaml))
    dep = next(d for d in docs if d["kind"] == "Deployment")
    assert dep["spec"]["replicas"] == 3


def test_memory_tagging_applied_to_all_client_results():
    node_pair = NodePair(
        source="kmain07",
        target="kmain08",
        hardware_class="10g",
        extra_sources=["kmain09"],
    )
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])

    logs = {
        "iperf3-client-kmain07-p5201": _fake_iperf_json(4e9),
        "iperf3-client-kmain09-p5202": _fake_iperf_json(5e9),
    }
    client = _make_client(logs)
    client.get_pod_name.return_value = "iperf3-server-xxx"
    client.top_pod_containers.return_value = [
        {"container": "iperf3-server-5201", "cpu": "10m", "memory": "50Mi"},
        {"container": "iperf3-server-5202", "cpu": "20m", "memory": "30Mi"},
    ]

    runner = BenchmarkRunner(node_pair, config, client=client)
    results = runner.run()

    assert len(results) == 2
    for r in results:
        assert r.memory_used_bytes == (50 + 30) * 1024 * 1024
