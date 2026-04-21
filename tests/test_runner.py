"""Unit tests for :class:`kube_autotuner.benchmark.runner.BenchmarkRunner`."""

from __future__ import annotations

import json
import shutil
import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import yaml

from kube_autotuner.benchmark import runner as runner_module
from kube_autotuner.benchmark.errors import ResultValidationError
from kube_autotuner.benchmark.parser import parse_iperf_json
from kube_autotuner.benchmark.runner import (
    CLIENT_LABEL,
    FORTIO_CLIENT_LABEL,
    BenchmarkRunner,
)
from kube_autotuner.experiment import (
    CniSection,
    IperfArgs,
    IperfSection,
    Patch,
    PatchTarget,
)
from kube_autotuner.k8s.client import K8sApiError
from kube_autotuner.models import BenchmarkConfig, NodePair

if TYPE_CHECKING:
    from collections.abc import Callable


def _fake_iperf_json(bps: float, remote_total: float = 12.3) -> str:
    return json.dumps({
        "start": {"timestamp": {"timesecs": 1700000000}},
        "end": {
            "sum_sent": {
                "bits_per_second": bps,
                "retransmits": 0,
                "bytes": 1_000_000_000,
            },
            "cpu_utilization_percent": {
                "host_total": 5.0,
                "remote_total": remote_total,
            },
        },
    })


def _fake_fortio_json(
    rps: float = 1000.0,
    p99_seconds: float = 0.01,
) -> str:
    return json.dumps({
        "ActualQPS": rps,
        "DurationHistogram": {
            "Count": 10000,
            "Percentiles": [
                {"Percentile": 50.0, "Value": p99_seconds / 4},
                {"Percentile": 90.0, "Value": p99_seconds / 2},
                {"Percentile": 99.0, "Value": p99_seconds},
            ],
        },
    })


def _make_client(logs_by_job: dict[str, str]) -> MagicMock:
    client = MagicMock()
    client.apply.return_value = None
    client.wait.return_value = None
    client.rollout_status.return_value = None
    client.delete.return_value = None
    client.delete_by_label.return_value = None
    # The runner now fetches the log-source pod name via _job_log_pod,
    # then calls logs("pod", <pod>, ns). Fake it by returning the job
    # name so the existing job-name keyed log fixtures still resolve.
    client._job_log_pod.side_effect = lambda job_name, _ns: job_name

    def _logs(_kind, name, _ns):
        if name in logs_by_job:
            return logs_by_job[name]
        if name.startswith("fortio-client-"):
            return _fake_fortio_json()
        raise KeyError(name)

    client.logs.side_effect = _logs
    client.describe_job_failure.return_value = {}
    client.top_pod_containers.return_value = []
    client.top_node.return_value = {}
    client.list_pods_by_selector_on_node.return_value = []
    return client


def _cni_disabled() -> CniSection:
    return CniSection(enabled=False)


def test_single_client_single_iteration():
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])

    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(9e9)}
    client = _make_client(logs)

    runner = BenchmarkRunner(node_pair, config, client=client)
    runner.setup_server()
    iteration_results = runner.run()

    assert len(iteration_results.bench) == 1
    r = iteration_results.bench[0]
    assert r.bits_per_second == pytest.approx(9e9)
    assert r.client_node == "kmain07"
    assert r.iteration == 0
    assert r.cpu_server_percent == pytest.approx(12.3)
    # Fortio sub-stages fire twice per iteration (saturation + fixed_qps).
    assert len(iteration_results.latency) == 2
    workloads = sorted(lr.workload for lr in iteration_results.latency)
    assert workloads == ["fixed_qps", "saturation"]


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
    iteration_results = runner.run()

    # 2 clients * 2 iterations.
    bench = iteration_results.bench
    assert len(bench) == 4
    iterations = sorted({(r.client_node, r.iteration) for r in bench})
    assert iterations == [
        ("kmain07", 0),
        ("kmain07", 1),
        ("kmain09", 0),
        ("kmain09", 1),
    ]
    # Latency: 2 clients * 2 iterations * 2 workloads = 8 records.
    assert len(iteration_results.latency) == 8

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
    assert ("job", FORTIO_CLIENT_LABEL) in labels_deleted
    assert ("deployment", "app.kubernetes.io/name=iperf3-server") in labels_deleted
    assert ("service", "app.kubernetes.io/name=iperf3-server") in labels_deleted
    assert ("deployment", "app.kubernetes.io/name=fortio-server") in labels_deleted
    assert ("service", "app.kubernetes.io/name=fortio-server") in labels_deleted


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


def _slow_logs(delay_seconds: float, logs_by_job: dict[str, str]):
    """Build a ``client.logs`` side_effect that waits before returning.

    Gives the background memory sampler time to poll at least once
    during the otherwise-instant mocked iteration. Unknown fortio
    client job names fall back to a canned fortio JSON response so
    sub-stages after the iperf3 stage can complete.

    Returns:
        A callable matching the ``client.logs`` signature.
    """

    def _impl(_kind, name, _ns):
        time.sleep(delay_seconds)
        if name in logs_by_job:
            return logs_by_job[name]
        if name.startswith("fortio-client-"):
            return _fake_fortio_json()
        raise KeyError(name)

    return _impl


def test_node_memory_tagging_applied_to_all_client_results(monkeypatch):
    monkeypatch.setattr(runner_module, "SAMPLE_INTERVAL_S", 0.001)

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
    client.logs.side_effect = _slow_logs(0.05, logs)
    client.top_node.return_value = {"cpu": "500m", "memory": "8000000Ki"}

    runner = BenchmarkRunner(node_pair, config, client=client, cni=_cni_disabled())
    iteration_results = runner.run()

    assert len(iteration_results.bench) == 2
    for r in iteration_results.bench:
        assert r.node_memory_used_bytes == 8_000_000 * 1024
        assert r.cni_memory_used_bytes is None
    # Latency records also see the iteration peak.
    for lr in iteration_results.latency:
        assert lr.node_memory_used_bytes == 8_000_000 * 1024
        assert lr.cni_memory_used_bytes is None


def test_node_memory_peak_not_last(monkeypatch):
    monkeypatch.setattr(runner_module, "SAMPLE_INTERVAL_S", 0.001)

    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])

    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(1e9)}
    client = _make_client(logs)
    client.logs.side_effect = _slow_logs(0.1, logs)

    series = iter([
        {"cpu": "0", "memory": "50Mi"},
        {"cpu": "0", "memory": "200Mi"},
        {"cpu": "0", "memory": "100Mi"},
    ])
    tail = {"cpu": "0", "memory": "100Mi"}

    def _top_node(_name):
        return next(series, tail)

    client.top_node.side_effect = _top_node

    runner = BenchmarkRunner(node_pair, config, client=client, cni=_cni_disabled())
    iteration_results = runner.run()

    assert len(iteration_results.bench) == 1
    assert iteration_results.bench[0].node_memory_used_bytes == 200 * 1024 * 1024


def test_node_memory_none_when_metrics_empty(monkeypatch):
    monkeypatch.setattr(runner_module, "SAMPLE_INTERVAL_S", 0.001)

    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])

    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(1e9)}
    client = _make_client(logs)
    client.logs.side_effect = _slow_logs(0.05, logs)
    client.top_node.return_value = {}

    runner = BenchmarkRunner(node_pair, config, client=client, cni=_cni_disabled())
    iteration_results = runner.run()

    assert len(iteration_results.bench) == 1
    assert iteration_results.bench[0].node_memory_used_bytes is None
    assert iteration_results.bench[0].cni_memory_used_bytes is None


def test_memory_sampler_survives_api_error(monkeypatch):
    monkeypatch.setattr(runner_module, "SAMPLE_INTERVAL_S", 0.001)

    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])

    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(1e9)}
    client = _make_client(logs)
    client.logs.side_effect = _slow_logs(0.1, logs)

    calls = {"n": 0}

    def _top_node(_name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise K8sApiError(
                op="top node",
                status=503,
                reason="ServiceUnavailable",
                message="metrics-server down",
            )
        return {"cpu": "100m", "memory": "75Mi"}

    client.top_node.side_effect = _top_node

    runner = BenchmarkRunner(node_pair, config, client=client, cni=_cni_disabled())
    iteration_results = runner.run()

    assert len(iteration_results.bench) == 1
    assert iteration_results.bench[0].node_memory_used_bytes == 75 * 1024 * 1024
    assert calls["n"] >= 2, "sampler thread did not survive initial API error"


def test_memory_sampler_thread_is_cleaned_up(monkeypatch):
    monkeypatch.setattr(runner_module, "SAMPLE_INTERVAL_S", 0.001)

    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=2, modes=["tcp"])

    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(1e9)}
    client = _make_client(logs)
    client.top_node.return_value = {"cpu": "10m", "memory": "10Mi"}

    runner = BenchmarkRunner(node_pair, config, client=client, cni=_cni_disabled())
    runner.run()

    lingering = [t for t in threading.enumerate() if t.name.startswith("mem-sampler-")]
    assert lingering == []


def test_cni_memory_populated_when_enabled(monkeypatch):
    monkeypatch.setattr(runner_module, "SAMPLE_INTERVAL_S", 0.001)

    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])

    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(1e9)}
    client = _make_client(logs)
    client.logs.side_effect = _slow_logs(0.05, logs)
    client.top_node.return_value = {"cpu": "1000m", "memory": "4194304Ki"}
    client.list_pods_by_selector_on_node.return_value = ["cilium-abc"]
    client.top_pod_containers.return_value = [
        {"container": "cilium-agent", "cpu": "100m", "memory": "64Mi"},
    ]

    cni = CniSection(
        enabled=True,
        namespace="kube-system",
        label_selector="k8s-app=cilium",
    )
    runner = BenchmarkRunner(node_pair, config, client=client, cni=cni)
    iteration_results = runner.run()

    assert len(iteration_results.bench) == 1
    assert iteration_results.bench[0].node_memory_used_bytes == 4_194_304 * 1024
    assert iteration_results.bench[0].cni_memory_used_bytes == 64 * 1024 * 1024
    client.list_pods_by_selector_on_node.assert_called()
    args = client.list_pods_by_selector_on_node.call_args
    assert args.args[0] == "k8s-app=cilium"
    assert args.args[1] == "kube-system"
    assert args.args[2] == "kmain08"


def test_cni_memory_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(runner_module, "SAMPLE_INTERVAL_S", 0.001)

    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])

    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(1e9)}
    client = _make_client(logs)
    client.logs.side_effect = _slow_logs(0.05, logs)
    client.top_node.return_value = {"cpu": "1000m", "memory": "1048576Ki"}

    runner = BenchmarkRunner(node_pair, config, client=client, cni=_cni_disabled())
    iteration_results = runner.run()

    assert len(iteration_results.bench) == 1
    assert iteration_results.bench[0].node_memory_used_bytes == 1_048_576 * 1024
    assert iteration_results.bench[0].cni_memory_used_bytes is None
    assert client.list_pods_by_selector_on_node.call_count == 0


def test_substages_run_sequentially_in_order():
    """bw, fortio-sat, fortio-fixed run in that order with no overlap."""
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])

    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(1e9)}

    client = _make_client(logs)

    runner = BenchmarkRunner(node_pair, config, client=client)
    runner.setup_server()
    iteration_results = runner.run()

    assert len(iteration_results.bench) == 1
    assert len(iteration_results.latency) == 2

    # Apply order: iperf server, fortio server (during setup_server),
    # then per iteration: iperf client, fortio saturation client,
    # fortio fixed_qps client.
    applied_yamls = [c.args[0] for c in client.apply.call_args_list]
    assert len(applied_yamls) == 5
    assert "iperf3-server-" in applied_yamls[0]
    assert "fortio-server-" in applied_yamls[1]
    assert "iperf3-client-" in applied_yamls[2]
    assert "saturation" in applied_yamls[3]
    assert "fixed_qps" in applied_yamls[4]


def test_fortio_failure_cleans_up_by_fortio_label():
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])

    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(1e9)}

    client = _make_client(logs)

    def _logs(_kind, name, _ns):
        if name in logs:
            return logs[name]
        if name.startswith("fortio-client-"):
            msg = "fortio failed"
            raise RuntimeError(msg)
        raise KeyError(name)

    client.logs.side_effect = _logs

    runner = BenchmarkRunner(node_pair, config, client=client)
    with pytest.raises(RuntimeError, match="fortio failed"):
        runner.run()

    label_delete_calls = [
        c
        for c in client.delete_by_label.call_args_list
        if FORTIO_CLIENT_LABEL in c.args
    ]
    assert label_delete_calls, "expected fortio client label cleanup on failure"


# ---- retry / failure-detection coverage ------------------------------------


def _retry_runner(
    *,
    iperf_logs: list[str],
    iperf_max_attempts: int = 3,
    iperf_log_pod: Callable | None = None,
) -> tuple[BenchmarkRunner, MagicMock]:
    """Build a single-iteration TCP-only runner with a retry harness.

    The fortio sub-stages stay happy-path so each iteration's retry
    behavior is entirely driven by ``iperf_logs`` / ``iperf_log_pod``.

    Returns:
        ``(runner, client_mock)``.
    """
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])

    iperf_iter = iter(iperf_logs)
    client = _make_client(logs_by_job={})

    def _logs(kind, name, _ns):
        if kind == "pod" and name.startswith("iperf3-client-"):
            return next(iperf_iter)
        if kind == "pod" and name.startswith("fortio-client-"):
            return _fake_fortio_json()
        raise KeyError((kind, name))

    client.logs.side_effect = _logs
    if iperf_log_pod is not None:

        def _log_pod(job_name, ns):
            if job_name.startswith("iperf3-client-"):
                return iperf_log_pod(job_name, ns)
            return job_name  # fortio: happy path

        client._job_log_pod.side_effect = _log_pod

    runner = BenchmarkRunner(
        node_pair,
        config,
        client=client,
        iperf_args=IperfSection(max_attempts=iperf_max_attempts),
    )
    return runner, client


def test_retry_succeeds_after_first_attempt_fails(caplog):
    """(a) First attempt raises; second attempt returns a valid payload."""
    runner, client = _retry_runner(
        iperf_logs=[
            json.dumps({"error": "connection refused"}),
            _fake_iperf_json(9e9),
        ],
    )
    with caplog.at_level("WARNING"):
        results = runner.run()
    assert len(results.bench) == 1
    assert results.bench[0].bits_per_second == pytest.approx(9e9)
    # Exactly one "attempt 1/3 failed" warning; no "after 3 attempts".
    attempt_failed = [
        r for r in caplog.records if "attempt 1/3 failed" in r.getMessage()
    ]
    assert len(attempt_failed) == 1
    assert all("after 3 attempts" not in r.getMessage() for r in caplog.records)
    # apply/delete happened twice (one per attempt).
    iperf_applies = [
        c for c in client.apply.call_args_list if "iperf3-client-" in c.args[0]
    ]
    assert len(iperf_applies) == 2


def test_retry_exhaustion_raises_runtime_error_with_cause():
    """(b) All attempts fail; final RuntimeError chains the last cause."""
    runner, _ = _retry_runner(
        iperf_logs=[json.dumps({"error": "x"}) for _ in range(3)],
    )
    with pytest.raises(RuntimeError, match="after 3 attempts") as exc_info:
        runner.run()
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, ResultValidationError)


def test_retry_when_no_succeeded_pod(caplog):
    """(c) Job Complete but _job_log_pod returns None → JobAttemptError retry."""
    # First call returns None (no Succeeded pod); second returns a name.
    pod_side = iter([None, "iperf3-client-kmain07-p5201"])
    runner, _ = _retry_runner(
        iperf_logs=[_fake_iperf_json(8e9)],  # single good log, used on 2nd attempt
        iperf_log_pod=lambda _jn, _ns: next(pod_side),
    )
    with caplog.at_level("WARNING"):
        results = runner.run()
    assert len(results.bench) == 1
    assert any("no Succeeded pod" in r.getMessage() for r in caplog.records)


def test_retry_on_iperf_error_payload():
    """(d) iperf3 payload has top-level error field → retry until clean."""
    runner, _ = _retry_runner(
        iperf_logs=[
            json.dumps({"error": "unable to connect"}),
            _fake_iperf_json(7e9),
        ],
    )
    results = runner.run()
    assert results.bench[0].bits_per_second == pytest.approx(7e9)


def test_retry_on_fortio_zero_count():
    """(e) fortio DurationHistogram.Count==0 → retry until Count > 0."""
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])

    fortio_iter = iter([
        json.dumps({
            "ActualQPS": 0.0,
            "DurationHistogram": {"Count": 0, "Percentiles": []},
        }),
        _fake_fortio_json(rps=500.0),
        _fake_fortio_json(rps=600.0),  # fixed_qps sub-stage
    ])
    client = _make_client(
        logs_by_job={"iperf3-client-kmain07-p5201": _fake_iperf_json(1e9)},
    )

    def _logs(kind, name, _ns):
        if name.startswith("iperf3-client-"):
            return _fake_iperf_json(1e9)
        if name.startswith("fortio-client-"):
            return next(fortio_iter)
        raise KeyError((kind, name))

    client.logs.side_effect = _logs
    runner = BenchmarkRunner(node_pair, config, client=client)
    results = runner.run()
    assert len(results.latency) == 2  # saturation + fixed_qps
    rps_values = sorted(lr.rps for lr in results.latency)
    assert rps_values == [pytest.approx(500.0), pytest.approx(600.0)]


def test_happy_path_emits_no_diagnostics(caplog):
    """(f) Diagnostics log is silent when every attempt succeeds first time."""
    runner, client = _retry_runner(iperf_logs=[_fake_iperf_json(9e9)])
    with caplog.at_level("WARNING"):
        runner.run()
    diag_records = [r for r in caplog.records if "diagnostics:" in r.getMessage()]
    assert diag_records == []
    client.describe_job_failure.assert_not_called()


def test_udp_zero_bps_nonzero_packets_does_not_raise():
    """(g) UDP false-positive regression: packets>0 + bps=0.0 must NOT raise."""
    r = parse_iperf_json(
        {"end": {"sum": {"packets": 42, "bits_per_second": 0.0}}}, "udp"
    )
    assert r.bits_per_second == pytest.approx(0.0)


def test_zero_pods_returns_none_triggers_retry(caplog):
    """(h) _job_log_pod returns None (Job deleted externally) → retry."""
    pod_side = iter([None, None, "iperf3-client-kmain07-p5201"])
    runner, _ = _retry_runner(
        iperf_logs=[_fake_iperf_json(6e9)],
        iperf_log_pod=lambda _jn, _ns: next(pod_side),
    )
    with caplog.at_level("WARNING"):
        results = runner.run()
    assert results.bench[0].bits_per_second == pytest.approx(6e9)
    # Two "no Succeeded pod" warnings: one per missing-pod attempt.
    msgs = [r.getMessage() for r in caplog.records]
    assert sum("no Succeeded pod" in m for m in msgs) == 2


def test_sibling_abort_caps_retry_amplification():
    """(i) Once one sibling exhausts, other siblings bail early.

    Assert ``sibling_attempts < max_attempts`` rather than an exact
    count; thread scheduling decides how far the siblings got before
    they observed the abort Event.
    """
    node_pair = NodePair(
        source="kmain07",
        target="kmain08",
        hardware_class="10g",
        extra_sources=["kmain09"],
    )
    config = BenchmarkConfig(duration=1, iterations=1, modes=["tcp"])

    max_attempts = 4
    attempts_per_client: dict[str, int] = {"kmain07": 0, "kmain09": 0}
    attempts_lock = threading.Lock()

    client = _make_client(logs_by_job={})

    def _logs(_kind, name, _ns):
        # All iperf3 attempts return a degenerate payload so every
        # attempt raises ResultValidationError. Fortio payloads are
        # happy-path but the stage never reaches them.
        if name.startswith("iperf3-client-"):
            # Extract the node slug from the job name
            # ("iperf3-client-<node>-p<port>").
            node = name.split("-")[2]
            with attempts_lock:
                attempts_per_client[node] += 1
            # Stall kmain09 so kmain07 exhausts first and sets abort.
            if node == "kmain09":
                time.sleep(0.05)
            return json.dumps({"error": "connection refused"})
        if name.startswith("fortio-client-"):
            return _fake_fortio_json()
        raise KeyError(name)

    client.logs.side_effect = _logs

    runner = BenchmarkRunner(
        node_pair,
        config,
        client=client,
        iperf_args=IperfSection(max_attempts=max_attempts),
    )
    with pytest.raises(RuntimeError, match="after"):
        runner.run()
    # The node whose thread exhausted first hits max_attempts. The
    # sibling observes the abort Event before using up its full budget.
    exhaustion_counts = sorted(attempts_per_client.values())
    assert exhaustion_counts[-1] == max_attempts
    assert exhaustion_counts[0] < max_attempts
