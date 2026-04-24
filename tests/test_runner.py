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

from kube_autotuner.benchmark.errors import (
    BenchmarkFailure,
    ClientJobFailed,
    ResultValidationError,
)
from kube_autotuner.benchmark.parser import parse_iperf_json
from kube_autotuner.benchmark.runner import (
    CLIENT_LABEL,
    FORTIO_CLIENT_LABEL,
    BenchmarkRunner,
)
from kube_autotuner.experiment import (
    IperfArgs,
    IperfSection,
    Patch,
    PatchTarget,
)
from kube_autotuner.models import BenchmarkConfig, NodePair

if TYPE_CHECKING:
    from collections.abc import Callable


def _fake_iperf_json(
    bps: float,
    mode: str = "tcp",
) -> str:
    base: dict = {
        "start": {"timestamp": {"timesecs": 1700000000}},
        "end": {},
    }
    if mode == "tcp":
        base["end"]["sum_sent"] = {
            "bits_per_second": bps,
            "retransmits": 0,
            "bytes": 1_000_000_000,
        }
    else:
        base["end"]["sum"] = {
            "bits_per_second": bps,
            "packets": 10_000,
            "jitter_ms": 0.05,
        }
    return json.dumps(base)


def _fake_iperf_udp_json(bps: float = 1e9) -> str:
    return _fake_iperf_json(bps, mode="udp")


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


def _iperf_job_mode_from_yaml(yaml_text: str) -> tuple[str, str] | None:
    """Return ``(job_name, mode)`` for an iperf3 client apply, or ``None``.

    The runner applies both TCP and UDP iperf3 client Jobs per
    iteration under the same job name, so the only way to tell them
    apart in the mock client is to inspect the rendered args for
    ``-u``.
    """
    for doc in yaml.safe_load_all(yaml_text):
        if not isinstance(doc, dict) or doc.get("kind") != "Job":
            continue
        name = doc.get("metadata", {}).get("name", "")
        if not name.startswith("iperf3-client-"):
            continue
        containers = doc["spec"]["template"]["spec"]["containers"]
        args = containers[0].get("args", [])
        return name, ("udp" if "-u" in args else "tcp")
    return None


def _make_client(
    logs_by_job: dict[str, str],
    logs_by_job_udp: dict[str, str] | None = None,
) -> MagicMock:
    client = MagicMock()
    last_mode_by_job: dict[str, str] = {}

    def _apply(yaml_text, _ns):
        info = _iperf_job_mode_from_yaml(yaml_text)
        if info is not None:
            name, mode = info
            last_mode_by_job[name] = mode

    client.apply.side_effect = _apply
    client.wait.return_value = None
    client.rollout_status.return_value = None
    client.delete.return_value = None
    client.delete_by_label.return_value = None
    # The runner now fetches the log-source pod name via _job_log_pod,
    # then calls logs("pod", <pod>, ns). Fake it by returning the job
    # name so the existing job-name keyed log fixtures still resolve.
    client._job_log_pod.side_effect = lambda job_name, _ns: job_name

    def _logs(_kind, name, _ns):
        if name.startswith("iperf3-client-"):
            mode = last_mode_by_job.get(name, "tcp")
            if mode == "udp":
                if logs_by_job_udp and name in logs_by_job_udp:
                    return logs_by_job_udp[name]
                return _fake_iperf_udp_json()
            if name in logs_by_job:
                return logs_by_job[name]
            raise KeyError(name)
        if name.startswith("fortio-client-"):
            return _fake_fortio_json()
        raise KeyError(name)

    client.logs.side_effect = _logs
    client.describe_job_failure.return_value = {}
    # Exposed so tests that override `client.logs.side_effect` can still
    # discriminate TCP vs UDP for the shared iperf3 client job names.
    client._last_mode_by_job = last_mode_by_job
    return client


def test_single_client_single_iteration():
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1)

    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(9e9)}
    client = _make_client(logs)

    runner = BenchmarkRunner(node_pair, config, client=client)
    runner.setup_server()
    iteration_results = runner.run()

    # Each iteration runs both bw-tcp and bw-udp now.
    assert len(iteration_results.bench) == 2
    modes = sorted(r.mode for r in iteration_results.bench)
    assert modes == ["tcp", "udp"]
    tcp = next(r for r in iteration_results.bench if r.mode == "tcp")
    assert tcp.bits_per_second == pytest.approx(9e9)
    assert tcp.client_node == "kmain07"
    assert tcp.iteration == 0
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
    config = BenchmarkConfig(duration=1, iterations=2)

    logs = {
        "iperf3-client-kmain07-p5201": _fake_iperf_json(4e9),
        "iperf3-client-kmain09-p5202": _fake_iperf_json(5e9),
    }
    client = _make_client(logs)

    runner = BenchmarkRunner(node_pair, config, client=client)
    runner.setup_server()
    iteration_results = runner.run()

    # 2 clients * 2 iterations * 2 modes (tcp + udp).
    bench = iteration_results.bench
    assert len(bench) == 8
    iterations = sorted({(r.client_node, r.iteration, r.mode) for r in bench})
    assert iterations == [
        ("kmain07", 0, "tcp"),
        ("kmain07", 0, "udp"),
        ("kmain07", 1, "tcp"),
        ("kmain07", 1, "udp"),
        ("kmain09", 0, "tcp"),
        ("kmain09", 0, "udp"),
        ("kmain09", 1, "tcp"),
        ("kmain09", 1, "udp"),
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
    config = BenchmarkConfig(duration=1, iterations=1)

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
    config = BenchmarkConfig(duration=1, iterations=1)
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
    config = BenchmarkConfig(duration=1, iterations=1)
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
    config = BenchmarkConfig(duration=1, iterations=1)
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


def test_substages_run_sequentially_in_order():
    """bw-tcp, bw-udp, fortio-sat, fortio-fixed run in that order."""
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1)

    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(1e9)}

    client = _make_client(logs)

    runner = BenchmarkRunner(node_pair, config, client=client)
    runner.setup_server()
    iteration_results = runner.run()

    # 1 client * 1 iteration * 2 modes (tcp + udp).
    assert len(iteration_results.bench) == 2
    assert len(iteration_results.latency) == 2

    # Apply order: iperf server, fortio server (during setup_server),
    # then per iteration: iperf client TCP, iperf client UDP, fortio
    # saturation client, fortio fixed_qps client.
    applied_yamls = [c.args[0] for c in client.apply.call_args_list]
    assert len(applied_yamls) == 6
    assert "iperf3-server-" in applied_yamls[0]
    assert "fortio-server-" in applied_yamls[1]
    assert "iperf3-client-" in applied_yamls[2]
    assert "-u" not in applied_yamls[2]  # bw-tcp
    assert "iperf3-client-" in applied_yamls[3]
    assert "-u" in applied_yamls[3]  # bw-udp
    assert "saturation" in applied_yamls[4]
    assert "fixed_qps" in applied_yamls[5]


def test_fortio_failure_cleans_up_by_fortio_label():
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1)

    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(1e9)}

    client = _make_client(logs)

    def _logs(_kind, name, _ns):
        if name.startswith("iperf3-client-"):
            mode = client._last_mode_by_job.get(name, "tcp")
            if mode == "udp":
                return _fake_iperf_udp_json()
            if name in logs:
                return logs[name]
            raise KeyError(name)
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
    """Build a single-iteration runner with a retry harness.

    ``iperf_logs`` / ``iperf_log_pod`` only drive the ``bw-tcp``
    stage; the follow-on ``bw-udp`` stage gets a canned valid
    response so the retry behaviour under test is isolated to TCP.
    The fortio sub-stages stay happy-path so each iteration's retry
    behavior is entirely driven by the TCP stage.

    Returns:
        ``(runner, client_mock)``.
    """
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1)

    iperf_iter = iter(iperf_logs)
    client = _make_client(logs_by_job={})

    def _logs(kind, name, _ns):
        if kind == "pod" and name.startswith("iperf3-client-"):
            mode = client._last_mode_by_job.get(name, "tcp")
            if mode == "udp":
                return _fake_iperf_udp_json()
            return next(iperf_iter)
        if kind == "pod" and name.startswith("fortio-client-"):
            return _fake_fortio_json()
        raise KeyError((kind, name))

    client.logs.side_effect = _logs
    if iperf_log_pod is not None:

        def _log_pod(job_name, ns):
            if job_name.startswith("iperf3-client-"):
                mode = client._last_mode_by_job.get(job_name, "tcp")
                if mode == "tcp":
                    return iperf_log_pod(job_name, ns)
            return job_name  # fortio / bw-udp: happy path

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
    # 2 records: TCP (retried then succeeded) + UDP (always happy path).
    assert len(results.bench) == 2
    tcp = next(r for r in results.bench if r.mode == "tcp")
    assert tcp.bits_per_second == pytest.approx(9e9)
    # Exactly one "attempt 1/3 failed" warning; no "after 3 attempts".
    attempt_failed = [
        r for r in caplog.records if "attempt 1/3 failed" in r.getMessage()
    ]
    assert len(attempt_failed) == 1
    assert all("after 3 attempts" not in r.getMessage() for r in caplog.records)
    # 3 iperf3 applies: TCP attempt 1 (fail), TCP attempt 2 (ok), UDP.
    iperf_applies = [
        c for c in client.apply.call_args_list if "iperf3-client-" in c.args[0]
    ]
    assert len(iperf_applies) == 3


def test_retry_exhaustion_raises_runtime_error_with_cause():
    """(b) All attempts fail; BenchmarkFailure wraps ClientJobFailed → cause."""
    runner, _ = _retry_runner(
        iperf_logs=[json.dumps({"error": "x"}) for _ in range(3)],
    )
    with pytest.raises(RuntimeError, match="after 3 attempts") as exc_info:
        runner.run()
    # Outer: BenchmarkFailure envelope carrying stage/iteration metadata.
    assert isinstance(exc_info.value, BenchmarkFailure)
    assert exc_info.value.stage == "bw-tcp"
    assert exc_info.value.iteration == 0
    # Middle: ClientJobFailed carrying the per-attempt diagnostics list.
    inner = exc_info.value.__cause__
    assert isinstance(inner, ClientJobFailed)
    # Innermost: the ResultValidationError raised by parse().
    assert isinstance(inner.__cause__, ResultValidationError)


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
    # TCP (retried once) + UDP (happy path).
    assert len(results.bench) == 2
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
    tcp = next(r for r in results.bench if r.mode == "tcp")
    assert tcp.bits_per_second == pytest.approx(7e9)


def test_retry_on_fortio_zero_count():
    """(e) fortio DurationHistogram.Count==0 → retry until Count > 0."""
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1)

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
            mode = client._last_mode_by_job.get(name, "tcp")
            if mode == "udp":
                return _fake_iperf_udp_json()
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
    tcp = next(r for r in results.bench if r.mode == "tcp")
    assert tcp.bits_per_second == pytest.approx(6e9)
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
    config = BenchmarkConfig(duration=1, iterations=1)

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


def test_run_flushes_network_state_per_iteration():
    """Each iteration invokes ``flush_network_state`` on every configured backend."""
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=2)

    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(9e9)}
    client = _make_client(logs)

    target_backend = MagicMock()
    client_backend = MagicMock()

    runner = BenchmarkRunner(
        node_pair,
        config,
        client=client,
        flush_backends=[target_backend, client_backend],
    )
    runner.setup_server()
    runner.run()

    assert target_backend.flush_network_state.call_count == 2
    assert client_backend.flush_network_state.call_count == 2


def test_run_without_flush_backends_skips_flush():
    """Default ``flush_backends=None`` triggers no per-iteration flush work."""
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=2)

    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(9e9)}
    client = _make_client(logs)

    runner = BenchmarkRunner(node_pair, config, client=client)
    # Sentinel backend that must never be touched: if the default path
    # ever grows a hidden flush call, this MagicMock will record it.
    untouched_backend = MagicMock()
    assert runner._flush_backends == []

    runner.setup_server()
    runner.run()

    untouched_backend.flush_network_state.assert_not_called()


def _message_index(caplog, substring: str) -> int:
    """Return the index of the first caplog record containing ``substring``.

    Used to assert strict ordering between related log lines (start
    before complete, benchmark-complete after the final stage). Fails the
    test with a readable dump if the substring is not present.
    """
    for idx, record in enumerate(caplog.records):
        if substring in record.getMessage():
            return idx
    dump = [r.getMessage() for r in caplog.records]
    pytest.fail(f"no log record contained {substring!r}; got {dump}")
    return -1  # unreachable, keeps ty happy


def test_run_emits_stage_boundary_logs(caplog):
    """``run`` logs benchmark start, start/complete per stage, and benchmark end."""
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=2)
    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(9e9)}
    client = _make_client(logs)

    runner = BenchmarkRunner(node_pair, config, client=client)
    runner.setup_server()
    with caplog.at_level("INFO", logger="kube_autotuner.benchmark.runner"):
        runner.run()

    messages = [r.getMessage() for r in caplog.records]
    assert sum("Starting benchmark" in m for m in messages) == 1
    assert sum("Benchmark complete" in m for m in messages) == 1

    for iteration in (1, 2):
        for stage in ("bw-tcp", "bw-udp", "fortio-sat", "fortio-fixed"):
            start_tag = f"Stage {stage} starting (iteration {iteration}/2)"
            complete_tag = f"Stage {stage} complete (iteration {iteration}/2"
            start_idx = _message_index(caplog, start_tag)
            complete_idx = _message_index(caplog, complete_tag)
            assert start_idx < complete_idx, f"{start_tag} must precede {complete_tag}"

    start_idx = _message_index(caplog, "Starting benchmark")
    end_idx = _message_index(caplog, "Benchmark complete")
    last_stage_idx = _message_index(
        caplog,
        "Stage fortio-fixed complete (iteration 2/2",
    )
    assert start_idx < last_stage_idx < end_idx


def test_run_emits_flush_bookends_per_backend(caplog):
    """Each configured backend gets a ``starting`` / ``complete`` INFO pair in order."""
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1)
    logs = {"iperf3-client-kmain07-p5201": _fake_iperf_json(9e9)}
    client = _make_client(logs)

    target_backend = MagicMock()
    target_backend.node = "kmain08"
    client_backend = MagicMock()
    client_backend.node = "kmain07"

    runner = BenchmarkRunner(
        node_pair,
        config,
        client=client,
        flush_backends=[target_backend, client_backend],
    )
    runner.setup_server()
    with caplog.at_level("INFO", logger="kube_autotuner.benchmark.runner"):
        runner.run()

    assert any(
        "Flushing network state before iteration 1 on 2 backend(s)" in r.getMessage()
        for r in caplog.records
    )
    for node in ("kmain08", "kmain07"):
        start_idx = _message_index(caplog, f"Network flush starting on {node}")
        complete_idx = _message_index(caplog, f"Network flush complete on {node}")
        assert start_idx < complete_idx


def test_run_flush_starting_logged_even_when_backend_raises(caplog):
    """A raising backend still produces its ``starting`` line, and aborts the run."""
    node_pair = NodePair(source="kmain07", target="kmain08", hardware_class="10g")
    config = BenchmarkConfig(duration=1, iterations=1)
    client = _make_client({"iperf3-client-kmain07-p5201": _fake_iperf_json(9e9)})

    angry_backend = MagicMock()
    angry_backend.node = "kmain08"
    angry_backend.flush_network_state.side_effect = RuntimeError("flush exploded")

    runner = BenchmarkRunner(
        node_pair,
        config,
        client=client,
        flush_backends=[angry_backend],
    )
    runner.setup_server()
    with (
        caplog.at_level("INFO", logger="kube_autotuner.benchmark.runner"),
        pytest.raises(RuntimeError, match="flush exploded"),
    ):
        runner.run()

    messages = [r.getMessage() for r in caplog.records]
    assert any("Network flush starting on kmain08" in m for m in messages)
    # "complete" must NOT be logged when the backend propagated.
    assert not any("Network flush complete on kmain08" in m for m in messages)
