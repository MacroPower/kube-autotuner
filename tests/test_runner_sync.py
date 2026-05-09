"""Stage-level start-time barrier plumbing in :class:`BenchmarkRunner`.

Verifies the invariants of :meth:`BenchmarkRunner._stage_barrier`:

* All clients in a multi-client stage share one ``start_at_epoch``.
* Each stage (TCP vs UDP vs fortio sub-stages) picks its own epoch.
* Single-client stages and ``sync_window_seconds == 0`` skip the
  barrier and retain the base ``_CLIENT_WAIT_TIMEOUT_SECONDS`` wait
  budget.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Literal
from unittest.mock import MagicMock

import pytest

from kube_autotuner.benchmark.runner import BenchmarkRunner
from kube_autotuner.experiment import IperfSection
from kube_autotuner.models import BenchmarkConfig, NodePair

if TYPE_CHECKING:
    import pathlib

_FIXED_NOW = 1_700_000_000.0
_BASE_WAIT = 180


def _fake_tcp_json() -> str:
    return json.dumps({
        "start": {"timestamp": {"timesecs": int(_FIXED_NOW)}},
        "end": {
            "sum_sent": {
                "bits_per_second": 1e9,
                "retransmits": 0,
                "bytes": 1_000_000_000,
            },
        },
    })


def _fake_udp_json() -> str:
    return json.dumps({
        "start": {"timestamp": {"timesecs": int(_FIXED_NOW)}},
        "end": {
            "sum": {
                "bits_per_second": 1e9,
                "packets": 10_000,
                "jitter_ms": 0.05,
            },
        },
    })


def _fake_fortio_json() -> str:
    return json.dumps({
        "ActualQPS": 1000.0,
        "DurationHistogram": {
            "Count": 10000,
            "Percentiles": [
                {"Percentile": 50.0, "Value": 0.001},
                {"Percentile": 90.0, "Value": 0.005},
                {"Percentile": 99.0, "Value": 0.01},
            ],
        },
    })


def _make_k8s_mock() -> MagicMock:
    """Capturing fake ``K8sClient`` for stage-level assertions.

    ``applied`` accumulates every ``(yaml, namespace)`` passed to
    ``apply``. ``wait`` calls are also captured so tests can assert the
    ``timeout`` kwarg each sub-stage passes.

    Returns:
        A ``MagicMock`` whose ``apply`` / ``wait`` / ``logs`` side
        effects are wired to return minimal but well-formed fixtures.
    """
    client = MagicMock()
    client.apply.return_value = None
    client.wait.return_value = None
    client.rollout_status.return_value = None
    client.delete.return_value = None
    client.delete_by_label.return_value = None
    client._job_log_pod.side_effect = lambda job_name, _ns: job_name

    def _logs(_kind: str, name: str, _ns: str) -> str:
        if name.startswith("iperf3-client-"):
            # Caller discriminates TCP vs UDP by inspecting the YAML;
            # both branches of the runner accept the same fake shape
            # minus the mode-specific ``end`` block, so we pick one
            # consistent with the last applied Job's iperf3 argv.
            last_yaml = client.apply.call_args.args[0]
            return _fake_udp_json() if " -u " in last_yaml else _fake_tcp_json()
        if name.startswith("fortio-client-"):
            return _fake_fortio_json()
        msg = name
        raise KeyError(msg)

    client.logs.side_effect = _logs
    client.describe_job_failure.return_value = {}
    return client


def _client_job_yamls(client: MagicMock, *, kind: str) -> list[str]:
    """Return every applied YAML whose Job metadata.name starts with ``kind``."""
    out: list[str] = []
    for call in client.apply.call_args_list:
        yaml_text = call.args[0]
        if f"name: {kind}-client-" in yaml_text:
            out.append(yaml_text)
    return out


def _wait_timeouts(client: MagicMock) -> list[int]:
    """Return the ``timeout`` kwarg from every ``client.wait`` call."""
    return [call.kwargs["timeout"] for call in client.wait.call_args_list]


def _extract_epoch(yaml_text: str) -> int | None:
    """Return the ``start_at_epoch`` literal baked into the YAML, or ``None``."""
    match = re.search(r"DELTA=\$\(\( (\d+) - NOW \)\)", yaml_text)
    return int(match.group(1)) if match else None


def _runner(
    tmp_path: pathlib.Path,
    *,
    extra_sources: list[str],
    sync_window_seconds: int,
    iperf_args: IperfSection | None = None,
) -> tuple[BenchmarkRunner, MagicMock]:
    """Build a BenchmarkRunner with the given source fan-out and sync window.

    Returns:
        A ``(runner, k8s)`` tuple -- the configured
        :class:`BenchmarkRunner` plus the capturing ``K8sClient`` mock
        so tests can assert on what was applied and waited on.
    """
    del tmp_path  # reserved for future log captures; unused today.
    node_pair = NodePair(
        source="src0",
        target="tgt",
        hardware_class="10g",
        extra_sources=extra_sources,
    )
    config = BenchmarkConfig(
        iterations=1,
        sync_window_seconds=sync_window_seconds,
    )
    k8s = _make_k8s_mock()
    runner = BenchmarkRunner(
        node_pair=node_pair,
        config=config,
        client=k8s,
        iperf_args=iperf_args,
    )
    return runner, k8s


def test_bandwidth_stage_shares_epoch_across_clients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    monkeypatch.setattr(
        "kube_autotuner.benchmark.runner.time.time",
        lambda: _FIXED_NOW,
    )
    runner, k8s = _runner(
        tmp_path,
        extra_sources=["src1"],
        sync_window_seconds=15,
    )
    runner._run_bandwidth_stage("tcp", 0)

    yamls = _client_job_yamls(k8s, kind="iperf3")
    assert len(yamls) == 2
    epochs = [_extract_epoch(y) for y in yamls]
    assert epochs[0] == epochs[1] == int(_FIXED_NOW) + 15

    # Both client Jobs waited with the padded timeout.
    timeouts = _wait_timeouts(k8s)
    assert timeouts == [_BASE_WAIT + 15, _BASE_WAIT + 15]


def test_bandwidth_tcp_and_udp_pick_distinct_epochs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """The barrier is stage-scoped: TCP and UDP in one iteration differ."""
    clock = [_FIXED_NOW]

    def _time() -> float:
        clock[0] += 1.0  # advance a second between stage calls
        return clock[0]

    monkeypatch.setattr("kube_autotuner.benchmark.runner.time.time", _time)
    runner, k8s = _runner(
        tmp_path,
        extra_sources=["src1"],
        sync_window_seconds=15,
    )
    runner._run_bandwidth_stage("tcp", 0)
    runner._run_bandwidth_stage("udp", 0)

    yamls = _client_job_yamls(k8s, kind="iperf3")
    epochs = {_extract_epoch(y) for y in yamls}
    # Two distinct epoch literals (one per stage), both non-None.
    assert None not in epochs
    assert len(epochs) == 2


def test_single_client_stage_has_no_barrier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    monkeypatch.setattr(
        "kube_autotuner.benchmark.runner.time.time",
        lambda: _FIXED_NOW,
    )
    runner, k8s = _runner(tmp_path, extra_sources=[], sync_window_seconds=15)
    runner._run_bandwidth_stage("tcp", 0)

    yamls = _client_job_yamls(k8s, kind="iperf3")
    assert len(yamls) == 1
    assert _extract_epoch(yamls[0]) is None
    assert "sleep" not in yamls[0]
    # Unpadded base wait budget.
    assert _wait_timeouts(k8s) == [_BASE_WAIT]


def test_single_source_with_two_slots_activates_barrier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """``clients_per_node=2`` on one source activates the wall-clock barrier."""
    monkeypatch.setattr(
        "kube_autotuner.benchmark.runner.time.time",
        lambda: _FIXED_NOW,
    )
    runner, k8s = _runner(
        tmp_path,
        extra_sources=[],
        sync_window_seconds=15,
        iperf_args=IperfSection(clients_per_node=2),
    )
    runner._run_bandwidth_stage("tcp", 0)

    yamls = _client_job_yamls(k8s, kind="iperf3")
    assert len(yamls) == 2
    epochs = [_extract_epoch(y) for y in yamls]
    # Both slots share one shared start_at_epoch.
    assert epochs[0] == epochs[1] == int(_FIXED_NOW) + 15
    # And both wait calls use the padded budget.
    assert _wait_timeouts(k8s) == [_BASE_WAIT + 15, _BASE_WAIT + 15]


def test_sync_window_seconds_zero_disables_barrier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    monkeypatch.setattr(
        "kube_autotuner.benchmark.runner.time.time",
        lambda: _FIXED_NOW,
    )
    runner, k8s = _runner(
        tmp_path,
        extra_sources=["src1"],
        sync_window_seconds=0,
    )
    runner._run_bandwidth_stage("tcp", 0)

    yamls = _client_job_yamls(k8s, kind="iperf3")
    assert len(yamls) == 2
    assert all(_extract_epoch(y) is None for y in yamls)
    assert all("sleep" not in y for y in yamls)
    # Both wait calls use the base (unpadded) budget.
    assert _wait_timeouts(k8s) == [_BASE_WAIT, _BASE_WAIT]


@pytest.mark.parametrize("workload", ["saturation", "fixed_qps"])
def test_latency_stage_shares_epoch_across_clients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    workload: Literal["saturation", "fixed_qps"],
) -> None:
    monkeypatch.setattr(
        "kube_autotuner.benchmark.runner.time.time",
        lambda: _FIXED_NOW,
    )
    runner, k8s = _runner(
        tmp_path,
        extra_sources=["src1"],
        sync_window_seconds=15,
    )
    runner._run_latency_stage(0, workload=workload)

    yamls = _client_job_yamls(k8s, kind="fortio")
    assert len(yamls) == 2
    epochs = [_extract_epoch(y) for y in yamls]
    assert epochs[0] == epochs[1] == int(_FIXED_NOW) + 15
    assert _wait_timeouts(k8s) == [_BASE_WAIT + 15, _BASE_WAIT + 15]
