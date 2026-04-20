"""Tests for :mod:`kube_autotuner.runs`.

The real ``NodeLease`` and ``BenchmarkRunner`` are patched out; the
sysctl backend is injected directly as a ``MagicMock`` through
:class:`~kube_autotuner.runs.RunContext`. Zone resolution is left real
and driven by stubbing ``kubectl.get_node_zone`` on the injected
kubectl mock, so the tests exercise the real ``_resolve_zones``
helper. Nothing imports Ax, so these tests pass without the
``optimize`` dependency group.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from kube_autotuner import runs
from kube_autotuner.experiment import ExperimentConfig
from kube_autotuner.models import BenchmarkResult
from kube_autotuner.sysctl.params import PARAM_SPACE

if TYPE_CHECKING:
    from pathlib import Path


def _results() -> list[BenchmarkResult]:
    return [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=9_000_000_000,
            retransmits=5,
            cpu_utilization_percent=30.0,
        ),
    ]


def _snapshot(names):
    return {n: ("6.1.0-talos" if n == "kernel.osrelease" else "212992") for n in names}


def _kubectl_stub() -> MagicMock:
    """Return a kubectl mock whose ``get_node_zone`` returns ``""``."""
    kubectl = MagicMock()
    kubectl.get_node_zone.return_value = ""
    return kubectl


@patch("kube_autotuner.runs.NodeLease")
@patch("kube_autotuner.runs.BenchmarkRunner")
def test_run_baseline_threads_iperf_args_and_patches(
    mock_runner_cls,
    mock_lease_cls,
    tmp_path: Path,
):
    backend = MagicMock()
    backend.snapshot.side_effect = _snapshot
    mock_runner_cls.return_value.run.return_value = _results()

    out = tmp_path / "r.jsonl"
    exp = ExperimentConfig.model_validate({
        "mode": "baseline",
        "nodes": {"sources": ["a"], "target": "b"},
        "benchmark": {"duration": 1, "iterations": 1},
        "iperf": {"client": {"extra_args": ["-Z"]}},
        "patches": [
            {"target": {"kind": "Job"}, "patch": {"spec": {"replicas": 1}}},
        ],
        "output": str(out),
    })
    ctx = runs.RunContext(
        exp=exp,
        kubectl=_kubectl_stub(),
        backend=backend,
        output=out,
    )
    runs.run_baseline(ctx)

    kwargs = mock_runner_cls.call_args.kwargs
    assert kwargs["iperf_args"].client.extra_args == ["-Z"]
    assert len(kwargs["patches"]) == 1
    assert kwargs["patches"][0].target.kind == "Job"
    assert mock_lease_cls.call_count == 2

    trial = json.loads(out.read_text().strip())
    assert trial["kernel_version"] == "6.1.0-talos"


@patch("kube_autotuner.runs.NodeLease")
@patch("kube_autotuner.runs.BenchmarkRunner")
def test_run_trial_snapshots_only_applied_keys(
    mock_runner_cls,
    mock_lease_cls,
    tmp_path: Path,
):
    backend = MagicMock()
    backend.snapshot.return_value = {
        "net.core.rmem_max": "67108864",
        "kernel.osrelease": "6.1.0",
    }
    mock_runner_cls.return_value.run.return_value = _results()

    out = tmp_path / "r.jsonl"
    exp = ExperimentConfig.model_validate({
        "mode": "trial",
        "nodes": {"sources": ["a"], "target": "b"},
        "benchmark": {"duration": 1, "iterations": 1},
        "trial": {"sysctls": {"net.core.rmem_max": "16777216"}},
        "output": str(out),
    })
    ctx = runs.RunContext(
        exp=exp,
        kubectl=_kubectl_stub(),
        backend=backend,
        output=out,
    )
    runs.run_trial(ctx)

    backend.snapshot.assert_called_once_with(
        ["net.core.rmem_max", "kernel.osrelease"],
    )
    backend.apply.assert_called_once_with({"net.core.rmem_max": "16777216"})
    backend.restore.assert_called_once_with({"net.core.rmem_max": "67108864"})
    assert mock_lease_cls.call_count == 2


@patch("kube_autotuner.runs.NodeLease")
@patch("kube_autotuner.runs.BenchmarkRunner")
def test_run_baseline_snapshots_full_param_space(
    mock_runner_cls,
    mock_lease_cls,
    tmp_path: Path,
):
    backend = MagicMock()
    backend.snapshot.side_effect = _snapshot
    mock_runner_cls.return_value.run.return_value = _results()

    out = tmp_path / "r.jsonl"
    exp = ExperimentConfig.model_validate({
        "mode": "baseline",
        "nodes": {"sources": ["a"], "target": "b"},
        "benchmark": {"duration": 1, "iterations": 1},
        "output": str(out),
    })
    ctx = runs.RunContext(
        exp=exp,
        kubectl=_kubectl_stub(),
        backend=backend,
        output=out,
    )
    runs.run_baseline(ctx)

    requested = backend.snapshot.call_args[0][0]
    for name in PARAM_SPACE.param_names():
        assert name in requested
    assert "kernel.osrelease" in requested
    assert mock_lease_cls.call_count == 2
