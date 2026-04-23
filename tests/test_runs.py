"""Tests for :mod:`kube_autotuner.runs`.

The real ``NodeLease`` and ``BenchmarkRunner`` are patched out; the
sysctl backend is injected directly as a ``MagicMock`` through
:class:`~kube_autotuner.runs.RunContext`. Zone resolution is left real
and driven by stubbing ``client.get_node_zone`` on the injected
:class:`K8sClient` mock, so the tests exercise the real
``_resolve_zones`` helper. Nothing imports Ax, so these tests pass
without the ``optimize`` dependency group.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import typer

from kube_autotuner import runs
from kube_autotuner.experiment import ExperimentConfig
from kube_autotuner.models import (
    BenchmarkConfig,
    BenchmarkResult,
    IterationResults,
    NodePair,
    ParamSpace,
    ResumeMetadata,
    SysctlParam,
    TrialLog,
    TrialResult,
)
from kube_autotuner.sysctl.params import PARAM_SPACE

if TYPE_CHECKING:
    from pathlib import Path


def _results() -> IterationResults:
    return IterationResults(
        bench=[
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=9_000_000_000,
                retransmits=5,
                bytes_sent=33_750_000_000,
            ),
        ],
        latency=[],
    )


def _snapshot(names):
    return {n: ("6.1.0-talos" if n == "kernel.osrelease" else "212992") for n in names}


def _client_stub() -> MagicMock:
    """Return a K8sClient mock whose ``get_node_zone`` returns ``""``."""
    client = MagicMock()
    client.get_node_zone.return_value = ""
    return client


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
        client=_client_stub(),
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
        client=_client_stub(),
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
        client=_client_stub(),
        backend=backend,
        output=out,
    )
    runs.run_baseline(ctx)

    requested = backend.snapshot.call_args[0][0]
    for name in PARAM_SPACE.param_names():
        assert name in requested
    assert "kernel.osrelease" in requested
    assert mock_lease_cls.call_count == 2


# --- resume + fresh semantics -------------------------------------------


def _prior_trial(sysctl_value: int = 1048576) -> TrialResult:
    return TrialResult(
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values={"net.core.rmem_max": sysctl_value},
        config=BenchmarkConfig(duration=1, iterations=1),
        results=[
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=9e9,
                retransmits=5,
                bytes_sent=1_000_000_000,
            ),
        ],
    )


def _optimize_exp(out: Path, n_trials: int = 5) -> ExperimentConfig:
    return ExperimentConfig.model_validate({
        "mode": "optimize",
        "nodes": {"sources": ["a"], "target": "b"},
        "benchmark": {"duration": 1, "iterations": 1},
        "optimize": {"n_trials": n_trials, "n_sobol": 2},
        "output": str(out),
    })


def _seed_prior_results(
    output: Path,
    exp: ExperimentConfig,
    trials: list[TrialResult],
    *,
    n_sobol: int | None = None,
) -> None:
    assert exp.optimize is not None
    for t in trials:
        TrialLog.append(output, t)
    TrialLog.write_resume_metadata(
        output,
        ResumeMetadata(
            objectives=exp.objectives,
            param_space=exp.effective_param_space(),
            benchmark=exp.benchmark,
            n_sobol=n_sobol if n_sobol is not None else exp.optimize.n_sobol,
        ),
    )


@patch("kube_autotuner.optimizer.OptimizationLoop")
def test_run_optimize_loads_prior_trials_when_jsonl_present(
    mock_loop_cls,
    tmp_path: Path,
):
    out = tmp_path / "opt.jsonl"
    exp = _optimize_exp(out, n_trials=5)
    priors = [_prior_trial(), _prior_trial(2097152)]
    _seed_prior_results(out, exp, priors)

    loop = MagicMock()
    loop.run.return_value = [*priors, _prior_trial(4194304)]
    loop.prior_count = 2
    loop.pareto_front.return_value = []
    mock_loop_cls.return_value = loop

    ctx = runs.RunContext(
        exp=exp,
        client=_client_stub(),
        backend=MagicMock(),
        output=out,
    )
    runs.run_optimize(ctx)

    kwargs = mock_loop_cls.call_args.kwargs
    assert len(kwargs["prior_trials"]) == 2
    assert kwargs["n_trials"] == 5


@patch("kube_autotuner.optimizer.OptimizationLoop")
def test_run_optimize_fresh_moves_prior_files(
    mock_loop_cls,
    tmp_path: Path,
):
    out = tmp_path / "opt.jsonl"
    exp = _optimize_exp(out, n_trials=5)
    priors = [_prior_trial()]
    _seed_prior_results(out, exp, priors)

    loop = MagicMock()
    loop.run.return_value = []
    loop.prior_count = 0
    loop.pareto_front.return_value = []
    mock_loop_cls.return_value = loop

    ctx = runs.RunContext(
        exp=exp,
        client=_client_stub(),
        backend=MagicMock(),
        output=out,
    )
    runs.run_optimize(ctx, fresh=True)

    kwargs = mock_loop_cls.call_args.kwargs
    assert kwargs["prior_trials"] == []

    # Archived files present with the exact naming scheme.
    jsonl_backups = [
        p for p in tmp_path.glob("opt.jsonl.*.bak") if ".meta.json" not in p.name
    ]
    assert len(jsonl_backups) == 1
    meta_backups = list(tmp_path.glob("opt.jsonl.meta.json.*.bak"))
    assert len(meta_backups) == 1
    # Current output does not carry any prior trials.
    assert not out.exists() or out.stat().st_size == 0


def test_run_optimize_incompatible_param_space_raises(tmp_path: Path):
    out = tmp_path / "opt.jsonl"
    exp = _optimize_exp(out, n_trials=5)
    assert exp.optimize is not None
    priors = [_prior_trial()]
    # Write a sidecar with a different param_space.
    for t in priors:
        TrialLog.append(out, t)
    TrialLog.write_resume_metadata(
        out,
        ResumeMetadata(
            objectives=exp.objectives,
            param_space=ParamSpace(
                params=[
                    SysctlParam(
                        name="other.sysctl",
                        values=[1, 2],
                        param_type="int",
                    ),
                ],
            ),
            benchmark=exp.benchmark,
            n_sobol=exp.optimize.n_sobol,
        ),
    )

    ctx = runs.RunContext(
        exp=exp,
        client=_client_stub(),
        backend=MagicMock(),
        output=out,
    )
    with pytest.raises(typer.BadParameter, match="param_space"):
        runs.run_optimize(ctx)


def test_run_optimize_incompatible_n_sobol_raises(tmp_path: Path):
    out = tmp_path / "opt.jsonl"
    exp = _optimize_exp(out, n_trials=5)
    assert exp.optimize is not None
    priors = [_prior_trial()]
    _seed_prior_results(out, exp, priors, n_sobol=exp.optimize.n_sobol + 1)

    ctx = runs.RunContext(
        exp=exp,
        client=_client_stub(),
        backend=MagicMock(),
        output=out,
    )
    with pytest.raises(typer.BadParameter, match="n_sobol"):
        runs.run_optimize(ctx)


@patch("kube_autotuner.optimizer.OptimizationLoop")
def test_run_optimize_short_circuits_when_budget_met(
    mock_loop_cls,
    tmp_path: Path,
    caplog,
):
    out = tmp_path / "opt.jsonl"
    exp = _optimize_exp(out, n_trials=2)
    priors = [_prior_trial(), _prior_trial(2097152)]
    _seed_prior_results(out, exp, priors)

    loop = MagicMock()
    loop.prior_count = 2
    loop.pareto_front.return_value = []
    mock_loop_cls.return_value = loop

    caplog.set_level("INFO")
    ctx = runs.RunContext(
        exp=exp,
        client=_client_stub(),
        backend=MagicMock(),
        output=out,
    )
    runs.run_optimize(ctx)

    loop.run.assert_not_called()
    assert any("Budget already met" in rec.message for rec in caplog.records)


def test_run_optimize_missing_sidecar_raises(tmp_path: Path):
    out = tmp_path / "opt.jsonl"
    exp = _optimize_exp(out, n_trials=5)
    # Write JSONL but not sidecar.
    for t in [_prior_trial()]:
        TrialLog.append(out, t)
    ctx = runs.RunContext(
        exp=exp,
        client=_client_stub(),
        backend=MagicMock(),
        output=out,
    )
    with pytest.raises(typer.BadParameter, match="sidecar"):
        runs.run_optimize(ctx)


def test_run_baseline_writes_resume_meta_without_n_sobol(tmp_path: Path):
    out = tmp_path / "r.jsonl"
    exp = ExperimentConfig.model_validate({
        "mode": "baseline",
        "nodes": {"sources": ["a"], "target": "b"},
        "benchmark": {"duration": 1, "iterations": 1},
        "output": str(out),
    })

    with (
        patch("kube_autotuner.runs.NodeLease"),
        patch("kube_autotuner.runs.BenchmarkRunner") as mock_runner_cls,
    ):
        backend = MagicMock()
        backend.snapshot.side_effect = _snapshot
        mock_runner_cls.return_value.run.return_value = _results()

        ctx = runs.RunContext(
            exp=exp,
            client=_client_stub(),
            backend=backend,
            output=out,
        )
        runs.run_baseline(ctx)

    loaded = TrialLog.load_resume_metadata(out)
    assert loaded is not None
    assert loaded.n_sobol is None


def test_run_trial_writes_resume_meta_without_n_sobol(tmp_path: Path):
    out = tmp_path / "r.jsonl"
    exp = ExperimentConfig.model_validate({
        "mode": "trial",
        "nodes": {"sources": ["a"], "target": "b"},
        "benchmark": {"duration": 1, "iterations": 1},
        "trial": {"sysctls": {"net.core.rmem_max": "16777216"}},
        "output": str(out),
    })

    with (
        patch("kube_autotuner.runs.NodeLease"),
        patch("kube_autotuner.runs.BenchmarkRunner") as mock_runner_cls,
    ):
        backend = MagicMock()
        backend.snapshot.return_value = {
            "net.core.rmem_max": "67108864",
            "kernel.osrelease": "6.1.0",
        }
        mock_runner_cls.return_value.run.return_value = _results()

        ctx = runs.RunContext(
            exp=exp,
            client=_client_stub(),
            backend=backend,
            output=out,
        )
        runs.run_trial(ctx)

    loaded = TrialLog.load_resume_metadata(out)
    assert loaded is not None
    assert loaded.n_sobol is None


class TestFormatMeanSem:
    """Pin the mean / SEM formatting rule used by the verification table."""

    def test_small_sem_renders_with_scientific_notation(self) -> None:
        """Regression: SEM around 5e-5 must not collapse to ``"0"``.

        The live table uses fixed-point for the mean to avoid
        scientific notation at seconds-scale p99 values, but the SEM
        keeps ``.3g`` so tiny uncertainties still show their
        magnitude rather than rounding to zero.
        """
        row = {"latency_p99": 0.010, "latency_p99_sem": 5e-5}
        out = runs._format_mean_sem(row, "latency_p99", scale=1.0)
        assert out.startswith("0.01 ± ")
        sem_str = out.split("± ", 1)[1]
        assert sem_str != "0"
        assert float(sem_str) == pytest.approx(5e-5)

    def test_nan_mean_returns_n_a(self) -> None:
        """NaN means render as ``"n/a"`` regardless of SEM."""
        row = {"latency_p99": float("nan"), "latency_p99_sem": 1e-5}
        assert runs._format_mean_sem(row, "latency_p99") == "n/a"

    def test_missing_column_returns_n_a(self) -> None:
        """An absent column is treated as missing and renders ``"n/a"``."""
        assert runs._format_mean_sem({}, "latency_p99") == "n/a"

    def test_mean_uses_fixed_point_at_seconds_scale(self) -> None:
        """Mean values scaled into the seconds range stay fixed-point."""
        row = {"throughput": 9.4e9, "throughput_sem": 1e7}
        # scale=1e-6 converts bits/sec -> Mbps: 9400 ± 10.
        out = runs._format_mean_sem(row, "throughput", scale=1e-6)
        assert "e" not in out.split(" ± ")[0]
        assert out.startswith("9400 ± ")
