"""Tests for the kube-autotuner Typer CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from kube_autotuner import __version__
from kube_autotuner.cli import app
from kube_autotuner.experiment import PreflightResult
from kube_autotuner.sysctl.fake import FakeSysctlBackend

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


# --- help / version ------------------------------------------------------


def test_root_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert result.stdout.strip()
    for sub in ("baseline", "trial", "optimize", "run", "sysctl"):
        assert sub in result.stdout


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_baseline_help() -> None:
    result = runner.invoke(app, ["baseline", "--help"])
    assert result.exit_code == 0
    assert result.stdout.strip()
    assert "--source" in result.stdout


def test_trial_help() -> None:
    result = runner.invoke(app, ["trial", "--help"])
    assert result.exit_code == 0
    assert result.stdout.strip()
    assert "--param" in result.stdout


def test_optimize_help() -> None:
    result = runner.invoke(app, ["optimize", "--help"])
    assert result.exit_code == 0
    assert result.stdout.strip()
    assert "--n-trials" in result.stdout


def test_run_help() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert result.stdout.strip()
    assert "--config" in result.stdout


def test_sysctl_help() -> None:
    result = runner.invoke(app, ["sysctl", "--help"])
    assert result.exit_code == 0
    assert result.stdout.strip()
    for sub in ("get", "set"):
        assert sub in result.stdout


def test_sysctl_get_help() -> None:
    result = runner.invoke(app, ["sysctl", "get", "--help"])
    assert result.exit_code == 0
    assert result.stdout.strip()
    assert "--node" in result.stdout


def test_sysctl_set_help() -> None:
    result = runner.invoke(app, ["sysctl", "set", "--help"])
    assert result.exit_code == 0
    assert result.stdout.strip()
    assert "--node" in result.stdout


# --- positive invocations ------------------------------------------------


def _fake_backend(tmp_path: Path) -> FakeSysctlBackend:
    return FakeSysctlBackend("node-a", tmp_path / "sysctl_state.json")


def test_baseline_invokes_run_baseline(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    with (
        patch("kube_autotuner.cli.K8sClient") as client_cls,
        patch("kube_autotuner.cli._resolve_backend") as resolve,
        patch("kube_autotuner.cli.runs.run_baseline") as run_baseline,
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = _fake_backend(tmp_path)

        result = runner.invoke(
            app,
            [
                "baseline",
                "--source",
                "a",
                "--target",
                "b",
                "--duration",
                "1",
                "--iterations",
                "1",
                "--output",
                str(out),
                "--backend",
                "fake",
                "--fake-state-path",
                str(tmp_path / "sysctl_state.json"),
            ],
        )
    assert result.exit_code == 0, result.output
    run_baseline.assert_called_once()
    ctx = run_baseline.call_args.args[0]
    assert ctx.exp.mode == "baseline"
    assert ctx.exp.nodes.sources == ["a"]
    assert ctx.exp.nodes.target == "b"
    assert ctx.output == out


def test_trial_parses_params_and_invokes_run_trial(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    with (
        patch("kube_autotuner.cli.K8sClient") as client_cls,
        patch("kube_autotuner.cli._resolve_backend") as resolve,
        patch("kube_autotuner.cli.runs.run_trial") as run_trial,
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = _fake_backend(tmp_path)

        result = runner.invoke(
            app,
            [
                "trial",
                "--source",
                "a",
                "--target",
                "b",
                "--duration",
                "1",
                "--iterations",
                "1",
                "--output",
                str(out),
                "-p",
                "net.core.rmem_max=16777216",
                "-p",
                "net.core.wmem_max=16777216",
            ],
        )
    assert result.exit_code == 0, result.output
    run_trial.assert_called_once()
    ctx = run_trial.call_args.args[0]
    assert ctx.exp.mode == "trial"
    assert ctx.exp.trial is not None
    assert ctx.exp.trial.sysctls == {
        "net.core.rmem_max": "16777216",
        "net.core.wmem_max": "16777216",
    }


def test_trial_rejects_malformed_param(tmp_path: Path) -> None:
    with (
        patch("kube_autotuner.cli.K8sClient"),
        patch("kube_autotuner.cli._resolve_backend"),
        patch("kube_autotuner.cli.runs.run_trial") as run_trial,
    ):
        result = runner.invoke(
            app,
            [
                "trial",
                "--source",
                "a",
                "--target",
                "b",
                "--output",
                str(tmp_path / "r.jsonl"),
                "-p",
                "no-equals-sign",
            ],
        )
    assert result.exit_code == 1
    assert "Invalid param format" in result.output
    run_trial.assert_not_called()


def test_optimize_invokes_run_optimize(tmp_path: Path) -> None:
    out = tmp_path / "opt.jsonl"
    with (
        patch("kube_autotuner.cli.K8sClient") as client_cls,
        patch("kube_autotuner.cli._resolve_backend") as resolve,
        patch("kube_autotuner.cli.runs.run_optimize") as run_optimize,
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = _fake_backend(tmp_path)

        result = runner.invoke(
            app,
            [
                "optimize",
                "--source",
                "a",
                "--target",
                "b",
                "--duration",
                "1",
                "--iterations",
                "1",
                "--output",
                str(out),
                "--n-trials",
                "4",
                "--n-sobol",
                "2",
            ],
        )
    assert result.exit_code == 0, result.output
    run_optimize.assert_called_once()
    ctx = run_optimize.call_args.args[0]
    assert ctx.exp.mode == "optimize"
    assert ctx.exp.optimize is not None
    assert ctx.exp.optimize.n_trials == 4
    assert ctx.exp.optimize.n_sobol == 2


def test_run_loads_yaml_and_dispatches_to_run_baseline(tmp_path: Path) -> None:
    config = tmp_path / "exp.yaml"
    out = tmp_path / "results.jsonl"
    config.write_text(
        f"mode: baseline\n"
        f"nodes:\n"
        f"  sources: [a]\n"
        f"  target: b\n"
        f"benchmark:\n"
        f"  duration: 1\n"
        f"  iterations: 1\n"
        f"output: {out}\n",
    )
    with (
        patch("kube_autotuner.cli.K8sClient") as client_cls,
        patch("kube_autotuner.cli._resolve_backend") as resolve,
        patch("kube_autotuner.cli.runs.run_baseline") as run_baseline,
        patch(
            "kube_autotuner.cli.ExperimentConfig.preflight",
            return_value=[],
        ),
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = _fake_backend(tmp_path)

        result = runner.invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == 0, result.output
    run_baseline.assert_called_once()


def test_run_reports_preflight_failures(tmp_path: Path) -> None:
    config = tmp_path / "exp.yaml"
    config.write_text(
        "mode: baseline\n"
        "nodes:\n"
        "  sources: [a]\n"
        "  target: b\n"
        "benchmark:\n"
        "  duration: 1\n"
        "  iterations: 1\n",
    )
    failing = PreflightResult(
        name="nodes-exist",
        passed=False,
        detail="node 'a' missing",
    )
    with (
        patch("kube_autotuner.cli.K8sClient"),
        patch(
            "kube_autotuner.cli.ExperimentConfig.preflight",
            return_value=[failing],
        ),
        patch("kube_autotuner.cli.runs.run_baseline") as run_baseline,
    ):
        result = runner.invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == 2
    assert "nodes-exist" in result.output
    assert "node 'a' missing" in result.output
    run_baseline.assert_not_called()


# --- sysctl get / set with the fake backend ------------------------------


def test_sysctl_set_apply_roundtrip(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    result = runner.invoke(
        app,
        [
            "sysctl",
            "set",
            "--node",
            "node-a",
            "--backend",
            "fake",
            "--fake-state-path",
            str(state),
            "-p",
            "net.core.rmem_max=16777216",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Applied 1 sysctl(s) on node-a" in result.output
    persisted = json.loads(state.read_text())
    assert persisted == {"net.core.rmem_max": "16777216"}


def test_sysctl_get_reads_persisted_state(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"net.core.rmem_max": "67108864"}))

    result = runner.invoke(
        app,
        [
            "sysctl",
            "get",
            "--node",
            "node-a",
            "--backend",
            "fake",
            "--fake-state-path",
            str(state),
            "-p",
            "net.core.rmem_max",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {"net.core.rmem_max": "67108864"}
