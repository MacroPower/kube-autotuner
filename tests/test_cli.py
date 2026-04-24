"""Tests for the kube-autotuner Typer CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from kube_autotuner import __version__
from kube_autotuner.cli import app
from kube_autotuner.experiment import PreflightResult
from kube_autotuner.progress import NullObserver, RichProgressObserver
from kube_autotuner.sysctl.fake import FakeSysctlBackend

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


_BASELINE_YAML = """\
nodes:
  sources: [a]
  target: b
benchmark:
  duration: 1
  iterations: 1
output: {output}
"""

_TRIAL_YAML = """\
nodes:
  sources: [a]
  target: b
benchmark:
  duration: 1
  iterations: 1
trial:
  sysctls:
    net.core.rmem_max: "16777216"
    net.core.wmem_max: "16777216"
output: {output}
"""

_OPTIMIZE_YAML = """\
nodes:
  sources: [a]
  target: b
benchmark:
  duration: 1
  iterations: 1
optimize:
  nTrials: 4
  nSobol: 2
output: {output}
"""


def _write_yaml(tmp_path: Path, body: str) -> Path:
    config = tmp_path / "exp.yaml"
    config.write_text(body)
    return config


def _fake_backend(tmp_path: Path) -> FakeSysctlBackend:
    return FakeSysctlBackend("node-a", tmp_path / "sysctl_state.json")


# --- help / version ------------------------------------------------------


def test_root_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert result.stdout.strip()
    for sub in ("baseline", "trial", "optimize", "analyze", "sysctl"):
        assert sub in result.stdout
    assert "run " not in result.stdout
    assert "--no-progress" in result.stdout


def test_run_command_no_longer_exists() -> None:
    """The deprecated ``run`` command must be gone after the refactor."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code != 0


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_baseline_help_shows_only_runtime_flags(tmp_path: Path) -> None:  # noqa: ARG001
    result = runner.invoke(app, ["baseline", "--help"])
    assert result.exit_code == 0
    assert "CONFIG_PATH" in result.stdout.upper() or "config_path" in result.stdout
    assert "--backend" in result.stdout
    assert "--fake-state-path" in result.stdout
    assert "--source" not in result.stdout
    assert "--duration" not in result.stdout


def test_trial_help_shows_only_runtime_flags(tmp_path: Path) -> None:  # noqa: ARG001
    result = runner.invoke(app, ["trial", "--help"])
    assert result.exit_code == 0
    assert "--backend" in result.stdout
    assert "--param" not in result.stdout


def test_optimize_help_shows_fresh_and_runtime_flags(
    tmp_path: Path,  # noqa: ARG001
) -> None:
    result = runner.invoke(app, ["optimize", "--help"])
    assert result.exit_code == 0
    assert "--fresh" in result.stdout
    assert "--backend" in result.stdout
    assert "--n-trials" not in result.stdout
    assert "--verification-trials" not in result.stdout


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


def test_no_progress_yields_null_observer(tmp_path: Path) -> None:
    out = tmp_path / "results"
    config = _write_yaml(tmp_path, _BASELINE_YAML.format(output=out))
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

        result = runner.invoke(
            app,
            [
                "--no-progress",
                "baseline",
                str(config),
                "--backend",
                "fake",
                "--fake-state-path",
                str(tmp_path / "sysctl_state.json"),
            ],
        )
    assert result.exit_code == 0, result.output
    ctx = run_baseline.call_args.args[0]
    assert isinstance(ctx.observer, NullObserver)
    # CliRunner captures a non-TTY buffer, so no rich escape codes leak in.
    assert "\x1b[" not in result.output


def test_progress_enabled_under_forced_terminal(tmp_path: Path) -> None:
    out = tmp_path / "results"
    config = _write_yaml(tmp_path, _BASELINE_YAML.format(output=out))
    with (
        patch("kube_autotuner.cli.K8sClient") as client_cls,
        patch("kube_autotuner.cli._resolve_backend") as resolve,
        patch("kube_autotuner.cli.runs.run_baseline") as run_baseline,
        patch("kube_autotuner.cli.Console") as console_cls,
        patch(
            "kube_autotuner.cli.ExperimentConfig.preflight",
            return_value=[],
        ),
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = _fake_backend(tmp_path)
        forced = MagicMock()
        forced.is_terminal = True
        console_cls.return_value = forced

        result = runner.invoke(
            app,
            [
                "baseline",
                str(config),
                "--backend",
                "fake",
                "--fake-state-path",
                str(tmp_path / "sysctl_state.json"),
            ],
        )
    assert result.exit_code == 0, result.output
    ctx = run_baseline.call_args.args[0]
    assert isinstance(ctx.observer, RichProgressObserver)


def test_baseline_invokes_run_baseline(tmp_path: Path) -> None:
    out = tmp_path / "results"
    config = _write_yaml(tmp_path, _BASELINE_YAML.format(output=out))
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

        result = runner.invoke(
            app,
            [
                "baseline",
                str(config),
                "--backend",
                "fake",
                "--fake-state-path",
                str(tmp_path / "sysctl_state.json"),
            ],
        )
    assert result.exit_code == 0, result.output
    run_baseline.assert_called_once()
    ctx = run_baseline.call_args.args[0]
    assert ctx.exp.nodes.sources == ["a"]
    assert ctx.exp.nodes.target == "b"
    assert ctx.output == out


def test_trial_loads_yaml_and_invokes_run_trial(tmp_path: Path) -> None:
    out = tmp_path / "results"
    config = _write_yaml(tmp_path, _TRIAL_YAML.format(output=out))
    with (
        patch("kube_autotuner.cli.K8sClient") as client_cls,
        patch("kube_autotuner.cli._resolve_backend") as resolve,
        patch("kube_autotuner.cli.runs.run_trial") as run_trial,
        patch(
            "kube_autotuner.cli.ExperimentConfig.preflight",
            return_value=[],
        ),
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = _fake_backend(tmp_path)

        result = runner.invoke(app, ["trial", str(config)])
    assert result.exit_code == 0, result.output
    run_trial.assert_called_once()
    ctx = run_trial.call_args.args[0]
    assert ctx.exp.trial is not None
    assert ctx.exp.trial.sysctls == {
        "net.core.rmem_max": "16777216",
        "net.core.wmem_max": "16777216",
    }


def test_trial_without_trial_section_errors(tmp_path: Path) -> None:
    """``trial`` rejects a YAML lacking a ``trial:`` section."""
    out = tmp_path / "results"
    config = _write_yaml(tmp_path, _BASELINE_YAML.format(output=out))
    with (
        patch("kube_autotuner.cli.K8sClient"),
        patch("kube_autotuner.cli._resolve_backend"),
        patch("kube_autotuner.cli.runs.run_trial") as run_trial,
    ):
        result = runner.invoke(app, ["trial", str(config)])
    assert result.exit_code == 2
    assert "trial" in result.output.lower()
    run_trial.assert_not_called()


def test_optimize_invokes_run_optimize(tmp_path: Path) -> None:
    out = tmp_path / "opt"
    config = _write_yaml(tmp_path, _OPTIMIZE_YAML.format(output=out))
    with (
        patch("kube_autotuner.cli.K8sClient") as client_cls,
        patch("kube_autotuner.cli._resolve_backend") as resolve,
        patch("kube_autotuner.cli.runs.run_optimize") as run_optimize,
        patch(
            "kube_autotuner.cli.ExperimentConfig.preflight",
            return_value=[],
        ),
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = _fake_backend(tmp_path)

        result = runner.invoke(app, ["optimize", str(config)])
    assert result.exit_code == 0, result.output
    run_optimize.assert_called_once()
    ctx = run_optimize.call_args.args[0]
    assert ctx.exp.optimize is not None
    assert ctx.exp.optimize.n_trials == 4
    assert ctx.exp.optimize.n_sobol == 2


def test_optimize_without_optimize_section_errors(tmp_path: Path) -> None:
    """``optimize`` rejects a YAML lacking an ``optimize:`` section."""
    out = tmp_path / "opt"
    config = _write_yaml(tmp_path, _BASELINE_YAML.format(output=out))
    with (
        patch("kube_autotuner.cli.K8sClient"),
        patch("kube_autotuner.cli._resolve_backend"),
        patch("kube_autotuner.cli.runs.run_optimize") as run_optimize,
    ):
        result = runner.invoke(app, ["optimize", str(config)])
    assert result.exit_code == 2
    assert "optimize" in result.output.lower()
    run_optimize.assert_not_called()


def test_baseline_reports_preflight_failures(tmp_path: Path) -> None:
    out = tmp_path / "results"
    config = _write_yaml(tmp_path, _BASELINE_YAML.format(output=out))
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
        result = runner.invoke(app, ["baseline", str(config)])
    assert result.exit_code == 2
    assert "nodes-exist" in result.output
    assert "node 'a' missing" in result.output
    run_baseline.assert_not_called()


def test_optimize_forwards_fresh_flag(tmp_path: Path) -> None:
    out = tmp_path / "opt"
    config = _write_yaml(tmp_path, _OPTIMIZE_YAML.format(output=out))
    with (
        patch("kube_autotuner.cli.K8sClient") as client_cls,
        patch("kube_autotuner.cli._resolve_backend") as resolve,
        patch("kube_autotuner.cli.runs.run_optimize") as run_optimize,
        patch(
            "kube_autotuner.cli.ExperimentConfig.preflight",
            return_value=[],
        ),
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = _fake_backend(tmp_path)

        result = runner.invoke(app, ["optimize", str(config), "--fresh"])
    assert result.exit_code == 0, result.output
    run_optimize.assert_called_once()
    assert run_optimize.call_args.kwargs == {"fresh": True}


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


# --- _format_top_recommendation tolerance -------------------------------


def test_format_top_recommendation_skips_missing_metrics() -> None:
    """Rows missing TCP/UDP fields render the remaining metrics only."""
    from kube_autotuner.cli import _format_top_recommendation  # noqa: PLC0415, PLC2701

    rendered = _format_top_recommendation({
        "mean_tcp_throughput": 2.5e9,
        "tcp_retransmit_rate": 0.1,
    })
    assert "Mbps TCP" in rendered
    assert "retx/GB" in rendered
    assert "UDP" not in rendered
    assert "rps" not in rendered


def test_format_top_recommendation_renders_full_row() -> None:
    """With every metric present the formatter covers every field."""
    from kube_autotuner.cli import _format_top_recommendation  # noqa: PLC0415, PLC2701

    rendered = _format_top_recommendation({
        "mean_tcp_throughput": 2.5e9,
        "mean_udp_throughput": 1.0e9,
        "tcp_retransmit_rate": 0.1,
        "udp_loss_rate": 0.01,
        "mean_udp_jitter": 5e-4,
        "mean_rps": 12345.6,
        "mean_latency_p50": 1e-3,
        "mean_latency_p90": 5e-3,
        "mean_latency_p99": 1e-2,
    })
    assert "Mbps TCP" in rendered
    assert "Mbps UDP" in rendered
    assert "UDP loss" in rendered
    assert "rps" in rendered
    assert "p99" in rendered
