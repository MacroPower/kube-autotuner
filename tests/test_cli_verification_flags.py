"""Tests for ``optimize.verification*`` YAML threading on the ``optimize`` command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner
import yaml

from kube_autotuner.cli import app
from kube_autotuner.sysctl.fake import FakeSysctlBackend

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _fake_backend(tmp_path: Path) -> FakeSysctlBackend:
    return FakeSysctlBackend("node-a", tmp_path / "sysctl_state.json")


def _optimize_yaml(out: Path, *, optimize_overrides: dict) -> str:
    body = {
        "nodes": {"sources": ["a"], "target": "b"},
        "benchmark": {"iterations": 1},
        "optimize": {"nTrials": 4, "nSobol": 2, **optimize_overrides},
        "output": str(out),
    }
    return yaml.safe_dump(body)


def test_verification_yaml_threads_into_optimize_section(tmp_path: Path) -> None:
    out = tmp_path / "opt"
    config = tmp_path / "exp.yaml"
    config.write_text(
        _optimize_yaml(
            out,
            optimize_overrides={
                "verificationTrials": 3,
                "verificationTopK": 2,
            },
        ),
    )
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
    ctx = run_optimize.call_args.args[0]
    assert ctx.exp.optimize is not None
    assert ctx.exp.optimize.verification_trials == 3
    assert ctx.exp.optimize.verification_top_k == 2


def test_defaults_when_yaml_omits_verification_keys(tmp_path: Path) -> None:
    """Omitted keys leave the ``OptimizeSection`` defaults (0 disables verification)."""
    out = tmp_path / "opt"
    config = tmp_path / "exp.yaml"
    config.write_text(_optimize_yaml(out, optimize_overrides={}))
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
    ctx = run_optimize.call_args.args[0]
    assert ctx.exp.optimize is not None
    assert ctx.exp.optimize.verification_trials == 0
    assert ctx.exp.optimize.verification_top_k == 3
