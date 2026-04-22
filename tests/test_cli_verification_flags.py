"""Tests for the ``--verification-*`` CLI flags on the ``optimize`` command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from kube_autotuner.cli import app
from kube_autotuner.sysctl.fake import FakeSysctlBackend

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _fake_backend(tmp_path: Path) -> FakeSysctlBackend:
    return FakeSysctlBackend("node-a", tmp_path / "sysctl_state.json")


def test_verification_flags_thread_into_optimize_section(tmp_path: Path) -> None:
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
                "--verification-trials",
                "3",
                "--verification-top-k",
                "2",
            ],
        )
    assert result.exit_code == 0, result.output
    ctx = run_optimize.call_args.args[0]
    assert ctx.exp.optimize is not None
    assert ctx.exp.optimize.verification_trials == 3
    assert ctx.exp.optimize.verification_top_k == 2


def test_defaults_when_flags_omitted(tmp_path: Path) -> None:
    """Omitted flags leave the defaults (0 disables verification)."""
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
                "2",
                "--n-sobol",
                "1",
            ],
        )
    assert result.exit_code == 0, result.output
    ctx = run_optimize.call_args.args[0]
    assert ctx.exp.optimize is not None
    assert ctx.exp.optimize.verification_trials == 0
    assert ctx.exp.optimize.verification_top_k == 3
