"""Tests for ``--collect-host-state`` plumbing through the four CLI commands."""

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


def _common_patches(tmp_path: Path):
    """Return (patch pair, fake backend) for the CLI command boundary.

    Returns:
        A ``((client_patch, resolve_patch), fake_backend)`` tuple: two
        uncommitted ``unittest.mock.patch`` objects that the caller
        enters in a ``with`` block, plus a pre-built
        :class:`FakeSysctlBackend` the ``_resolve_backend`` patch can
        return.
    """
    return (
        patch("kube_autotuner.cli.K8sClient"),
        patch("kube_autotuner.cli._resolve_backend"),
    ), _fake_backend(tmp_path)


def test_baseline_flag_threaded_into_context(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    (client_patch, resolve_patch), fake = _common_patches(tmp_path)
    with (
        client_patch as client_cls,
        resolve_patch as resolve,
        patch("kube_autotuner.cli.runs.run_baseline") as run_baseline,
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = fake

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
                "--collect-host-state",
            ],
        )
    assert result.exit_code == 0, result.output
    ctx = run_baseline.call_args.args[0]
    assert ctx.collect_host_state is True


def test_trial_flag_threaded_into_context(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    (client_patch, resolve_patch), fake = _common_patches(tmp_path)
    with (
        client_patch as client_cls,
        resolve_patch as resolve,
        patch("kube_autotuner.cli.runs.run_trial") as run_trial,
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = fake

        result = runner.invoke(
            app,
            [
                "trial",
                "--source",
                "a",
                "--target",
                "b",
                "-p",
                "net.core.rmem_max=16777216",
                "--duration",
                "1",
                "--iterations",
                "1",
                "--output",
                str(out),
                "-H",
            ],
        )
    assert result.exit_code == 0, result.output
    ctx = run_trial.call_args.args[0]
    assert ctx.collect_host_state is True


def test_optimize_flag_threaded_into_context(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    (client_patch, resolve_patch), fake = _common_patches(tmp_path)
    with (
        client_patch as client_cls,
        resolve_patch as resolve,
        patch("kube_autotuner.cli.runs.run_optimize") as run_optimize,
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = fake

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
                "1",
                "--n-sobol",
                "1",
                "--collect-host-state",
            ],
        )
    assert result.exit_code == 0, result.output
    ctx = run_optimize.call_args.args[0]
    assert ctx.collect_host_state is True


def test_run_yaml_flag_threaded_into_context(tmp_path: Path) -> None:
    """The YAML-driven ``run`` command also accepts the flag."""
    out = tmp_path / "r.jsonl"
    config = tmp_path / "exp.yaml"
    config.write_text(
        yaml.safe_dump({
            "mode": "baseline",
            "nodes": {"sources": ["a"], "target": "b"},
            "benchmark": {"duration": 1, "iterations": 1},
            "output": str(out),
        })
    )
    client_mock = MagicMock()
    client_mock.preflight = MagicMock(return_value=[])
    (client_patch, resolve_patch), fake = _common_patches(tmp_path)
    with (
        client_patch as client_cls,
        resolve_patch as resolve,
        patch("kube_autotuner.cli.runs.run_baseline") as run_baseline,
    ):
        client_cls.return_value = client_mock
        resolve.return_value = fake

        result = runner.invoke(
            app,
            [
                "run",
                "-c",
                str(config),
                "--collect-host-state",
            ],
        )
    assert result.exit_code == 0, result.output
    ctx = run_baseline.call_args.args[0]
    assert ctx.collect_host_state is True


def test_baseline_default_is_false(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    (client_patch, resolve_patch), fake = _common_patches(tmp_path)
    with (
        client_patch as client_cls,
        resolve_patch as resolve,
        patch("kube_autotuner.cli.runs.run_baseline") as run_baseline,
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = fake

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
            ],
        )
    assert result.exit_code == 0, result.output
    ctx = run_baseline.call_args.args[0]
    assert ctx.collect_host_state is False
