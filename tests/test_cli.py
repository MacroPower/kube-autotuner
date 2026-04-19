"""Tests for the kube-autotuner CLI."""

from typer.testing import CliRunner

from kube_autotuner import __version__
from kube_autotuner.cli import app

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_tune_placeholder() -> None:
    result = runner.invoke(app, ["tune", "node-1"])
    assert result.exit_code == 0
    assert "node-1" in result.stdout
