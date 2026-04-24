"""End-to-end integration test for the baseline CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from kube_autotuner.cli import app

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(300),
]


def test_baseline_produces_results(
    kubeconfig_env: str,  # noqa: ARG001 - activates KUBECONFIG env var
    node_names: dict[str, str],
    test_namespace: str,
    fake_sysctl_env: Path,  # noqa: ARG001 - activates fake backend env vars
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "results"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "baseline",
            "--source",
            node_names["source"],
            "--target",
            node_names["target"],
            "--hardware-class",
            "1g",
            "--ip-family-policy",
            "SingleStack",
            "--namespace",
            test_namespace,
            "--duration",
            "5",
            "--iterations",
            "1",
            "--output",
            str(output_file),
        ],
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"

    from kube_autotuner.trial_log import TrialLog  # noqa: PLC0415

    trials = TrialLog.load(output_file)
    assert len(trials) == 1
    trial = trials[0]
    # Each iteration runs both bw-tcp and bw-udp (no `modes:` dimension).
    assert len(trial.results) == 2
    assert {r.mode for r in trial.results} == {"tcp", "udp"}
    assert trial.results[0].mode == "tcp"  # bw-tcp runs first
    assert trial.results[0].bits_per_second > 0
    udp = next(r for r in trial.results if r.mode == "udp")
    assert udp.jitter is not None
    assert trial.sysctl_values
