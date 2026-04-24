"""End-to-end integration test for the trial CLI command."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from kube_autotuner.cli import app

if TYPE_CHECKING:
    from collections.abc import Callable

    from kube_autotuner.k8s.client import K8sClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(300),
]


def test_trial_snapshots_applies_benchmarks_restores(
    kubeconfig_env: str,  # noqa: ARG001 - activates KUBECONFIG env var
    k8s_client: K8sClient,  # noqa: ARG001 - fixture kept for parity with other e2e tests
    node_names: dict[str, str],
    test_namespace: str,
    fake_sysctl_env: Path,  # noqa: ARG001 - activates fake backend env vars
    tmp_path: Path,
    write_experiment_yaml: Callable[..., Path],
) -> None:
    output_file = tmp_path / "trial"
    config_path = write_experiment_yaml(
        node_names=node_names,
        namespace=test_namespace,
        output=output_file,
        trial_sysctls={"net.core.rmem_max": "16777216"},
    )

    runner = CliRunner()
    result = runner.invoke(app, ["trial", str(config_path)])
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    assert "Snapshotted" in result.stderr
    assert "Applied 1 sysctl(s)" in result.stderr
    assert "Restored original sysctls" in result.stderr

    from kube_autotuner.trial_log import TrialLog  # noqa: PLC0415

    trials = TrialLog.load(output_file)
    assert len(trials) == 1
    trial = trials[0]
    assert trial.sysctl_values == {"net.core.rmem_max": "16777216"}
    # Each iteration now runs both bw-tcp and bw-udp.
    assert {r.mode for r in trial.results} == {"tcp", "udp"}
    assert trial.results[0].mode == "tcp"  # bw-tcp runs first
    assert trial.results[0].bits_per_second > 0
    # UDP jitter is the signal that the new bw-udp stage actually ran.
    assert trial.mean_udp_jitter() > 0.0

    # Fake-state witness: after restore, the state file holds the pre-trial
    # default that snapshot captured (not the trial's 16777216), proving
    # `setter.restore` actually ran instead of the stdout print firing alone.
    state_path = Path(os.environ["KUBE_AUTOTUNER_SYSCTL_FAKE_STATE"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["net.core.rmem_max"] == "212992"
