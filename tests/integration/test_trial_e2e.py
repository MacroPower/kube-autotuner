"""End-to-end integration test for the trial CLI command."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from kube_autotuner.cli import app
from kube_autotuner.models import TrialResult

if TYPE_CHECKING:
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
) -> None:
    output_file = tmp_path / "trial.jsonl"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "trial",
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
            "-p",
            "net.core.rmem_max=16777216",
        ],
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    assert "Snapshotted" in result.stderr
    assert "Applied 1 sysctl(s)" in result.stderr
    assert "Restored original sysctls" in result.stderr

    lines = output_file.read_text().strip().splitlines()
    assert len(lines) == 1

    trial = TrialResult.model_validate_json(lines[0])
    assert trial.sysctl_values == {"net.core.rmem_max": "16777216"}
    assert trial.results[0].bits_per_second > 0

    # Fake-state witness: after restore, the state file holds the pre-trial
    # default that snapshot captured (not the trial's 16777216), proving
    # `setter.restore` actually ran instead of the stdout print firing alone.
    state_path = Path(os.environ["KUBE_AUTOTUNER_SYSCTL_FAKE_STATE"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["net.core.rmem_max"] == "212992"
