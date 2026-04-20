"""End-to-end integration test for the optimize CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from kube_autotuner.cli import app
from kube_autotuner.k8s.lease import NodeLease
from kube_autotuner.models import TrialResult

if TYPE_CHECKING:
    from pathlib import Path

    from kube_autotuner.k8s.client import K8sClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(900),
]


def test_optimize_runs_trials_and_writes_jsonl(
    kubeconfig_env: str,  # noqa: ARG001 - activates KUBECONFIG env var
    k8s_client: K8sClient,
    node_names: dict[str, str],
    test_namespace: str,
    fake_sysctl_env: Path,  # noqa: ARG001 - activates fake backend env vars
    tmp_path: Path,
) -> None:
    pytest.importorskip("ax")

    output_file = tmp_path / "optimize.jsonl"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "optimize",
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
            "--n-trials",
            "2",
            "--n-sobol",
            "2",
            "--output",
            str(output_file),
        ],
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    assert "Completed 2 trials" in result.output

    lines = output_file.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        trial = TrialResult.model_validate_json(line)
        assert trial.sysctl_values
        assert trial.results[0].bits_per_second > 0

    lease_name = f"{NodeLease.LEASE_PREFIX}-{node_names['target']}"
    assert k8s_client.get_json("lease", lease_name, test_namespace) is None
