"""End-to-end integration test for the analyze CLI command (chained off baseline)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from kube_autotuner.cli import app

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(600),
]

REQUIRED_RECOMMENDATION_KEYS = {
    "rank",
    "trial_id",
    "sysctl_values",
    "mean_throughput",
    "mean_cpu",
    "mean_node_memory",
    "mean_cni_memory",
    "retransmit_rate",
    "score",
}


def test_analyze_generates_reports_from_baseline_output(
    kubeconfig_env: str,  # noqa: ARG001 - activates KUBECONFIG env var
    node_names: dict[str, str],
    test_namespace: str,
    fake_sysctl_env: Path,  # noqa: ARG001 - activates fake backend env vars
    tmp_path: Path,
) -> None:
    trials_file = tmp_path / "trials.jsonl"
    analysis_dir = tmp_path / "analysis"

    runner = CliRunner()
    baseline_result = runner.invoke(
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
            str(trials_file),
        ],
    )
    assert baseline_result.exit_code == 0, f"baseline failed:\n{baseline_result.output}"

    analyze_result = runner.invoke(
        app,
        [
            "analyze",
            "-i",
            str(trials_file),
            "-o",
            str(analysis_dir),
            "--hardware-class",
            "1g",
            "--top-n",
            "1",
        ],
    )
    assert analyze_result.exit_code == 0, f"analyze failed:\n{analyze_result.output}"
    assert "=== 1g (1 trials," in analyze_result.output

    hw_dir = analysis_dir / "1g"

    pareto_html = hw_dir / "pareto_scatter_matrix.html"
    assert pareto_html.exists()
    assert pareto_html.stat().st_size > 0

    recs = json.loads((hw_dir / "recommendations.json").read_text(encoding="utf-8"))
    assert isinstance(recs, list)
    assert len(recs) == 1
    assert REQUIRED_RECOMMENDATION_KEYS.issubset(recs[0])

    importance = json.loads((hw_dir / "importance.json").read_text(encoding="utf-8"))
    assert importance == []

    index_html = analysis_dir / "index.html"
    assert index_html.exists()
    assert index_html.stat().st_size > 0
    index_text = index_html.read_text(encoding="utf-8")
    assert "Hardware class: 1g" in index_text
    expected_labels = [
        "Objective space (scatter matrix)",
        "Pareto: mean_throughput vs mean_cpu",
        "Pareto: mean_throughput vs mean_node_memory",
        "Pareto: mean_throughput vs mean_cni_memory",
        "Pareto: mean_throughput vs retransmit_rate",
        "Pareto: mean_cpu vs mean_node_memory",
        "Pareto: mean_cpu vs mean_cni_memory",
        "Pareto: mean_cpu vs retransmit_rate",
    ]
    for label in expected_labels:
        assert label in index_text, f"missing figure label in index.html: {label}"
