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
    "mean_tcp_throughput",
    "mean_udp_throughput",
    "tcp_retransmit_rate",
    "udp_loss_rate",
    "mean_udp_jitter",
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
    # Labels that are always produced: iperf3 bw-tcp emits throughput
    # and tcp_retransmit_rate; bw-udp emits udp_loss_rate and jitter.
    required_labels = [
        "Objective space (scatter matrix)",
        "Pareto: mean_tcp_throughput vs tcp_retransmit_rate",
        "Pareto: mean_tcp_throughput vs mean_udp_jitter",
        "Pareto: mean_udp_throughput vs udp_loss_rate",
        "Pareto: mean_udp_throughput vs mean_udp_jitter",
    ]
    for label in required_labels:
        assert label in index_text, f"missing figure label in index.html: {label}"
