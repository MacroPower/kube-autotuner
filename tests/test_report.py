"""Unit tests for the consolidated HTML report."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

pd = pytest.importorskip("pandas")
go = pytest.importorskip("plotly.graph_objects")

from kube_autotuner import report  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


def _minimal_section(
    hw: str,
    *,
    with_importance: bool = True,
    n_figures: int = 6,
) -> dict[str, Any]:
    fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 4, 9])])
    figures = [(f"Chart {i}", go.Figure(fig)) for i in range(n_figures)]
    importance_df = (
        pd.DataFrame(
            [
                {
                    "param": "net.core.rmem_max",
                    "category": "tcp_buffer",
                    "spearman_r": 0.8,
                    "rf_importance": 0.4,
                },
                {
                    "param": "net.ipv4.tcp_wmem",
                    "category": "tcp_buffer",
                    "spearman_r": 0.5,
                    "rf_importance": 0.2,
                },
                {
                    "param": "net.ipv4.tcp_congestion_control",
                    "category": "congestion",
                    "spearman_r": 0.1,
                    "rf_importance": 0.05,
                },
            ],
        )
        if with_importance
        else pd.DataFrame()
    )
    return {
        "hardware_class": hw,
        "trial_count": 12,
        "pareto_count": 3,
        "topology": None,
        "recommendations": [
            {
                "rank": 1,
                "trial_id": f"trial-{hw}-abc",
                "sysctl_values": {
                    "net.core.rmem_max": "67108864",
                    "net.ipv4.tcp_congestion_control": "bbr",
                },
                "mean_throughput": 4.2e10,
                "mean_cpu": 33.9,
                "mean_memory": 67108864,
                "retransmit_rate": 1e-8,
                "score": 0.95,
            },
        ],
        "importance": importance_df,
        "figures": figures,
    }


@pytest.mark.parametrize("hw_classes", [["10g"], ["10g", "1g"]])
def test_write_index_html_renders_all_sections(
    tmp_path: Path,
    hw_classes: list[str],
) -> None:
    sections = [_minimal_section(hw) for hw in hw_classes]
    path = report.write_index_html(tmp_path, sections)

    assert path == tmp_path / "index.html"
    assert path.exists()
    html_text = path.read_text()

    assert "<title>kube-autotuner analysis report</title>" in html_text
    assert "memory (MiB)" in html_text
    for hw in hw_classes:
        assert f"href='#hw-{hw}'" in html_text
        assert f"id='hw-{hw}'" in html_text
        assert f"Hardware class: {hw}" in html_text
        assert f"trial-{hw}-abc" in html_text


def test_write_index_html_slugifies_hardware_class_in_ids(tmp_path: Path) -> None:
    section = _minimal_section("10G NIC", n_figures=1)
    path = report.write_index_html(tmp_path, [section])
    html_text = path.read_text()

    assert "href='#hw-10g-nic'" in html_text
    assert "id='hw-10g-nic'" in html_text
    assert 'id="fig-10g-nic-chart-0"' in html_text
    assert "Hardware class: 10G NIC" in html_text


def test_write_index_html_handles_empty_importance(tmp_path: Path) -> None:
    section = _minimal_section("10g", with_importance=False, n_figures=4)
    path = report.write_index_html(tmp_path, [section])
    html_text = path.read_text()

    assert "Parameter importance unavailable" in html_text
    for i in range(4):
        assert f"Chart {i}" in html_text


def test_write_index_html_uses_cdn_not_inlined_plotly(tmp_path: Path) -> None:
    sections = [
        _minimal_section("10g", n_figures=2),
        _minimal_section("1g", n_figures=2),
    ]
    path = report.write_index_html(tmp_path, sections)
    html_text = path.read_text()

    cdn_matches = re.findall(r'src="https://cdn\.plot\.ly/[^"]+"', html_text)
    assert len(cdn_matches) == 1, (
        f"expected exactly one CDN script tag, got {cdn_matches}"
    )

    size_bytes = path.stat().st_size
    assert size_bytes < 500_000, (
        f"index.html is {size_bytes} bytes, plotly.js likely inlined"
    )
