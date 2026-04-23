"""Unit tests for kube_autotuner.plots dark styling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("plotly.graph_objects")

from kube_autotuner import plots  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


def _tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_id": ["a", "b"],
            "mean_tcp_throughput": [1.0, 2.0],
            "mean_cpu": [10.0, 20.0],
        },
    )


def test_plot_pareto_2d_applies_dark_template() -> None:
    df = _tiny_df()
    fig = plots.plot_pareto_2d(df, df, "mean_tcp_throughput", "mean_cpu")
    assert fig.layout.template.layout.paper_bgcolor == "rgb(17,17,17)"
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.font.color == "#abb2bf"


def test_write_standalone_html_injects_dark_page_style(tmp_path: Path) -> None:
    df = _tiny_df()
    fig = plots.plot_pareto_2d(df, df, "mean_tcp_throughput", "mean_cpu")
    out = tmp_path / "fig.html"
    plots.write_standalone_html(fig, out)
    text = out.read_text()
    assert "background:#21252b" in text
    assert "color-scheme:dark" in text
    assert "</head>" in text
