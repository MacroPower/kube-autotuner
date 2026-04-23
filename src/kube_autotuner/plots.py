"""Plotly chart generation for offline trial analysis.

``plotly`` and ``pandas`` live in the optional ``analysis`` dependency
group. Every imported symbol is referenced lazily (either in annotations
under ``TYPE_CHECKING`` or inside a function body) so that
``import kube_autotuner.plots`` succeeds under the base ``dev`` sync.
Runtime helpers raise :class:`RuntimeError` with the
``uv sync --group analysis`` hint when the group is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go

_ANALYSIS_HINT = "install analysis group: uv sync --group analysis"

_DARK_LAYOUT: dict[str, Any] = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#abb2bf"},
}

_STANDALONE_STYLE = (
    "<style>html,body{background:#21252b;color:#abb2bf;margin:0;"
    "color-scheme:dark}</style>"
)


def _require_plotly() -> tuple[Any, Any]:
    """Return the plotly sub-modules this module uses.

    Returns:
        A ``(plotly.express, plotly.graph_objects)`` tuple.

    Raises:
        RuntimeError: ``plotly`` is not installed.
    """
    try:
        import plotly.express as px  # noqa: PLC0415
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(_ANALYSIS_HINT) from e
    return px, go


def plot_pareto_scatter_matrix(
    df: pd.DataFrame,
    pareto_mask: pd.Series,
) -> go.Figure:
    """Scatter matrix of the objective columns with Pareto points highlighted.

    Lazy-imports ``plotly`` and raises :exc:`RuntimeError` with the
    ``uv sync --group analysis`` hint when the group is missing.

    Args:
        df: Frame produced by
            :func:`kube_autotuner.analysis.trials_to_dataframe`.
        pareto_mask: Boolean Series aligned with ``df``; ``True`` marks
            Pareto-optimal rows.

    Returns:
        A :class:`plotly.graph_objects.Figure` rendering the subset of
        the eight objective columns (TCP throughput, UDP throughput,
        CPU, node memory, CNI memory, TCP retransmit_rate,
        udp_loss_rate, UDP jitter) that have at least one non-null
        value in ``df``. Columns with no data (e.g. ``mean_cni_memory``
        when ``cni.enabled=false``) are skipped so the matrix does
        not show a dead axis.
    """
    px, _ = _require_plotly()
    candidate_cols = [
        "mean_tcp_throughput",
        "mean_udp_throughput",
        "mean_cpu",
        "mean_node_memory",
        "mean_cni_memory",
        "tcp_retransmit_rate",
        "udp_loss_rate",
        "mean_udp_jitter_ms",
    ]
    cols = [c for c in candidate_cols if c in df.columns and df[c].notna().any()]
    plot_df = df[[*cols, "trial_id"]].copy()
    plot_df["pareto"] = pareto_mask.map({True: "pareto", False: "other"})
    fig = px.scatter_matrix(
        plot_df,
        dimensions=cols,
        color="pareto",
        color_discrete_map={"pareto": "#EF553B", "other": "#BABBBD"},
        hover_data=["trial_id"],
        custom_data=["trial_id"],
        title="Objective Space (Pareto-optimal highlighted)",
    )
    fig.update_traces(diagonal_visible=False, marker={"size": 5})
    fig.update_layout(width=900, height=900, **_DARK_LAYOUT)
    return fig


def plot_pareto_2d(
    df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    x: str,
    y: str,
) -> go.Figure:
    """2-D scatter of all trials with the Pareto frontier drawn as a line.

    Lazy-imports ``plotly`` and raises :exc:`RuntimeError` with the
    ``uv sync --group analysis`` hint when the group is missing.

    Args:
        df: Frame produced by
            :func:`kube_autotuner.analysis.trials_to_dataframe`.
        pareto_df: Frame produced by
            :func:`kube_autotuner.analysis.pareto_front`.
        x: Column name to place on the x-axis.
        y: Column name to place on the y-axis.

    Returns:
        A :class:`plotly.graph_objects.Figure` rendering the scatter
        and the connected Pareto frontier.
    """
    _, go = _require_plotly()
    fig = go.Figure()

    hover_template = (
        f"trial %{{customdata}}<br>{x}=%{{x:.3g}}<br>{y}=%{{y:.3g}}<extra></extra>"
    )

    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="markers",
            marker={"color": "#BABBBD", "size": 6},
            name="all trials",
            customdata=df["trial_id"],
            hovertemplate=hover_template,
        ),
    )

    frontier = pareto_df.sort_values(x)
    fig.add_trace(
        go.Scatter(
            x=frontier[x],
            y=frontier[y],
            mode="markers+lines",
            marker={"color": "#EF553B", "size": 9},
            line={"color": "#EF553B", "width": 2},
            name="pareto front",
            customdata=frontier["trial_id"],
            hovertemplate=hover_template,
        ),
    )

    fig.update_layout(
        title=f"Pareto Frontier: {x} vs {y}",
        xaxis_title=x,
        yaxis_title=y,
        width=800,
        height=600,
        **_DARK_LAYOUT,
    )
    return fig


def write_standalone_html(fig: go.Figure, path: str | Path) -> None:
    """Write a dark-themed stand-alone figure HTML file.

    Plotly's ``write_html`` emits a default ``<body>`` with a white UA
    background; ``paper_bgcolor`` only fills the plot area, so the
    page margins around the figure would stay white. This helper
    injects a small ``<style>`` block into the ``<head>`` so the
    margins are dark too and the figure blends into the surrounding
    page.

    Args:
        fig: The Plotly figure to serialize.
        path: Destination path for the HTML file.
    """
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    html = html.replace("</head>", _STANDALONE_STYLE + "</head>", 1)
    Path(path).write_text(html, encoding="utf-8")
