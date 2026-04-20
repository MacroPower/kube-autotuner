"""Plotly chart generation for offline trial analysis.

``plotly`` and ``pandas`` live in the optional ``analysis`` dependency
group. Every imported symbol is referenced lazily (either in annotations
under ``TYPE_CHECKING`` or inside a function body) so that
``import kube_autotuner.plots`` succeeds under the base ``dev`` sync.
Runtime helpers raise :class:`RuntimeError` with the
``uv sync --group analysis`` hint when the group is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go

_ANALYSIS_HINT = "install analysis group: uv sync --group analysis"


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
        A :class:`plotly.graph_objects.Figure` rendering a 4-objective
        scatter matrix (throughput, CPU, memory, retransmits).
    """
    px, _ = _require_plotly()
    cols = ["mean_throughput", "mean_cpu", "mean_memory", "total_retransmits"]
    plot_df = df[[*cols, "trial_id"]].copy()
    plot_df["pareto"] = pareto_mask.map({True: "pareto", False: "other"})
    fig = px.scatter_matrix(
        plot_df,
        dimensions=cols,
        color="pareto",
        color_discrete_map={"pareto": "#EF553B", "other": "#BABBBD"},
        hover_data=["trial_id"],
        title="Objective Space (Pareto-optimal highlighted)",
    )
    fig.update_traces(diagonal_visible=False, marker={"size": 5})
    fig.update_layout(width=900, height=900)
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

    hover_template = f"%{{text}}<br>{x}=%{{x:.3g}}<br>{y}=%{{y:.3g}}<extra></extra>"

    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="markers",
            marker={"color": "#BABBBD", "size": 6},
            name="all trials",
            text=df["trial_id"],
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
            text=frontier["trial_id"],
            hovertemplate=hover_template,
        ),
    )

    fig.update_layout(
        title=f"Pareto Frontier: {x} vs {y}",
        xaxis_title=x,
        yaxis_title=y,
        width=800,
        height=600,
    )
    return fig


def plot_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of parameter importance, coloured by category.

    Lazy-imports ``plotly`` and raises :exc:`RuntimeError` with the
    ``uv sync --group analysis`` hint when the group is missing.

    Args:
        importance_df: Frame produced by
            :func:`kube_autotuner.analysis.parameter_importance`.
        top_n: Number of top-importance parameters to include.

    Returns:
        A :class:`plotly.graph_objects.Figure` rendering the bar chart.
    """
    px, _ = _require_plotly()
    top = importance_df.head(top_n).iloc[::-1]
    fig = px.bar(
        top,
        x="rf_importance",
        y="param",
        color="category",
        orientation="h",
        title=f"Parameter Importance (top {min(top_n, len(top))})",
    )
    fig.update_layout(
        yaxis_title="",
        xaxis_title="Random Forest Importance",
        width=900,
        height=max(400, 30 * len(top)),
    )
    return fig


def plot_param_heatmap(
    df: pd.DataFrame,  # noqa: ARG001 - kept for API symmetry with callers
    pareto_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    top_n: int = 10,
) -> go.Figure:
    """Heatmap of top-N params across Pareto trials (min-max normalized).

    Args:
        df: All-trial frame (unused; accepted for call-signature
            parity with sibling plot helpers).
        pareto_df: Frame produced by
            :func:`kube_autotuner.analysis.pareto_front`.
        importance_df: Frame produced by
            :func:`kube_autotuner.analysis.parameter_importance`.
        top_n: Number of top-importance parameters to include.

    Lazy-imports ``plotly`` and raises :exc:`RuntimeError` with the
    ``uv sync --group analysis`` hint when the group is missing.

    Returns:
        A :class:`plotly.graph_objects.Figure` rendering the
        normalized heatmap, or an empty figure when no parameters are
        available.
    """
    _, go = _require_plotly()
    top_params = importance_df.head(top_n)["param"].tolist()
    if not top_params:
        return go.Figure()

    subset = pareto_df[top_params].copy()
    for col in subset.columns:
        lo, hi = subset[col].min(), subset[col].max()
        if hi != lo:
            subset[col] = (subset[col] - lo) / (hi - lo)
        else:
            subset[col] = 0.5

    fig = go.Figure(
        data=go.Heatmap(
            z=subset.T.values,
            x=[f"trial {tid}" for tid in pareto_df["trial_id"]],
            y=top_params,
            colorscale="RdYlGn",
            hovertemplate=(
                "param=%{y}<br>trial=%{x}<br>normalized=%{z:.2f}<extra></extra>"
            ),
        ),
    )
    fig.update_layout(
        title=f"Pareto Configurations: Top {top_n} Parameters (normalized)",
        width=max(600, 80 * len(pareto_df)),
        height=max(400, 35 * len(top_params)),
    )
    return fig
