"""Interactive HTML report combining per-hardware-class analysis output.

Each hardware-class section embeds its Pareto-frontier rows as JSON and
a vanilla-JS module renders a slider panel, a re-ranked top-N table,
and lazy score-decomposition charts. Weights are applied in the
browser via a direct port of
:func:`kube_autotuner.scoring.score_rows`; the Pareto frontier itself
is weight-invariant so no Python round-trip is needed.

``pandas`` and ``plotly`` live in the optional ``analysis`` dependency
group. All runtime uses are lazy-imported inside function bodies so
that ``import kube_autotuner.report.render`` remains cheap under the
base ``dev`` sync. Runtime helpers raise :class:`RuntimeError` with the
``uv sync --group analysis`` hint when the group is missing.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

_ANALYSIS_HINT = "install analysis group: uv sync --group analysis"

_TOP_IMPORTANCE_ROWS = 20

_MIN_AXIS_COLUMNS = 2

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


_STYLE = """
:root {
  color-scheme: dark;
  --bg: #21252b;
  --panel: #282c34;
  --panel-2: #2c313a;
  --border: #3e4451;
  --divider: #3e4451;
  --fg: #abb2bf;
  --fg-muted: #828997;
  --fg-dim: #5c6370;
  --accent: #61afef;
  --accent-fg: #282c34;
  --pos: #98c379;
  --neg: #e06c75;
  --pos-bar: rgba(152, 195, 121, 0.28);
  --neg-bar: rgba(224, 108, 117, 0.28);
  --imp-bar: rgba(97, 175, 239, 0.32);
  --top-rank: rgba(229, 192, 123, 0.16);
}
body { font-family: Inter, -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
       margin: 0; color: var(--fg); background: var(--bg); }
header { padding: 1rem 2rem; background: var(--panel);
         border-bottom: 1px solid var(--border); }
header h1 { margin: 0; font-size: 1.4rem; color: var(--fg); }
nav.top { position: sticky; top: 0; z-index: 10;
          background: color-mix(in srgb, var(--panel) 75%, transparent);
          backdrop-filter: blur(8px);
          border-bottom: 1px solid var(--border); padding: 0.5rem 2rem; }
nav.top a { margin-right: 1rem; text-decoration: none; color: var(--accent); }
main { padding: 1rem 2rem 4rem; max-width: 1280px; margin: 0 auto; }
section.hw { margin-top: 2rem; padding-top: 1rem;
             border-top: 2px solid var(--divider); }
section.fig { margin: 1.5rem 0; background: var(--panel);
              border: 1px solid var(--border); border-radius: 8px;
              padding: 0.75rem 1rem;
              box-shadow: 0 1px 2px rgba(0,0,0,0.4); }
h2 { font-size: 1.2rem; }
h3 { font-size: 1rem; color: var(--fg-muted); }
table.report-table { border-collapse: collapse; margin: 0.5rem 0 1rem;
                     font-size: 0.9rem; background: var(--panel); width: 100%; }
table.report-table th, table.report-table td {
    border: 1px solid var(--border); padding: 4px 8px; text-align: left;
    white-space: nowrap; }
table.report-table th { background: var(--panel-2); }
table.report-table td.numeric { text-align: right; font-variant-numeric: tabular-nums; }
table.report-table tbody tr:hover {
    background: color-mix(in srgb, var(--panel) 92%, var(--accent)); }
tr.top-rank { background: var(--top-rank); }
details { margin: 0.25rem 0 0.75rem; }
summary { cursor: pointer; color: var(--accent); }
pre { background: var(--panel-2); padding: 0.5rem; overflow-x: auto;
      font-size: 0.85rem; border-radius: 6px; color: var(--fg); }
.meta { color: var(--fg-dim); font-size: 0.9rem; }
.panel { background: var(--panel); border: 1px solid var(--border);
         border-radius: 8px; padding: 0.75rem 1rem; margin: 0.75rem 0 1.25rem;
         box-shadow: 0 1px 2px rgba(0,0,0,0.4); }
.panel .presets { margin-bottom: 0.5rem; }
.panel .presets button {
    margin-right: 0.35rem; margin-bottom: 0.25rem;
    padding: 0.25rem 0.6rem; font-size: 0.85rem;
    border: 1px solid var(--border); background: var(--panel-2);
    border-radius: 6px; color: var(--fg);
    cursor: pointer; transition: background 120ms ease; }
.panel .presets button:hover {
    background: color-mix(in srgb, var(--panel-2) 80%, white); }
.panel .presets button:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 1px; }
.panel .presets button.active {
    background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
.panel .sliders {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 0.5rem 1rem;
    margin: 0.5rem 0; }
.panel .slider-row {
    display: grid; grid-template-columns: 1fr auto; align-items: center;
    gap: 0.25rem; font-size: 0.85rem; }
.panel .slider-row label { color: var(--fg-muted); }
.panel .slider-row .weight-value {
    font-variant-numeric: tabular-nums; color: var(--accent); font-weight: 600; }
.panel .slider-row input[type=range] { grid-column: 1 / -1; width: 100%; }
.panel .topn-row { margin-top: 0.5rem; font-size: 0.85rem; }
.panel .topn-row input { width: 4em; }
.decomposition-wrapper { padding: 0.5rem 0.75rem; }
table.decomposition-table { table-layout: fixed; width: 100%; margin: 0; }
table.decomposition-table col.metric-w { width: 38%; }
table.decomposition-table col.norm-w { width: 18%; }
table.decomposition-table col.weight-w { width: 14%; }
table.decomposition-table col.contrib-w { width: 30%; }
table.decomposition-table td.metric-col { font-family: ui-monospace,
    SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.82rem;
    white-space: normal; word-break: break-all; }
.axis-controls { display: flex; gap: 0.75rem; margin-bottom: 0.5rem;
    font-size: 0.85rem; align-items: center; flex-wrap: wrap; }
.axis-controls label { display: inline-flex; align-items: center;
    gap: 0.35rem; color: var(--fg-muted); }
.axis-controls select { background: var(--panel-2); color: var(--fg);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 0.2rem 0.4rem; font-size: 0.85rem; }
.axis-chart { width: 100%; min-height: 480px; }
.importance-controls { margin: 0.5rem 0 0.75rem; font-size: 0.85rem;
    display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; }
.importance-controls .meta { margin: 0; }
.target-buttons button {
    margin-right: 0.25rem; margin-bottom: 0.25rem;
    padding: 0.2rem 0.55rem; font-size: 0.8rem;
    border: 1px solid var(--border); background: var(--panel-2);
    border-radius: 6px; cursor: pointer; color: var(--fg);
    transition: background 120ms ease; }
.target-buttons button:hover {
    background: color-mix(in srgb, var(--panel-2) 80%, white); }
.target-buttons button:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 1px; }
.target-buttons button.active {
    background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
table.importance-table { table-layout: fixed; width: 100%; }
table.importance-table td.param-col { font-family: ui-monospace, SFMono-Regular,
    Menlo, Consolas, monospace; font-size: 0.82rem; white-space: normal;
    word-break: break-all; }
table.importance-table td.category-col { color: var(--fg-muted); }
table.report-table td.bar-cell { position: relative; padding: 4px 8px;
    text-align: right; font-variant-numeric: tabular-nums; }
table.importance-table col.param-w { width: 42%; }
table.importance-table col.category-w { width: 18%; }
table.importance-table col.corr-w { width: 20%; }
table.importance-table col.imp-w { width: 20%; }
.num-pos { color: var(--pos); }
.num-neg { color: var(--neg); }
.section-metadata { margin: 0.25rem 0 0.75rem; font-size: 0.85rem;
    color: var(--fg-muted); display: flex; flex-wrap: wrap;
    gap: 0.25rem 1rem; }
.section-metadata .field { white-space: nowrap; }
.section-metadata .field.mixed { color: var(--neg); }
.section-metadata .field .label { color: var(--fg-dim); margin-right: 0.3em; }
.baseline-card table { width: 100%; }
.baseline-card td.numeric { text-align: right; font-variant-numeric: tabular-nums; }
.stability-dot { display: inline-block; width: 0.7em; height: 0.7em;
    border-radius: 50%; margin-right: 0.3em;
    vertical-align: middle; }
.stability-dot.green { background: var(--pos); }
.stability-dot.amber { background: #e5c07b; }
.stability-dot.red { background: var(--neg); }
.stability-dot.unverified { background: var(--fg-dim); }
.trajectory-chart { width: 100%; min-height: 320px; }
.correlation-chart { width: 100%; min-height: 420px; }
.category-bar-chart { width: 100%; min-height: 200px; }
.host-state-issues ul { margin: 0.25rem 0; padding-left: 1.25rem;
    font-size: 0.85rem; }
.host-state-issues code { background: var(--panel-2);
    padding: 0.1rem 0.3rem; border-radius: 3px; }
"""


def _require_pandas() -> Any:  # noqa: ANN401
    """Return the ``pandas`` module, raising a hint when it is missing.

    Returns:
        The imported ``pandas`` module.

    Raises:
        RuntimeError: ``pandas`` is not installed.
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(_ANALYSIS_HINT) from e
    return pd


def _slug(s: str) -> str:
    """Return a lowercase, dash-separated slug suitable for HTML ids.

    Args:
        s: Source string (e.g. a hardware-class label).

    Returns:
        ``s`` lowercased, with every non-alphanumeric run collapsed to
        a single dash, with leading and trailing dashes stripped.
    """
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _finite_or_none(v: Any) -> float | None:  # noqa: ANN401
    """Coerce a numeric value to a finite float or ``None``.

    Mirrors the guard applied inside
    :func:`kube_autotuner.cli._build_axis_payload` so every new
    payload field survives ``json.dumps(allow_nan=False)``.

    Args:
        v: Any value; ``None``, non-numeric, and non-finite inputs
            all map to ``None``.

    Returns:
        A finite float, or ``None``.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except TypeError, ValueError:
        return None
    return f if math.isfinite(f) else None


_TARGET_METRIC_LABELS: dict[str, str] = {
    "mean_tcp_throughput": "TCP throughput",
    "mean_udp_throughput": "UDP throughput",
    "tcp_retransmit_rate": "TCP retransmit rate",
    "udp_loss_rate": "UDP loss rate",
    "mean_udp_jitter": "UDP jitter",
    "mean_rps": "RPS",
    "mean_latency_p50": "latency p50",
    "mean_latency_p90": "latency p90",
    "mean_latency_p99": "latency p99",
}


def _corr_bar_background(r: float) -> str:
    """Return a CSS linear-gradient for a diverging correlation bar.

    The bar is centered at 50%; positive ``r`` paints green to the
    right, negative ``r`` paints red to the left. The magnitude of
    ``|r|`` drives the filled width.
    """
    magnitude = max(0.0, min(1.0, abs(r))) * 50.0
    if r >= 0:
        start, end, color = 50.0, 50.0 + magnitude, "var(--pos-bar)"
    else:
        start, end, color = 50.0 - magnitude, 50.0, "var(--neg-bar)"
    return (
        f"linear-gradient(to right, transparent {start:.2f}%, "
        f"{color} {start:.2f}%, {color} {end:.2f}%, "
        f"transparent {end:.2f}%)"
    )


def _imp_bar_background(imp: float) -> str:
    """Return a CSS linear-gradient for a 0-anchored importance bar."""
    pct = max(0.0, min(1.0, imp)) * 100.0
    return (
        f"linear-gradient(to right, var(--imp-bar) {pct:.2f}%, transparent {pct:.2f}%)"
    )


def _render_importance(df: pd.DataFrame, top_n: int = _TOP_IMPORTANCE_ROWS) -> str:
    """Render a parameter-importance table with inline bar visualization.

    The rendered table adds a diverging bar behind the Spearman-r cell
    (green for positive, red for negative, width proportional to
    ``|r|``) and a left-anchored bar behind the RF-importance cell
    (width proportional to the normalized score). Numbers stay visible
    on top, right-aligned with tabular numerics, so the reader can
    both scan the ranking at a glance and read exact values.

    Args:
        df: Frame produced by
            :func:`kube_autotuner.report.analysis.parameter_importance`.
        top_n: Maximum number of rows to include.

    Returns:
        An HTML ``<table>`` fragment, or a short "unavailable" note
        when ``df`` is empty.
    """
    if df.empty:
        return (
            "<p class='meta'>Parameter importance unavailable "
            "(not enough variance).</p>"
        )

    rows_html: list[str] = []
    for _, row in df.head(top_n).iterrows():
        param = str(row["param"])
        category = str(row["category"])
        r = float(row["spearman_r"]) if row["spearman_r"] is not None else 0.0
        imp = float(row["rf_importance"]) if row["rf_importance"] is not None else 0.0
        r_cls = "num-pos" if r > 0 else ("num-neg" if r < 0 else "")
        rows_html.append(
            "<tr>"
            f"<td class='param-col'>{html.escape(param)}</td>"
            f"<td class='category-col'>{html.escape(category)}</td>"
            f"<td class='bar-cell' style='background: {_corr_bar_background(r)}'>"
            f"<span class='{r_cls}'>{r:+.2f}</span></td>"
            f"<td class='bar-cell' style='background: {_imp_bar_background(imp)}'>"
            f"{imp:.3f}</td>"
            "</tr>"
        )

    return (
        "<table class='report-table importance-table'>"
        "<colgroup>"
        "<col class='param-w'><col class='category-w'>"
        "<col class='corr-w'><col class='imp-w'>"
        "</colgroup>"
        "<thead><tr>"
        "<th>param</th><th>category</th>"
        "<th title='Spearman rank correlation with the target metric'>"
        "corr (r)</th>"
        "<th title='Random Forest feature importance, 0 to 1'>"
        "importance</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )


def _render_importance_block(
    hw_slug: str,
    importance_by_target: dict[str, pd.DataFrame],
    fallback: pd.DataFrame,
) -> str:
    """Render the importance block with a target-metric dropdown.

    Renders one ``<div>`` per target, all but the default hidden via
    the ``hidden`` attribute. A row of buttons above lets the user
    swap the visible table; the JS module wires the click handler.
    When ``importance_by_target`` is empty, falls back to a single
    table rendered from ``fallback`` (which may itself be an empty
    frame).

    Args:
        hw_slug: Slugified hardware-class label; used to namespace ids.
        importance_by_target: Per-target importance frames. Keys are
            DataFrame column names (e.g. ``"mean_tcp_throughput"``).
        fallback: Legacy single-frame fallback used when the per-target
            dict is empty.

    Returns:
        An HTML fragment covering the dropdown plus every table.
    """
    if not importance_by_target:
        return _render_importance(fallback)

    target_order = [col for col in _TARGET_METRIC_LABELS if col in importance_by_target]
    # Any unknown targets (future metrics) append in insertion order.
    target_order.extend(
        col for col in importance_by_target if col not in _TARGET_METRIC_LABELS
    )
    default_target = (
        "mean_tcp_throughput"
        if "mean_tcp_throughput" in importance_by_target
        else target_order[0]
    )

    buttons = "".join(
        (
            f'<button type="button" '
            f'class="target-btn{" active" if col == default_target else ""}" '
            f'data-hw-slug="{html.escape(hw_slug)}" '
            f'data-target="{html.escape(col)}">'
            f"{html.escape(_TARGET_METRIC_LABELS.get(col, col))}"
            "</button>"
        )
        for col in target_order
    )

    tables: list[str] = []
    for col in target_order:
        hidden = "" if col == default_target else " hidden"
        tables.append(
            f'<div class="importance-panel" '
            f'data-hw-slug="{html.escape(hw_slug)}" '
            f'data-target="{html.escape(col)}"{hidden}>'
            f"{_render_importance(importance_by_target[col])}</div>"
        )

    return (
        f'<div class="importance-controls">'
        f'<span class="meta">Target metric:</span>'
        f'<div class="target-buttons">{buttons}</div>'
        f"</div>" + "".join(tables)
    )


def _axis_chart_id(hw_slug: str) -> str:
    """Return the stable div id for a per-hardware-class axis chart."""
    return f"axis-chart-{hw_slug}"


def _host_state_chart_id(hw_slug: str) -> str:
    """Return the stable div id for a per-hardware-class host-state chart."""
    return f"host-state-chart-{hw_slug}"


_HOST_STATE_PREFERRED_METRICS: tuple[str, ...] = (
    "conntrack_count",
    "slab_nf_conntrack_active_objs",
    "sockstat_tcp_inuse",
)
"""Metric keys pre-selected in the host-state multi-select when present.

All three come from :mod:`kube_autotuner.sysctl.setter`: ``conntrack_count``
and the ``slab_nf_conntrack_*`` pair are written by the conntrack and
slabinfo parsers; ``sockstat_tcp_inuse`` is one of the keys emitted by
the generic ``/proc/net/sockstat`` parser (``sockstat_<proto>_<key>``).
"""


def _trajectory_chart_id(hw_slug: str) -> str:
    """Return the stable div id for the per-hardware-class trajectory chart."""
    return f"trajectory-chart-{hw_slug}"


def _correlation_chart_id(hw_slug: str) -> str:
    """Return the stable div id for the per-hardware-class correlation heatmap."""
    return f"correlation-chart-{hw_slug}"


def _category_bar_chart_id(hw_slug: str) -> str:
    """Return the stable div id for the per-hardware-class category bar."""
    return f"category-bar-{hw_slug}"


def _correlation_matrix_payload(matrix: pd.DataFrame | None) -> dict[str, Any] | None:
    """Lower a Spearman correlation DataFrame into a JSON-safe payload.

    Args:
        matrix: Square DataFrame produced by
            :func:`kube_autotuner.report.analysis.sysctl_correlation_matrix`,
            or ``None`` when there was not enough variance.

    Returns:
        ``{"columns": [...], "values": [[...]]}`` with non-finite
        cells coerced to ``None``, or ``None`` when the input was
        ``None``.
    """
    if matrix is None:
        return None
    pd_mod = _require_pandas()
    columns = [str(c) for c in matrix.columns]
    values: list[list[float | None]] = []
    for _, row in matrix.iterrows():
        row_out: list[float | None] = []
        for v in row:
            if v is None or pd_mod.isna(v):
                row_out.append(None)
            else:
                row_out.append(_finite_or_none(v))
        values.append(row_out)
    return {"columns": columns, "values": values}


def _clean_verification_stats(
    stats: dict[str, dict[str, dict[str, float | None]]] | None,
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Coerce every numeric cell in ``stats`` to a finite float or ``None``.

    Returns:
        A copy of ``stats`` with ``mean``/``stdev``/``cv`` scrubbed.
    """
    if not stats:
        return {}
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for parent, per_metric in stats.items():
        cleaned_metrics: dict[str, dict[str, float | None]] = {}
        for metric, entry in per_metric.items():
            cleaned_metrics[metric] = {
                "mean": _finite_or_none(entry.get("mean")),
                "stdev": _finite_or_none(entry.get("stdev")),
                "cv": _finite_or_none(entry.get("cv")),
            }
        out[parent] = cleaned_metrics
    return out


def _clean_baseline_comparison(
    entries: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Coerce numeric fields in ``entries`` to finite-or-``None``.

    Returns:
        A scrubbed list of per-objective comparison dicts, or ``None``
        when ``entries`` is ``None``.
    """
    if entries is None:
        return None
    return [
        {
            "metric": e["metric"],
            "direction": e["direction"],
            "baseline": _finite_or_none(e.get("baseline")),
            "recommended": _finite_or_none(e.get("recommended")),
            "abs_delta": _finite_or_none(e.get("abs_delta")),
            "pct_delta": _finite_or_none(e.get("pct_delta")),
        }
        for e in entries
    ]


def _clean_trajectory_rows(
    rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Coerce every ``<col>_best_so_far`` field to finite-or-``None``.

    Returns:
        The scrubbed trajectory rows; empty list when ``rows`` is
        falsy.
    """
    if not rows:
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        cleaned: dict[str, Any] = {}
        for k, v in r.items():
            if k.endswith("_best_so_far"):
                cleaned[k] = _finite_or_none(v)
            else:
                cleaned[k] = v
        out.append(cleaned)
    return out


def _clean_metadata(md: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce numeric fields in section metadata to finite-or-``None``.

    Returns:
        A scrubbed copy of ``md``; empty dict when ``md`` is falsy.
    """
    if not md:
        return {}
    result: dict[str, Any] = dict(md)
    trial_count = md.get("trial_count")
    if isinstance(trial_count, int):
        result["trial_count"] = trial_count
    for key in ("iperf_duration", "fortio_duration", "iterations"):
        v = md.get(key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            result[key] = float(v) if isinstance(v, float) else int(v)
        elif isinstance(v, str):
            result[key] = v
        else:
            result[key] = None
    return result


def _clean_category_rollup(
    rollup: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Coerce ``rf_sum`` cells to finite-or-``None``.

    Returns:
        A scrubbed copy of ``rollup``; empty dict when ``rollup`` is
        falsy.
    """
    if not rollup:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for target, entries in rollup.items():
        out[target] = [
            {
                "category": e["category"],
                "rf_sum": _finite_or_none(e.get("rf_sum")) or 0.0,
            }
            for e in entries
        ]
    return out


def _section_payload(section: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON-serializable payload for one hardware-class section.

    Args:
        section: Section dict produced by ``_analyze_one_class``.

    Returns:
        A dict safe to pass to :func:`json.dumps` with
        ``allow_nan=False``: every unmeasured metric is ``None`` (from
        :func:`kube_autotuner.report.analysis.pareto_recommendation_rows`
        and :func:`kube_autotuner.cli._build_axis_payload`) and every
        Pydantic ``ParetoObjective`` has already been lowered to a
        plain dict upstream.
    """
    hw = section["hardware_class"]
    hw_slug = _slug(hw)
    return {
        "hardwareClass": hw,
        "hwSlug": hw_slug,
        "topology": section.get("topology"),
        "trialCount": section["trial_count"],
        "paretoCount": section["pareto_count"],
        "topN": section.get("top_n", 3),
        "objectives": section["objectives"],
        "defaultWeights": section["default_weights"],
        "memoryCostWeight": section["memory_cost_weight"],
        "paretoRows": section["pareto_rows"],
        "allRows": section["all_rows"],
        "axisColumns": section["axis_columns"],
        # figureDivIds is consumed by highlightInPlots to restyle the
        # Pareto top-N on charts that carry trial_id customdata; the
        # host-state chart deliberately has none, so keep it out of this
        # list to avoid a misaimed Plotly.restyle loop.
        "figureDivIds": [_axis_chart_id(hw_slug)],
        "hostState": section.get("host_state"),
        "hostStateChartId": _host_state_chart_id(hw_slug),
        "hostStatePreferredMetrics": list(_HOST_STATE_PREFERRED_METRICS),
        "baselineComparison": _clean_baseline_comparison(
            section.get("baseline_comparison"),
        ),
        "verificationStats": _clean_verification_stats(
            section.get("verification_stats"),
        ),
        "trajectoryRows": _clean_trajectory_rows(
            section.get("trajectory_rows"),
        ),
        "trajectoryChartId": _trajectory_chart_id(hw_slug),
        "metadata": _clean_metadata(section.get("metadata")),
        "correlationMatrix": _correlation_matrix_payload(
            section.get("correlation_matrix"),
        ),
        "correlationChartId": _correlation_chart_id(hw_slug),
        "importanceCategoryRollup": _clean_category_rollup(
            section.get("importance_category_rollup"),
        ),
        "categoryBarChartId": _category_bar_chart_id(hw_slug),
        "hostStateIssues": section.get("host_state_issues") or [],
    }


def _embed_json(data: dict[str, Any], script_id: str) -> str:
    """Embed a JSON payload inside a ``<script type="application/json">`` tag.

    Emits with ``allow_nan=False`` so a stray ``float('nan')`` fails
    fast in Python instead of producing JSON that ``JSON.parse``
    rejects in the browser. The ``</`` sequence is escaped so the
    payload cannot close the enclosing script tag.

    Args:
        data: The payload to serialize.
        script_id: ``id`` attribute for the script tag.

    Returns:
        A ``<script>`` tag string.
    """
    payload = json.dumps(data, allow_nan=False, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    return (
        f'<script type="application/json" id="{html.escape(script_id)}">'
        f"{payload}</script>"
    )


def _render_interactive_panel(hw_slug: str) -> str:
    """Return the empty skeleton the JS module fills in on load."""
    return (
        f'<div class="panel" data-hw-slug="{html.escape(hw_slug)}">\n'
        "<h3>Top recommendations</h3>\n"
        '<div class="controls">\n'
        '<div class="presets" role="group" aria-label="Weight presets"></div>\n'
        '<div class="sliders"></div>\n'
        '<div class="topn-row">'
        "<label>Show top "
        f'<input type="number" min="1" step="1" data-hw-slug="{html.escape(hw_slug)}" '
        'class="topn-input"> rows</label>'
        "</div>\n"
        "</div>\n"
        '<div class="ranked-table-wrapper"></div>\n'
        "</div>"
    )


def _render_axis_chart(hw_slug: str, axis_columns: list[str]) -> str:
    """Render the skeleton for the interactive axis-selector chart.

    The JS module (see :data:`_JS_MODULE`) populates the select
    options, wires change listeners, and calls Plotly on the chart
    div. Python's only job here is to emit a stable skeleton with the
    initial ``selected`` ``<option>`` set so the JS-side default
    state (X=``mean_tcp_throughput`` else first, Y=``tcp_retransmit_rate``
    else first-differs-from-X) agrees with the DOM.

    Args:
        hw_slug: Slugified hardware-class label; used to namespace
            ids and data attributes.
        axis_columns: Metric columns with at least one non-null value
            for this hardware class, in canonical order.

    Returns:
        An HTML ``<section>`` fragment. When fewer than two axis
        columns are available, renders a "not enough metrics" meta
        paragraph in place of the controls so the page does not carry
        a degenerate chart the user cannot configure.
    """
    chart_id = _axis_chart_id(hw_slug)
    if len(axis_columns) < _MIN_AXIS_COLUMNS:
        return (
            "<section class='fig'>\n"
            "<h3>Objective space</h3>\n"
            "<p class='meta'>Not enough metrics with data to render the "
            "objective-space chart.</p>\n"
            f'<div class="axis-chart" id="{html.escape(chart_id)}" hidden></div>\n'
            "</section>"
        )

    default_x = (
        "mean_tcp_throughput"
        if "mean_tcp_throughput" in axis_columns
        else axis_columns[0]
    )
    default_y = (
        "tcp_retransmit_rate"
        if "tcp_retransmit_rate" in axis_columns
        else next((c for c in axis_columns if c != default_x), default_x)
    )

    def _options(selected: str) -> str:
        parts: list[str] = []
        for col in axis_columns:
            label = _TARGET_METRIC_LABELS.get(col, col)
            sel = " selected" if col == selected else ""
            parts.append(
                f'<option value="{html.escape(col)}"{sel}>{html.escape(label)}</option>'
            )
        return "".join(parts)

    slug_attr = html.escape(hw_slug)
    return (
        "<section class='fig'>\n"
        "<h3>Objective space</h3>\n"
        "<div class='axis-controls'>\n"
        f'<label>X <select class="axis-select-x" data-hw-slug="{slug_attr}">'
        f"{_options(default_x)}</select></label>\n"
        f'<label>Y <select class="axis-select-y" data-hw-slug="{slug_attr}">'
        f"{_options(default_y)}</select></label>\n"
        f'<label><input type="checkbox" class="axis-trend-toggle"'
        f' data-hw-slug="{slug_attr}"> Trend line</label>\n'
        "</div>\n"
        f'<div class="axis-chart" id="{html.escape(chart_id)}"></div>\n'
        "</section>"
    )


def _render_host_state_chart(
    hw_slug: str,
    payload: dict[str, Any] | None,
) -> str:
    """Render the skeleton for the interactive host-state chart.

    Mirrors :func:`_render_axis_chart`: emit a stable DOM with a
    multi-select and an empty chart div, let the JS module fill in
    traces via Plotly on load.

    The skeleton is omitted entirely when ``payload`` is ``None`` --
    i.e. when no trial in the hardware class carries any
    :class:`~kube_autotuner.models.HostStateSnapshot`, or every
    snapshot's ``metrics`` dict was empty. In that case there is
    nothing meaningful to chart, and rendering an empty section
    would just clutter the page.

    Args:
        hw_slug: Slugified hardware-class label; used to namespace
            ids and data attributes.
        payload: The host-state payload produced by
            :func:`kube_autotuner.report.analysis.host_state_series`, or
            ``None``.

    Returns:
        An HTML ``<section>`` fragment, or the empty string when
        ``payload`` is ``None``.
    """
    if payload is None:
        return ""

    metrics = list(payload.get("metrics", []))
    if not metrics:
        return ""

    preferred = [m for m in _HOST_STATE_PREFERRED_METRICS if m in metrics]
    preselected = set(preferred) if preferred else {metrics[0]}

    options: list[str] = []
    for metric in metrics:
        sel = " selected" if metric in preselected else ""
        options.append(
            f'<option value="{html.escape(metric)}"{sel}>'
            f"{html.escape(metric)}</option>",
        )

    slug_attr = html.escape(hw_slug)
    chart_id = _host_state_chart_id(hw_slug)
    return (
        "<section class='fig'>\n"
        "<h3>Host state</h3>\n"
        "<p class='meta'>Per-iteration snapshots collected when "
        "<code>--collect-host-state</code> is enabled. "
        "X-axis is UTC capture timestamp; phases are distinguished by "
        "marker shape "
        "(diamond=baseline, circle=post-flush, square=post-iteration).</p>\n"
        "<div class='axis-controls'>\n"
        f'<label>metrics <select multiple size="6" '
        f'class="host-state-metric-select" data-hw-slug="{slug_attr}">'
        f"{''.join(options)}</select></label>\n"
        "</div>\n"
        f'<div class="axis-chart" id="{html.escape(chart_id)}"></div>\n'
        "</section>"
    )


_METRIC_LABEL_BY_SHORT: dict[str, str] = {
    "tcp_throughput": "TCP throughput",
    "udp_throughput": "UDP throughput",
    "tcp_retransmit_rate": "TCP retransmit rate",
    "udp_loss_rate": "UDP loss rate",
    "udp_jitter": "UDP jitter",
    "rps": "RPS",
    "latency_p50": "latency p50",
    "latency_p90": "latency p90",
    "latency_p99": "latency p99",
}


def _render_baseline_card(section: dict[str, Any]) -> str:
    """Render the baseline-vs-recommended comparison card.

    Emits the empty string when ``section["baseline_comparison"]`` is
    ``None`` (no defaults-match trial exists).

    Args:
        section: Section dict; consults ``baseline_comparison``.

    Returns:
        An HTML ``<section>`` fragment, or the empty string.
    """
    entries = section.get("baseline_comparison")
    if not entries:
        return ""
    rows_html: list[str] = []
    for e in entries:
        metric = str(e["metric"])
        direction = str(e["direction"])
        label = _METRIC_LABEL_BY_SHORT.get(metric, metric)
        baseline = e.get("baseline")
        recommended = e.get("recommended")
        abs_delta = e.get("abs_delta")
        pct_delta = e.get("pct_delta")

        def _fmt(v: float | None) -> str:
            if v is None:
                return "-"
            return f"{v:.4g}"

        if abs_delta is None:
            cls = ""
            abs_text = "-"
            pct_text = "-"
        else:
            improving = (direction == "maximize" and abs_delta > 0) or (
                direction == "minimize" and abs_delta < 0
            )
            cls = "num-pos" if improving else ("num-neg" if abs_delta != 0 else "")
            sign = "+" if abs_delta > 0 else ""
            abs_text = f"{sign}{abs_delta:.4g}"
            pct_text = f"{sign}{pct_delta * 100:.2f}%" if pct_delta is not None else "-"
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(label)} "
            f"<span class='meta'>({html.escape(direction)})</span></td>"
            f"<td class='numeric'>{_fmt(baseline)}</td>"
            f"<td class='numeric'>{_fmt(recommended)}</td>"
            f"<td class='numeric'><span class='{cls}'>{abs_text}</span></td>"
            f"<td class='numeric'><span class='{cls}'>{pct_text}</span></td>"
            "</tr>",
        )
    return (
        "<section class='fig baseline-card'>\n"
        "<h3>Baseline comparison</h3>\n"
        "<p class='meta'>Top recommendation vs the "
        "<code>RECOMMENDED_DEFAULTS</code> baseline (per objective).</p>\n"
        "<table class='report-table'>\n"
        "<thead><tr><th>metric</th><th>baseline</th><th>recommended</th>"
        "<th>delta</th><th>%</th></tr></thead>\n"
        f"<tbody>{''.join(rows_html)}</tbody>\n"
        "</table>\n"
        "</section>"
    )


def _render_metadata_header(section: dict[str, Any]) -> str:
    """Render the experiment-metadata strip above the hardware-class heading.

    Fields that are ``None`` are omitted; fields whose value is the
    string ``"mixed"`` render with a muted visual weight so the
    inconsistency is visible.

    Args:
        section: Section dict; consults ``metadata``.

    Returns:
        An HTML fragment with one ``<span>`` per field, or the empty
        string when no field survives.
    """
    md = section.get("metadata") or {}
    if not md:
        return ""
    parts: list[str] = []

    def _field(label: str, value: Any, *, always_str: bool = False) -> None:  # noqa: ANN401
        if value is None:
            return
        mixed = value == "mixed"
        text = value if always_str or isinstance(value, str) else str(value)
        cls = "field mixed" if mixed else "field"
        parts.append(
            f"<span class='{cls}'><span class='label'>{html.escape(label)}</span>"
            f"{html.escape(str(text))}</span>",
        )

    trial_count = md.get("trial_count")
    if isinstance(trial_count, int) and trial_count > 0:
        _field("trials", trial_count)
    phase_counts = md.get("phase_counts") or {}
    phase_chunks = [
        f"{label}={count}" for label, count in phase_counts.items() if count
    ]
    if phase_chunks:
        parts.append(
            "<span class='field'><span class='label'>phases</span>"
            f"{html.escape(', '.join(phase_chunks))}</span>",
        )
    _field("kernel", md.get("kernel_version"))
    _field("iperf duration", md.get("iperf_duration"))
    _field("fortio duration", md.get("fortio_duration"))
    _field("iterations", md.get("iterations"))
    stages = md.get("stages")
    if stages is not None:
        if isinstance(stages, list):
            _field("stages", ", ".join(stages))
        else:
            _field("stages", stages)
    first = md.get("first_created_at_iso")
    last = md.get("last_created_at_iso")
    if first:
        _field("first", first)
    if last and last != first:
        _field("last", last)
    if not parts:
        return ""
    return f"<div class='section-metadata'>{''.join(parts)}</div>"


def _render_trajectory_chart(hw_slug: str, rows: list[dict[str, Any]]) -> str:
    """Render the skeleton for the optimization-trajectory chart.

    Args:
        hw_slug: Slugified hardware-class label.
        rows: Trajectory rows; the fragment is omitted when empty.

    Returns:
        An HTML ``<section>`` fragment, or the empty string.
    """
    if not rows:
        return ""
    chart_id = _trajectory_chart_id(hw_slug)
    return (
        "<section class='fig'>\n"
        "<h3>Optimization trajectory</h3>\n"
        "<p class='meta'>Running best-so-far per objective over the "
        "primary-trial sequence (ordered by <code>created_at</code>). "
        "Markers are colored by phase.</p>\n"
        f'<div class="trajectory-chart" id="{html.escape(chart_id)}"></div>\n'
        "</section>"
    )


def _render_correlation_heatmap(
    hw_slug: str,
    payload: dict[str, Any] | None,
) -> str:
    """Render the skeleton for the sysctl-correlation heatmap.

    Wrapped in a collapsed ``<details>`` so the heatmap does not
    dominate the page by default.

    Args:
        hw_slug: Slugified hardware-class label.
        payload: Correlation payload from
            :func:`_correlation_matrix_payload`, or ``None``.

    Returns:
        An HTML ``<section>`` fragment, or the empty string when
        ``payload`` is ``None``.
    """
    if payload is None:
        return ""
    chart_id = _correlation_chart_id(hw_slug)
    return (
        "<section class='fig'>\n"
        "<details>\n"
        "<summary>Sysctl-sysctl correlation heatmap</summary>\n"
        "<p class='meta'>Strong off-diagonal correlations mean these "
        "knobs moved together; attribution to a single knob is less "
        "reliable.</p>\n"
        f'<div class="correlation-chart" id="{html.escape(chart_id)}"></div>\n'
        "</details>\n"
        "</section>"
    )


def _render_category_bar(
    hw_slug: str,
    rollup: dict[str, list[dict[str, Any]]] | None,
) -> str:
    """Render the skeleton for the per-target category-importance bar.

    Args:
        hw_slug: Slugified hardware-class label.
        rollup: Per-target category rollup; the fragment is omitted
            when ``rollup`` is empty.

    Returns:
        An HTML ``<section>`` fragment, or the empty string.
    """
    if not rollup:
        return ""
    chart_id = _category_bar_chart_id(hw_slug)
    return (
        "<section class='fig importance-category'>\n"
        "<h3>Importance by category</h3>\n"
        "<p class='meta'>Random Forest importance summed by sysctl "
        "category for the active target metric.</p>\n"
        f'<div class="category-bar-chart" id="{html.escape(chart_id)}"></div>\n'
        "</section>"
    )


def _render_host_state_issues(section: dict[str, Any]) -> str:
    """Render a ``<details>`` listing :attr:`HostStateSnapshot.errors` lines.

    Args:
        section: Section dict; consults ``host_state_issues``.

    Returns:
        An HTML ``<section>`` fragment, or the empty string when no
        snapshot carries any errors.
    """
    issues = section.get("host_state_issues") or []
    if not issues:
        return ""
    items: list[str] = []
    for it in issues:
        trial_id = html.escape(str(it.get("trial_id", "")))
        node = html.escape(str(it.get("node", "")))
        phase = html.escape(str(it.get("phase", "")))
        iteration = it.get("iteration")
        iter_str = f" iter {int(iteration)}" if isinstance(iteration, int) else ""
        text = html.escape(str(it.get("error_text", "")))
        items.append(
            "<li>"
            f"<code>{trial_id}</code> "
            f"<span class='meta'>{node} / {phase}{html.escape(iter_str)}</span>: "
            f"{text}"
            "</li>",
        )
    return (
        "<section class='fig host-state-issues'>\n"
        "<details>\n"
        f"<summary>Data-collection issues ({len(issues)} snapshots affected)"
        "</summary>\n"
        f"<ul>{''.join(items)}</ul>\n"
        "</details>\n"
        "</section>"
    )


def _render_section(  # noqa: PLR0914 - one-pass composer of independent fragments
    section: dict[str, Any],
) -> str:
    """Render one per-hardware-class section.

    Args:
        section: Section payload with ``hardware_class``, ``topology``,
            ``trial_count``, ``pareto_count``, ``recommendations``,
            ``pareto_rows``, ``objectives``, ``default_weights``,
            ``top_n``, ``importance``, ``all_rows``, and
            ``axis_columns``.

    Returns:
        An HTML fragment for the section.
    """
    hw = section["hardware_class"]
    hw_slug = _slug(hw)
    topology = section.get("topology")
    topology_suffix = f", topology={html.escape(topology)}" if topology else ""
    meta = (
        f"<p class='meta'>{section['trial_count']} trials, "
        f"{section['pareto_count']} Pareto-optimal{topology_suffix}</p>"
    )

    data_script = _embed_json(
        _section_payload(section),
        script_id=f"section-data-{hw_slug}",
    )

    importance_by_target = section.get("importance_by_target", {})
    importance_block = _render_importance_block(
        hw_slug,
        importance_by_target,
        section["importance"],
    )

    metadata_header = _render_metadata_header(section)
    baseline_card = _render_baseline_card(section)
    axis_chart = _render_axis_chart(hw_slug, section["axis_columns"])
    trajectory_chart = _render_trajectory_chart(
        hw_slug,
        section.get("trajectory_rows") or [],
    )
    host_state_chart = _render_host_state_chart(hw_slug, section.get("host_state"))
    correlation_heatmap = _render_correlation_heatmap(
        hw_slug,
        _correlation_matrix_payload(section.get("correlation_matrix")),
    )
    category_bar = _render_category_bar(
        hw_slug,
        section.get("importance_category_rollup"),
    )
    host_state_issues = _render_host_state_issues(section)

    return (
        f"<section class='hw' id='hw-{hw_slug}'>\n"
        f"<h2>Hardware class: {html.escape(hw)}</h2>\n"
        f"{metadata_header}\n"
        f"{meta}\n"
        f"{baseline_card}\n"
        f"{_render_interactive_panel(hw_slug)}\n"
        f"{data_script}\n"
        f"<h3>Parameter importance (top {_TOP_IMPORTANCE_ROWS})</h3>\n"
        f"{importance_block}\n"
        f"{category_bar}\n"
        f"{axis_chart}\n"
        f"{trajectory_chart}\n"
        f"{correlation_heatmap}\n"
        f"{host_state_chart}\n"
        f"{host_state_issues}\n"
        f"</section>"
    )


# The JS module is a single constant so that a small regex test in
# ``tests/test_report.py`` can pluck out the preset literals and
# verify they stay aligned with ``_DEFAULT_WEIGHTS`` in experiment.py.
_JS_MODULE = r"""
const DEGENERATE = 0.5;

const METRIC_TO_DF_COLUMN = {
  tcp_throughput: "mean_tcp_throughput",
  udp_throughput: "mean_udp_throughput",
  tcp_retransmit_rate: "tcp_retransmit_rate",
  udp_loss_rate: "udp_loss_rate",
  udp_jitter: "mean_udp_jitter",
  rps: "mean_rps",
  latency_p50: "mean_latency_p50",
  latency_p90: "mean_latency_p90",
  latency_p99: "mean_latency_p99",
};

const PRESETS = {
  "default": null,
  "latency-sensitive": {latency_p99: 0.4, latency_p90: 0.2, tcp_retransmit_rate: 0.1},
  "throughput-only": {}
};

const PRESET_LABELS = {
  "default": "Default",
  "latency-sensitive": "Latency-sensitive",
  "throughput-only": "Throughput-only"
};

const METRIC_DISPLAY = {
  mean_tcp_throughput: {label: "TCP throughput", unit: "Mbps",
                        format: v => (v / 1e6).toFixed(1)},
  mean_udp_throughput: {label: "UDP throughput", unit: "Mbps",
                        format: v => (v / 1e6).toFixed(1)},
  tcp_retransmit_rate: {label: "TCP retx", unit: "/GB",
                        format: v => v.toFixed(2)},
  udp_loss_rate: {label: "UDP loss", unit: "%",
                  format: v => (v * 100).toFixed(2)},
  mean_rps: {label: "RPS", unit: "",
             format: v => v.toLocaleString("en-US", {maximumFractionDigits: 1})},
  mean_udp_jitter: {label: "UDP jitter", unit: "ms",
                    format: v => (v * 1000).toFixed(3)},
  mean_latency_p50: {label: "p50", unit: "ms", format: v => (v * 1000).toFixed(1)},
  mean_latency_p90: {label: "p90", unit: "ms", format: v => (v * 1000).toFixed(1)},
  mean_latency_p99: {label: "p99", unit: "ms", format: v => (v * 1000).toFixed(1)},
};

const MS_SCALED = new Set([
  "mean_udp_jitter", "mean_latency_p50",
  "mean_latency_p90", "mean_latency_p99"
]);

function axisValue(row, col) {
  const v = row[col];
  if (v === null || v === undefined) return null;
  return MS_SCALED.has(col) ? v * 1000 : v;
}

function axisLabel(col) {
  const d = METRIC_DISPLAY[col];
  if (!d) return col;
  return d.unit ? `${d.label} (${d.unit})` : d.label;
}

function toFloatOrNaN(v) {
  if (v === null || v === undefined) return NaN;
  const f = Number(v);
  return Number.isFinite(f) ? f : NaN;
}

function normalizeColumn(values) {
  const finite = values.filter(v => !Number.isNaN(v));
  if (finite.length === 0) return values.map(() => DEGENERATE);
  let lo = Infinity, hi = -Infinity;
  for (const v of finite) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi === lo) {
    return values.map(() => DEGENERATE);
  }
  const span = hi - lo;
  return values.map(v => Number.isNaN(v) ? DEGENERATE : (v - lo) / span);
}

// Port of kube_autotuner.scoring.score_rows with per-metric contribution
// bookkeeping for the decomposition panel.
function scoreRows(rows, objectives, weights, memoryCostWeight) {
  const n = rows.length;
  const scores = new Array(n).fill(0);
  const contributions = Array.from({length: n}, () => ({}));
  if (n === 0) return {scores, contributions};
  for (const obj of objectives) {
    const col = METRIC_TO_DF_COLUMN[obj.metric];
    const raw = rows.map(r => toFloatOrNaN(r[col]));
    const norm = normalizeColumn(raw);
    if (obj.direction === "maximize") {
      const w = weights[obj.metric] ?? 1.0;
      for (let i = 0; i < n; i++) {
        const c = w * norm[i];
        scores[i] += c;
        contributions[i][obj.metric] = {
          norm: norm[i], contribution: c, direction: "maximize", weight: w,
        };
      }
    } else {
      const w = weights[obj.metric] ?? 0.0;
      for (let i = 0; i < n; i++) {
        const c = -w * norm[i];
        scores[i] += c;
        contributions[i][obj.metric] = {
          norm: norm[i], contribution: c, direction: "minimize", weight: w,
        };
      }
    }
  }
  // Memory-cost term: a separate top-level state field, not keyed in
  // state.weights, so preset parity (activePresetKey / applyPreset)
  // stays untouched. Matches kube_autotuner.scoring.score_rows.
  const mw = memoryCostWeight ?? 0.0;
  if (mw > 0) {
    const raw = rows.map(r => toFloatOrNaN(r.memory_cost));
    const norm = normalizeColumn(raw);
    for (let i = 0; i < n; i++) {
      const c = -mw * norm[i];
      scores[i] += c;
      contributions[i].memory_cost = {
        norm: norm[i], contribution: c, direction: "minimize", weight: mw,
      };
    }
  }
  return {scores, contributions};
}

function formatMetric(col, value) {
  if (value === null || value === undefined) return "-";
  const fmt = METRIC_DISPLAY[col];
  if (!fmt) return String(value);
  return fmt.format(value) + (fmt.unit ? " " + fmt.unit : "");
}

// The keyset that drives sliders and presets: every pareto objective
// in the section, regardless of direction. defaultWeights fills in
// initial minimize-direction values; maximize objectives and metrics
// without an explicit default fall back to the direction default
// (1.0 for maximize, 0.0 for minimize).
function weightedMetricKeys(section) {
  return section.objectives.map(o => o.metric);
}

function directionDefault(section, metric) {
  const obj = section.objectives.find(o => o.metric === metric);
  return obj && obj.direction === "maximize" ? 1.0 : 0.0;
}

function activePresetKey(weights, section) {
  const keys = weightedMetricKeys(section);
  for (const [key, overrides] of Object.entries(PRESETS)) {
    let target;
    if (overrides === null) {
      target = {};
      for (const k of keys) {
        target[k] = section.defaultWeights[k] ?? directionDefault(section, k);
      }
    } else {
      target = {};
      for (const k of keys) target[k] = directionDefault(section, k);
      for (const [k, v] of Object.entries(overrides)) {
        if (k in target) target[k] = v;
      }
    }
    let match = true;
    for (const k of keys) {
      const fallback = directionDefault(section, k);
      const wv = weights[k] ?? fallback;
      const tv = target[k] ?? fallback;
      if (Math.abs(wv - tv) > 1e-9) {
        match = false;
        break;
      }
    }
    if (match) return key;
  }
  return null;
}

function applyPreset(key, section) {
  const keys = weightedMetricKeys(section);
  const overrides = PRESETS[key];
  const next = {};
  if (overrides === null) {
    for (const k of keys) {
      next[k] = section.defaultWeights[k] ?? directionDefault(section, k);
    }
    return next;
  }
  for (const k of keys) next[k] = directionDefault(section, k);
  for (const [k, v] of Object.entries(overrides)) {
    if (k in next) next[k] = v;
  }
  return next;
}

function renderPresets(container, state, rerank) {
  container.innerHTML = "";
  for (const key of Object.keys(PRESETS)) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = PRESET_LABELS[key];
    btn.dataset.preset = key;
    btn.addEventListener("click", () => {
      state.weights = applyPreset(key, state.section);
      syncSliderValues(state);
      rerank();
    });
    container.appendChild(btn);
  }
}

function renderSliders(container, state, rerank) {
  container.innerHTML = "";
  const metrics = weightedMetricKeys(state.section);
  // Compute sliderMax from the initial weights map, which now
  // includes 1.0 entries for maximize metrics; the 1.5x multiplier
  // keeps maximize defaults with visible headroom above 1.0 so users
  // can bias a maximize weight upward.
  const initialValues = metrics.map(
    m => state.weights[m] ?? directionDefault(state.section, m));
  const maxInitial = initialValues.length ? Math.max(0, ...initialValues) : 0;
  const sliderMax = Math.max(1.0, 1.5 * maxInitial);
  state.sliderMax = sliderMax;
  for (const metric of metrics) {
    const row = document.createElement("div");
    row.className = "slider-row";
    const label = document.createElement("label");
    label.textContent = metric + ": ";
    const valueSpan = document.createElement("span");
    valueSpan.className = "weight-value";
    label.appendChild(valueSpan);
    const input = document.createElement("input");
    input.type = "range";
    input.min = "0";
    input.max = String(sliderMax);
    input.step = "0.01";
    input.dataset.metric = metric;
    input.value = String(
      state.weights[metric] ?? directionDefault(state.section, metric));
    valueSpan.textContent = Number(input.value).toFixed(2);
    input.addEventListener("input", () => {
      const v = Number(input.value);
      state.weights[metric] = v;
      valueSpan.textContent = v.toFixed(2);
      rerank();
    });
    row.appendChild(label);
    row.appendChild(input);
    container.appendChild(row);
  }
}

function syncSliderValues(state) {
  const panel = state.panel;
  for (const input of panel.querySelectorAll("input[type=range]")) {
    const metric = input.dataset.metric;
    const v = state.weights[metric] ?? directionDefault(state.section, metric);
    input.value = String(v);
    const valueSpan = input.parentElement.querySelector(".weight-value");
    if (valueSpan) valueSpan.textContent = v.toFixed(2);
  }
}

function updatePresetHighlight(state) {
  const active = activePresetKey(state.weights, state.section);
  for (const btn of state.panel.querySelectorAll(".presets button")) {
    if (btn.dataset.preset === active) btn.classList.add("active");
    else btn.classList.remove("active");
  }
}

// Build the ranked-table skeleton once per section (stable DOM nodes
// keyed by trial_id). Subsequent reranks only update cell text, reorder
// existing <tr> pairs in tbody, and toggle row visibility -- the
// <details> expansion state and any already-rendered Plotly chart are
// preserved across slider drags.
function buildRankedTable(wrapper, state, visibleCols) {
  const table = document.createElement("table");
  table.className = "report-table";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const h of ["rank", "trial"]) {
    const th = document.createElement("th");
    th.textContent = h;
    headRow.appendChild(th);
  }
  for (const col of visibleCols) {
    const th = document.createElement("th");
    th.textContent = METRIC_DISPLAY[col].label
      + (METRIC_DISPLAY[col].unit ? " (" + METRIC_DISPLAY[col].unit + ")" : "");
    headRow.appendChild(th);
  }
  const scoreTh = document.createElement("th");
  scoreTh.textContent = "score";
  headRow.appendChild(scoreTh);
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  const rowRefs = new Map();
  for (let i = 0; i < state.section.paretoRows.length; i++) {
    const row = state.section.paretoRows[i];
    const tr = document.createElement("tr");
    const rankTd = document.createElement("td");
    tr.appendChild(rankTd);
    const trialTd = document.createElement("td");
    const badge = row.stability_badge || "unverified";
    const dot = document.createElement("span");
    dot.className = "stability-dot " + badge;
    dot.title = "stability: " + badge;
    trialTd.appendChild(dot);
    trialTd.appendChild(document.createTextNode(row.trial_id));
    tr.appendChild(trialTd);
    const metricTds = {};
    for (const col of visibleCols) {
      const td = document.createElement("td");
      td.className = "numeric";
      td.textContent = formatMetric(col, row[col]);
      metricTds[col] = td;
      tr.appendChild(td);
    }
    const scoreTd = document.createElement("td");
    scoreTd.className = "numeric";
    tr.appendChild(scoreTd);

    const detailsTr = document.createElement("tr");
    const detailsTd = document.createElement("td");
    detailsTd.colSpan = visibleCols.length + 3;
    detailsTd.style.padding = "0";
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "details for trial " + row.trial_id;
    details.appendChild(summary);

    const wrapper = document.createElement("div");
    wrapper.className = "decomposition-wrapper";
    const table = document.createElement("table");
    table.className = "report-table decomposition-table";
    table.id = "decomp-" + state.section.hwSlug + "-" + i;
    wrapper.appendChild(table);
    details.appendChild(wrapper);

    const verifStats = (state.section.verificationStats || {})[row.trial_id];
    if (verifStats) {
      const verifDetails = document.createElement("details");
      const verifSummary = document.createElement("summary");
      verifSummary.textContent = "verification stability";
      verifDetails.appendChild(verifSummary);
      const verifWrapper = document.createElement("div");
      verifWrapper.className = "decomposition-wrapper";
      const verifTable = document.createElement("table");
      verifTable.className = "report-table decomposition-table";
      let verifBody = "<thead><tr><th>metric</th>"
        + "<th>mean</th><th>± stdev</th><th>CV</th></tr></thead><tbody>";
      for (const [metric, entry] of Object.entries(verifStats)) {
        const mean = entry.mean === null || entry.mean === undefined
          ? "-" : Number(entry.mean).toPrecision(4);
        const sd = entry.stdev === null || entry.stdev === undefined
          ? "-" : Number(entry.stdev).toPrecision(3);
        const cv = entry.cv === null || entry.cv === undefined
          ? "—" : (Number(entry.cv) * 100).toFixed(2) + "%";
        verifBody += `<tr><td class="metric-col">${metric}</td>`
          + `<td class="numeric">${mean}</td>`
          + `<td class="numeric">${sd}</td>`
          + `<td class="numeric">${cv}</td></tr>`;
      }
      verifBody += "</tbody>";
      verifTable.innerHTML = verifBody;
      verifWrapper.appendChild(verifTable);
      verifDetails.appendChild(verifWrapper);
      details.appendChild(verifDetails);
    }

    const sysctlDetails = document.createElement("details");
    const sysctlSummary = document.createElement("summary");
    sysctlSummary.textContent = "sysctl values";
    sysctlDetails.appendChild(sysctlSummary);
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(row.sysctl_values, null, 2);
    sysctlDetails.appendChild(pre);
    details.appendChild(sysctlDetails);

    detailsTd.appendChild(details);
    detailsTr.appendChild(detailsTd);
    tbody.appendChild(tr);
    tbody.appendChild(detailsTr);

    const ref = {tr, rankTd, scoreTd, detailsTr, details,
                 tableEl: table, entry: null};
    details.addEventListener("toggle", () => {
      if (details.open && ref.entry) ensureDecomposition(ref);
    });
    rowRefs.set(row.trial_id, ref);
  }
  table.appendChild(tbody);
  wrapper.innerHTML = "";
  wrapper.appendChild(table);
  return {tbody, rowRefs};
}

function updateRankedTable(state, ranked) {
  const tbody = state.tbody;
  const topN = state.topN;
  for (let rank = 0; rank < ranked.length; rank++) {
    const entry = ranked[rank];
    const ref = state.rowRefs.get(entry.row.trial_id);
    if (!ref) continue;
    ref.entry = entry;
    const visible = rank < topN;
    ref.tr.hidden = !visible;
    ref.detailsTr.hidden = !visible;
    ref.tr.classList.toggle("top-rank", visible && rank === 0);
    if (visible) {
      ref.rankTd.textContent = String(rank + 1);
      ref.scoreTd.textContent = entry.score.toFixed(4);
      // Reorder: place this row pair next in tbody (stable even when
      // appending an element that already lives in the tree).
      tbody.appendChild(ref.tr);
      tbody.appendChild(ref.detailsTr);
      if (ref.details.open) ensureDecomposition(ref);
    }
  }
}

function ensureDecomposition(ref) {
  if (!ref.entry) return;
  renderDecompositionTable(ref.tableEl, ref.entry);
}

function renderDecompositionTable(tableEl, entry) {
  const contribs = entry.contributions;
  const metrics = Object.keys(contribs);
  // Largest |contribution| sets the bar scale so the longest bar fills
  // the cell. Recomputed on every rerank because slider changes shift
  // contributions; the ~20-row max keeps this cheap.
  let maxAbs = 0;
  for (const m of metrics) {
    const v = Math.abs(contribs[m].contribution);
    if (v > maxAbs) maxAbs = v;
  }
  tableEl.innerHTML =
    `<colgroup>`
    + `<col class="metric-w"><col class="norm-w">`
    + `<col class="weight-w"><col class="contrib-w">`
    + `</colgroup>`
    + `<thead><tr><th>metric</th>`
    + `<th title="Normalized metric value, 0 to 1">norm</th>`
    + `<th title="Slider weight">weight</th>`
    + `<th title="Signed score contribution = norm \u00d7 weight">`
    + `contribution</th>`
    + `</tr></thead><tbody></tbody>`;
  const tbody = tableEl.querySelector("tbody");
  for (const m of metrics) {
    const c = contribs[m];
    const sign = c.contribution >= 0 ? "+" : "";
    const numCls = c.contribution > 0 ? "num-pos"
      : (c.contribution < 0 ? "num-neg" : "");
    const barColor = c.direction === "maximize"
      ? "var(--pos-bar)" : "var(--neg-bar)";
    const pct = maxAbs > 0
      ? Math.min(1, Math.abs(c.contribution) / maxAbs) * 100 : 0;
    const weightCell = c.weight.toFixed(2);
    const tr = document.createElement("tr");
    const metricTd = document.createElement("td");
    metricTd.className = "metric-col";
    // textContent avoids HTML injection from arbitrary metric keys.
    metricTd.textContent = m;
    tr.appendChild(metricTd);
    tr.insertAdjacentHTML("beforeend",
      `<td class="numeric">${c.norm.toFixed(3)}</td>`
      + `<td class="numeric">${weightCell}</td>`
      + `<td class="bar-cell" style="background: linear-gradient(`
      + `to right, ${barColor} ${pct.toFixed(2)}%, transparent `
      + `${pct.toFixed(2)}%)">`
      + `<span class="${numCls}">${sign}`
      + `${c.contribution.toFixed(3)}</span></td>`);
    tbody.appendChild(tr);
  }
}

function highlightInPlots(figureDivIds, topTrialIds) {
  const topSet = new Set(topTrialIds.map(String));
  for (const id of figureDivIds) {
    const el = document.getElementById(id);
    if (!el || !el.data) continue;
    for (let t = 0; t < el.data.length; t++) {
      const trace = el.data[t];
      const cd = trace.customdata;
      if (!cd) continue;
      // Plotly customdata can be a flat array or an array of arrays.
      const size = [];
      const opacity = [];
      for (let i = 0; i < cd.length; i++) {
        const raw = cd[i];
        const tid = Array.isArray(raw) ? raw[0] : raw;
        if (topSet.has(String(tid))) {
          size.push(14);
          opacity.push(1.0);
        } else {
          size.push(5);
          opacity.push(0.35);
        }
      }
      Plotly.restyle(el, {"marker.size": [size], "marker.opacity": [opacity]}, [t]);
    }
  }
}

const PHASE_SYMBOL = {
  sobol: "circle",
  bayesian: "square",
  verification: "diamond",
  unknown: "triangle-up",
};

const PHASE_COLOR = {
  sobol: "#BABBBD",
  bayesian: "#5B9BD5",
  verification: "#98C379",
  unknown: "#828997",
};

function axisStd(row, col) {
  const v = row[col + "_std"];
  if (v === null || v === undefined) return null;
  return MS_SCALED.has(col) ? v * 1000 : v;
}

// Closed-form OLS over finite (x, y) pairs. Returns null if fewer than
// two points or zero x-variance (vertical fit is ill-defined).
function linearFit(xs, ys) {
  let n = 0, sx = 0, sy = 0;
  let xmin = Infinity, xmax = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    const a = xs[i], b = ys[i];
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    sx += a; sy += b;
    if (a < xmin) xmin = a;
    if (a > xmax) xmax = a;
    n++;
  }
  if (n < 2) return null;
  const mx = sx / n, my = sy / n;
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < xs.length; i++) {
    const a = xs[i], b = ys[i];
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    const dx = a - mx, dy = b - my;
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  if (sxx === 0) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let r2 = syy === 0 ? 1 : (sxy * sxy) / (sxx * syy);
  if (r2 > 1) r2 = 1;
  if (r2 < 0) r2 = 0;
  return {slope, intercept, r2, n, xmin, xmax};
}

// Build phase-split non-pareto traces plus a single pareto trace for
// the interactive axis chart and hand them to Plotly. Error bars are
// keyed off the per-row <col>_std arrays (null disables the bar for
// that point). Preserves the top-N marker-size highlight across axis
// swaps by re-applying ``highlightInPlots`` with ``state.lastTopIds``
// after every render.
function renderAxisChart(state) {
  const section = state.section;
  const div = document.getElementById("axis-chart-" + section.hwSlug);
  if (!div) return;
  const x = state.axisX;
  const y = state.axisY;
  const byPhase = {};
  const paretos = [];
  for (const row of section.allRows) {
    if (row.pareto) {
      paretos.push(row);
    } else {
      const phase = row.phase || "unknown";
      (byPhase[phase] = byPhase[phase] || []).push(row);
    }
  }
  const traces = [];
  const phaseOrder = ["sobol", "bayesian", "verification", "unknown"]
    .filter(p => byPhase[p] && byPhase[p].length);
  for (const p of Object.keys(byPhase).sort()) {
    if (!phaseOrder.includes(p)) phaseOrder.push(p);
  }
  for (const phase of phaseOrder) {
    const rows = byPhase[phase];
    if (!rows.length) continue;
    traces.push({
      type: "scatter", mode: "markers", name: phase,
      x: rows.map(r => axisValue(r, x)),
      y: rows.map(r => axisValue(r, y)),
      customdata: rows.map(r => r.trial_id),
      error_x: {
        type: "data",
        array: rows.map(r => axisStd(r, x)),
        visible: true,
        color: "rgba(170,170,170,0.35)",
        thickness: 1,
        width: 2,
      },
      error_y: {
        type: "data",
        array: rows.map(r => axisStd(r, y)),
        visible: true,
        color: "rgba(170,170,170,0.35)",
        thickness: 1,
        width: 2,
      },
      marker: {
        color: PHASE_COLOR[phase] || "#BABBBD",
        symbol: PHASE_SYMBOL[phase] || "circle",
        size: 6,
      },
      hovertemplate:
        `trial %{customdata}<br>phase ${phase}<br>`
        + `${axisLabel(x)}=%{x:.3g}<br>`
        + `${axisLabel(y)}=%{y:.3g}<extra></extra>`,
    });
  }
  traces.push({
    type: "scatter", mode: "markers", name: "pareto",
    x: paretos.map(r => axisValue(r, x)),
    y: paretos.map(r => axisValue(r, y)),
    customdata: paretos.map(r => r.trial_id),
    error_x: {
      type: "data",
      array: paretos.map(r => axisStd(r, x)),
      visible: true,
      color: "rgba(239,85,59,0.4)",
      thickness: 1,
      width: 2,
    },
    error_y: {
      type: "data",
      array: paretos.map(r => axisStd(r, y)),
      visible: true,
      color: "rgba(239,85,59,0.4)",
      thickness: 1,
      width: 2,
    },
    marker: {color: "#EF553B", size: 9},
    hovertemplate:
      `trial %{customdata}<br>${axisLabel(x)}=%{x:.3g}<br>`
      + `${axisLabel(y)}=%{y:.3g}<extra></extra>`,
  });
  if (state.showTrend) {
    const xs = section.allRows.map(r => axisValue(r, x));
    const ys = section.allRows.map(r => axisValue(r, y));
    const fit = linearFit(xs, ys);
    if (fit) {
      traces.push({
        type: "scatter", mode: "lines",
        name: `trend (R^2=${fit.r2.toFixed(2)})`,
        x: [fit.xmin, fit.xmax],
        y: [
          fit.intercept + fit.slope * fit.xmin,
          fit.intercept + fit.slope * fit.xmax,
        ],
        line: {color: "#E5C07B", width: 2, dash: "dash"},
        hoverinfo: "skip",
        showlegend: true,
      });
    }
  }
  const layout = {
    template: "plotly_dark",
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: "#abb2bf"},
    xaxis: {title: axisLabel(x)},
    yaxis: {title: axisLabel(y)},
    autosize: true,
    height: 480,
    margin: {l: 60, r: 20, t: 10, b: 50},
    showlegend: true,
    legend: {orientation: "h", y: -0.2},
  };
  const config = {responsive: true, displayModeBar: false};
  if (div._fullLayout) {
    Plotly.react(div, traces, layout, config);
  } else {
    Plotly.newPlot(div, traces, layout, config);
  }
  if (state.lastTopIds) {
    highlightInPlots(section.figureDivIds, state.lastTopIds);
  }
}

function setupAxisChart(section, state) {
  const axisCols = section.axisColumns || [];
  if (axisCols.length < 2) return;
  const selX = document.querySelector(
    `select.axis-select-x[data-hw-slug="${section.hwSlug}"]`);
  const selY = document.querySelector(
    `select.axis-select-y[data-hw-slug="${section.hwSlug}"]`);
  if (!selX || !selY) return;
  selX.value = state.axisX;
  selY.value = state.axisY;
  selX.addEventListener("change", () => {
    state.axisX = selX.value;
    renderAxisChart(state);
  });
  selY.addEventListener("change", () => {
    state.axisY = selY.value;
    renderAxisChart(state);
  });
  const trendEl = document.querySelector(
    `input.axis-trend-toggle[data-hw-slug="${section.hwSlug}"]`);
  if (trendEl) {
    trendEl.checked = state.showTrend;
    trendEl.addEventListener("change", () => {
      state.showTrend = trendEl.checked;
      renderAxisChart(state);
    });
  }
  renderAxisChart(state);
}

function renderSection(panel, section) {
  // Column visibility is computed once from the full frontier so the
  // table's column set is stable under slider and top-N changes.
  const visibleCols = Object.keys(METRIC_DISPLAY).filter(col =>
    section.paretoRows.some(r => r[col] !== null && r[col] !== undefined)
  );
  // Every pareto objective gets a slider. defaultWeights supplies
  // the initial minimize-direction values; maximize objectives (and
  // any metric without an explicit default) fall back to the
  // direction default -- 1.0 for maximize, 0.0 for minimize.
  const initialWeights = {};
  for (const metric of weightedMetricKeys(section)) {
    initialWeights[metric] =
      section.defaultWeights[metric] ?? directionDefault(section, metric);
  }
  const axisCols = section.axisColumns || [];
  const axisX = axisCols.includes("mean_tcp_throughput")
    ? "mean_tcp_throughput" : axisCols[0];
  const axisY = axisCols.includes("tcp_retransmit_rate")
    ? "tcp_retransmit_rate"
    : (axisCols.find(c => c !== axisX) || axisX);
  const state = {
    section,
    panel,
    visibleCols,
    weights: initialWeights,
    // Kept outside state.weights so preset parity (activePresetKey /
    // applyPreset) is untouched; memory cost is not a Pareto metric.
    memoryCostWeight: section.memoryCostWeight ?? 0.0,
    topN: Math.max(1, Math.min(section.topN || 3, section.paretoRows.length)),
    axisX,
    axisY,
    showTrend: false,
    lastTopIds: null,
  };
  const presetsEl = panel.querySelector(".presets");
  const slidersEl = panel.querySelector(".sliders");
  const tableWrapper = panel.querySelector(".ranked-table-wrapper");
  const topNInput = panel.querySelector(".topn-input");
  topNInput.value = String(state.topN);
  topNInput.max = String(Math.max(1, section.paretoRows.length));

  const {tbody, rowRefs} = buildRankedTable(tableWrapper, state, visibleCols);
  state.tbody = tbody;
  state.rowRefs = rowRefs;

  function rerank() {
    const {scores, contributions} = scoreRows(
      section.paretoRows,
      section.objectives,
      state.weights,
      state.memoryCostWeight);
    const ranked = section.paretoRows
      .map((row, i) => ({row, score: scores[i], contributions: contributions[i]}))
      .sort((a, b) => {
        if (b.score !== a.score) return b.score - a.score;
        return String(a.row.trial_id).localeCompare(String(b.row.trial_id));
      });
    updateRankedTable(state, ranked);
    updatePresetHighlight(state);
    const topIds = ranked.slice(0, state.topN).map(r => r.row.trial_id);
    state.lastTopIds = topIds;
    highlightInPlots(section.figureDivIds, topIds);
  }

  renderPresets(presetsEl, state, rerank);
  renderSliders(slidersEl, state, rerank);
  topNInput.addEventListener("input", () => {
    const v = Math.max(1, Math.min(
      parseInt(topNInput.value, 10) || 1,
      section.paretoRows.length));
    state.topN = v;
    rerank();
  });
  rerank();
  setupAxisChart(section, state);
  // Open the rank-1 details after the first rerank so ``ref.entry`` is
  // populated when the toggle listener fires -- avoids the first-load
  // double-render the reviewer flagged.
  const firstRef = rowRefs.get(section.paretoRows[0].trial_id);
  if (firstRef) firstRef.details.open = true;
}

// Plotly marker symbol per HostStatePhase; any phase not listed here
// would fall back to "circle" (which also happens to be Plotly's
// default), but the model's Literal type keeps this set closed.
const HOST_STATE_PHASE_SYMBOL = {
  "baseline": "diamond",
  "post-flush": "circle",
  "post-iteration": "square",
};

function hostStateSelectedMetrics(section) {
  const sel = document.querySelector(
    `select.host-state-metric-select[data-hw-slug="${section.hwSlug}"]`);
  if (!sel) return [];
  const out = [];
  for (const opt of sel.options) {
    if (opt.selected) out.push(opt.value);
  }
  return out;
}

function renderHostStateChart(section) {
  const div = document.getElementById(section.hostStateChartId);
  if (!div) return;
  const payload = section.hostState;
  if (!payload) return;
  const metrics = hostStateSelectedMetrics(section);
  const points = payload.points;
  const x = points.map(p => p.timestamp);
  const symbols = points.map(
    p => HOST_STATE_PHASE_SYMBOL[p.phase] || "circle");
  const text = points.map(p => {
    const iter = p.iteration !== null && p.iteration !== undefined
      ? " iter " + p.iteration : "";
    return `trial ${p.trial_id}<br>${p.phase}${iter}`;
  });
  const traces = [];
  for (const metric of metrics) {
    const y = points.map(p => {
      const v = p.metrics[metric];
      return (v === null || v === undefined) ? null : v;
    });
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      name: metric,
      x,
      y,
      text,
      marker: {symbol: symbols, size: 8},
      connectgaps: false,
      hovertemplate:
        `metric ${metric}<br>%{text}<br>%{x}<br>value=%{y}<extra></extra>`,
    });
  }
  const layout = {
    template: "plotly_dark",
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: "#abb2bf"},
    xaxis: {title: "timestamp", type: "date"},
    yaxis: {title: "counter value"},
    autosize: true,
    height: 480,
    margin: {l: 70, r: 20, t: 10, b: 50},
    showlegend: true,
    legend: {orientation: "h", y: -0.2},
  };
  const config = {responsive: true, displayModeBar: false};
  if (div._fullLayout) {
    Plotly.react(div, traces, layout, config);
  } else {
    Plotly.newPlot(div, traces, layout, config);
  }
}

function setupHostStateChart(section) {
  if (!section.hostState) return;
  const sel = document.querySelector(
    `select.host-state-metric-select[data-hw-slug="${section.hwSlug}"]`);
  if (!sel) return;
  sel.addEventListener("change", () => renderHostStateChart(section));
  renderHostStateChart(section);
}

function wireImportanceSelectors() {
  for (const btn of document.querySelectorAll(".target-btn")) {
    btn.addEventListener("click", () => {
      const hwSlug = btn.dataset.hwSlug;
      const target = btn.dataset.target;
      for (const sibling of document.querySelectorAll(
          `.target-btn[data-hw-slug="${hwSlug}"]`)) {
        sibling.classList.toggle("active", sibling === btn);
      }
      for (const panel of document.querySelectorAll(
          `.importance-panel[data-hw-slug="${hwSlug}"]`)) {
        panel.hidden = panel.dataset.target !== target;
      }
      const section = SECTIONS[hwSlug];
      if (section) renderCategoryRollup(section, target);
    });
  }
}

const SECTIONS = {};

// One subplot per objective so each series keeps its own natural
// units (bits/sec vs retx-rate vs ms latency), which makes the
// running-best curves readable — a single shared y-axis would squash
// every series against the bits/sec one.
function renderTrajectoryChart(section) {
  const div = document.getElementById(section.trajectoryChartId);
  if (!div) return;
  const rows = section.trajectoryRows || [];
  if (!rows.length) return;
  const objectives = section.objectives || [];
  if (!objectives.length) return;
  const x = rows.map(r => r.created_at_iso);
  const phases = rows.map(r => r.phase_effective || "unknown");
  const symbols = phases.map(p => PHASE_SYMBOL[p] || "circle");
  const colors = phases.map(p => PHASE_COLOR[p] || "#BABBBD");
  const text = rows.map(r => `trial ${r.trial_id}<br>${r.phase_effective}`);
  const traces = [];
  const layout = {
    template: "plotly_dark",
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: "#abb2bf"},
    autosize: true,
    height: Math.max(220, objectives.length * 170),
    margin: {l: 70, r: 20, t: 28, b: 50},
    showlegend: false,
    grid: {rows: objectives.length, columns: 1, pattern: "independent"},
    annotations: [],
  };
  for (let i = 0; i < objectives.length; i++) {
    const obj = objectives[i];
    const col = METRIC_TO_DF_COLUMN[obj.metric];
    const key = col + "_best_so_far";
    const y = rows.map(r => {
      const v = r[key];
      if (v === null || v === undefined) return null;
      return MS_SCALED.has(col) ? v * 1000 : v;
    });
    const xaxis = i === 0 ? "x" : "x" + (i + 1);
    const yaxis = i === 0 ? "y" : "y" + (i + 1);
    const xaxisKey = i === 0 ? "xaxis" : "xaxis" + (i + 1);
    const yaxisKey = i === 0 ? "yaxis" : "yaxis" + (i + 1);
    layout[xaxisKey] = {
      type: "date",
      title: i === objectives.length - 1 ? "trial timestamp" : "",
      showticklabels: i === objectives.length - 1,
    };
    layout[yaxisKey] = {title: axisLabel(col)};
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      name: axisLabel(col),
      x,
      y,
      xaxis,
      yaxis,
      text,
      marker: {symbol: symbols, color: colors, size: 7},
      line: {width: 1.2, color: "rgba(97,175,239,0.45)"},
      connectgaps: true,
      hovertemplate: `%{text}<br>%{x}<br>best=%{y:.3g}<extra>${axisLabel(col)}</extra>`,
    });
  }
  const config = {responsive: true, displayModeBar: false};
  if (div._fullLayout) {
    Plotly.react(div, traces, layout, config);
  } else {
    Plotly.newPlot(div, traces, layout, config);
  }
}

function renderCorrelationHeatmap(section) {
  const div = document.getElementById(section.correlationChartId);
  if (!div) return;
  const payload = section.correlationMatrix;
  if (!payload) return;
  const z = payload.values.map(row => row.map(v => v === null ? null : v));
  const traces = [{
    type: "heatmap",
    z,
    x: payload.columns,
    y: payload.columns,
    colorscale: [
      [0, "#e06c75"], [0.5, "#2c313a"], [1, "#98c379"],
    ],
    zmin: -1,
    zmax: 1,
    zauto: false,
    hoverongaps: false,
    hovertemplate: "%{y} x %{x}<br>r=%{z:.2f}<extra></extra>",
  }];
  const layout = {
    template: "plotly_dark",
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: "#abb2bf", size: 10},
    xaxis: {automargin: true, tickangle: -45},
    yaxis: {automargin: true},
    autosize: true,
    height: Math.max(360, payload.columns.length * 28),
    margin: {l: 140, r: 20, t: 20, b: 120},
  };
  const config = {responsive: true, displayModeBar: false};
  if (div._fullLayout) {
    Plotly.react(div, traces, layout, config);
  } else {
    Plotly.newPlot(div, traces, layout, config);
  }
}

function renderCategoryRollup(section, target) {
  const div = document.getElementById(section.categoryBarChartId);
  if (!div) return;
  const rollup = section.importanceCategoryRollup || {};
  const entries = rollup[target] || rollup[Object.keys(rollup)[0]];
  if (!entries) {
    Plotly.purge(div);
    return;
  }
  const cats = entries.map(e => e.category);
  const vals = entries.map(e => e.rf_sum);
  const traces = [{
    type: "bar",
    orientation: "h",
    x: vals,
    y: cats,
    marker: {color: "#61afef"},
    hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
  }];
  const layout = {
    template: "plotly_dark",
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: "#abb2bf"},
    xaxis: {title: "Σ rf_importance", rangemode: "tozero"},
    yaxis: {automargin: true, autorange: "reversed"},
    autosize: true,
    height: Math.max(180, cats.length * 32),
    margin: {l: 120, r: 20, t: 10, b: 40},
  };
  const config = {responsive: true, displayModeBar: false};
  if (div._fullLayout) {
    Plotly.react(div, traces, layout, config);
  } else {
    Plotly.newPlot(div, traces, layout, config);
  }
}

function defaultCategoryTarget(section) {
  const rollup = section.importanceCategoryRollup || {};
  if ("mean_tcp_throughput" in rollup) return "mean_tcp_throughput";
  const keys = Object.keys(rollup);
  return keys.length ? keys[0] : null;
}

function boot() {
  const dataScripts = document.querySelectorAll(
    'script[type="application/json"][id^="section-data-"]');
  for (const dataEl of dataScripts) {
    const section = JSON.parse(dataEl.textContent);
    SECTIONS[section.hwSlug] = section;
    const panel = document.querySelector(
      `.panel[data-hw-slug="${section.hwSlug}"]`);
    // Host state, trajectory, correlation, and category-rollup are
    // wired independently of Pareto-row availability: a hardware
    // class without a usable frontier can still have per-iteration
    // snapshots, a trajectory, or an importance rollup worth
    // inspecting.
    setupHostStateChart(section);
    renderTrajectoryChart(section);
    renderCorrelationHeatmap(section);
    const catTarget = defaultCategoryTarget(section);
    if (catTarget) renderCategoryRollup(section, catTarget);
    if (!panel) continue;
    if (!section.paretoRows || section.paretoRows.length === 0) {
      panel.querySelector(".ranked-table-wrapper").innerHTML =
        "<p class='meta'>No Pareto-frontier rows.</p>";
      continue;
    }
    renderSection(panel, section);
  }
  wireImportanceSelectors();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
"""


def write_index_html(output_dir: Path, sections: list[dict[str, Any]]) -> Path:
    """Write a consolidated interactive HTML index.

    Args:
        output_dir: Destination directory; created if necessary.
        sections: One dict per hardware-class section (see
            :func:`_render_section`).

    Returns:
        The path of the written ``index.html`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nav_links = " ".join(
        f"<a href='#hw-{_slug(s['hardware_class'])}'>"
        f"{html.escape(s['hardware_class'])}</a>"
        for s in sections
    )

    section_html = "\n".join(_render_section(s) for s in sections)

    doc = (
        "<!doctype html>\n<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        '<meta name="color-scheme" content="dark">\n'
        '<meta name="theme-color" content="#21252b">\n'
        "<title>kube-autotuner analysis report</title>\n"
        f"<style>{_STYLE}</style>\n"
        f'<script src="{_PLOTLY_CDN}"></script>\n'
        "</head>\n<body>\n"
        "<header><h1>kube-autotuner analysis report</h1></header>\n"
        f"<nav class='top'>Jump to: {nav_links}</nav>\n"
        "<main>\n" + section_html + "\n</main>\n"
        f"<script type='module'>{_JS_MODULE}</script>\n"
        "</body>\n</html>\n"
    )

    path = output_dir / "index.html"
    path.write_text(doc)
    return path
