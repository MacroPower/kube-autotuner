"""Interactive HTML report combining per-hardware-class analysis output.

Each hardware-class section embeds its Pareto-frontier rows as JSON and
a vanilla-JS module renders a slider panel, a re-ranked top-N table,
and lazy score-decomposition charts. Weights are applied in the
browser via a direct port of
:func:`kube_autotuner.scoring.score_rows`; the Pareto frontier itself
is weight-invariant so no Python round-trip is needed.

``pandas`` and ``plotly`` live in the optional ``analysis`` dependency
group. All runtime uses are lazy-imported inside function bodies so
that ``import kube_autotuner.report`` remains cheap under the base
``dev`` sync. Runtime helpers raise :class:`RuntimeError` with the
``uv sync --group analysis`` hint when the group is missing.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

_ANALYSIS_HINT = "install analysis group: uv sync --group analysis"

_TOP_IMPORTANCE_ROWS = 20

_MIN_AXIS_COLUMNS = 2

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def format_retransmit_rate(rate: float | None) -> str:
    """Format a retx-per-byte rate for display as retransmits per MB.

    Args:
        rate: Retransmits per byte, or ``None`` when the trial had no
            observable rate (UDP-only, empty TCP).

    Returns:
        ``"-"`` when ``rate`` is ``None``; otherwise ``rate * 1e6``
        formatted with two decimals.
    """
    if rate is None:
        return "-"
    return f"{rate * 1e6:.2f}"


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
            :func:`kube_autotuner.analysis.parameter_importance`.
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


def _section_payload(section: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON-serializable payload for one hardware-class section.

    Args:
        section: Section dict produced by ``_analyze_one_class``.

    Returns:
        A dict safe to pass to :func:`json.dumps` with
        ``allow_nan=False``: every unmeasured metric is ``None`` (from
        :func:`kube_autotuner.analysis.pareto_recommendation_rows`
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
        "figureDivIds": [_axis_chart_id(hw_slug)],
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
        "</div>\n"
        f'<div class="axis-chart" id="{html.escape(chart_id)}"></div>\n'
        "</section>"
    )


def _render_section(section: dict[str, Any]) -> str:
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

    axis_chart = _render_axis_chart(hw_slug, section["axis_columns"])

    return (
        f"<section class='hw' id='hw-{hw_slug}'>\n"
        f"<h2>Hardware class: {html.escape(hw)}</h2>\n"
        f"{meta}\n"
        f"{_render_interactive_panel(hw_slug)}\n"
        f"{data_script}\n"
        f"<h3>Parameter importance (top {_TOP_IMPORTANCE_ROWS})</h3>\n"
        f"{importance_block}\n"
        f"{axis_chart}\n"
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
  tcp_retransmit_rate: {label: "TCP retx", unit: "/MB",
                        format: v => (v * 1e6).toFixed(2)},
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
    trialTd.textContent = row.trial_id;
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

// Build the two traces (non-pareto grey + pareto red) for the
// interactive axis chart and hand them to Plotly. Preserves the
// top-N marker-size highlight across axis swaps by re-applying
// ``highlightInPlots`` with ``state.lastTopIds`` after every render.
function renderAxisChart(state) {
  const section = state.section;
  const div = document.getElementById("axis-chart-" + section.hwSlug);
  if (!div) return;
  const x = state.axisX;
  const y = state.axisY;
  const others = [];
  const paretos = [];
  for (const row of section.allRows) {
    (row.pareto ? paretos : others).push(row);
  }
  const traces = [
    {
      type: "scatter", mode: "markers", name: "other",
      x: others.map(r => axisValue(r, x)),
      y: others.map(r => axisValue(r, y)),
      customdata: others.map(r => r.trial_id),
      marker: {color: "#BABBBD", size: 6},
      hovertemplate:
        `trial %{customdata}<br>${axisLabel(x)}=%{x:.3g}<br>`
        + `${axisLabel(y)}=%{y:.3g}<extra></extra>`,
    },
    {
      type: "scatter", mode: "markers", name: "pareto",
      x: paretos.map(r => axisValue(r, x)),
      y: paretos.map(r => axisValue(r, y)),
      customdata: paretos.map(r => r.trial_id),
      marker: {color: "#EF553B", size: 9},
      hovertemplate:
        `trial %{customdata}<br>${axisLabel(x)}=%{x:.3g}<br>`
        + `${axisLabel(y)}=%{y:.3g}<extra></extra>`,
    },
  ];
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
    });
  }
}

function boot() {
  const dataScripts = document.querySelectorAll(
    'script[type="application/json"][id^="section-data-"]');
  for (const dataEl of dataScripts) {
    const section = JSON.parse(dataEl.textContent);
    const panel = document.querySelector(
      `.panel[data-hw-slug="${section.hwSlug}"]`);
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
