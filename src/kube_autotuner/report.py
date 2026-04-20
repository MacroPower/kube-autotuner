"""Consolidated HTML report combining per-hardware-class analysis output.

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
    import plotly.graph_objects as go

_ANALYSIS_HINT = "install analysis group: uv sync --group analysis"

_TOP_IMPORTANCE_ROWS = 20


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
body { font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif;
       margin: 0; color: #222; background: #fafafa; }
header { padding: 1rem 2rem; background: #fff; border-bottom: 1px solid #ddd; }
header h1 { margin: 0; font-size: 1.4rem; }
nav.top { position: sticky; top: 0; z-index: 10; background: #fff;
          border-bottom: 1px solid #ddd; padding: 0.5rem 2rem; }
nav.top a { margin-right: 1rem; text-decoration: none; color: #06c; }
main { padding: 1rem 2rem 4rem; max-width: 1200px; margin: 0 auto; }
section.hw { margin-top: 2rem; padding-top: 1rem; border-top: 2px solid #ccc; }
section.fig { margin: 1.5rem 0; }
h2 { font-size: 1.2rem; }
h3 { font-size: 1rem; color: #444; }
table.report-table { border-collapse: collapse; margin: 0.5rem 0 1rem;
                     font-size: 0.9rem; background: #fff; }
table.report-table th, table.report-table td {
    border: 1px solid #ddd; padding: 4px 8px; text-align: left; }
table.report-table th { background: #f0f0f0; }
details { margin: 0.25rem 0 0.75rem; }
summary { cursor: pointer; color: #06c; }
pre { background: #f4f4f4; padding: 0.5rem; overflow-x: auto;
      font-size: 0.85rem; }
.meta { color: #666; font-size: 0.9rem; }
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


def _render_recommendations(recs: list[dict[str, Any]]) -> str:
    """Render the recommendations table and per-trial sysctl details.

    Args:
        recs: Recommendation dicts as produced by
            :func:`kube_autotuner.analysis.recommend_configs`.

    Lazy-imports ``pandas`` and raises :exc:`RuntimeError` with the
    ``uv sync --group analysis`` hint when the group is missing.

    Returns:
        An HTML fragment combining a summary table with one expandable
        ``<details>`` block per recommendation. When ``recs`` is
        empty, returns a short "No recommendations." note.
    """
    if not recs:
        return "<p class='meta'>No recommendations.</p>"

    pd = _require_pandas()

    display = pd.DataFrame(
        [
            {
                "rank": r["rank"],
                "trial_id": r["trial_id"],
                "throughput (Mbps)": round(r["mean_throughput"] / 1e6, 1),
                "cpu": f"{r['mean_cpu']:.1f}%",
                "node memory (MiB)": f"{r['mean_node_memory'] / 1024 / 1024:.0f}",
                "cni memory (MiB)": f"{r['mean_cni_memory'] / 1024 / 1024:.0f}",
                "retx/MB": format_retransmit_rate(r["retransmit_rate"]),
                "score": r["score"],
            }
            for r in recs
        ],
    )
    parts = [display.to_html(index=False, classes="report-table", border=0)]
    for r in recs:
        sysctl_json = json.dumps(r["sysctl_values"], indent=2)
        parts.append(
            f"<details><summary>sysctl values (trial "
            f"{html.escape(str(r['trial_id']))})"
            f"</summary><pre>{html.escape(sysctl_json)}</pre></details>",
        )
    return "\n".join(parts)


def _render_importance(df: pd.DataFrame, top_n: int = _TOP_IMPORTANCE_ROWS) -> str:
    """Render the parameter-importance table.

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
    return df.head(top_n).to_html(index=False, classes="report-table", border=0)


def _render_figure(
    hw_slug: str,
    label: str,
    fig: go.Figure,
    *,
    include_js: bool,
) -> str:
    """Render a single Plotly figure inside a ``<section class='fig'>`` block.

    Args:
        hw_slug: Slugified hardware-class label.
        label: Human-readable chart label (rendered as ``<h3>``).
        fig: The Plotly figure.
        include_js: When ``True``, emit the CDN ``<script>`` tag for
            the plotly bundle; the caller is responsible for passing
            ``True`` exactly once per document.

    Returns:
        An HTML fragment containing the chart.
    """
    div = fig.to_html(
        include_plotlyjs=("cdn" if include_js else False),
        full_html=False,
        div_id=f"fig-{hw_slug}-{_slug(label)}",
    )
    return f"<section class='fig'><h3>{html.escape(label)}</h3>\n{div}\n</section>"


def _render_section(
    section: dict[str, Any],
    *,
    include_js: bool,
) -> tuple[str, bool]:
    """Render one per-hardware-class section of the report.

    Args:
        section: Section payload with ``hardware_class``,
            ``topology``, ``trial_count``, ``pareto_count``,
            ``recommendations``, ``importance``, and ``figures``.
        include_js: ``True`` when the plotly CDN tag has not yet been
            emitted.

    Returns:
        An ``(html_fragment, include_js_remaining)`` pair. The second
        entry is ``False`` after this call consumes the CDN slot.
    """
    hw = section["hardware_class"]
    hw_slug = _slug(hw)
    topology = section.get("topology")
    topology_suffix = f", topology={html.escape(topology)}" if topology else ""
    meta = (
        f"<p class='meta'>{section['trial_count']} trials, "
        f"{section['pareto_count']} Pareto-optimal{topology_suffix}</p>"
    )

    fig_html: list[str] = []
    js_remaining = include_js
    for label, fig in section["figures"]:
        fig_html.append(_render_figure(hw_slug, label, fig, include_js=js_remaining))
        js_remaining = False

    body = (
        f"<section class='hw' id='hw-{hw_slug}'>\n"
        f"<h2>Hardware class: {html.escape(hw)}</h2>\n"
        f"{meta}\n"
        f"<h3>Top recommendations</h3>\n"
        f"{_render_recommendations(section['recommendations'])}\n"
        f"<h3>Parameter importance (top {_TOP_IMPORTANCE_ROWS})</h3>\n"
        f"{_render_importance(section['importance'])}\n"
        f"<h3>Charts</h3>\n{''.join(fig_html)}\n"
        f"</section>"
    )
    return body, js_remaining


def write_index_html(output_dir: Path, sections: list[dict[str, Any]]) -> Path:
    """Write a consolidated HTML index combining all hardware-class sections.

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

    include_js = True
    section_html: list[str] = []
    for s in sections:
        rendered, include_js = _render_section(s, include_js=include_js)
        section_html.append(rendered)

    doc = (
        "<!doctype html>\n<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<title>kube-autotuner analysis report</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n<body>\n"
        "<header><h1>kube-autotuner analysis report</h1></header>\n"
        f"<nav class='top'>Jump to: {nav_links}</nav>\n"
        "<main>\n" + "\n".join(section_html) + "\n</main>\n</body>\n</html>\n"
    )

    path = output_dir / "index.html"
    path.write_text(doc)
    return path
