"""Unit tests for the interactive HTML report."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import pytest

pd = pytest.importorskip("pandas")

from kube_autotuner import report  # noqa: E402
from kube_autotuner.experiment import ObjectivesSection, ParetoObjective  # noqa: E402

_DEFAULT_WEIGHTS = ObjectivesSection().recommendation_weights

if TYPE_CHECKING:
    from pathlib import Path


def _minimal_section(
    hw: str,
    *,
    with_importance: bool = True,
    n_pareto_rows: int = 3,
    trial_count: int = 12,
) -> dict[str, Any]:
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
    pareto_rows = [
        {
            "trial_id": f"trial-{hw}-{i:02d}",
            "sysctl_values": {
                "net.core.rmem_max": "67108864",
                "net.ipv4.tcp_congestion_control": "bbr",
            },
            "mean_tcp_throughput": 4.2e10 - 1e9 * i,
            "mean_udp_throughput": 9.5e9 - 1e8 * i,
            "tcp_retransmit_rate": 1e-8 * (i + 1),
            "udp_loss_rate": 0.001 * (i + 1),
            "mean_udp_jitter": 0.0001,
            "mean_rps": 12345.0,
            "mean_latency_p50_ms": 1.0,
            "mean_latency_p90_ms": 2.0,
            "mean_latency_p99_ms": 3.0 + i,
            "memory_cost": float((i + 1) * 1_000_000),
            "score": 1.0 - 0.1 * i,
        }
        for i in range(n_pareto_rows)
    ]
    all_rows = [
        {
            "trial_id": f"trial-{hw}-{i:02d}",
            "pareto": i < n_pareto_rows,
            "mean_tcp_throughput": 4.2e10 - 1e9 * i,
            "tcp_retransmit_rate": 1e-8 * (i + 1),
        }
        for i in range(trial_count)
    ]
    axis_columns = ["mean_tcp_throughput", "tcp_retransmit_rate"]
    objectives = [
        {"metric": "tcp_throughput", "direction": "maximize"},
        {"metric": "udp_throughput", "direction": "maximize"},
        {"metric": "tcp_retransmit_rate", "direction": "minimize"},
        {"metric": "udp_loss_rate", "direction": "minimize"},
        {"metric": "udp_jitter", "direction": "minimize"},
        {"metric": "rps", "direction": "maximize"},
        {"metric": "latency_p50", "direction": "minimize"},
        {"metric": "latency_p90", "direction": "minimize"},
        {"metric": "latency_p99", "direction": "minimize"},
    ]
    return {
        "hardware_class": hw,
        "trial_count": trial_count,
        "pareto_count": n_pareto_rows,
        "topology": None,
        "top_n": 3,
        "recommendations": [
            {
                "rank": 1,
                "trial_id": pareto_rows[0]["trial_id"] if pareto_rows else "",
                "sysctl_values": {},
                "mean_tcp_throughput": 4.2e10,
                "score": 0.95,
            },
        ],
        "pareto_rows": pareto_rows,
        "objectives": objectives,
        "default_weights": dict(_DEFAULT_WEIGHTS),
        "memory_cost_weight": 0.1,
        "importance": importance_df,
        "all_rows": all_rows,
        "axis_columns": axis_columns,
    }


def _section_payload_from_html(html_text: str, hw_slug: str) -> dict[str, Any]:
    pattern = (
        r'<script type="application/json" id="section-data-'
        + re.escape(hw_slug)
        + r'">(.*?)</script>'
    )
    match = re.search(pattern, html_text, re.DOTALL)
    assert match is not None, f"section-data-{hw_slug} script not found"
    # The template escapes `</` as `<\/` so the payload can't close its own tag.
    raw = match.group(1).replace("<\\/", "</")
    return json.loads(raw)


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
    for hw in hw_classes:
        assert f"href='#hw-{hw}'" in html_text
        assert f"id='hw-{hw}'" in html_text
        assert f"Hardware class: {hw}" in html_text
        payload = _section_payload_from_html(html_text, hw)
        trial_ids = [r["trial_id"] for r in payload["paretoRows"]]
        assert f"trial-{hw}-00" in trial_ids


def test_write_index_html_slugifies_hardware_class_in_ids(tmp_path: Path) -> None:
    section = _minimal_section("10G NIC")
    path = report.write_index_html(tmp_path, [section])
    html_text = path.read_text()

    assert "href='#hw-10g-nic'" in html_text
    assert "id='hw-10g-nic'" in html_text
    assert 'id="axis-chart-10g-nic"' in html_text
    assert 'id="section-data-10g-nic"' in html_text
    assert "Hardware class: 10G NIC" in html_text


def test_write_index_html_handles_empty_importance(tmp_path: Path) -> None:
    section = _minimal_section("10g", with_importance=False)
    path = report.write_index_html(tmp_path, [section])
    html_text = path.read_text()

    assert "Parameter importance unavailable" in html_text
    assert "Objective space" in html_text
    assert 'id="axis-chart-10g"' in html_text


def test_write_index_html_axis_chart_defaults_are_selected(tmp_path: Path) -> None:
    section = _minimal_section("10g")
    path = report.write_index_html(tmp_path, [section])
    html_text = path.read_text()

    # Python's `selected` attribute must agree with the JS-side default
    # computation so the first paint does not mismatch the <select>
    # values displayed in the DOM.
    assert (
        '<select class="axis-select-x" data-hw-slug="10g">'
        '<option value="mean_tcp_throughput" selected>'
    ) in html_text
    assert '<option value="tcp_retransmit_rate" selected>' in html_text


def test_write_index_html_axis_chart_degenerates_below_two_columns(
    tmp_path: Path,
) -> None:
    section = _minimal_section("10g")
    section["axis_columns"] = ["mean_tcp_throughput"]
    path = report.write_index_html(tmp_path, [section])
    html_text = path.read_text()

    # Fallback message replaces the controls, but the div keeps its
    # stable id (hidden) so highlightInPlots can tolerate it.
    assert "Not enough metrics with data" in html_text
    assert re.search(
        r'<div[^>]*id="axis-chart-10g"[^>]*hidden',
        html_text,
    )
    assert 'class="axis-select-x"' not in html_text
    assert 'class="axis-select-y"' not in html_text


def test_write_index_html_no_legacy_pareto_labels(tmp_path: Path) -> None:
    section = _minimal_section("10g")
    path = report.write_index_html(tmp_path, [section])
    html_text = path.read_text()

    # Guard against regressing the scatter-matrix / Pareto-2D plots.
    assert "Pareto: " not in html_text
    assert "scatter matrix" not in html_text.lower()


def test_write_index_html_uses_cdn_not_inlined_plotly(tmp_path: Path) -> None:
    sections = [
        _minimal_section("10g"),
        _minimal_section("1g"),
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


def test_write_index_html_embeds_pareto_rows(tmp_path: Path) -> None:
    sections = [
        _minimal_section("10g", n_pareto_rows=4),
        _minimal_section("1g", n_pareto_rows=2),
    ]
    path = report.write_index_html(tmp_path, sections)
    html_text = path.read_text()

    data_scripts = re.findall(
        r'<script type="application/json" id="section-data-[^"]+">',
        html_text,
    )
    assert len(data_scripts) == len(sections)

    for section in sections:
        hw_slug = section["hardware_class"].lower()
        payload = _section_payload_from_html(html_text, hw_slug)
        assert payload["hardwareClass"] == section["hardware_class"]
        assert payload["hwSlug"] == hw_slug
        assert payload["trialCount"] == section["trial_count"]
        assert payload["paretoCount"] == section["pareto_count"]
        assert payload["defaultWeights"] == section["default_weights"]
        # memoryCostWeight is a top-level state field, NOT folded into
        # defaultWeights -- doing so would break preset parity in the
        # browser-side slider panel.
        assert payload["memoryCostWeight"] == section["memory_cost_weight"]
        assert "memoryCostWeight" not in payload["defaultWeights"]
        assert len(payload["paretoRows"]) == len(section["pareto_rows"])
        expected_ids = [r["trial_id"] for r in section["pareto_rows"]]
        actual_ids = [r["trial_id"] for r in payload["paretoRows"]]
        assert actual_ids == expected_ids
        # Every pareto row carries a ``memory_cost`` so the JS
        # ``scoreRows`` port can read it off the row dict directly.
        for row in payload["paretoRows"]:
            assert "memory_cost" in row
        expected_scores = [r["score"] for r in section["pareto_rows"]]
        actual_scores = [r["score"] for r in payload["paretoRows"]]
        assert actual_scores == expected_scores

        for obj in payload["objectives"]:
            ParetoObjective.model_validate(obj)

        assert payload["figureDivIds"] == [f"axis-chart-{hw_slug}"]
        assert payload["axisColumns"] == section["axis_columns"]
        assert len(payload["allRows"]) == len(section["all_rows"])

    js_module_matches = re.findall(
        r"<script type='module'>",
        html_text,
    )
    assert len(js_module_matches) == 1


def test_write_index_html_escapes_script_close_in_payload(tmp_path: Path) -> None:
    section = _minimal_section("10g", n_pareto_rows=1)
    section["pareto_rows"][0]["sysctl_values"]["payload"] = "</script><b>xss</b>"
    path = report.write_index_html(tmp_path, [section])
    html_text = path.read_text()

    # The verbatim `</script>` must not appear inside our data payload.
    assert "</script><b>xss</b>" not in html_text
    assert "<\\/script><b>xss<\\/b>" in html_text

    payload = _section_payload_from_html(html_text, "10g")
    assert payload["paretoRows"][0]["sysctl_values"]["payload"] == "</script><b>xss</b>"


def test_write_index_html_rejects_nan_in_payload(tmp_path: Path) -> None:
    section = _minimal_section("10g", n_pareto_rows=1)
    section["pareto_rows"][0]["mean_tcp_throughput"] = float("nan")

    with pytest.raises(ValueError, match="Out of range float values"):
        report.write_index_html(tmp_path, [section])


def _extract_js_object_literal(js: str, identifier: str) -> dict[str, Any]:
    """Pluck a ``const IDENTIFIER = {...};`` object literal out of JS source.

    The JS module is deliberately written so preset definitions are
    plain object literals (no variable references, no computed keys)
    so this shallow parser suffices.

    Returns:
        The parsed object literal as a Python dict.
    """
    # Match an outer { ... } with at most one level of nested braces;
    # tighter than the previous non-greedy `{.*?}` which would have
    # stopped at the first `};` inside a comment or string.
    match = re.search(
        r"const\s+" + re.escape(identifier) + r"\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\})\s*;",
        js,
        re.DOTALL,
    )
    assert match is not None, f"{identifier} not found in JS module"
    literal = match.group(1)
    # Quote bare keys so json.loads accepts the object.
    key_pattern = r"(?P<pre>[{,\s])(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:"
    quoted = re.sub(key_pattern, r'\g<pre>"\g<key>":', literal)
    return json.loads(quoted)


def test_presets_subset_of_defaults() -> None:
    presets = _extract_js_object_literal(report._JS_MODULE, "PRESETS")
    default_keys = set(_DEFAULT_WEIGHTS.keys())
    for name, overrides in presets.items():
        if overrides is None:
            # The "default" preset copies defaultWeights verbatim at runtime.
            continue
        unknown = set(overrides.keys()) - default_keys
        assert not unknown, (
            f"preset {name!r} references non-default keys {sorted(unknown)}"
        )


def test_presets_labels_cover_every_preset() -> None:
    presets = _extract_js_object_literal(report._JS_MODULE, "PRESETS")
    labels = _extract_js_object_literal(report._JS_MODULE, "PRESET_LABELS")
    assert set(presets.keys()) == set(labels.keys())


def test_decomposition_table_weight_cell_is_always_numeric() -> None:
    """The JS decomposition table no longer emits ``-`` for maximize rows.

    Before weights applied to both directions, the decomposition
    table rendered ``"-"`` in the weight column for maximize metrics.
    After the change every row shows a numeric weight; guard against
    a future regression that reintroduces the placeholder.
    """
    js = report._JS_MODULE
    assert 'c.direction === "maximize"\n      ? "-"' not in js
    assert '? "-" : c.weight.toFixed(2)' not in js


def test_render_sliders_does_not_filter_by_direction() -> None:
    """``renderSliders`` must iterate every pareto objective.

    The historical ``minimizeMetricKeys`` helper restricted sliders
    to minimize metrics. Confirm the rename landed and no stale
    direction filter survives inside the renderer.
    """
    js = report._JS_MODULE
    assert "minimizeMetricKeys" not in js
    assert "weightedMetricKeys" in js
    # renderSliders iterates every pareto objective; no direction
    # filter should live inside the renderer.
    render_start = js.index("function renderSliders(")
    render_end = js.index("function syncSliderValues(")
    render_src = js[render_start:render_end]
    assert ".filter(" not in render_src


# The three helpers below mirror the JS ``directionDefault``,
# ``applyPreset``, and ``activePresetKey`` in ``report._JS_MODULE``.
# The JS is the source of truth; these Python mirrors exist so the
# preset round-trip can be asserted without an in-test JS runtime.
# Update both sides in lockstep when the preset logic changes.
def _direction_default(section: dict[str, Any], metric: str) -> float:
    obj = next(o for o in section["objectives"] if o["metric"] == metric)
    return 1.0 if obj["direction"] == "maximize" else 0.0


def _apply_preset_py(
    key: str,
    section: dict[str, Any],
    presets: dict[str, dict[str, float] | None],
) -> dict[str, float]:
    keys = [o["metric"] for o in section["objectives"]]
    overrides = presets[key]
    if overrides is None:
        return {
            k: section["default_weights"].get(k, _direction_default(section, k))
            for k in keys
        }
    result = {k: _direction_default(section, k) for k in keys}
    for k, v in overrides.items():
        if k in result:
            result[k] = v
    return result


def _active_preset_key_py(
    weights: dict[str, float],
    section: dict[str, Any],
    presets: dict[str, dict[str, float] | None],
) -> str | None:
    keys = [o["metric"] for o in section["objectives"]]
    for key in presets:
        target = _apply_preset_py(key, section, presets)
        match = True
        for k in keys:
            fallback = _direction_default(section, k)
            wv = weights.get(k, fallback)
            tv = target.get(k, fallback)
            if abs(wv - tv) > 1e-9:
                match = False
                break
        if match:
            return key
    return None


def test_apply_preset_throughput_only_sets_maximize_to_one() -> None:
    """``throughput-only`` zeros minimize weights and keeps maximize at 1.0."""
    section = _minimal_section("10g", n_pareto_rows=1)
    presets = _extract_js_object_literal(report._JS_MODULE, "PRESETS")
    weights = _apply_preset_py("throughput-only", section, presets)
    for obj in section["objectives"]:
        if obj["direction"] == "maximize":
            assert weights[obj["metric"]] == pytest.approx(1.0)
        else:
            assert weights[obj["metric"]] == pytest.approx(0.0)


def test_active_preset_key_round_trip_throughput_only() -> None:
    """The weights produced by ``throughput-only`` round-trip back to it."""
    section = _minimal_section("10g", n_pareto_rows=1)
    presets = _extract_js_object_literal(report._JS_MODULE, "PRESETS")
    weights = _apply_preset_py("throughput-only", section, presets)
    assert _active_preset_key_py(weights, section, presets) == "throughput-only"


def test_apply_preset_latency_sensitive_keeps_maximize_at_one() -> None:
    """``latency-sensitive`` zeros non-overridden minimize weights.

    Maximize weights stay at their 1.0 default.
    """
    section = _minimal_section("10g", n_pareto_rows=1)
    presets = _extract_js_object_literal(report._JS_MODULE, "PRESETS")
    overrides = presets["latency-sensitive"]
    assert overrides is not None
    weights = _apply_preset_py("latency-sensitive", section, presets)
    for obj in section["objectives"]:
        metric = obj["metric"]
        if obj["direction"] == "maximize":
            assert weights[metric] == pytest.approx(1.0)
        elif metric in overrides:
            assert weights[metric] == pytest.approx(overrides[metric])
        else:
            assert weights[metric] == pytest.approx(0.0)


def test_write_index_html_is_dark_themed(tmp_path: Path) -> None:
    path = report.write_index_html(tmp_path, [_minimal_section("10g")])
    html_text = path.read_text()
    assert "color-scheme: dark" in html_text
    for forbidden in (
        "#fafafa",
        "#fff7e6",
        "#f4f4f4",
        "#c6f6c0",
        "#f8c3c3",
        "#cfe2ff",
        "#2b8a3e",
        "#c92a2a",
        "#06c",
    ):
        assert forbidden not in html_text, (
            f"light-palette literal {forbidden} leaked into report"
        )


def test_decomposition_uses_table_not_plotly() -> None:
    js = report._JS_MODULE
    assert "renderDecompositionTable(" in js
    assert "decomposition-table" in js
    # Match the function definition with a word boundary so the new
    # name `renderDecompositionTable` does not satisfy the assertion.
    assert not re.search(r"\bfunction\s+renderDecomposition\s*\(", js)
    assert "Plotly.newPlot(divEl" not in js
    assert "decomposition-chart" not in js


def test_decomposition_table_is_styled() -> None:
    css = report._STYLE
    assert "decomposition-wrapper" in css
    assert "table.decomposition-table" in css
    assert ".decomposition-chart" not in css


def test_write_index_html_emits_decomposition_wrapper(
    tmp_path: Path,
) -> None:
    path = report.write_index_html(tmp_path, [_minimal_section("10g")])
    html_text = path.read_text()
    assert "decomposition-wrapper" in html_text
    assert "decomposition-table" in html_text


def test_js_score_rows_port_matches_python(tmp_path: Path) -> None:  # noqa: PLR0914 - parity replay needs both JS and Python bookkeeping
    """Replay the JS ``scoreRows`` arithmetic and check it matches Python.

    The browser-side decomposition panel has a standalone port of
    :func:`kube_autotuner.scoring.score_rows` (``report.py:scoreRows``)
    that must match the Python scorer for the same embedded payload.
    We recompute the JS formula here against the embedded JSON and
    assert the ranking agrees with Python's ``score_rows``.
    """
    from kube_autotuner.experiment import ParetoObjective as Obj  # noqa: PLC0415
    from kube_autotuner.scoring import METRIC_TO_DF_COLUMN, score_rows  # noqa: PLC0415

    section = _minimal_section("10g", n_pareto_rows=5)
    # Give rows distinct-enough metric spreads that the ranking is not
    # determined by a single axis.
    for i, row in enumerate(section["pareto_rows"]):
        row["mean_tcp_throughput"] = 1.0e10 + 1e8 * i
        row["tcp_retransmit_rate"] = 1e-6 * (5 - i)
        row["memory_cost"] = float((i + 1) * 1_000_000_000)
    section["memory_cost_weight"] = 0.1

    path = report.write_index_html(tmp_path, [section])
    payload = _section_payload_from_html(path.read_text(), "10g")

    rows = payload["paretoRows"]
    objectives = payload["objectives"]
    weights = payload["defaultWeights"]
    mw = payload["memoryCostWeight"]

    def _normalize(values: list[float]) -> list[float]:
        finite = [v for v in values if v is not None]
        if not finite:
            return [0.5] * len(values)
        lo, hi = min(finite), max(finite)
        if lo == hi:
            return [0.5] * len(values)
        span = hi - lo
        return [0.5 if v is None else (v - lo) / span for v in values]

    # JS port arithmetic, stdlib-only. Mirrors the post-fa45690
    # direction-sensitive weight defaults (maximize -> 1.0, minimize -> 0.0).
    n = len(rows)
    js_scores = [0.0] * n
    for obj in objectives:
        col = METRIC_TO_DF_COLUMN[obj["metric"]]
        raw = [r.get(col) for r in rows]
        norm = _normalize(raw)
        if obj["direction"] == "maximize":
            w = weights.get(obj["metric"], 1.0)
            for i, v in enumerate(norm):
                js_scores[i] += w * v
        else:
            w = weights.get(obj["metric"], 0.0)
            for i, v in enumerate(norm):
                js_scores[i] -= w * v
    cost_norm = _normalize([r.get("memory_cost") for r in rows])
    for i, v in enumerate(cost_norm):
        js_scores[i] -= mw * v

    # Python scorer over the same payload.
    py_objectives = [Obj.model_validate(o) for o in objectives]
    py_scores = score_rows(
        rows,
        py_objectives,
        weights,
        memory_costs=[r["memory_cost"] for r in rows],
        memory_cost_weight=mw,
    )
    # Rankings must match even if absolute scores differ by rounding.
    js_order = sorted(range(n), key=lambda i: (-js_scores[i], i))
    py_order = sorted(range(n), key=lambda i: (-py_scores[i], i))
    assert js_order == py_order
    for a, b in zip(js_scores, py_scores, strict=True):
        assert a == pytest.approx(b)
