"""Unit tests for host-state rendering in the HTML report."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import pytest

pd = pytest.importorskip("pandas")

from kube_autotuner import report  # noqa: E402
from kube_autotuner.experiment import ObjectivesSection  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


_DEFAULT_WEIGHTS = ObjectivesSection().recommendation_weights


def _minimal_section(hw: str) -> dict[str, Any]:
    """Build a minimal per-hardware-class section dict for report tests.

    Mirrors the fixture in ``tests/test_report.py`` but trimmed to only
    the fields :func:`report._render_section` reads when rendering the
    host-state block.

    Returns:
        A section dict suitable for :func:`report.write_index_html` or
        :func:`report._render_section`.
    """
    pareto_rows = [
        {
            "trial_id": f"trial-{hw}-00",
            "sysctl_values": {"net.core.rmem_max": "67108864"},
            "mean_tcp_throughput": 4.2e10,
            "mean_udp_throughput": 9.5e9,
            "tcp_retransmit_rate": 1e-8,
            "udp_loss_rate": 0.001,
            "mean_udp_jitter": 0.0001,
            "mean_rps": 12345.0,
            "mean_latency_p50_ms": 1.0,
            "mean_latency_p90_ms": 2.0,
            "mean_latency_p99_ms": 3.0,
            "memory_cost": 1_000_000.0,
            "score": 1.0,
        },
    ]
    all_rows = [
        {
            "trial_id": pareto_rows[0]["trial_id"],
            "pareto": True,
            "mean_tcp_throughput": 4.2e10,
            "tcp_retransmit_rate": 1e-8,
        },
    ]
    objectives = [
        {"metric": "tcp_throughput", "direction": "maximize"},
        {"metric": "tcp_retransmit_rate", "direction": "minimize"},
    ]
    return {
        "hardware_class": hw,
        "trial_count": 1,
        "pareto_count": 1,
        "topology": None,
        "top_n": 1,
        "recommendations": [
            {
                "rank": 1,
                "trial_id": pareto_rows[0]["trial_id"],
                "sysctl_values": {},
                "mean_tcp_throughput": 4.2e10,
                "score": 0.95,
            },
        ],
        "pareto_rows": pareto_rows,
        "objectives": objectives,
        "default_weights": dict(_DEFAULT_WEIGHTS),
        "memory_cost_weight": 0.1,
        "importance": pd.DataFrame(),
        "all_rows": all_rows,
        "axis_columns": ["mean_tcp_throughput", "tcp_retransmit_rate"],
    }


def _host_state_payload(
    *,
    metrics: list[str] | None = None,
    points: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    default_metrics = [
        "conntrack_count",
        "slab_nf_conntrack_active_objs",
        "sockstat_tcp_inuse",
    ]
    default_points = [
        {
            "timestamp": "2026-04-24T10:00:00+00:00",
            "trial_id": "t1",
            "sysctl_hash": "abc123",
            "iteration": None,
            "phase": "baseline",
            "metrics": {
                "conntrack_count": 10,
                "slab_nf_conntrack_active_objs": 100,
                "sockstat_tcp_inuse": 3,
            },
        },
        {
            "timestamp": "2026-04-24T10:00:01+00:00",
            "trial_id": "t1",
            "sysctl_hash": "abc123",
            "iteration": 0,
            "phase": "post-flush",
            "metrics": {
                "conntrack_count": 12,
                "slab_nf_conntrack_active_objs": 110,
                "sockstat_tcp_inuse": 5,
            },
        },
        {
            "timestamp": "2026-04-24T10:00:02+00:00",
            "trial_id": "t1",
            "sysctl_hash": "abc123",
            "iteration": 0,
            "phase": "post-iteration",
            "metrics": {
                "conntrack_count": 18,
                "slab_nf_conntrack_active_objs": 130,
                "sockstat_tcp_inuse": 7,
            },
        },
    ]
    return {
        "metrics": metrics if metrics is not None else default_metrics,
        "points": points if points is not None else default_points,
    }


def _section_payload_from_html(html_text: str, hw_slug: str) -> dict[str, Any]:
    pattern = (
        r'<script type="application/json" id="section-data-'
        + re.escape(hw_slug)
        + r'">(.*?)</script>'
    )
    match = re.search(pattern, html_text, re.DOTALL)
    assert match is not None, f"section-data-{hw_slug} script not found"
    raw = match.group(1).replace("<\\/", "</")
    return json.loads(raw)


def test_render_section_omits_host_state_when_missing(tmp_path: Path) -> None:
    section = _minimal_section("10g")
    section["host_state"] = None
    path = report.write_index_html(tmp_path, [section])
    html_text = path.read_text()

    assert "<h3>Host state</h3>" not in html_text
    assert 'id="host-state-chart-10g"' not in html_text


def test_render_section_omits_host_state_when_key_absent(tmp_path: Path) -> None:
    section = _minimal_section("10g")
    # No host_state key at all; _render_section's .get() must tolerate it.
    assert "host_state" not in section
    path = report.write_index_html(tmp_path, [section])
    html_text = path.read_text()

    assert "<h3>Host state</h3>" not in html_text


def test_render_section_emits_host_state_skeleton(tmp_path: Path) -> None:
    section = _minimal_section("10g")
    section["host_state"] = _host_state_payload()
    path = report.write_index_html(tmp_path, [section])
    html_text = path.read_text()

    assert "<h3>Host state</h3>" in html_text
    assert 'id="host-state-chart-10g"' in html_text
    # Every metric in the payload shows up as an <option>.
    for metric in section["host_state"]["metrics"]:
        assert f'value="{metric}"' in html_text
    # Preferred metrics are pre-selected.
    assert 'value="conntrack_count" selected' in html_text
    assert 'value="slab_nf_conntrack_active_objs" selected' in html_text
    assert 'value="sockstat_tcp_inuse" selected' in html_text


def test_host_state_payload_embedded_for_js(tmp_path: Path) -> None:
    section = _minimal_section("10g")
    section["host_state"] = _host_state_payload()
    path = report.write_index_html(tmp_path, [section])
    html_text = path.read_text()

    payload = _section_payload_from_html(html_text, "10g")
    assert payload["hostStateChartId"] == "host-state-chart-10g"
    # figureDivIds must NOT include the host-state chart id (keeps
    # highlightInPlots from iterating traces without customdata).
    assert "host-state-chart-10g" not in payload["figureDivIds"]
    assert payload["hostState"] is not None
    assert payload["hostState"]["metrics"] == section["host_state"]["metrics"]
    # Baseline iteration is preserved as null through the JSON embed.
    first_point = payload["hostState"]["points"][0]
    assert first_point["iteration"] is None
    assert first_point["phase"] == "baseline"


def test_host_state_payload_survives_allow_nan_false() -> None:
    section = _minimal_section("10g")
    section["host_state"] = _host_state_payload()
    payload = report._section_payload(section)  # type: ignore[attr-defined]

    # Mirrors _embed_json's serialization guard; host_state contributes
    # only int metrics, so this must round-trip cleanly.
    dumped = json.dumps(payload, allow_nan=False, ensure_ascii=False)
    restored = json.loads(dumped)
    assert restored["hostState"]["metrics"] == section["host_state"]["metrics"]


def test_render_section_pre_select_falls_back_when_preferred_absent(
    tmp_path: Path,
) -> None:
    section = _minimal_section("10g")
    section["host_state"] = _host_state_payload(
        metrics=["netstat_ip_InSegs", "netstat_tcp_OutSegs"],
        points=[
            {
                "timestamp": "2026-04-24T10:00:00+00:00",
                "trial_id": "t1",
                "sysctl_hash": "abc123",
                "iteration": 0,
                "phase": "post-flush",
                "metrics": {
                    "netstat_ip_InSegs": 1000,
                    "netstat_tcp_OutSegs": 500,
                },
            },
        ],
    )
    path = report.write_index_html(tmp_path, [section])
    html_text = path.read_text()

    # No preferred metric present -> the first metric is pre-selected.
    assert 'value="netstat_ip_InSegs" selected' in html_text
    # The second option must render without 'selected' so the
    # multi-select opens with exactly one default choice.
    assert 'value="netstat_tcp_OutSegs">' in html_text


def test_js_module_wires_host_state_setup() -> None:
    # Guardrail against silent removal of the JS boot wiring.
    js = report._JS_MODULE  # type: ignore[attr-defined]
    assert "setupHostStateChart" in js
    assert "renderHostStateChart" in js
    assert "HOST_STATE_PHASE_SYMBOL" in js
    # Flat timestamp-sorted payload shape: the JS must read payload.points
    # with a date x-axis and must not regress to the old per-trial layout
    # or the -0.5 baseline hack.
    assert "payload.points" in js
    assert 'type: "date"' in js
    assert "payload.trials" not in js
    assert "-0.5" not in js
