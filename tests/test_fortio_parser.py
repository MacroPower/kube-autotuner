"""Tests for :mod:`kube_autotuner.benchmark.fortio_parser`."""

from __future__ import annotations

import pytest

from kube_autotuner.benchmark.fortio_parser import parse_fortio_json


def _full_payload() -> dict:
    return {
        "ActualQPS": 1234.5,
        "DurationHistogram": {
            "Count": 5000,
            "Percentiles": [
                {"Percentile": 50.0, "Value": 0.001},
                {"Percentile": 75.0, "Value": 0.002},
                {"Percentile": 90.0, "Value": 0.004},
                {"Percentile": 99.0, "Value": 0.010},
                {"Percentile": 99.9, "Value": 0.050},
            ],
        },
    }


def test_parses_rps_and_percentiles():
    result = parse_fortio_json(
        _full_payload(),
        client_node="kmain07",
        iteration=2,
        workload="fixed_qps",
    )
    assert result.rps == pytest.approx(1234.5)
    assert result.total_requests == 5000
    assert result.latency_p50_ms == pytest.approx(1.0)
    assert result.latency_p90_ms == pytest.approx(4.0)
    assert result.latency_p99_ms == pytest.approx(10.0)
    assert result.client_node == "kmain07"
    assert result.iteration == 2
    assert result.workload == "fixed_qps"


def test_missing_percentile_is_none():
    payload = _full_payload()
    payload["DurationHistogram"]["Percentiles"] = [
        {"Percentile": 50.0, "Value": 0.001},
    ]
    result = parse_fortio_json(
        payload,
        client_node="c1",
        iteration=0,
        workload="saturation",
    )
    assert result.latency_p50_ms == pytest.approx(1.0)
    assert result.latency_p90_ms is None
    assert result.latency_p99_ms is None


def test_missing_duration_histogram():
    result = parse_fortio_json(
        {"ActualQPS": 500.0},
        client_node="c1",
        iteration=0,
        workload="saturation",
    )
    assert result.rps == pytest.approx(500.0)
    assert result.total_requests is None
    assert result.latency_p50_ms is None
    assert result.latency_p90_ms is None
    assert result.latency_p99_ms is None


def test_missing_actual_qps_defaults_to_zero():
    result = parse_fortio_json(
        {"DurationHistogram": {"Count": 0, "Percentiles": []}},
        client_node="c1",
        iteration=0,
        workload="saturation",
    )
    assert result.rps == pytest.approx(0.0)
    assert result.total_requests == 0


def test_workload_tag_preserved():
    sat = parse_fortio_json(
        _full_payload(),
        client_node="c1",
        iteration=0,
        workload="saturation",
    )
    fixed = parse_fortio_json(
        _full_payload(),
        client_node="c1",
        iteration=0,
        workload="fixed_qps",
    )
    assert sat.workload == "saturation"
    assert fixed.workload == "fixed_qps"


def test_raw_json_preserved():
    payload = _full_payload()
    result = parse_fortio_json(
        payload,
        client_node="c1",
        iteration=0,
        workload="fixed_qps",
    )
    assert result.raw_json == payload


def test_percentile_float_tolerance_matches_near_integers():
    """Fortio's floating-point percentile keys within 1e-6 of the target match."""
    payload = {
        "ActualQPS": 100.0,
        "DurationHistogram": {
            "Count": 1,
            "Percentiles": [
                {"Percentile": 99.0 + 1e-9, "Value": 0.005},
            ],
        },
    }
    result = parse_fortio_json(
        payload,
        client_node="c1",
        iteration=0,
        workload="fixed_qps",
    )
    assert result.latency_p99_ms == pytest.approx(5.0)


def test_percentile_entry_without_value_returns_none():
    payload = {
        "ActualQPS": 100.0,
        "DurationHistogram": {
            "Count": 1,
            "Percentiles": [
                {"Percentile": 99.0},
            ],
        },
    }
    result = parse_fortio_json(
        payload,
        client_node="c1",
        iteration=0,
        workload="fixed_qps",
    )
    assert result.latency_p99_ms is None
