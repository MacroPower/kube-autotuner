"""Tests for :mod:`kube_autotuner.benchmark.fortio_parser`."""

from __future__ import annotations

import json

import pytest

from kube_autotuner.benchmark.errors import ResultValidationError
from kube_autotuner.benchmark.fortio_parser import (
    extract_fortio_result_json,
    parse_fortio_json,
)


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
    assert result.latency_p50 == pytest.approx(0.001)
    assert result.latency_p90 == pytest.approx(0.004)
    assert result.latency_p99 == pytest.approx(0.010)
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
    assert result.latency_p50 == pytest.approx(0.001)
    assert result.latency_p90 is None
    assert result.latency_p99 is None


def test_missing_duration_histogram_raises():
    """A payload without ``DurationHistogram`` is degenerate; parser raises."""
    with pytest.raises(ResultValidationError, match="Count="):
        parse_fortio_json(
            {"ActualQPS": 500.0},
            client_node="c1",
            iteration=0,
            workload="saturation",
        )


def test_zero_count_histogram_raises():
    """``Count=0`` means no requests were issued; parser raises."""
    with pytest.raises(ResultValidationError, match="Count=0"):
        parse_fortio_json(
            {"DurationHistogram": {"Count": 0, "Percentiles": []}},
            client_node="c1",
            iteration=0,
            workload="saturation",
        )


def test_missing_actual_qps_defaults_to_zero():
    """Missing ``ActualQPS`` is tolerated when the histogram is populated."""
    result = parse_fortio_json(
        {"DurationHistogram": {"Count": 3, "Percentiles": []}},
        client_node="c1",
        iteration=0,
        workload="saturation",
    )
    assert result.rps == pytest.approx(0.0)
    assert result.total_requests == 3


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
    assert result.latency_p99 == pytest.approx(0.005)


def test_extract_returns_pure_json_unchanged():
    body = json.dumps(_full_payload(), indent=2)
    assert extract_fortio_result_json(body)["ActualQPS"] == pytest.approx(1234.5)


def test_extract_skips_leading_json_log_line():
    """Reproduces the prod failure: a JSON log record precedes the result."""
    log_line = json.dumps(
        {"ts": "2026-04-20T22:02:10Z", "level": "info", "msg": "starting load"},
    )
    body = log_line + "\n" + json.dumps(_full_payload(), indent=2)
    result = extract_fortio_result_json(body)
    assert "DurationHistogram" in result
    assert result["ActualQPS"] == pytest.approx(1234.5)


def test_extract_skips_non_json_banner_and_trailing_log():
    body = (
        "Fortio 1.69.0 running...\n"
        "Aki: target 1000 QPS\n"
        + json.dumps(_full_payload(), indent=2)
        + "\nAll done 5000 calls\n"
    )
    result = extract_fortio_result_json(body)
    assert result["DurationHistogram"]["Count"] == 5000


def test_extract_picks_result_when_multiple_json_objects_present():
    """An incidental log object before the result must not win."""
    earlier = json.dumps({"msg": "warmup", "qps": 100})
    body = earlier + "\n" + json.dumps(_full_payload()) + "\n"
    result = extract_fortio_result_json(body)
    assert "DurationHistogram" in result


def test_extract_raises_when_no_result_object_present():
    body = json.dumps({"msg": "fortio crashed"}) + "\nstack trace...\n"
    with pytest.raises(ValueError, match="DurationHistogram"):
        extract_fortio_result_json(body)


def test_extract_error_includes_log_snippet_and_size():
    body = "Aborting because of lookup nonexistent.local: no such host\n"
    with pytest.raises(ValueError, match="DurationHistogram") as exc:
        extract_fortio_result_json(body)
    msg = str(exc.value)
    assert f"total {len(body)} bytes" in msg
    assert "Aborting" in msg


def test_extract_error_marks_empty_log_clearly():
    with pytest.raises(ValueError, match="<empty log>") as exc:
        extract_fortio_result_json("")
    assert "total 0 bytes" in str(exc.value)


def test_extract_error_truncates_huge_log_with_byte_count():
    body = "x" * 5000
    with pytest.raises(ValueError, match="bytes omitted") as exc:
        extract_fortio_result_json(body)
    msg = str(exc.value)
    assert "total 5000 bytes" in msg
    assert len(msg) < len(body)


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
    assert result.latency_p99 is None
