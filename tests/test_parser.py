"""Unit tests for :mod:`kube_autotuner.benchmark.parser`."""

from __future__ import annotations

import pytest

from kube_autotuner.benchmark.errors import ResultValidationError
from kube_autotuner.benchmark.parser import parse_iperf_json, parse_k8s_memory

SAMPLE_TCP_JSON = {
    "start": {
        "timestamp": {"timesecs": 1700000000},
    },
    "end": {
        "sum_sent": {
            "bits_per_second": 9_400_000_000.0,
            "retransmits": 12,
            "bytes": 35_250_000_000,
        },
        "sum_received": {
            "bits_per_second": 9_380_000_000.0,
        },
        "cpu_utilization_percent": {
            "host_total": 35.2,
            "remote_total": 22.1,
        },
    },
}

SAMPLE_UDP_JSON = {
    "start": {
        "timestamp": {"timesecs": 1700000100},
    },
    "end": {
        "sum": {
            "bits_per_second": 1_000_000_000.0,
            "jitter_ms": 0.025,
            "lost_packets": 3,
            "packets": 85000,
        },
        "cpu_utilization_percent": {
            "host_total": 12.5,
            "remote_total": 8.3,
        },
    },
}


def test_parse_tcp():
    result = parse_iperf_json(SAMPLE_TCP_JSON, "tcp")
    assert result.mode == "tcp"
    assert result.bits_per_second == pytest.approx(9_400_000_000.0)
    assert result.retransmits == 12
    assert result.bytes_sent == 35_250_000_000
    assert result.cpu_utilization_percent == pytest.approx(35.2)
    assert result.cpu_server_percent == pytest.approx(22.1)
    assert result.jitter is None
    assert result.timestamp.year == 2023


def test_parse_tcp_bytes_missing():
    raw = {
        "start": {"timestamp": {"timesecs": 1700000000}},
        "end": {"sum_sent": {"bits_per_second": 1e9, "retransmits": 0}},
    }
    result = parse_iperf_json(raw, "tcp")
    assert result.bytes_sent is None


def test_parse_cpu_server_percent_missing():
    raw = {
        "start": {"timestamp": {"timesecs": 1700000000}},
        "end": {
            "sum_sent": {"bits_per_second": 1e9, "retransmits": 0},
            "cpu_utilization_percent": {"host_total": 10.0},
        },
    }
    result = parse_iperf_json(raw, "tcp")
    assert result.cpu_server_percent is None


def test_parse_tags_client_and_iteration():
    result = parse_iperf_json(
        SAMPLE_TCP_JSON,
        "tcp",
        client_node="kmain09",
        iteration=2,
    )
    assert result.client_node == "kmain09"
    assert result.iteration == 2


def test_parse_udp():
    result = parse_iperf_json(SAMPLE_UDP_JSON, "udp")
    assert result.mode == "udp"
    assert result.bits_per_second == pytest.approx(1_000_000_000.0)
    assert result.jitter == pytest.approx(0.025 / 1000.0)
    assert result.retransmits is None
    assert result.bytes_sent is None
    assert result.packets == 85000
    assert result.lost_packets == 3
    assert result.cpu_utilization_percent == pytest.approx(12.5)


def test_parse_tcp_leaves_udp_only_fields_none():
    result = parse_iperf_json(SAMPLE_TCP_JSON, "tcp")
    assert result.packets is None
    assert result.lost_packets is None


def test_parse_udp_lost_packets_missing():
    raw = {
        "start": {"timestamp": {"timesecs": 1700000000}},
        "end": {"sum": {"packets": 100, "bits_per_second": 1e9}},
    }
    result = parse_iperf_json(raw, "udp")
    assert result.packets == 100
    assert result.lost_packets is None


def test_parse_minimal_json_raises_on_missing_sum_sent():
    """TCP without ``end.sum_sent`` is degenerate; the parser raises."""
    with pytest.raises(ResultValidationError, match="sum_sent"):
        parse_iperf_json({"end": {}}, "tcp")


def test_parse_raises_on_iperf_error_field():
    with pytest.raises(ResultValidationError, match="connection refused"):
        parse_iperf_json({"error": "unable to connect: connection refused"}, "tcp")


def test_parse_raises_on_missing_end_block():
    with pytest.raises(ResultValidationError, match="'end' block"):
        parse_iperf_json({}, "tcp")


def test_parse_raises_on_tcp_zero_bytes_and_zero_bps():
    raw = {"end": {"sum_sent": {"bytes": 0, "bits_per_second": 0.0}}}
    with pytest.raises(ResultValidationError, match="zero bytes"):
        parse_iperf_json(raw, "tcp")


def test_parse_tcp_accepts_nonzero_bytes_even_if_bps_zero():
    raw = {"end": {"sum_sent": {"bytes": 1024, "bits_per_second": 0.0}}}
    result = parse_iperf_json(raw, "tcp")
    assert result.bytes_sent == 1024


def test_parse_udp_accepts_nonzero_packets_even_if_bps_zero():
    raw = {"end": {"sum": {"packets": 7, "bits_per_second": 0.0}}}
    result = parse_iperf_json(raw, "udp")
    assert result.bits_per_second == pytest.approx(0.0)


def test_parse_raises_on_udp_zero_packets_and_zero_bps():
    raw = {"end": {"sum": {"packets": 0, "bits_per_second": 0.0}}}
    with pytest.raises(ResultValidationError, match="zero packets"):
        parse_iperf_json(raw, "udp")


def test_parse_preserves_raw():
    result = parse_iperf_json(SAMPLE_TCP_JSON, "tcp")
    assert result.raw_json == SAMPLE_TCP_JSON


def test_parse_k8s_memory_mi():
    assert parse_k8s_memory("45Mi") == 45 * 1024**2


def test_parse_k8s_memory_gi():
    assert parse_k8s_memory("1Gi") == 1024**3


def test_parse_k8s_memory_ki():
    assert parse_k8s_memory("65536Ki") == 65536 * 1024


def test_parse_k8s_memory_bare_number():
    assert parse_k8s_memory("1048576") == 1048576


def test_parse_k8s_memory_invalid():
    with pytest.raises(ValueError, match="Cannot parse memory string"):
        parse_k8s_memory("invalid")
