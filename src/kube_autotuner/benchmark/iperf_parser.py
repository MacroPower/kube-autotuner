"""Parser for ``iperf3 --json`` output."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Any, Literal

from kube_autotuner.benchmark.errors import ResultValidationError
from kube_autotuner.models import BenchmarkResult


def parse_iperf_json(  # noqa: PLR0915 - linear sanity-check ladder, no branching
    raw: dict[str, Any],
    mode: Literal["tcp", "udp"],
    client_node: str = "",
    iteration: int = 0,
) -> BenchmarkResult:
    """Extract metrics from ``iperf3 --json`` output.

    Raises :class:`ResultValidationError` on unambiguously degenerate
    payloads so the benchmark runner's retry loop detects them before
    they pollute the optimizer's objective:

    * iperf3 reported a connection-level ``error`` field.
    * The ``end`` block is missing entirely.
    * For TCP, ``end.sum_sent`` is missing, or both ``bytes`` and
      ``bits_per_second`` are zero.
    * For UDP, ``end.sum`` is missing, or both ``packets`` and
      ``bits_per_second`` are zero.

    Cross-checking throughput against an independent counter
    (``bytes``/``packets``) avoids false positives from legitimate
    interval rounding that can push the end-of-run ``bits_per_second``
    to ``0.0`` on very short or UDP-heavy runs.

    Args:
        raw: Parsed iperf3 JSON document.
        mode: ``"tcp"`` or ``"udp"``; controls which branch of ``end`` is
            consulted (``sum_sent`` vs ``sum``) and whether retransmit or
            jitter metrics are populated.
        client_node: Name of the node that ran this iperf3 client;
            recorded on the returned :class:`BenchmarkResult`.
        iteration: Iteration index (zero-based); recorded on the result.

    Returns:
        A :class:`BenchmarkResult` with the parsed metrics. The original
        ``raw`` document is preserved in ``raw_json`` for downstream
        analysis. Missing timestamps default to :func:`datetime.now` in
        UTC.

    Raises:
        ResultValidationError: Payload is degenerate (see above).
    """
    err = raw.get("error")
    if err:
        msg = f"iperf3 reported error: {err!r}"
        raise ResultValidationError(msg)
    end = raw.get("end")
    if end is None:
        msg = "iperf3 result missing 'end' block"
        raise ResultValidationError(msg)

    if mode == "tcp":
        sum_sent = end.get("sum_sent")
        if not sum_sent:
            msg = "iperf3 TCP result missing 'end.sum_sent'"
            raise ResultValidationError(msg)
        bits_per_second = sum_sent.get("bits_per_second", 0.0)
        retransmits = sum_sent.get("retransmits")
        bytes_sent = sum_sent.get("bytes")
        jitter = None
        packets = None
        lost_packets = None
        if not bytes_sent and not bits_per_second:
            msg = "iperf3 TCP produced zero bytes and zero bits_per_second"
            raise ResultValidationError(msg)
    else:
        sum_data = end.get("sum")
        if not sum_data:
            msg = "iperf3 UDP result missing 'end.sum'"
            raise ResultValidationError(msg)
        bits_per_second = sum_data.get("bits_per_second", 0.0)
        jitter_raw = sum_data.get("jitter_ms")
        jitter = None if jitter_raw is None else jitter_raw / 1000.0
        retransmits = None
        bytes_sent = None
        packets = sum_data.get("packets")
        lost_packets = sum_data.get("lost_packets")
        if not packets and not bits_per_second:
            msg = "iperf3 UDP produced zero packets and zero bits_per_second"
            raise ResultValidationError(msg)

    ts_str = raw.get("start", {}).get("timestamp", {}).get("timesecs")
    if ts_str is not None:
        timestamp = datetime.fromtimestamp(ts_str, tz=UTC)
    else:
        timestamp = datetime.now(UTC)

    return BenchmarkResult(
        timestamp=timestamp,
        mode=mode,
        bits_per_second=bits_per_second,
        retransmits=retransmits,
        bytes_sent=bytes_sent,
        jitter=jitter,
        packets=packets,
        lost_packets=lost_packets,
        client_node=client_node,
        iteration=iteration,
        raw_json=raw,
    )


def parse_iperf_output(
    output: str,
    mode: Literal["tcp", "udp"],
    *,
    client_node: str,
    iteration: int,
) -> BenchmarkResult:
    """Parse a raw iperf3 client log into a :class:`BenchmarkResult`.

    Wraps :class:`json.JSONDecodeError` (and any other ``ValueError``
    raised by :func:`json.loads`) as
    :class:`ResultValidationError` so the benchmark runner's retry loop
    treats malformed JSON the same as semantically degenerate payloads.

    Args:
        output: Raw container log body from an iperf3 client pod.
        mode: ``"tcp"`` or ``"udp"``; forwarded to
            :func:`parse_iperf_json`.
        client_node: Name of the node that ran this iperf3 client.
        iteration: Zero-based iteration index.

    Returns:
        The parsed :class:`BenchmarkResult`.

    Raises:
        ResultValidationError: ``output`` is not valid JSON, or the
            decoded document is degenerate per :func:`parse_iperf_json`.
    """
    try:
        raw = json.loads(output)
    except ValueError as exc:
        msg = f"iperf3 log is not valid JSON: {exc}"
        raise ResultValidationError(msg) from exc
    return parse_iperf_json(raw, mode, client_node=client_node, iteration=iteration)
