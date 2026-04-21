"""Parser for ``fortio load -json`` output."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from kube_autotuner.models import LatencyResult

_MS_PER_SECOND = 1000.0
_PERCENTILE_MATCH_TOLERANCE = 1e-6

Workload = Literal["saturation", "fixed_qps"]


def _percentile_value_ms(
    percentiles: list[dict[str, Any]],
    target: float,
) -> float | None:
    """Return the ``target`` percentile value from fortio's histogram.

    Fortio emits ``DurationHistogram.Percentiles`` as a list of
    ``{"Percentile": <pct>, "Value": <seconds>}`` objects. The exact
    float representation depends on how fortio was invoked, so the
    match is done with a small tolerance.

    Args:
        percentiles: The list under ``DurationHistogram.Percentiles``.
        target: Percentile requested (e.g. ``50.0``).

    Returns:
        The latency value in milliseconds, or ``None`` when ``target``
        was not requested at load time.
    """
    for entry in percentiles:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("Percentile")
        if raw is None:
            continue
        try:
            pct = float(raw)
        except TypeError, ValueError:
            continue
        if abs(pct - target) < _PERCENTILE_MATCH_TOLERANCE:
            value = entry.get("Value")
            if value is None:
                return None
            try:
                return float(value) * _MS_PER_SECOND
            except TypeError, ValueError:
                return None
    return None


def parse_fortio_json(
    raw: dict[str, Any],
    client_node: str,
    iteration: int,
    workload: Workload,
) -> LatencyResult:
    """Extract latency metrics from ``fortio load -json`` output.

    Args:
        raw: Parsed fortio JSON document (``HTTPRunnerResults`` shape
            at the top level; ``ActualQPS`` is embedded from
            ``RunnerResults``).
        client_node: Name of the node that ran this fortio client;
            recorded on the returned :class:`LatencyResult`.
        iteration: Iteration index (zero-based); recorded on the
            result.
        workload: ``"saturation"`` or ``"fixed_qps"``; recorded on the
            result so downstream aggregators know which metric the
            record is authoritative for.

    Returns:
        A :class:`LatencyResult` with parsed metrics. Missing
        percentile entries map to ``None`` rather than zero to
        preserve the "unobserved" signal for aggregation.
    """
    actual_qps = raw.get("ActualQPS")
    try:
        rps = float(actual_qps) if actual_qps is not None else 0.0
    except TypeError, ValueError:
        rps = 0.0

    histogram = raw.get("DurationHistogram") or {}
    total_requests_raw = histogram.get("Count")
    try:
        total_requests = (
            int(total_requests_raw) if total_requests_raw is not None else None
        )
    except TypeError, ValueError:
        total_requests = None

    percentiles = histogram.get("Percentiles") or []
    if not isinstance(percentiles, list):
        percentiles = []

    p50 = _percentile_value_ms(percentiles, 50.0)
    p90 = _percentile_value_ms(percentiles, 90.0)
    p99 = _percentile_value_ms(percentiles, 99.0)

    return LatencyResult(
        timestamp=datetime.now(UTC),
        workload=workload,
        client_node=client_node,
        iteration=iteration,
        rps=rps,
        total_requests=total_requests,
        latency_p50_ms=p50,
        latency_p90_ms=p90,
        latency_p99_ms=p99,
        raw_json=raw,
    )
