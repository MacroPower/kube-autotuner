"""Parser for ``fortio load -json`` output."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Any, Literal

from kube_autotuner.benchmark.errors import ResultValidationError
from kube_autotuner.models import LatencyResult

_PERCENTILE_MATCH_TOLERANCE = 1e-6
_FORTIO_RESULT_MARKER = "DurationHistogram"
_LOG_SNIPPET_HEAD = 400
_LOG_SNIPPET_TAIL = 400

Workload = Literal["saturation", "fixed_qps"]


def extract_fortio_result_json(output: str) -> dict[str, Any]:
    """Find the fortio results JSON document inside a noisy log stream.

    ``kubectl logs`` merges container stdout and stderr, so any
    non-result output (a startup banner, a structured log line, a
    warning) can appear before or after fortio's pretty-printed
    ``-json -`` document. A naive ``json.loads`` on the whole stream
    then fails with ``Extra data`` once the parser eats the first valid
    JSON value it encounters.

    This helper walks every ``{`` in ``output``, attempts to parse a
    JSON object starting at that position with
    :class:`json.JSONDecoder`'s ``raw_decode``, and returns the first
    object that carries the ``DurationHistogram`` key. That key
    distinguishes a fortio ``HTTPRunnerResults`` document from any
    incidental log object.

    Args:
        output: Raw container log body from a fortio client pod.

    Returns:
        The decoded results document.

    Raises:
        ValueError: When no JSON object containing ``DurationHistogram``
            is found in ``output``.
    """
    decoder = json.JSONDecoder()
    idx = 0
    length = len(output)
    while idx < length:
        next_brace = output.find("{", idx)
        if next_brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(output, next_brace)
        except json.JSONDecodeError:
            idx = next_brace + 1
            continue
        if isinstance(obj, dict) and _FORTIO_RESULT_MARKER in obj:
            return obj
        idx = end
    raise ValueError(_no_result_message(output))


def _no_result_message(output: str) -> str:
    """Build the diagnostic error for a log stream missing the result JSON.

    Includes the head and tail of the log so the operator can tell the
    difference between an empty pod log (a failed first-attempt pod that
    the Job logs path picked up), a fortio crash with only stderr text,
    and a JSON document with the wrong shape.

    Args:
        output: The raw log body that was scanned.

    Returns:
        A multi-line error message.
    """
    length = len(output)
    if length == 0:
        snippet = "<empty log>"
    elif length <= _LOG_SNIPPET_HEAD + _LOG_SNIPPET_TAIL:
        snippet = output
    else:
        omitted = length - _LOG_SNIPPET_HEAD - _LOG_SNIPPET_TAIL
        snippet = (
            output[:_LOG_SNIPPET_HEAD]
            + f"\n... [{omitted} bytes omitted] ...\n"
            + output[-_LOG_SNIPPET_TAIL:]
        )
    return (
        f"no fortio results JSON object (with {_FORTIO_RESULT_MARKER!r}) found "
        f"in container log output (total {length} bytes); log snippet:\n{snippet}"
    )


def _percentile_value_seconds(
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
        The latency value in seconds, or ``None`` when ``target``
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
                return float(value)
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

    Raises :class:`ResultValidationError` when the histogram carries no
    samples (``Count`` missing or zero). That is unambiguously a
    degenerate run (fortio always records at least one request when
    the test actually executed) and the benchmark runner's retry loop
    detects it before the zero-valued latencies pollute the optimizer's
    objective.

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

    Raises:
        ResultValidationError: ``DurationHistogram.Count`` is missing
            or zero.
    """
    histogram = raw.get("DurationHistogram") or {}
    total_requests_raw = histogram.get("Count")
    try:
        total_requests = (
            int(total_requests_raw) if total_requests_raw is not None else None
        )
    except TypeError, ValueError:
        total_requests = None
    if not total_requests:
        msg = f"fortio histogram is empty (Count={total_requests_raw!r})"
        raise ResultValidationError(msg)

    actual_qps = raw.get("ActualQPS")
    try:
        rps = float(actual_qps) if actual_qps is not None else 0.0
    except TypeError, ValueError:
        rps = 0.0

    percentiles = histogram.get("Percentiles") or []
    if not isinstance(percentiles, list):
        percentiles = []

    p50 = _percentile_value_seconds(percentiles, 50.0)
    p90 = _percentile_value_seconds(percentiles, 90.0)
    p99 = _percentile_value_seconds(percentiles, 99.0)

    return LatencyResult(
        timestamp=datetime.now(UTC),
        workload=workload,
        client_node=client_node,
        iteration=iteration,
        rps=rps,
        total_requests=total_requests,
        latency_p50=p50,
        latency_p90=p90,
        latency_p99=p99,
        raw_json=raw,
    )
