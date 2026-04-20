"""Parsers for iperf3 JSON output and Kubernetes memory strings."""

from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import Any, Literal

from kube_autotuner.models import BenchmarkResult

_K8S_MEM_RE = re.compile(r"^(\d+)(Ki|Mi|Gi|Ti|k|M|G|T)?$")
_K8S_MEM_MULTIPLIERS: dict[str | None, int] = {
    None: 1,
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "k": 1000,
    "M": 1_000_000,
    "G": 1_000_000_000,
    "T": 1_000_000_000_000,
}


def parse_k8s_memory(mem_str: str) -> int:
    """Parse a Kubernetes memory string (e.g. ``"45Mi"``) to bytes.

    Args:
        mem_str: Memory string with optional IEC (``Ki``/``Mi``/``Gi``/
            ``Ti``) or SI (``k``/``M``/``G``/``T``) suffix. Bare integers
            are returned as-is.

    Returns:
        The decoded byte count.

    Raises:
        ValueError: When the string does not match the expected format.
    """
    m = _K8S_MEM_RE.match(mem_str.strip())
    if not m:
        msg = f"Cannot parse memory string: {mem_str!r}"
        raise ValueError(msg)
    value = int(m.group(1))
    suffix = m.group(2)
    return value * _K8S_MEM_MULTIPLIERS[suffix]


def parse_iperf_json(
    raw: dict[str, Any],
    mode: Literal["tcp", "udp"],
    client_node: str = "",
    iteration: int = 0,
) -> BenchmarkResult:
    """Extract metrics from ``iperf3 --json`` output.

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
    """
    end = raw.get("end", {})

    if mode == "tcp":
        sum_sent = end.get("sum_sent", {})
        bits_per_second = sum_sent.get("bits_per_second", 0.0)
        retransmits = sum_sent.get("retransmits")
        jitter_ms = None
    else:
        sum_data = end.get("sum", {})
        bits_per_second = sum_data.get("bits_per_second", 0.0)
        jitter_ms = sum_data.get("jitter_ms")
        retransmits = None

    cpu = end.get("cpu_utilization_percent", {})
    cpu_pct = cpu.get("host_total", 0.0)
    cpu_server = cpu.get("remote_total")

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
        cpu_utilization_percent=cpu_pct,
        cpu_server_percent=cpu_server,
        jitter_ms=jitter_ms,
        client_node=client_node,
        iteration=iteration,
        raw_json=raw,
    )
