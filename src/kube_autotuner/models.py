"""Pydantic data models for trial, benchmark, and node-pair records.

JSONL field names produced by :class:`TrialLog` are kept stable on a
best-effort basis -- they are not a frozen public contract. Downstream
consumers that parse trial JSONL should be prepared to re-analyse when
field names or structure evolve.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class SysctlParam(BaseModel):
    """A single sysctl knob with its discrete search space."""

    name: str
    values: list[int | str]
    param_type: Literal["int", "choice"]


class ParamSpace(BaseModel):
    """Ordered collection of :class:`SysctlParam` forming a search space."""

    params: list[SysctlParam]

    def param_names(self) -> list[str]:
        """Return the parameter names in declaration order."""
        return [p.name for p in self.params]


class NodePair(BaseModel):
    """Source and destination nodes for a benchmark run."""

    source: str
    target: str
    hardware_class: Literal["1g", "10g"]
    namespace: str = "default"
    source_zone: str = ""
    target_zone: str = ""
    ip_family_policy: str = "RequireDualStack"
    extra_sources: list[str] = Field(default_factory=list)
    extra_source_zones: dict[str, str] = Field(default_factory=dict)

    @property
    def topology(self) -> Literal["intra-az", "inter-az", "unknown"]:
        """Return the topology label derived from source/target zones."""
        if not self.source_zone or not self.target_zone:
            return "unknown"
        return "intra-az" if self.source_zone == self.target_zone else "inter-az"

    @property
    def all_sources(self) -> list[str]:
        """Return the primary source followed by any extra source clients."""
        return [self.source, *self.extra_sources]

    def zone_for(self, client: str) -> str:
        """Return the zone recorded for ``client``, or ``""`` if unknown.

        Args:
            client: Node name of a client participating in the benchmark.

        Returns:
            The zone string, or an empty string when no zone is known.
        """
        if client == self.source:
            return self.source_zone
        return self.extra_source_zones.get(client, "")


def _default_modes() -> list[Literal["tcp", "udp"]]:
    """Return the default benchmark modes list (TCP only)."""
    return ["tcp"]


class BenchmarkConfig(BaseModel):
    """Configuration for a single benchmark session."""

    duration: int = 30
    omit: int = 5
    iterations: int = 3
    parallel: int = 16
    window: str | None = None
    modes: list[Literal["tcp", "udp"]] = Field(default_factory=_default_modes)


class BenchmarkResult(BaseModel):
    """Parsed metrics from one iperf3 run."""

    timestamp: datetime
    mode: Literal["tcp", "udp"]
    bits_per_second: float
    retransmits: int | None = None
    cpu_utilization_percent: float = 0.0
    cpu_server_percent: float | None = None
    jitter_ms: float | None = None
    memory_used_bytes: int | None = Field(
        default=None,
        description="Deferred: requires cgroup/kubectl-top integration.",
    )
    client_node: str = ""
    iteration: int = 0
    raw_json: dict[str, Any] = Field(default_factory=dict)


def compute_sysctl_hash(sysctl_values: Mapping[str, str | int]) -> str:
    """Compute a short SHA-256 hash over sorted sysctl key-value pairs.

    The hash is used as a deduplication key for trials that apply the same
    sysctl set.

    Args:
        sysctl_values: Mapping of sysctl key to value.

    Returns:
        The first 16 hex characters of the SHA-256 digest over a canonical,
        sorted JSON representation of ``sysctl_values``.
    """
    canonical = json.dumps(sorted(sysctl_values.items()), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _group_by_iteration(
    results: list[BenchmarkResult],
) -> dict[int, list[BenchmarkResult]]:
    """Group ``results`` by their ``iteration`` index.

    Returns:
        A mapping from iteration index to the results recorded at that
        iteration, preserving insertion order within each group.
    """
    grouped: dict[int, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        grouped[r.iteration].append(r)
    return grouped


class TrialResult(BaseModel):
    """Aggregated result for one set of sysctls across iterations."""

    trial_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    node_pair: NodePair
    sysctl_values: dict[str, str | int]
    sysctl_hash: str = ""
    kernel_version: str = ""
    topology: Literal["intra-az", "inter-az", "unknown"] = "unknown"
    config: BenchmarkConfig
    results: list[BenchmarkResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, _context: Any, /) -> None:  # noqa: ANN401
        """Populate derived fields (``sysctl_hash`` and ``topology``)."""
        if not self.sysctl_hash:
            self.sysctl_hash = compute_sysctl_hash(self.sysctl_values)
        if self.topology == "unknown":
            self.topology = self.node_pair.topology

    def mean_throughput(self) -> float:
        """Return the mean total throughput in bits per second.

        Throughput is summed across clients within each iteration, then
        averaged across iterations.

        Returns:
            The averaged throughput, or ``0.0`` when there are no results.
        """
        if not self.results:
            return 0.0
        per_iter_sums = [
            sum(r.bits_per_second for r in group)
            for group in _group_by_iteration(self.results).values()
        ]
        return sum(per_iter_sums) / len(per_iter_sums)

    def mean_cpu(self) -> float:
        """Return the mean CPU utilization percent across iterations.

        Uses the server-side CPU when every record reports it (since all
        clients share the same server); otherwise falls back to the
        per-client host CPU.

        Returns:
            The averaged CPU percent, or ``0.0`` when there are no results.
        """
        if not self.results:
            return 0.0
        use_server = all(r.cpu_server_percent is not None for r in self.results)
        per_iter_means: list[float] = []
        for group in _group_by_iteration(self.results).values():
            if use_server:
                values = [cast("float", r.cpu_server_percent) for r in group]
            else:
                values = [r.cpu_utilization_percent for r in group]
            per_iter_means.append(sum(values) / len(values))
        return sum(per_iter_means) / len(per_iter_means)

    def total_retransmits(self) -> int:
        """Return the total retransmit count summed across all results."""
        return sum(r.retransmits or 0 for r in self.results)

    def mean_jitter_ms(self) -> float:
        """Return the mean jitter in milliseconds.

        Takes the per-iteration mean across clients that reported jitter,
        then averages across iterations.

        Returns:
            The averaged jitter, or ``0.0`` when no results report jitter.
        """
        per_iter_means: list[float] = []
        for group in _group_by_iteration(self.results).values():
            vals = [r.jitter_ms for r in group if r.jitter_ms is not None]
            if vals:
                per_iter_means.append(sum(vals) / len(vals))
        if not per_iter_means:
            return 0.0
        return sum(per_iter_means) / len(per_iter_means)


class TrialLog:
    """Append-only JSON-lines persistence for :class:`TrialResult` records."""

    @staticmethod
    def append(path: Path, trial: TrialResult) -> None:
        """Append ``trial`` as a single JSON line to ``path``.

        Args:
            path: Target JSONL file. Created if it does not exist.
            trial: The trial record to persist.
        """
        with path.open("a", encoding="utf-8") as f:
            f.write(trial.model_dump_json() + "\n")

    @staticmethod
    def load(path: Path) -> list[TrialResult]:
        """Load every trial record from the JSONL file at ``path``.

        Args:
            path: Source JSONL file. If it does not exist, an empty list is
                returned.

        Returns:
            The decoded trials, in file order.
        """
        if not path.exists():
            return []
        trials: list[TrialResult] = []
        with path.open(encoding="utf-8") as f:
            for raw in f:
                stripped = raw.strip()
                if stripped:
                    trials.append(TrialResult.model_validate_json(stripped))
        return trials
