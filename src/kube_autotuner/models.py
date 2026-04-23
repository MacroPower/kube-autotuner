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
import logging
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.alias_generators import to_camel

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    # Imported for the :class:`ResumeMetadata` forward reference only.
    # The actual import cannot live at module level because
    # :mod:`kube_autotuner.experiment` already imports from this module.
    # :func:`_ensure_resume_metadata_built` resolves the forward ref
    # lazily and is triggered from ``experiment``'s module initialiser.
    from kube_autotuner.experiment import ObjectivesSection  # noqa: TC004

logger = logging.getLogger(__name__)


class SysctlParam(BaseModel):
    """A single sysctl knob with its discrete search space."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

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
    hardware_class: str = Field(min_length=1)
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


class BenchmarkConfig(BaseModel):
    """Configuration for a single benchmark session."""

    duration: int = 30
    omit: int = 5
    iterations: int = 3
    parallel: int = 16
    window: str | None = None


class BenchmarkResult(BaseModel):
    """Parsed metrics from one iperf3 run."""

    timestamp: datetime
    mode: Literal["tcp", "udp"]
    bits_per_second: float
    retransmits: int | None = None
    bytes_sent: int | None = Field(
        default=None,
        description=(
            "Total bytes sent during the iperf3 run; populated from "
            "end.sum_sent.bytes for TCP, None for UDP or legacy records."
        ),
    )
    cpu_utilization_percent: float = 0.0
    cpu_server_percent: float | None = None
    jitter_ms: float | None = None
    packets: int | None = Field(
        default=None,
        description=(
            "Total UDP packets sent during the iperf3 run; populated from "
            "end.sum.packets for UDP, None for TCP."
        ),
    )
    lost_packets: int | None = Field(
        default=None,
        description=(
            "UDP datagrams reported lost during the iperf3 run, from "
            "end.sum.lost_packets; None for TCP."
        ),
    )
    node_memory_used_bytes: int | None = Field(
        default=None,
        description=(
            "Peak memory observed on the iperf target node during the "
            "iteration, sampled from metrics.k8s.io/v1beta1 nodes/<name>."
        ),
    )
    cni_memory_used_bytes: int | None = Field(
        default=None,
        description=(
            "Peak memory summed across CNI pods on the target node during "
            "the iteration, selected via the experiment's cni selector."
        ),
    )
    client_node: str = ""
    iteration: int = 0
    raw_json: dict[str, Any] = Field(default_factory=dict)


class LatencyResult(BaseModel):
    """Parsed metrics from one fortio load run.

    Fortio drives a request/response workload against the fortio server
    pod and returns per-run latency percentiles plus the achieved QPS.
    ``workload`` disambiguates the two sub-stages: ``"saturation"``
    runs fortio with ``-qps 0`` and is the sole source of the ``rps``
    metric, while ``"fixed_qps"`` runs at a stable offered rate and is
    the sole source of the latency percentiles.
    """

    timestamp: datetime
    workload: Literal["saturation", "fixed_qps"]
    client_node: str = ""
    iteration: int = 0
    rps: float = 0.0
    total_requests: int | None = None
    latency_p50_ms: float | None = None
    latency_p90_ms: float | None = None
    latency_p99_ms: float | None = None
    node_memory_used_bytes: int | None = None
    cni_memory_used_bytes: int | None = None
    raw_json: dict[str, Any] = Field(default_factory=dict)


class IterationResults(BaseModel):
    """Container for the two record streams produced by a single run.

    :class:`~kube_autotuner.benchmark.runner.BenchmarkRunner.run`
    returns one of these per call so callers can thread both the
    iperf3 ``bench`` records and the fortio ``latency`` records into
    the :class:`TrialResult` without adding a second return value.
    """

    bench: list[BenchmarkResult] = Field(default_factory=list)
    latency: list[LatencyResult] = Field(default_factory=list)


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


def tcp_retransmit_rate_by_iteration(
    results: list[BenchmarkResult],
) -> list[float]:
    """Return one TCP retransmit rate (retx per byte) per iteration.

    Each iteration's rate is ``sum(retransmits) / sum(bytes_sent)``
    over its records. Every iteration now runs both TCP and UDP
    bandwidth stages, so the filter's job is to skip iterations where
    no TCP record reported ``retransmits`` / ``bytes_sent`` (e.g. a
    failed ``bw-tcp`` stage). UDP records are dropped naturally
    because they never report ``bytes_sent``.

    Args:
        results: Raw benchmark records for a single trial.

    Returns:
        The per-iteration rates, in iteration-index order.
    """
    per_iter_retx: dict[int, int] = defaultdict(int)
    per_iter_bytes: dict[int, int] = defaultdict(int)
    per_iter_saw_retx: dict[int, bool] = defaultdict(bool)
    for r in results:
        if r.retransmits is not None:
            per_iter_retx[r.iteration] += r.retransmits
            per_iter_saw_retx[r.iteration] = True
        if r.bytes_sent is not None and r.bytes_sent > 0:
            per_iter_bytes[r.iteration] += r.bytes_sent
    return [
        per_iter_retx[it] / bytes_
        for it, bytes_ in per_iter_bytes.items()
        if bytes_ > 0 and per_iter_saw_retx[it]
    ]


def udp_loss_rate_by_iteration(results: list[BenchmarkResult]) -> list[float]:
    """Return one UDP loss rate (lost packets per packet) per iteration.

    Each iteration's rate is ``sum(lost_packets) / sum(packets)`` over
    its UDP records. Iterations where no UDP record reported a
    non-zero ``packets`` total are skipped; TCP records are dropped
    naturally because they never report ``packets`` / ``lost_packets``.

    Args:
        results: Raw benchmark records for a single trial.

    Returns:
        The per-iteration loss rates, in iteration-index order.
    """
    per_iter_lost: dict[int, int] = defaultdict(int)
    per_iter_packets: dict[int, int] = defaultdict(int)
    per_iter_saw_lost: dict[int, bool] = defaultdict(bool)
    for r in results:
        if r.lost_packets is not None:
            per_iter_lost[r.iteration] += r.lost_packets
            per_iter_saw_lost[r.iteration] = True
        if r.packets is not None and r.packets > 0:
            per_iter_packets[r.iteration] += r.packets
    return [
        per_iter_lost[it] / packets
        for it, packets in per_iter_packets.items()
        if packets > 0 and per_iter_saw_lost[it]
    ]


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


def _group_latency_by_iteration(
    results: list[LatencyResult],
) -> dict[int, list[LatencyResult]]:
    """Group fortio ``results`` by their ``iteration`` index.

    Returns:
        A mapping from iteration index to the latency records recorded
        at that iteration, preserving insertion order within each
        group.
    """
    grouped: dict[int, list[LatencyResult]] = defaultdict(list)
    for r in results:
        grouped[r.iteration].append(r)
    return grouped


def is_primary(t: TrialResult) -> bool:
    """Return ``True`` when ``t`` is a primary (sobol/bayesian/legacy) trial.

    Legacy rows written before this feature have ``phase is None`` and
    are primary by construction -- the verification loop did not exist
    at the time they were produced, so they cannot be verification
    repeats.

    Args:
        t: Trial record to classify.

    Returns:
        ``True`` for rows that belong in the primary Ax population;
        ``False`` for rows emitted by the verification pass.
    """
    return t.phase != "verification"


def effective_phase(t: TrialResult, index: int, n_sobol: int) -> str:
    """Return ``t.phase``, or infer from index for legacy primary rows.

    Callers MUST filter to primary rows first (``is_primary(t)``) --
    this helper is only valid for primary trials in file order and
    will happily mislabel a verification row if one slips in.

    Args:
        t: Trial record whose phase label is needed.
        index: Position of ``t`` within the primary-only subsequence.
        n_sobol: The experiment's Sobol initialization budget.

    Returns:
        The stored ``phase`` when present; otherwise ``"sobol"`` for
        indices below ``n_sobol`` and ``"bayesian"`` above.
    """
    if t.phase is not None:
        return t.phase
    return "sobol" if index < n_sobol else "bayesian"


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
    latency_results: list[LatencyResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    phase: Literal["sobol", "bayesian", "verification"] | None = None
    parent_trial_id: str | None = None

    def model_post_init(self, _context: Any, /) -> None:  # noqa: ANN401
        """Populate derived fields (``sysctl_hash`` and ``topology``)."""
        if not self.sysctl_hash:
            self.sysctl_hash = compute_sysctl_hash(self.sysctl_values)
        if self.topology == "unknown":
            self.topology = self.node_pair.topology

    def mean_tcp_throughput(self) -> float:
        """Return the mean total TCP throughput in bits per second.

        Filters to ``mode == "tcp"`` records before aggregating; UDP
        throughput lives on :meth:`mean_udp_throughput`. Throughput is
        summed across clients within each iteration, then averaged
        across iterations.

        Returns:
            The averaged TCP throughput, or ``0.0`` when there are no
            TCP results.
        """
        tcp_results = [r for r in self.results if r.mode == "tcp"]
        if not tcp_results:
            return 0.0
        per_iter_sums = [
            sum(r.bits_per_second for r in group)
            for group in _group_by_iteration(tcp_results).values()
        ]
        return sum(per_iter_sums) / len(per_iter_sums)

    def mean_udp_throughput(self) -> float:
        """Return the mean total UDP throughput in bits per second.

        Mirrors :meth:`mean_tcp_throughput` but filters to ``mode ==
        "udp"`` records. Throughput is summed across clients within
        each iteration, then averaged across iterations.

        Returns:
            The averaged UDP throughput, or ``0.0`` when there are no
            UDP results.
        """
        udp_results = [r for r in self.results if r.mode == "udp"]
        if not udp_results:
            return 0.0
        per_iter_sums = [
            sum(r.bits_per_second for r in group)
            for group in _group_by_iteration(udp_results).values()
        ]
        return sum(per_iter_sums) / len(per_iter_sums)

    def mean_cpu(self) -> float:
        """Return the mean CPU utilization percent across TCP iterations.

        Filters to ``mode == "tcp"`` records before aggregating -- the
        ``cpu`` objective has always meant CPU observed during the TCP
        bandwidth stage. Uses the server-side CPU when every TCP
        record reports it (since all clients share the same server);
        otherwise falls back to the per-client host CPU.

        Returns:
            The averaged CPU percent, or ``0.0`` when there are no TCP
            results.
        """
        tcp_results = [r for r in self.results if r.mode == "tcp"]
        if not tcp_results:
            return 0.0
        use_server = all(r.cpu_server_percent is not None for r in tcp_results)
        per_iter_means: list[float] = []
        for group in _group_by_iteration(tcp_results).values():
            if use_server:
                values = [cast("float", r.cpu_server_percent) for r in group]
            else:
                values = [r.cpu_utilization_percent for r in group]
            per_iter_means.append(sum(values) / len(values))
        return sum(per_iter_means) / len(per_iter_means)

    def total_bytes_sent(self) -> int:
        """Return total bytes sent, summed across every record.

        Returns:
            Sum of :attr:`BenchmarkResult.bytes_sent` over
            ``self.results``; zero when no record reports it.
        """
        return sum(r.bytes_sent or 0 for r in self.results)

    def udp_loss_rate(self) -> float:
        """Return UDP lost packets per packet sent as a per-trial rate.

        Aggregates per-iteration ratio-of-sums then means across
        iterations -- mirroring :meth:`tcp_retransmit_rate` but feeding
        from UDP-mode records via ``lost_packets`` / ``packets``.
        Iterations where no UDP record reported a non-zero ``packets``
        total are dropped from the mean.

        Returns:
            The averaged UDP loss rate (e.g. ``0.01`` is 1% loss), or
            ``0.0`` when no UDP iteration contributed (no UDP stage,
            or every UDP stage failed before reporting packet counts).
        """
        rate_vals = udp_loss_rate_by_iteration(self.results)
        if not rate_vals:
            return 0.0
        return sum(rate_vals) / len(rate_vals)

    def tcp_retransmit_rate(self) -> float | None:
        """Return TCP retransmits per byte as a per-trial rate.

        Aggregates per-iteration ratio-of-sums then means across
        iterations -- matching how :meth:`mean_tcp_throughput` folds
        clients within an iteration and averages across iterations.
        Iterations where no record reported ``bytes_sent`` (or the
        per-iteration byte total was zero) are dropped from the mean
        rather than treated as perfect-rate zero. UDP records never
        carry ``bytes_sent``, so they are filtered out by this same
        check.

        Returns:
            Retransmits per byte sent as a ``float`` (e.g. ``1e-6``
            is approximately 1 retransmit per MB) -- the default
            case, since every iteration now runs a TCP bandwidth
            stage. Returns ``None`` only when every ``bw-tcp`` stage
            failed.
        """
        rate_vals = tcp_retransmit_rate_by_iteration(self.results)
        if not rate_vals:
            return None
        return sum(rate_vals) / len(rate_vals)

    def mean_udp_jitter_ms(self) -> float:
        """Return the mean UDP inter-arrival jitter in milliseconds.

        Takes the per-iteration mean across clients that reported jitter,
        then averages across iterations. Only UDP records carry jitter.

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

    def mean_node_memory(self) -> float | None:
        """Return the mean target-node memory usage in bytes.

        Takes the per-iteration mean across records that reported
        node memory, then averages across iterations.

        Returns:
            The averaged node memory in bytes, or ``None`` when no
            results report node memory. ``None`` signals "not
            measured" and is distinct from a real reading of ``0.0``.
        """
        per_iter_means: list[float] = []
        for group in _group_by_iteration(self.results).values():
            vals = [
                r.node_memory_used_bytes
                for r in group
                if r.node_memory_used_bytes is not None
            ]
            if vals:
                per_iter_means.append(sum(vals) / len(vals))
        if not per_iter_means:
            return None
        return sum(per_iter_means) / len(per_iter_means)

    def mean_cni_memory(self) -> float | None:
        """Return the mean CNI memory usage in bytes on the target node.

        Takes the per-iteration mean across records that reported CNI
        memory, then averages across iterations.

        Returns:
            The averaged CNI memory in bytes, or ``None`` when no
            results report CNI memory (e.g. ``cni.enabled=false`` or
            the selector matched no pods). ``None`` signals "not
            measured" and is distinct from a real reading of ``0.0``.
        """
        per_iter_means: list[float] = []
        for group in _group_by_iteration(self.results).values():
            vals = [
                r.cni_memory_used_bytes
                for r in group
                if r.cni_memory_used_bytes is not None
            ]
            if vals:
                per_iter_means.append(sum(vals) / len(vals))
        if not per_iter_means:
            return None
        return sum(per_iter_means) / len(per_iter_means)

    def mean_rps(self) -> float:
        """Return the mean achieved RPS from fortio saturation runs.

        RPS is summed across source clients within each iteration
        (mirroring the throughput aggregation) and then averaged
        across iterations. Only records tagged
        ``workload == "saturation"`` contribute; fixed-QPS runs are
        ignored because their RPS is clamped to the offered load.

        Returns:
            The averaged achieved RPS, or ``0.0`` when no saturation
            record is available.
        """
        saturation = [r for r in self.latency_results if r.workload == "saturation"]
        if not saturation:
            return 0.0
        per_iter_sums = [
            sum(r.rps for r in group)
            for group in _group_latency_by_iteration(saturation).values()
        ]
        return sum(per_iter_sums) / len(per_iter_sums)

    def mean_latency_p50_ms(self) -> float:
        """Return the mean p50 latency from fortio fixed-QPS runs.

        Returns:
            The averaged p50 latency in milliseconds, or ``0.0`` when
            no fixed-QPS record reports p50.
        """
        return self._mean_fixed_qps_latency(lambda r: r.latency_p50_ms)

    def mean_latency_p90_ms(self) -> float:
        """Return the mean p90 latency from fortio fixed-QPS runs.

        Returns:
            The averaged p90 latency in milliseconds, or ``0.0`` when
            no fixed-QPS record reports p90.
        """
        return self._mean_fixed_qps_latency(lambda r: r.latency_p90_ms)

    def mean_latency_p99_ms(self) -> float:
        """Return the mean p99 latency from fortio fixed-QPS runs.

        Returns:
            The averaged p99 latency in milliseconds, or ``0.0`` when
            no fixed-QPS record reports p99.
        """
        return self._mean_fixed_qps_latency(lambda r: r.latency_p99_ms)

    def _mean_fixed_qps_latency(
        self,
        extractor: Callable[[LatencyResult], float | None],
    ) -> float:
        """Average a fixed-QPS latency field across clients then iterations.

        Args:
            extractor: Callable returning the latency field for one
                record, or ``None`` when the record did not report it.

        Returns:
            The averaged latency in milliseconds, or ``0.0`` when no
            fixed-QPS record supplied a value.
        """
        fixed = [r for r in self.latency_results if r.workload == "fixed_qps"]
        per_iter_means: list[float] = []
        for group in _group_latency_by_iteration(fixed).values():
            vals = [v for r in group if (v := extractor(r)) is not None]
            if vals:
                per_iter_means.append(sum(vals) / len(vals))
        if not per_iter_means:
            return 0.0
        return sum(per_iter_means) / len(per_iter_means)


class ResumeMetadata(BaseModel):
    """Sidecar describing the experiment that produced a JSONL log.

    Written next to the JSONL file at ``<path>.meta.json`` on every run
    and consulted by :func:`kube_autotuner.runs.run_optimize` to decide
    whether the prior trials are compatible with the incoming
    experiment. ``objectives``, ``param_space``, ``benchmark`` are the
    compatibility keys; ``n_sobol``, ``verification_trials``, and
    ``verification_top_k`` are only populated by ``optimize`` mode
    (baseline / trial leave them ``None``). ``verification_*`` fields
    default to ``None`` for sidecars written by pre-feature binaries;
    the compatibility check treats that shape as tolerant rather than
    drift.
    """

    objectives: ObjectivesSection
    param_space: ParamSpace
    benchmark: BenchmarkConfig
    n_sobol: int | None = None
    verification_trials: int | None = None
    verification_top_k: int | None = None


_resume_metadata_built = False


def _ensure_resume_metadata_built() -> None:
    """Resolve :class:`ResumeMetadata`'s ``ObjectivesSection`` forward ref.

    The annotation lives under ``TYPE_CHECKING`` to avoid an import
    cycle with :mod:`kube_autotuner.experiment`. Callers invoke this
    helper before the first validate/dump so Pydantic can bind the
    real class into the model's type namespace.
    """
    global _resume_metadata_built  # noqa: PLW0603
    if _resume_metadata_built:
        return
    from kube_autotuner.experiment import ObjectivesSection  # noqa: PLC0415

    ResumeMetadata.model_rebuild(
        _types_namespace={"ObjectivesSection": ObjectivesSection},
    )
    _resume_metadata_built = True


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

        A single partially-written trailing line (from a Ctrl-C during
        ``append``) is tolerated: the last line is parsed leniently and
        a warning is logged if it fails. Any mid-file failure still
        raises, because that signals real corruption rather than an
        interrupted write.

        Args:
            path: Source JSONL file. If it does not exist, an empty list is
                returned.

        Returns:
            The decoded trials, in file order.

        Raises:
            json.JSONDecodeError: Mid-file JSON decoding failure
                (real corruption, not an interrupted write).
            pydantic.ValidationError: Mid-file record fails schema
                validation.
        """
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as f:
            raw_lines = f.readlines()
        lines = [line for line in raw_lines if line.strip()]
        if not lines:
            return []
        trials: list[TrialResult] = []
        last_idx = len(lines) - 1
        for idx, raw in enumerate(lines):
            stripped = raw.strip()
            try:
                trials.append(TrialResult.model_validate_json(stripped))
            except ValidationError, json.JSONDecodeError:
                if idx == last_idx:
                    logger.warning(
                        "dropping truncated final line in %s (likely from "
                        "an interrupted write)",
                        path,
                    )
                    break
                raise
        return trials

    @staticmethod
    def _metadata_path(path: Path) -> Path:
        """Return the sibling metadata path for a JSONL log at ``path``."""
        return path.parent / f"{path.name}.meta.json"

    @staticmethod
    def write_resume_metadata(path: Path, meta: ResumeMetadata) -> None:
        """Persist ``meta`` as a sidecar next to the JSONL log.

        The sidecar lives at ``<path>.meta.json``. Writes are
        idempotent for identical content; drift detection now happens
        at resume time in
        :func:`kube_autotuner.runs._check_compatibility`, not here.

        Args:
            path: JSONL log path; the sidecar is written beside it.
            meta: :class:`ResumeMetadata` describing the run that is
                about to write ``path``.
        """
        _ensure_resume_metadata_built()
        meta_path = TrialLog._metadata_path(path)
        meta_path.write_text(meta.model_dump_json() + "\n", encoding="utf-8")

    @staticmethod
    def load_resume_metadata(path: Path) -> ResumeMetadata | None:
        """Return the sidecar :class:`ResumeMetadata` for ``path``.

        Args:
            path: JSONL log path whose sibling ``<path>.meta.json`` is
                consulted.

        Returns:
            The parsed :class:`ResumeMetadata`, or ``None`` when no
            sidecar exists. A malformed sidecar raises
            :class:`pydantic.ValidationError`.
        """
        _ensure_resume_metadata_built()
        meta_path = TrialLog._metadata_path(path)
        if not meta_path.exists():
            return None
        return ResumeMetadata.model_validate_json(
            meta_path.read_text(encoding="utf-8"),
        )
