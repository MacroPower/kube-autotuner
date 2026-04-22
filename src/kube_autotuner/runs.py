"""Baseline / trial / optimize orchestration entry points.

The three functions in this module -- :func:`run_baseline`,
:func:`run_trial`, and :func:`run_optimize` -- glue together the data
models, the Kubernetes client / lease primitives, the sysctl backends,
the benchmark runner,
:class:`~kube_autotuner.experiment.ExperimentConfig`, and the Ax
optimizer. They are the single source of truth for run behaviour; the
Typer CLI delegates into them rather than reimplementing the
lease/snapshot/apply/restore dance.

Run-mode JSONL output
---------------------

Every mode appends :class:`~kube_autotuner.models.TrialResult` records
to ``ctx.output`` via :meth:`~kube_autotuner.models.TrialLog.append`.
Readers tailing the file see one JSON object per line whose field
names and nesting mirror the Pydantic models in
:mod:`kube_autotuner.models` (``TrialResult`` → ``BenchmarkResult`` →
``NodePair``). Downstream analysers revalidate against the Pydantic
models on read rather than relying on a byte-for-byte compatibility
contract.

* **baseline** -- one :class:`TrialResult` line. ``sysctl_values``
  captures the full
  :data:`~kube_autotuner.sysctl.params.PARAM_SPACE` snapshot taken on
  the target node; nothing is written back. Use this to record a
  "current kernel config" entry before a tuning pass.
* **trial** -- one :class:`TrialResult` line. ``sysctl_values`` holds
  the keys from ``exp.trial.sysctls`` that were actually applied; the
  original values are restored through a ``finally`` block before the
  run returns.
* **optimize** -- one :class:`TrialResult` line per completed Ax
  trial, emitted as the loop progresses. Failed trials are *not*
  written to the file (they are marked failed with the Ax client
  instead).

Backend injection
-----------------

Every entry point takes a :class:`RunContext` carrying an explicit
:class:`~kube_autotuner.sysctl.backend.SysctlBackend`. No environment
variables are read from this module; the CLI is the single sanctioned
place to call
:func:`~kube_autotuner.sysctl.setter.make_sysctl_setter_from_env`.
"""

from __future__ import annotations

from collections import Counter
import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
import math
from typing import TYPE_CHECKING, cast

from rich.console import Console
from rich.table import Table
import typer

from kube_autotuner.benchmark.runner import BenchmarkRunner
from kube_autotuner.k8s.lease import NodeLease
from kube_autotuner.models import ResumeMetadata, TrialLog, TrialResult, is_primary
from kube_autotuner.progress import NullObserver
from kube_autotuner.report import format_retransmit_rate
from kube_autotuner.scoring import (
    METRIC_TO_DF_COLUMN,
    aggregate_verification,
    score_rows,
)
from kube_autotuner.sysctl.params import PARAM_SPACE

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from kube_autotuner.experiment import ExperimentConfig, ObjectivesSection
    from kube_autotuner.k8s.client import K8sClient
    from kube_autotuner.models import NodePair
    from kube_autotuner.progress import ProgressObserver
    from kube_autotuner.sysctl.backend import SysctlBackend

logger = logging.getLogger(__name__)

_OPTIMIZE_HINT = (
    "ax-platform is required for optimization. Install with: uv sync --group optimize"
)


@dataclass(frozen=True)
class RunContext:
    """Collaborators threaded into every ``run_*`` entry point.

    Bundling these into a dataclass keeps the run functions under the
    ``pylint.max-args = 6`` ceiling without forcing per-call signatures
    to repeat the same five or six arguments.

    Attributes:
        exp: Validated experiment configuration produced by
            :meth:`ExperimentConfig.from_yaml` or equivalent.
        client: Injected :class:`K8sClient` shared with the lease,
            benchmark, and (indirectly) backend layers.
        backend: Sysctl backend targeting ``exp.nodes.target``. Used
            by :func:`run_baseline` and :func:`run_trial`.
            :func:`run_optimize` leaves this field unused because
            :class:`~kube_autotuner.optimizer.OptimizationLoop` builds
            its own per-node setters (one per source when
            ``apply_source=True``).
        output: JSONL destination for :class:`TrialResult` records.
            Usually ``Path(exp.output)``; decoupled from ``exp`` so
            callers can redirect the sink without rewriting the
            config.
        observer: Progress callback threaded into
            :class:`BenchmarkRunner` / :class:`OptimizationLoop`.
            Defaults to :class:`NullObserver` so library consumers
            and tests get zero output side-effects; the CLI overrides
            it with a :class:`RichProgressObserver` under a TTY.
    """

    exp: ExperimentConfig
    client: K8sClient
    backend: SysctlBackend
    output: Path
    observer: ProgressObserver = field(default_factory=NullObserver)


def _resolve_zones(node_pair: NodePair, client: K8sClient) -> NodePair:
    """Populate zone fields on ``node_pair`` via Kubernetes node labels.

    Args:
        node_pair: Pair to enrich.
        client: Client used to query
            ``topology.kubernetes.io/zone`` on each node.

    Returns:
        A copy of ``node_pair`` with ``source_zone``, ``target_zone``,
        and ``extra_source_zones`` filled in. Unresolvable nodes map
        to ``""`` rather than aborting the run -- topology metadata is
        informational.
    """

    def _zone(node: str) -> str:
        try:
            return client.get_node_zone(node)
        except Exception:  # noqa: BLE001 - topology is best-effort
            logger.warning("Could not resolve zone for node %s", node)
            return ""

    return node_pair.model_copy(
        update={
            "source_zone": _zone(node_pair.source),
            "target_zone": _zone(node_pair.target),
            "extra_source_zones": {c: _zone(c) for c in node_pair.extra_sources},
        },
    )


@contextlib.contextmanager
def _acquire_all_leases(
    nodes: list[str],
    namespace: str,
    client: K8sClient,
) -> Iterator[None]:
    """Acquire a :class:`NodeLease` on every node in sorted order.

    Sorted entry order produces a total ordering across concurrent
    callers so two runs contending on the same node set cannot
    deadlock.

    Args:
        nodes: Node names to lock. Duplicates collapse.
        namespace: Namespace hosting the ``coordination.k8s.io/v1``
            Lease resources.
        client: Client forwarded to :class:`NodeLease`.

    Yields:
        ``None`` once every lease is held. Leases release in reverse
        order on exit, via :class:`contextlib.ExitStack`.
    """
    with contextlib.ExitStack() as stack:
        for node in sorted(set(nodes)):
            stack.enter_context(
                NodeLease(node, namespace=namespace, client=client),
            )
        yield


@dataclass(frozen=True)
class _ResumeState:
    """Result of a resume-preparation pass.

    Attributes:
        prior_trials: Successful :class:`TrialResult` records loaded
            from the prior session, in file order. Includes both
            primary and verification rows when present; empty when no
            resume occurred.
        remaining_trials: Live-loop primary-attempt budget: ``max(0,
            n_trials - len(primary_priors))``. Counts successful
            primary priors only; a prior run that had failures leaves
            the remaining budget wider than a strict
            ``n_trials - attempts_so_far`` would -- accepted as the
            simplest interpretation. See the module docstring.
        verification_done_by_parent: Map of primary ``trial_id`` ->
            count of verification rows already in the JSONL for that
            parent. Fed to
            :meth:`~kube_autotuner.optimizer.OptimizationLoop.run_verification`
            so the verification loop skips work that a prior session
            already finished.
    """

    prior_trials: list[TrialResult]
    remaining_trials: int
    verification_done_by_parent: dict[str, int] = field(default_factory=dict)


def _move_prior_artifacts(output: Path) -> None:
    """Rename the prior JSONL and sidecar aside with a UTC timestamp.

    Given ``output = <dir>/results.jsonl``, renames to
    ``<dir>/results.jsonl.<T>.bak`` and the sidecar to
    ``<dir>/results.jsonl.meta.json.<T>.bak`` where ``T`` is a UTC
    ISO-like timestamp (``YYYYMMDDTHHMMSSZ``). Non-existent files are
    skipped silently.

    Args:
        output: JSONL log path whose prior artefacts should be
            archived.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    meta = TrialLog._metadata_path(output)  # noqa: SLF001 sibling path helper
    for path in (output, meta):
        if not path.exists():
            continue
        backup = path.with_name(f"{path.name}.{stamp}.bak")
        path.rename(backup)
        logger.info("moved prior results to %s", backup)


def _check_compatibility(  # noqa: C901, PLR0912
    meta: ResumeMetadata | None,
    exp: ExperimentConfig,
) -> None:
    """Validate that ``meta`` matches ``exp`` on the compatibility keys.

    Compatibility keys: ``objectives``, ``param_space``,
    ``benchmark``, and (when the sidecar carries one) ``n_sobol``,
    ``verification_trials``, ``verification_top_k``. Node identity,
    patches, iperf/fortio args, and ``apply_source`` can change
    silently between runs.

    Sidecars written by pre-feature binaries have ``verification_*``
    fields set to ``None``. When the current run enables verification
    against such a sidecar, no drift is flagged: the JSONL stays
    usable and a refreshed sidecar will be written afterwards.

    A ``None`` sidecar alongside a non-empty JSONL signals a prior
    run that wrote no metadata (or a manually crafted JSONL); the
    user must pass ``--fresh`` or supply a sidecar before resume.

    Args:
        meta: The loaded sidecar, or ``None`` when absent.
        exp: The incoming experiment configuration.

    Raises:
        typer.BadParameter: The sidecar is missing or any
            compatibility key diverges.
    """
    if meta is None:
        msg = (
            "prior JSONL has no sidecar to validate against; pass "
            "--fresh to move it aside or supply a sidecar before "
            "resuming."
        )
        raise typer.BadParameter(msg)

    changed: list[str] = []
    if meta.objectives.model_dump() != exp.objectives.model_dump():
        changed.append("objectives")
    if meta.param_space.model_dump() != exp.effective_param_space().model_dump():
        changed.append("param_space")
    if meta.benchmark.model_dump() != exp.benchmark.model_dump():
        changed.append("benchmark")
    if exp.optimize is not None and meta.n_sobol is not None:
        if meta.n_sobol != exp.optimize.n_sobol:
            changed.append("n_sobol")
    elif exp.optimize is not None and meta.n_sobol is None:
        logger.warning("sidecar has no n_sobol; not verified")

    if exp.optimize is not None:
        legacy_verification = (
            meta.verification_trials is None and meta.verification_top_k is None
        )
        if legacy_verification and (
            exp.optimize.verification_trials > 0 or exp.optimize.verification_top_k != 3  # noqa: PLR2004 - schema default
        ):
            logger.info(
                "prior sidecar has no verification record; proceeding and "
                "rewriting sidecar",
            )
        else:
            if (
                meta.verification_trials is not None
                and meta.verification_trials != exp.optimize.verification_trials
            ):
                changed.append("verification_trials")
            if (
                meta.verification_top_k is not None
                and meta.verification_top_k != exp.optimize.verification_top_k
            ):
                changed.append("verification_top_k")

    if changed:
        msg = (
            "prior results in --output are incompatible with the current "
            f"experiment; changed fields: {sorted(changed)}. Realign the "
            "config or pass --fresh to archive the prior results."
        )
        raise typer.BadParameter(msg)


def _prepare_resume(
    output: Path,
    exp: ExperimentConfig,
    *,
    fresh: bool,
) -> _ResumeState:
    """Decide whether to resume and return the state the loop needs.

    Budget counts successful priors only; a prior run that had
    failures will have a remaining budget equal to ``n_trials -
    successful_priors``, which over-estimates the remaining attempts
    compared to a strict ``n_trials - attempts_so_far`` budget. This
    is the simplest interpretation -- users can always stop early.

    Args:
        output: JSONL destination path for the run.
        exp: Validated experiment. Must have ``exp.optimize`` set.
        fresh: When ``True``, archive any prior artefacts and return
            a fresh state unconditionally.

    Returns:
        A :class:`_ResumeState` describing the prior trials to replay
        and the remaining live-loop budget.

    Raises:
        RuntimeError: ``exp.optimize`` is ``None`` -- a caller
            invariant has been violated.
        typer.BadParameter: The prior JSONL exists but has no
            sidecar, or the sidecar's compatibility keys diverge from
            ``exp``. Raised indirectly via
            :func:`_check_compatibility`.
    """  # noqa: DOC502 - typer.BadParameter raised via _check_compatibility
    if exp.optimize is None:
        msg = "_prepare_resume requires exp.optimize to be populated"
        raise RuntimeError(msg)

    if fresh:
        _move_prior_artifacts(output)
        return _ResumeState(prior_trials=[], remaining_trials=exp.optimize.n_trials)

    if not output.exists():
        return _ResumeState(prior_trials=[], remaining_trials=exp.optimize.n_trials)

    prior = TrialLog.load(output)
    if not prior:
        return _ResumeState(prior_trials=[], remaining_trials=exp.optimize.n_trials)

    meta = TrialLog.load_resume_metadata(output)
    _check_compatibility(meta, exp)

    primary_prior = [t for t in prior if is_primary(t)]
    verification_done = Counter(t.parent_trial_id for t in prior if t.parent_trial_id)
    remaining = max(0, exp.optimize.n_trials - len(primary_prior))
    logger.info(
        "Resuming: %d prior trials (%d primary + %d verification); "
        "running %d more primary (budget=%d)",
        len(prior),
        len(primary_prior),
        len(prior) - len(primary_prior),
        remaining,
        exp.optimize.n_trials,
    )
    return _ResumeState(
        prior_trials=prior,
        remaining_trials=remaining,
        verification_done_by_parent={k: v for k, v in verification_done.items() if k},
    )


def run_baseline(ctx: RunContext) -> None:
    """Record a baseline :class:`TrialResult` without touching the node.

    Snapshots the full
    :data:`~kube_autotuner.sysctl.params.PARAM_SPACE` (plus
    ``kernel.osrelease``) on the target node so the recorded trial
    captures the live configuration verbatim, then runs the benchmark
    and appends a single :class:`TrialResult` to ``ctx.output``. The
    sysctls are never mutated.

    Args:
        ctx: Orchestration context. ``ctx.backend`` must target
            ``ctx.exp.nodes.target``.
    """
    exp = ctx.exp
    TrialLog.write_resume_metadata(
        ctx.output,
        ResumeMetadata(
            objectives=exp.objectives,
            param_space=exp.effective_param_space(),
            benchmark=exp.benchmark,
        ),
    )
    node_pair = _resolve_zones(exp.to_node_pair(), ctx.client)
    config = exp.benchmark
    all_params = [*PARAM_SPACE.param_names(), "kernel.osrelease"]

    lease_nodes = [node_pair.target, *node_pair.all_sources]
    with _acquire_all_leases(lease_nodes, node_pair.namespace, ctx.client):
        snapshot = ctx.backend.snapshot(all_params)
        kernel_version = snapshot.pop("kernel.osrelease", "")

        runner = BenchmarkRunner(
            node_pair,
            config,
            client=ctx.client,
            iperf_args=exp.iperf,
            patches=exp.patches,
            cni=exp.cni,
            fortio_args=exp.fortio,
            observer=ctx.observer,
        )
        try:
            runner.setup_server()
            results = runner.run()
        finally:
            runner.cleanup()

    trial = TrialResult(
        node_pair=node_pair,
        sysctl_values=cast("dict[str, str | int]", snapshot),
        kernel_version=kernel_version,
        config=config,
        results=results.bench,
        latency_results=results.latency,
    )
    TrialLog.append(ctx.output, trial)
    logger.info(
        "Wrote %d bench + %d latency results to %s",
        len(results.bench),
        len(results.latency),
        ctx.output,
    )
    logger.info("Mean throughput: %.1f Mbps", trial.mean_throughput() / 1e6)
    logger.info("Mean CPU: %.1f%%", trial.mean_cpu())
    nmem = trial.mean_node_memory()
    if nmem is not None:
        logger.info("Mean node memory: %.1f MiB", nmem / 1024 / 1024)
    cmem = trial.mean_cni_memory()
    if cmem is not None:
        logger.info("Mean CNI memory: %.1f MiB", cmem / 1024 / 1024)
    rps = trial.mean_rps()
    if rps > 0:
        logger.info("Mean RPS (saturation): %.1f", rps)
    p99 = trial.mean_latency_p99_ms()
    if p99 > 0:
        logger.info("Mean p99 latency (fixed_qps): %.2f ms", p99)


def run_trial(ctx: RunContext) -> None:  # noqa: PLR0914, PLR0915
    """Apply a fixed sysctl set, benchmark, and restore.

    Snapshots only the keys being applied (plus ``kernel.osrelease``),
    writes ``ctx.exp.trial.sysctls`` to the target node, runs the
    benchmark, and restores the original values through a ``finally``
    block. Appends one :class:`TrialResult` to ``ctx.output``.

    Args:
        ctx: Orchestration context. ``ctx.exp.trial`` must be
            populated (guaranteed by
            :class:`~kube_autotuner.experiment.ExperimentConfig`'s
            mode validator).

    Raises:
        RuntimeError: ``ctx.exp.trial`` is ``None`` -- an
            :class:`ExperimentConfig` invariant has been violated.
    """
    exp = ctx.exp
    if exp.trial is None:
        msg = (
            "run_trial requires exp.trial to be populated; "
            "ExperimentConfig's mode validator should have rejected this config"
        )
        raise RuntimeError(msg)

    TrialLog.write_resume_metadata(
        ctx.output,
        ResumeMetadata(
            objectives=exp.objectives,
            param_space=exp.effective_param_space(),
            benchmark=exp.benchmark,
        ),
    )
    params: dict[str, str] = {k: str(v) for k, v in exp.trial.sysctls.items()}
    node_pair = _resolve_zones(exp.to_node_pair(), ctx.client)
    config = exp.benchmark

    lease_nodes = [node_pair.target, *node_pair.all_sources]
    with _acquire_all_leases(lease_nodes, node_pair.namespace, ctx.client):
        snapshot_params = [*params.keys(), "kernel.osrelease"]
        original = ctx.backend.snapshot(snapshot_params)
        kernel_version = original.pop("kernel.osrelease", "")
        logger.info(
            "Snapshotted %d sysctl(s) on %s",
            len(original),
            node_pair.target,
        )

        ctx.backend.apply(params)
        logger.info(
            "Applied %d sysctl(s) on %s",
            len(params),
            node_pair.target,
        )

        runner = BenchmarkRunner(
            node_pair,
            config,
            client=ctx.client,
            iperf_args=exp.iperf,
            patches=exp.patches,
            cni=exp.cni,
            fortio_args=exp.fortio,
            observer=ctx.observer,
        )
        try:
            runner.setup_server()
            results = runner.run()
        finally:
            runner.cleanup()
            ctx.backend.restore(original)
            logger.info("Restored original sysctls on %s", node_pair.target)

    trial_result = TrialResult(
        node_pair=node_pair,
        sysctl_values=cast("dict[str, str | int]", params),
        kernel_version=kernel_version,
        config=config,
        results=results.bench,
        latency_results=results.latency,
    )
    TrialLog.append(ctx.output, trial_result)
    logger.info(
        "Wrote %d bench + %d latency results to %s",
        len(results.bench),
        len(results.latency),
        ctx.output,
    )
    logger.info(
        "Mean throughput: %.1f Mbps",
        trial_result.mean_throughput() / 1e6,
    )
    logger.info("Mean CPU: %.1f%%", trial_result.mean_cpu())
    nmem = trial_result.mean_node_memory()
    if nmem is not None:
        logger.info("Mean node memory: %.1f MiB", nmem / 1024 / 1024)
    cmem = trial_result.mean_cni_memory()
    if cmem is not None:
        logger.info("Mean CNI memory: %.1f MiB", cmem / 1024 / 1024)
    rps = trial_result.mean_rps()
    if rps > 0:
        logger.info("Mean RPS (saturation): %.1f", rps)
    p99 = trial_result.mean_latency_p99_ms()
    if p99 > 0:
        logger.info("Mean p99 latency (fixed_qps): %.2f ms", p99)


def run_optimize(  # noqa: PLR0914, PLR0915
    ctx: RunContext,
    *,
    fresh: bool = False,
) -> None:
    """Drive the Ax Bayesian optimization loop.

    Per-trial lease acquisition lives inside
    :class:`~kube_autotuner.optimizer.OptimizationLoop` so Ctrl-C
    releases leases between trials; this function deliberately does
    *not* wrap the loop in an outer lease. ``ctx.backend`` is unused
    here -- the optimizer builds its own per-node setters (one per
    source node when ``apply_source=True``) because a single fan-out
    setter does not fit a single target-only backend.

    Resume is automatic: when ``ctx.output`` already exists and its
    sidecar is compatible with ``ctx.exp``, the prior
    :class:`~kube_autotuner.models.TrialResult` records are replayed
    into Ax via ``attach_trial`` + ``complete_trial`` and the live
    loop only runs the remaining
    ``max(0, n_trials - len(prior))`` attempts. Budget invariant:
    when the budget is already met the loop is short-circuited
    entirely, which is safe because
    :meth:`OptimizationLoop.__init__` constructs only light objects
    (no server pods, leases, or similar per-run resources) -- this
    invariant must hold for future refactors of ``__init__``.

    Concurrency: two operators pointing at the same ``ctx.output``
    already race on :meth:`TrialLog.append`; resume widens the window
    but per-trial leases still serialise execution against the nodes.

    Args:
        ctx: Orchestration context. ``ctx.exp.optimize`` must be
            populated (guaranteed by
            :class:`~kube_autotuner.experiment.ExperimentConfig`'s
            mode validator).
        fresh: When ``True``, move any pre-existing ``ctx.output`` and
            sidecar aside (timestamped ``.bak`` rename) before
            starting, so the run begins from zero.

    Raises:
        RuntimeError: ``ctx.exp.optimize`` is ``None``, or
            ``ax-platform`` is not installed (the ``optimize`` dep
            group provides it).
        typer.BadParameter: The prior JSONL exists but its sidecar is
            missing or incompatible with the current experiment.
            Raised indirectly via :func:`_prepare_resume`.
    """  # noqa: DOC502 - typer.BadParameter raised via _prepare_resume
    try:
        from kube_autotuner.optimizer import OptimizationLoop  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(_OPTIMIZE_HINT) from e

    exp = ctx.exp
    if exp.optimize is None:
        msg = (
            "run_optimize requires exp.optimize to be populated; "
            "ExperimentConfig's mode validator should have rejected this config"
        )
        raise RuntimeError(msg)

    resume = _prepare_resume(ctx.output, exp, fresh=fresh)
    # Rewrite the sidecar on every run so a resume against a
    # pre-feature sidecar (missing verification_*) picks up the new
    # fields idempotently. _check_compatibility has already rejected
    # any drift, so the write is either a no-op or a refresh of
    # previously-None fields.
    TrialLog.write_resume_metadata(
        ctx.output,
        ResumeMetadata(
            objectives=exp.objectives,
            param_space=exp.effective_param_space(),
            benchmark=exp.benchmark,
            n_sobol=exp.optimize.n_sobol,
            verification_trials=exp.optimize.verification_trials,
            verification_top_k=exp.optimize.verification_top_k,
        ),
    )
    ctx.observer.seed_history(resume.prior_trials, exp.optimize.n_sobol)
    node_pair = _resolve_zones(exp.to_node_pair(), ctx.client)
    config = exp.benchmark

    loop = OptimizationLoop(
        node_pair=node_pair,
        config=config,
        param_space=exp.effective_param_space(),
        output=ctx.output,
        n_trials=exp.optimize.n_trials,
        n_sobol=exp.optimize.n_sobol,
        apply_source=exp.optimize.apply_source,
        iperf_args=exp.iperf,
        fortio_args=exp.fortio,
        patches=exp.patches,
        objectives=exp.objectives,
        cni=exp.cni,
        observer=ctx.observer,
        prior_trials=resume.prior_trials,
    )
    if resume.remaining_trials == 0:
        logger.info(
            "Budget already met (%d >= %d); skipping loop",
            len(resume.prior_trials),
            exp.optimize.n_trials,
        )
        trials = list(resume.prior_trials)
    else:
        trials = loop.run()
    new_count = len(trials) - loop.prior_count
    logger.info(
        "Completed %d trials (%d prior + %d new). Results in %s",
        len(trials),
        loop.prior_count,
        new_count,
        ctx.output,
    )
    if not trials:
        return

    if exp.optimize.verification_trials > 0:
        loop.run_verification(
            top_k=exp.optimize.verification_top_k,
            repeats=exp.optimize.verification_trials,
            already_done_by_parent=resume.verification_done_by_parent,
        )
        _log_verification_summary(loop._completed, exp.objectives)  # noqa: SLF001

    try:
        pareto = loop.pareto_front()
    except Exception:
        logger.warning("Could not compute Pareto front", exc_info=True)
        return

    logger.info("=== Pareto-optimal configurations (%d) ===", len(pareto))
    for _params, metrics, trial_idx, _arm in pareto:
        tp = metrics.get("throughput", 0)
        cpu = metrics.get("cpu", 0)
        rate = metrics.get("retransmit_rate")
        rps = metrics.get("rps")
        p99 = metrics.get("latency_p99")
        tp_val = tp[0] if isinstance(tp, tuple) else tp
        cpu_val = cpu[0] if isinstance(cpu, tuple) else cpu
        if rate is None:
            rate_val: float | None = None
        else:
            raw = rate[0] if isinstance(rate, tuple) else rate
            rate_val = None if math.isnan(raw) else float(raw)
        rps_val = _scalar_or_nan(rps)
        p99_val = _scalar_or_nan(p99)
        rps_str = "n/a" if math.isnan(rps_val) else f"{rps_val:.1f}"
        p99_str = "n/a" if math.isnan(p99_val) else f"{p99_val:.1f}"
        logger.info(
            "  [%d] throughput=%.1f Mbps cpu=%.1f%% rate=%s retx/MB rps=%s p99=%s ms",
            trial_idx,
            float(tp_val) / 1e6,
            float(cpu_val),
            format_retransmit_rate(rate_val),
            rps_str,
            p99_str,
        )


def _log_verification_summary(
    all_trials: list[TrialResult],
    objectives: ObjectivesSection,
) -> None:
    """Print a Rich table comparing primary vs combined scores.

    Groups ``all_trials`` by ``parent_trial_id or trial_id`` via
    :func:`kube_autotuner.scoring.aggregate_verification`, scores both
    the primary-only population and the combined (primary +
    verification) population through
    :func:`kube_autotuner.scoring.score_rows`, and prints one row per
    verified parent showing the primary score, the combined score,
    the score delta, and per-metric ``mean ± SEM`` for the headline
    metrics (throughput, cpu, retransmit_rate, latency_p99).

    Only parents with at least one verification child are rendered --
    listing every primary would dilute the table and confuse the
    user about which rows were actually re-run.

    Args:
        all_trials: Every :class:`TrialResult` from the run (primary
            and verification, in file order).
        objectives: The experiment's objective configuration; drives
            the weighted score used for the "primary score" /
            "combined score" comparison.
    """
    verification_parents = {t.parent_trial_id for t in all_trials if t.parent_trial_id}
    if not verification_parents:
        return

    primary_only = [t for t in all_trials if is_primary(t)]
    combined_rows = aggregate_verification(all_trials)
    primary_rows = aggregate_verification(primary_only)

    combined_scores = score_rows(
        combined_rows,
        objectives.pareto,
        objectives.recommendation_weights,
    )
    primary_scores = score_rows(
        primary_rows,
        objectives.pareto,
        objectives.recommendation_weights,
    )
    primary_score_by_id = {
        str(r["trial_id"]): s for r, s in zip(primary_rows, primary_scores, strict=True)
    }

    ordered = sorted(
        range(len(combined_rows)),
        key=lambda i: (
            -combined_scores[i],
            str(combined_rows[i]["trial_id"]),
        ),
    )

    table = Table(
        title="Verification summary",
        title_style="bold",
        show_header=True,
        header_style="bold",
        expand=False,
    )
    table.add_column("rank", justify="right", no_wrap=True)
    table.add_column("trial_id", no_wrap=True)
    table.add_column("primary", justify="right")
    table.add_column("combined", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("throughput", justify="right")
    table.add_column("cpu", justify="right")
    table.add_column("retx_rate", justify="right")
    table.add_column("p99 ms", justify="right")

    rank = 0
    for i in ordered:
        row = combined_rows[i]
        trial_id = str(row["trial_id"])
        if trial_id not in verification_parents:
            continue
        rank += 1
        primary = primary_score_by_id.get(trial_id, float("nan"))
        combined = combined_scores[i]
        delta = combined - primary if not math.isnan(primary) else float("nan")
        table.add_row(
            str(rank),
            trial_id,
            "n/a" if math.isnan(primary) else f"{primary:.4f}",
            f"{combined:.4f}",
            "n/a" if math.isnan(delta) else f"{delta:+.4f}",
            _format_mean_sem(row, METRIC_TO_DF_COLUMN["throughput"], scale=1e-6),
            _format_mean_sem(row, METRIC_TO_DF_COLUMN["cpu"]),
            _format_mean_sem(
                row,
                METRIC_TO_DF_COLUMN["retransmit_rate"],
                scale=1e6,
            ),
            _format_mean_sem(row, METRIC_TO_DF_COLUMN["latency_p99"]),
        )

    Console().print(table)


def _format_mean_sem(
    row: dict[str, float | int | str],
    col: str,
    *,
    scale: float = 1.0,
) -> str:
    """Render ``<mean> ± <SEM>`` for one metric in an aggregation row.

    Args:
        row: An :func:`aggregate_verification` row.
        col: DataFrame-column name for the metric (see
            :data:`METRIC_TO_DF_COLUMN`).
        scale: Multiplier applied to both mean and SEM before
            formatting (e.g. ``1e-6`` to convert bits/sec to Mbps).

    Returns:
        A human-readable ``"mean ± sem"`` string, or ``"n/a"`` when
        the mean is NaN.
    """
    mean_raw = row.get(col)
    sem_raw = row.get(f"{col}_sem", 0.0)
    try:
        mean = float(mean_raw)  # ty: ignore[invalid-argument-type]
    except TypeError, ValueError:
        return "n/a"
    if math.isnan(mean):
        return "n/a"
    sem = float(sem_raw) if sem_raw is not None else 0.0
    return f"{mean * scale:.3g} ± {sem * scale:.2g}"


def _scalar_or_nan(
    raw: float | tuple[int | float, int | float] | None,
) -> float:
    """Extract the scalar mean from an Ax metric entry, defaulting to NaN.

    Ax's :meth:`get_pareto_frontier` mixes scalar and ``(mean, SEM)``
    values per metric. The optimize pareto log needs the mean only;
    missing keys map to NaN so the caller can render ``"n/a"``.

    Args:
        raw: Ax metric entry (scalar, tuple, or ``None``).

    Returns:
        The scalar mean as a float, or ``math.nan`` when missing.
    """
    if raw is None:
        return math.nan
    if isinstance(raw, tuple):
        return float(raw[0])
    return float(raw)
