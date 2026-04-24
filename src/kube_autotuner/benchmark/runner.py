"""Concurrent iperf3 + fortio benchmark runner.

:class:`BenchmarkRunner` launches one iperf3 server Deployment and one
fortio server Deployment on the target node, then drives each iteration
as four sub-stages executed sequentially:

1. **bw-tcp** (iperf3 TCP fan-out).
2. **bw-udp** (iperf3 UDP fan-out; sole source of ``jitter`` and
   the residual pressure that exercises the UDP-tuning dimensions).
3. **fortio-saturation** (fortio ``-qps 0`` fan-out; sole source of
   the ``rps`` metric).
4. **fortio-fixed-qps** (fortio at the configured ``fixed_qps``; sole
   source of the latency percentiles).

Sub-stages run sequentially so fortio never contends with iperf3 for
NIC, CPU, or CNI state. Within each sub-stage the source nodes still
fan out in parallel through
:class:`concurrent.futures.ThreadPoolExecutor`. The wait loop uses
:data:`concurrent.futures.FIRST_EXCEPTION` on purpose:

* On a clean run, ``wait(... FIRST_EXCEPTION)`` returns only once every
  future has either completed or raised, so it collapses to the usual
  "wait for all" semantics.
* When any client raises, ``FIRST_EXCEPTION`` returns early with that
  future in ``done`` and the still-running ones in ``not_done``. We then
  cancel the pending futures, fire a label-based cleanup against the
  namespace (belt-and-braces -- each per-client runner also cleans up
  its own Job in ``finally``), drain ``not_done`` so their own cleanup
  paths get a chance to run, and finally re-raise the primary failure.

The cleanup block *no longer swallows* ``K8sApiError``: if the label
sweep itself fails while a primary client failure is in flight, we log
at ``warning`` and raise a wrapping :class:`RuntimeError` chained from
the primary exception so neither failure is hidden from the operator.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from kube_autotuner.benchmark.client_spec import build_client_yaml
from kube_autotuner.benchmark.errors import (
    BenchmarkFailure,
    ClientJobFailed,
    JobAttemptError,
    ResultValidationError,
)
from kube_autotuner.benchmark.fortio_client_spec import (
    build_fortio_client_yaml,
    fortio_client_job_name,
)
from kube_autotuner.benchmark.fortio_parser import (
    extract_fortio_result_json,
    parse_fortio_json,
)
from kube_autotuner.benchmark.fortio_server_spec import build_fortio_server_yaml
from kube_autotuner.benchmark.parser import parse_iperf_json
from kube_autotuner.benchmark.patch import apply_patches
from kube_autotuner.benchmark.server_spec import build_server_yaml
from kube_autotuner.experiment import FortioSection, IperfSection
from kube_autotuner.k8s.client import (
    JobFailedConditionError,
    K8sApiError,
    K8sClient,
)
from kube_autotuner.models import IterationResults
from kube_autotuner.progress import NullObserver

if TYPE_CHECKING:
    from collections.abc import Callable

    from kube_autotuner.benchmark.fortio_client_spec import Workload
    from kube_autotuner.experiment import Patch
    from kube_autotuner.k8s.client import JobFailureDiagnostics
    from kube_autotuner.models import (
        BenchmarkConfig,
        BenchmarkResult,
        HostStatePhase,
        HostStateSnapshot,
        LatencyResult,
        NodePair,
    )
    from kube_autotuner.progress import ProgressObserver
    from kube_autotuner.sysctl.backend import SysctlBackend

_T = TypeVar("_T")

logger = logging.getLogger(__name__)


def _diagnostics_from(exc: BaseException) -> list[JobFailureDiagnostics]:
    """Extract per-attempt diagnostics from a :class:`ClientJobFailed`.

    Returns ``[]`` for any other exception type so the stage method
    can unconditionally call this on ``first_exc``.

    Args:
        exc: The first exception raised by a stage's per-client
            future.

    Returns:
        The per-attempt diagnostics list, or ``[]`` when ``exc`` is
        not a :class:`ClientJobFailed` (e.g. a cleanup
        :class:`RuntimeError` surfaced from the retry loop's
        ``finally`` block).
    """
    if isinstance(exc, ClientJobFailed):
        return list(exc.diagnostics)
    return []


def _server_container_status_rows(pod: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Flatten a server pod's ``containerStatuses`` to plain dicts.

    Captures the fields the plan calls out for the stage snapshot:
    container name, ``ready``, ``restartCount``, and the
    ``lastState.terminated`` reason/exit code so a recently crashed
    iperf3 server is visible as a non-zero ``restartCount`` plus a
    ``last_terminated`` payload.

    Args:
        pod: Typed pod object from the CoreV1 API.

    Returns:
        A list of dicts, one per container. Empty list when the pod
        has no container statuses yet.
    """
    rows: list[dict[str, Any]] = []
    statuses = getattr(getattr(pod, "status", None), "container_statuses", None) or []
    for cs in statuses:
        last_state = getattr(cs, "last_state", None)
        term = getattr(last_state, "terminated", None) if last_state else None
        last_terminated: dict[str, Any] = {}
        if term is not None:
            last_terminated = {
                "reason": str(getattr(term, "reason", "") or ""),
                "exit_code": (
                    int(getattr(term, "exit_code", 0) or 0)
                    if getattr(term, "exit_code", None) is not None
                    else None
                ),
                "message": str(getattr(term, "message", "") or ""),
            }
        rows.append({
            "name": str(getattr(cs, "name", "") or ""),
            "ready": bool(getattr(cs, "ready", False)),
            "restart_count": int(getattr(cs, "restart_count", 0) or 0),
            "last_terminated": last_terminated,
        })
    return rows


CLIENT_LABEL = "app.kubernetes.io/name=iperf3-client"
SERVER_LABEL = "app.kubernetes.io/name=iperf3-server"
FORTIO_CLIENT_LABEL = "app.kubernetes.io/name=fortio-client"
FORTIO_SERVER_LABEL = "app.kubernetes.io/name=fortio-server"
_IPERF_BASE_PORT = 5201
_CLIENT_WAIT_TIMEOUT_SECONDS = 180


class BenchmarkRunner:
    """Orchestrates iperf3 server/client lifecycle via the Kubernetes API."""

    def __init__(  # noqa: PLR0913 - cohesive runner wiring (node pair, config, and every side-effect dep)
        self,
        node_pair: NodePair,
        config: BenchmarkConfig,
        client: K8sClient | None = None,
        iperf_args: IperfSection | None = None,
        patches: list[Patch] | None = None,
        *,
        fortio_args: FortioSection | None = None,
        observer: ProgressObserver | None = None,
        flush_backends: list[SysctlBackend] | None = None,
        snapshot_backends: list[SysctlBackend] | None = None,
        collect_host_state: bool = False,
    ) -> None:
        """Wire the runner to a node pair and benchmark config.

        Args:
            node_pair: Source/target nodes and namespace for this run.
            config: :class:`BenchmarkConfig` (duration, iterations,
                parallel streams, TCP window).
            client: Injected :class:`K8sClient`. Defaults to a freshly
                constructed real client.
            iperf_args: Optional per-role ``extra_args`` for the iperf3
                client and server commands.
            patches: Optional kustomize patches applied to every rendered
                manifest (server Deployment/Service and every client
                Job) via :func:`kube_autotuner.benchmark.patch.apply_patches`.
            fortio_args: Optional fortio per-role ``extra_args`` plus
                ``fixed_qps`` / ``connections`` / ``duration`` for the
                latency sub-stages.
            observer: Optional progress callback. Defaults to
                :class:`~kube_autotuner.progress.NullObserver` so
                library consumers and tests see no output side-effects.
            flush_backends: Optional sysctl backends whose
                :meth:`~kube_autotuner.sysctl.backend.SysctlBackend.flush_network_state`
                is invoked at the top of each iteration in
                :meth:`run` (typically the target setter plus every
                client setter). Defaults to ``None`` -- used by
                non-optimizer callers (``run_benchmark`` /
                ``run_trial``) and unit tests that do not need the
                per-iteration flush.
            snapshot_backends: Optional sysctl backends whose
                :meth:`~kube_autotuner.sysctl.backend.SysctlBackend.collect_host_state`
                is invoked three times per iteration (baseline /
                post-flush / post-iteration) when
                ``collect_host_state`` is ``True``. Kept separate from
                ``flush_backends`` so future callers can opt into one
                without the other. Ignored entirely when
                ``collect_host_state`` is ``False``.
            collect_host_state: When ``True``, schedule the
                host-state snapshot calls above and return them on
                :attr:`IterationResults.host_state_snapshots`. Off
                by default because each snapshot is an extra
                privileged-pod schedule.
        """
        self.node_pair = node_pair
        self.config = config
        self.client = client or K8sClient()
        self.iperf_args = iperf_args or IperfSection()
        self.fortio_args = fortio_args or FortioSection()
        self.patches = patches or []
        self.observer: ProgressObserver = observer or NullObserver()
        self._flush_backends: list[SysctlBackend] = flush_backends or []
        self._snapshot_backends: list[SysctlBackend] = snapshot_backends or []
        self._collect_host_state: bool = collect_host_state
        self._no_op_warned: bool = False
        self._server_name = f"iperf3-server-{node_pair.target}"
        self._fortio_server_name = f"fortio-server-{node_pair.target}"
        self._clients = list(node_pair.all_sources)
        self._ports = {c: _IPERF_BASE_PORT + i for i, c in enumerate(self._clients)}

    def setup_server(self) -> None:
        """Deploy the iperf3 + fortio servers on the target node and await rollout."""
        server_yaml = build_server_yaml(
            node=self.node_pair.target,
            ports=list(self._ports.values()),
            ip_family_policy=self.node_pair.ip_family_policy,
            extra_args=self.iperf_args.server.extra_args,
        )
        server_yaml = apply_patches(server_yaml, self.patches)
        self.client.apply(server_yaml, self.node_pair.namespace)
        self.client.rollout_status(
            "deployment", self._server_name, self.node_pair.namespace
        )
        logger.info(
            "iperf3 server ready on %s (ports=%s)",
            self.node_pair.target,
            sorted(self._ports.values()),
        )

        fortio_server_yaml = build_fortio_server_yaml(
            node=self.node_pair.target,
            ip_family_policy=self.node_pair.ip_family_policy,
            extra_args=self.fortio_args.server.extra_args,
        )
        fortio_server_yaml = apply_patches(fortio_server_yaml, self.patches)
        self.client.apply(fortio_server_yaml, self.node_pair.namespace)
        self.client.rollout_status(
            "deployment",
            self._fortio_server_name,
            self.node_pair.namespace,
        )
        logger.info("fortio server ready on %s", self.node_pair.target)

    def _run_one_client(
        self,
        client: str,
        port: int,
        mode: Literal["tcp", "udp"],
        iteration: int,
        abort: threading.Event | None = None,
        *,
        start_at_epoch: int | None,
        wait_timeout_seconds: int,
    ) -> BenchmarkResult:
        """Run a single iperf3 client Job with retries.

        Delegates to :meth:`_run_client_job_with_retries`; see there
        for the detection/retry/abort semantics and the raise contract.

        Args:
            client: Source node name.
            port: Server-side port for this client.
            mode: ``"tcp"`` or ``"udp"``.
            iteration: Iteration index (zero-based).
            abort: Stage-level early-abort signal shared with sibling
                threads. When set, the retry loop bails out on the next
                iteration instead of burning its remaining budget.
            start_at_epoch: Absolute Unix timestamp the client should
                sleep until before exec'ing iperf3. ``None`` disables
                the barrier (single-client stage or
                ``sync_window_seconds == 0``).
            wait_timeout_seconds: Per-attempt ``condition=complete`` wait
                budget, already padded by the stage for the sync window
                when the barrier is active.

        Returns:
            The parsed :class:`BenchmarkResult` for this client and
            iteration.
        """
        job_name = f"iperf3-client-{client}-p{port}"
        client_yaml = apply_patches(
            build_client_yaml(
                node=client,
                target=self.node_pair.target,
                port=port,
                duration=self.config.duration,
                omit=self.config.omit,
                parallel=self.config.parallel,
                mode=mode,
                window=self.config.window,
                extra_args=self.iperf_args.client.extra_args,
                start_at_epoch=start_at_epoch,
            ),
            self.patches,
        )

        def parse(output: str) -> BenchmarkResult:
            try:
                raw = json.loads(output)
            except ValueError as exc:
                msg = f"iperf3 log is not valid JSON: {exc}"
                raise ResultValidationError(msg) from exc
            return parse_iperf_json(
                raw,
                mode,
                client_node=client,
                iteration=iteration,
            )

        return self._run_client_job_with_retries(
            job_name=job_name,
            job_yaml=client_yaml,
            max_attempts=self.iperf_args.max_attempts,
            parse=parse,
            kind="iperf3",
            abort=abort,
            stage_label=f"bw-{mode}",
            iteration=iteration,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    def _run_one_fortio_client(
        self,
        client: str,
        iteration: int,
        workload: Workload,
        abort: threading.Event | None = None,
        *,
        start_at_epoch: int | None,
        wait_timeout_seconds: int,
    ) -> LatencyResult:
        """Run a single fortio client Job with retries.

        Delegates to :meth:`_run_client_job_with_retries`; see there
        for the detection/retry/abort semantics and the raise contract.

        Args:
            client: Source node name.
            iteration: Iteration index (zero-based).
            workload: ``"saturation"`` or ``"fixed_qps"``.
            abort: Stage-level early-abort signal shared with sibling
                threads. When set, the retry loop bails out on the next
                iteration instead of burning its remaining budget.
            start_at_epoch: Absolute Unix timestamp the client should
                sleep until before invoking ``fortio load``. ``None``
                disables the barrier.
            wait_timeout_seconds: Per-attempt ``condition=complete``
                wait budget, already padded by the stage for the sync
                window when the barrier is active.

        Returns:
            The parsed :class:`LatencyResult` for this client,
            iteration, and sub-stage workload.
        """
        job_name = fortio_client_job_name(client, workload, iteration)
        qps = 0 if workload == "saturation" else self.fortio_args.fixed_qps
        client_yaml = apply_patches(
            build_fortio_client_yaml(
                node=client,
                target=self.node_pair.target,
                iteration=iteration,
                workload=workload,
                qps=qps,
                connections=self.fortio_args.connections,
                duration=self.fortio_args.duration,
                extra_args=self.fortio_args.client.extra_args,
                start_at_epoch=start_at_epoch,
            ),
            self.patches,
        )

        def parse(output: str) -> LatencyResult:
            try:
                raw = extract_fortio_result_json(output)
            except ValueError as exc:
                raise ResultValidationError(str(exc)) from exc
            return parse_fortio_json(
                raw,
                client_node=client,
                iteration=iteration,
                workload=workload,
            )

        return self._run_client_job_with_retries(
            job_name=job_name,
            job_yaml=client_yaml,
            max_attempts=self.fortio_args.max_attempts,
            parse=parse,
            kind="fortio",
            abort=abort,
            stage_label=f"fortio-{workload}",
            iteration=iteration,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    def _run_client_job_with_retries(  # noqa: PLR0913, PLR0915 - one cohesive retry loop with three interleaved signals
        self,
        *,
        job_name: str,
        job_yaml: str,
        max_attempts: int,
        parse: Callable[[str], _T],
        kind: str,
        abort: threading.Event | None,
        stage_label: str,
        iteration: int,
        wait_timeout_seconds: int,
    ) -> _T:
        """Run one client Job with the three-signal detection + retry loop.

        Detection signals on every attempt, in order:

        1. Job status: :meth:`K8sClient.wait` with both success
           (``Complete``) and failure (``Failed``) predicates. On a
           Failed transition we exit promptly with
           :class:`JobFailedConditionError` rather than waiting the full
           watch timeout.
        2. Log-source pod phase: :meth:`K8sClient._job_log_pod`
           returns ``None`` unless a ``Succeeded`` pod exists, so a
           Job that only managed Failed/Error pods but still hit
           ``Complete=True`` is treated as an attempt failure.
        3. Payload sanity: ``parse`` is expected to raise
           :class:`ResultValidationError` when the decoded payload is
           structurally valid JSON but semantically degenerate.

        The ``finally`` block deletes the Job every attempt so retries
        land on a fresh object; diagnostics are captured **before**
        delete because pod events vanish with the pod's garbage
        collection. The ``attempt_ok`` flag gates the diagnostic log so
        the happy path produces zero warnings.

        ``abort`` is a shared :class:`threading.Event` set by the first
        sibling client in the stage to exhaust its retry budget.
        Siblings checking at loop-top bail out immediately so a
        cluster-wide outage does not multiply wall time by N. Setting
        ``abort`` on heterogeneous failures (one flaky source among
        healthy ones) is the intentional policy given the runner's
        "raise and fail the trial" contract: the trial is going to be
        marked failed either way once one client has exhausted.

        Args:
            job_name: Job metadata.name.
            job_yaml: Rendered Job manifest.
            max_attempts: Total attempts including the first try.
            parse: Callable converting the container log body into
                ``_T``. Must raise :class:`ResultValidationError` on
                degenerate payloads.
            kind: Short label (``"iperf3"`` / ``"fortio"``) used in log
                messages.
            abort: Optional stage-level early-abort signal.
            stage_label: Sub-stage label threaded into every warning
                (``"bw-tcp"`` / ``"bw-udp"`` / ``"fortio-saturation"``
                / ``"fortio-fixed_qps"``).
            iteration: Zero-based iteration index threaded into every
                warning.
            wait_timeout_seconds: Per-attempt ``condition=complete``
                wait budget, sourced from the calling stage so
                single-client stages retain the base
                ``_CLIENT_WAIT_TIMEOUT_SECONDS`` budget and multi-client
                stages pad it by ``sync_window_seconds``.

        Returns:
            The value produced by ``parse`` on the first successful
            attempt.

        Raises:
            ClientJobFailed: ``max_attempts`` were exhausted; carries
                the accumulated per-attempt
                :class:`JobFailureDiagnostics` list for the stage
                method to fold into :class:`BenchmarkFailure`.
            RuntimeError: The per-attempt ``delete`` failed, or the
                retry loop exited without recording any error (should
                not happen; signals a logic bug).
            JobAttemptError: Stage-level ``abort`` fired before the
                first attempt completed successfully.
        """
        ns = self.node_pair.namespace
        last_error: Exception | None = None
        diagnostics: list[JobFailureDiagnostics] = []
        for attempt in range(1, max_attempts + 1):
            if abort is not None and abort.is_set():
                break
            attempt_ok = False
            try:
                self.client.apply(job_yaml, ns)
                try:
                    self.client.wait(
                        "job",
                        job_name,
                        "condition=complete",
                        ns,
                        timeout=wait_timeout_seconds,
                        failure_condition="condition=failed",
                    )
                    log_pod = self.client._job_log_pod(job_name, ns)  # noqa: SLF001
                    if log_pod is None:
                        msg = (
                            f"Job {job_name} completed but has no Succeeded pod "
                            f"(attempt {attempt}/{max_attempts})"
                        )
                        raise JobAttemptError(msg)
                    output = self.client.logs("pod", log_pod, ns)
                    result = parse(output)
                    attempt_ok = True
                    return result
                finally:
                    if not attempt_ok:
                        diag = self._log_job_diagnostics(
                            job_name,
                            ns,
                            kind,
                            attempt,
                            stage_label=stage_label,
                            iteration=iteration,
                        )
                        if diag is not None:
                            diagnostics.append(diag)
                    try:
                        self.client.delete("job", job_name, ns, ignore_not_found=True)
                    except K8sApiError as cleanup_err:
                        logger.warning(
                            "Best-effort delete of %s client job %s failed",
                            kind,
                            job_name,
                            exc_info=True,
                        )
                        msg = f"cleanup of {kind} client job {job_name} failed"
                        raise RuntimeError(msg) from cleanup_err
            except (
                K8sApiError,
                JobFailedConditionError,
                JobAttemptError,
                ResultValidationError,
            ) as exc:
                last_error = exc
                logger.warning(
                    "%s client Job %s [stage=%s iter=%d] attempt %d/%d failed: %s",
                    kind,
                    job_name,
                    stage_label,
                    iteration,
                    attempt,
                    max_attempts,
                    exc,
                )
                continue
        if abort is not None and abort.is_set():
            msg = f"{kind} client Job {job_name} aborted by sibling failure"
            if last_error is not None:
                raise JobAttemptError(msg) from last_error
            raise JobAttemptError(msg)
        if abort is not None:
            abort.set()
        if last_error is None:
            msg = (
                f"{kind} client Job {job_name}: retry loop exited with no error"
                f" recorded (max_attempts={max_attempts})"
            )
            raise RuntimeError(msg)
        msg = (
            f"{kind} client Job {job_name} [stage={stage_label} iter={iteration}] "
            f"failed after {max_attempts} attempts"
        )
        raise ClientJobFailed(msg, diagnostics=diagnostics) from last_error

    def _log_job_diagnostics(
        self,
        job_name: str,
        namespace: str,
        kind: str,
        attempt: int,
        *,
        stage_label: str,
        iteration: int,
    ) -> JobFailureDiagnostics | None:
        """Emit a single warning line describing a failed Job attempt.

        Pulls :meth:`K8sClient.describe_job_failure` and renders the
        returned :class:`JobFailureDiagnostics` into one structured log
        record. Diagnostics must never mask the primary failure: every
        downstream call in the diagnostic path is best-effort and we
        log at ``warning`` rather than raising.

        Invariant: this method MUST NOT propagate any exception. The
        retry loop's per-attempt ``finally`` block runs this before the
        Job ``delete``; if this raised, ``delete`` would be skipped and
        the Job would leak until the stage-level label sweep runs.

        Args:
            job_name: Job metadata.name.
            namespace: Target namespace.
            kind: Short label for log grouping (``"iperf3"``, ``"fortio"``).
            attempt: 1-based attempt index.
            stage_label: Sub-stage label (``"bw-tcp"`` / ``"bw-udp"`` /
                ``"fortio-saturation"`` / ``"fortio-fixed_qps"``).
            iteration: Zero-based iteration index.

        Returns:
            The :class:`JobFailureDiagnostics` payload on success, or
            ``None`` if the describe call itself failed. The retry loop
            accumulates non-``None`` returns into the envelope carried
            by :class:`ClientJobFailed`.
        """
        try:
            diag = self.client.describe_job_failure(job_name, namespace)
        except Exception:
            logger.warning(
                "Could not describe %s client job %s [stage=%s iter=%d] (attempt %d)",
                kind,
                job_name,
                stage_label,
                iteration,
                attempt,
                exc_info=True,
            )
            return None
        logger.warning(
            "%s client Job %s [stage=%s iter=%d] attempt %d diagnostics: %s",
            kind,
            job_name,
            stage_label,
            iteration,
            attempt,
            diag,
        )
        return diag

    def _flush_network_state(self, iteration: int) -> None:
        """Flush per-iteration kernel network state on every configured backend.

        No-op when the runner was constructed without
        ``flush_backends`` (non-optimizer callers and pure unit tests).
        Each backend's implementation is contractually log-and-continue
        on failure, so a flush failure never aborts the iteration loop.

        Emits one INFO banner naming the iteration and the backend count,
        plus a start / complete INFO pair around every backend so
        operators can correlate flush activity with the surrounding stage
        logs. ``SysctlSetter.flush_network_state`` runs a real privileged
        pod; ``TalosSysctlBackend`` and ``FakeSysctlBackend`` are
        effectively no-ops, so ``complete`` on those will follow
        ``starting`` with near-zero elapsed -- the bracketing is emitted
        uniformly so operators can confirm every configured backend was
        attempted, but the log does not imply real work on the Talos or
        Fake paths.

        Args:
            iteration: Zero-based iteration index; presented as
                ``iteration + 1`` in the banner to match the adjacent
                "Running iteration N/M" one-based convention so
                operators do not have to reconcile two numbering
                schemes.
        """
        if not self._flush_backends:
            return
        logger.info(
            "Flushing network state before iteration %d on %d backend(s)",
            iteration + 1,
            len(self._flush_backends),
        )
        for backend in self._flush_backends:
            node = getattr(backend, "node", type(backend).__name__)
            flush_t0 = time.monotonic()
            logger.info("Network flush starting on %s", node)
            backend.flush_network_state()
            logger.info(
                "Network flush complete on %s (elapsed=%.1fs)",
                node,
                time.monotonic() - flush_t0,
            )

    def _snapshot_host_state(
        self,
        iteration: int | None,
        phase: HostStatePhase,
        out: list[HostStateSnapshot],
    ) -> None:
        """Collect one snapshot per snapshot backend, skipping ``None`` returns.

        Guarded by :attr:`_collect_host_state`; no-ops cleanly when the
        flag is off. When all backends return ``None`` for the
        ``"baseline"`` phase (Talos-only deployment), logs a single
        warning so operators know the flag is a no-op.

        Args:
            iteration: Iteration index; ``None`` for baseline.
            phase: Which collection point is firing.
            out: List to append collected snapshots to.
        """
        if not self._collect_host_state:
            return
        collected_any = False
        for backend in self._snapshot_backends:
            snapshot = backend.collect_host_state(iteration, phase)
            if snapshot is None:
                continue
            collected_any = True
            out.append(snapshot)
        if phase == "baseline" and not collected_any and not self._no_op_warned:
            logger.warning(
                "collect_host_state=True but every snapshot backend returned "
                "None on baseline (Talos-only deployment?); no snapshots "
                "will be recorded this run",
            )
            self._no_op_warned = True

    def run(self) -> IterationResults:  # noqa: PLR0915 - one cohesive iteration driver (flush + 4 sub-stages + observer fan-out)
        """Run every configured benchmark iteration.

        Each iteration expands into up to four sub-stages executed
        sequentially so fortio never contends with iperf3 for NIC,
        CPU, or CNI state: ``bw-tcp`` -> ``bw-udp`` -> ``fortio-sat``
        -> ``fortio-fixed``. Sub-stages absent from
        :attr:`BenchmarkConfig.stages` are skipped entirely; the
        observer callbacks and stage logging only fire for enabled
        stages.

        Before every iteration starts, :meth:`_flush_network_state`
        evicts cached TCP metrics and conntrack entries on every
        configured backend so the previous iteration's state cannot
        bleed into the next. Relies on the caller having applied
        ``tcp_no_metrics_save=1`` before calling :meth:`run`; the pin
        stays in effect across all iterations (no restore happens
        mid-run), so the ordering invariant holds from iteration 0
        onward.

        Fires :meth:`ProgressObserver.on_benchmark_start` once up
        front with the trial-wide iteration budget
        (``config.iterations``) so observers can size a trial-scoped
        progress bar before the per-iteration callbacks start.

        Returns:
            An :class:`IterationResults` holding every
            :class:`BenchmarkResult` and :class:`LatencyResult`
            produced across iterations.
        """
        self.observer.on_benchmark_start(self.config.iterations)
        logger.info(
            "Starting benchmark: %d iteration(s), target=%s, sources=%s",
            self.config.iterations,
            self.node_pair.target,
            self._clients,
        )
        all_bench: list[BenchmarkResult] = []
        all_latency: list[LatencyResult] = []
        snapshots: list[HostStateSnapshot] = []
        # Reset so each run() re-evaluates the no-op triggers. The runner
        # is reused across trials in OptimizationLoop; without this reset
        # operators would miss the second+ warning when the condition
        # still holds.
        self._no_op_warned = False
        if self._collect_host_state and not self._snapshot_backends:
            logger.warning(
                "collect_host_state=True but snapshot_backends is empty; "
                "no host-state snapshots will be recorded",
            )
            self._no_op_warned = True
        self._snapshot_host_state(None, "baseline", snapshots)
        for i in range(self.config.iterations):
            self._flush_network_state(i)
            self._snapshot_host_state(i, "post-flush", snapshots)
            logger.info(
                "Running iteration %d/%d: %s -> %s",
                i + 1,
                self.config.iterations,
                self._clients,
                self.node_pair.target,
            )
            self.observer.on_iteration_start(i)
            bench_tcp: list[BenchmarkResult] = []
            bench_udp: list[BenchmarkResult] = []
            sat: list[LatencyResult] = []
            fixed: list[LatencyResult] = []
            try:
                if "bw-tcp" in self.config.stages:
                    self.observer.on_stage_start("bw-tcp", i)
                    bw_tcp_t0 = time.monotonic()
                    logger.info(
                        "Stage %s starting (iteration %d/%d)",
                        "bw-tcp",
                        i + 1,
                        self.config.iterations,
                    )
                    try:
                        bench_tcp = self._run_bandwidth_stage("tcp", i)
                    finally:
                        logger.info(
                            "Stage %s complete (iteration %d/%d, elapsed=%.1fs)",
                            "bw-tcp",
                            i + 1,
                            self.config.iterations,
                            time.monotonic() - bw_tcp_t0,
                        )
                        self.observer.on_stage_end("bw-tcp", i)
                if "bw-udp" in self.config.stages:
                    self.observer.on_stage_start("bw-udp", i)
                    bw_udp_t0 = time.monotonic()
                    logger.info(
                        "Stage %s starting (iteration %d/%d)",
                        "bw-udp",
                        i + 1,
                        self.config.iterations,
                    )
                    try:
                        bench_udp = self._run_bandwidth_stage("udp", i)
                    finally:
                        logger.info(
                            "Stage %s complete (iteration %d/%d, elapsed=%.1fs)",
                            "bw-udp",
                            i + 1,
                            self.config.iterations,
                            time.monotonic() - bw_udp_t0,
                        )
                        self.observer.on_stage_end("bw-udp", i)
                if "fortio-sat" in self.config.stages:
                    self.observer.on_stage_start("fortio-sat", i)
                    sat_t0 = time.monotonic()
                    logger.info(
                        "Stage %s starting (iteration %d/%d)",
                        "fortio-sat",
                        i + 1,
                        self.config.iterations,
                    )
                    try:
                        sat = self._run_latency_stage(i, workload="saturation")
                    finally:
                        logger.info(
                            "Stage %s complete (iteration %d/%d, elapsed=%.1fs)",
                            "fortio-sat",
                            i + 1,
                            self.config.iterations,
                            time.monotonic() - sat_t0,
                        )
                        self.observer.on_stage_end("fortio-sat", i)
                if "fortio-fixed" in self.config.stages:
                    self.observer.on_stage_start("fortio-fixed", i)
                    fixed_t0 = time.monotonic()
                    logger.info(
                        "Stage %s starting (iteration %d/%d)",
                        "fortio-fixed",
                        i + 1,
                        self.config.iterations,
                    )
                    try:
                        fixed = self._run_latency_stage(i, workload="fixed_qps")
                    finally:
                        logger.info(
                            "Stage %s complete (iteration %d/%d, elapsed=%.1fs)",
                            "fortio-fixed",
                            i + 1,
                            self.config.iterations,
                            time.monotonic() - fixed_t0,
                        )
                        self.observer.on_stage_end("fortio-fixed", i)
            finally:
                self.observer.on_iteration_end(i)
                self._snapshot_host_state(i, "post-iteration", snapshots)
            all_bench.extend([*bench_tcp, *bench_udp])
            all_latency.extend([*sat, *fixed])
        logger.info(
            "Benchmark complete: %d iteration(s), %d bench + %d latency result(s)",
            self.config.iterations,
            len(all_bench),
            len(all_latency),
        )
        return IterationResults(
            bench=all_bench,
            latency=all_latency,
            host_state_snapshots=snapshots,
        )

    def _stage_barrier(self) -> tuple[int | None, int]:
        """Compute the per-stage start-time barrier and wait-timeout budget.

        The barrier is active only when ``sync_window_seconds > 0`` and
        the stage has more than one client to align. Single-client
        stages skip the sleep entirely and retain the unpadded
        ``_CLIENT_WAIT_TIMEOUT_SECONDS`` budget so they do not pay for
        slack they cannot use.

        Returns:
            A tuple ``(start_at_epoch, wait_timeout_seconds)``.
            ``start_at_epoch`` is the shared Unix timestamp every client
            in this stage should sleep until (``None`` when the barrier
            is inactive). ``wait_timeout_seconds`` is the
            ``condition=complete`` budget passed to
            :meth:`K8sClient.wait` -- base 180 s when the barrier is
            inactive, padded by ``sync_window_seconds`` otherwise.
        """
        barrier_active = self.config.sync_window_seconds > 0 and len(self._clients) > 1
        if not barrier_active:
            return (None, _CLIENT_WAIT_TIMEOUT_SECONDS)
        start_at_epoch = int(time.time()) + self.config.sync_window_seconds
        wait_timeout_seconds = (
            _CLIENT_WAIT_TIMEOUT_SECONDS + self.config.sync_window_seconds
        )
        return (start_at_epoch, wait_timeout_seconds)

    def _run_bandwidth_stage(
        self,
        mode: Literal["tcp", "udp"],
        iteration: int,
    ) -> list[BenchmarkResult]:
        """Launch one iperf3 client Job per client concurrently.

        Called twice per iteration from :meth:`run` -- once with
        ``mode="tcp"`` (the ``bw-tcp`` sub-stage), once with
        ``mode="udp"`` (the ``bw-udp`` sub-stage).

        Returns:
            The list of :class:`BenchmarkResult`, one per client, for
            this iteration. On any client failure a
            :class:`BenchmarkFailure` envelope is raised (carrying the
            per-attempt client diagnostics plus a server-side snapshot)
            and no partial results are returned.

        Raises:
            BenchmarkFailure: A client Job's future raised. The
                envelope chains the original exception via ``from`` and
                exposes per-attempt diagnostics plus the server-side
                snapshot for the optimizer to persist.
        """
        stage_label = f"bw-{mode}"
        abort = threading.Event()
        start_at_epoch, wait_timeout_seconds = self._stage_barrier()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self._clients),
        ) as executor:
            future_to_client = {
                executor.submit(
                    self._run_one_client,
                    client,
                    self._ports[client],
                    mode,
                    iteration,
                    abort,
                    start_at_epoch=start_at_epoch,
                    wait_timeout_seconds=wait_timeout_seconds,
                ): client
                for client in self._clients
            }
            done, not_done = concurrent.futures.wait(
                future_to_client,
                return_when=concurrent.futures.FIRST_EXCEPTION,
            )

            first_exc: BaseException | None = None
            for f in done:
                exc = f.exception()
                if exc is not None:
                    first_exc = exc
                    break

            if first_exc is not None:
                server_snapshots = self._collect_server_snapshot(label=SERVER_LABEL)
                self._cleanup_after_failure(
                    cast(
                        "set[concurrent.futures.Future[BenchmarkResult | LatencyResult]]",  # noqa: E501
                        not_done,
                    ),
                    first_exc,
                    CLIENT_LABEL,
                )
                raise BenchmarkFailure(
                    cause=first_exc,
                    attempt_diagnostics=_diagnostics_from(first_exc),
                    server_snapshots=server_snapshots,
                    stage=stage_label,
                    iteration=iteration,
                ) from first_exc

            return [f.result() for f in done]

    def _run_latency_stage(
        self,
        iteration: int,
        *,
        workload: Workload,
    ) -> list[LatencyResult]:
        """Launch one fortio client Job per client concurrently.

        Returns:
            The list of :class:`LatencyResult`, one per client, for
            this iteration and sub-stage. On any client failure a
            :class:`BenchmarkFailure` envelope is raised (carrying the
            per-attempt client diagnostics plus a fortio-server-side
            snapshot) and no partial results are returned.

        Raises:
            BenchmarkFailure: A fortio client Job's future raised.
                The envelope chains the original exception via
                ``from`` and exposes per-attempt diagnostics plus the
                fortio-server-side snapshot for the optimizer to
                persist.
        """
        stage_label = f"fortio-{workload}"
        abort = threading.Event()
        start_at_epoch, wait_timeout_seconds = self._stage_barrier()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self._clients),
        ) as executor:
            future_to_client = {
                executor.submit(
                    self._run_one_fortio_client,
                    client,
                    iteration,
                    workload,
                    abort,
                    start_at_epoch=start_at_epoch,
                    wait_timeout_seconds=wait_timeout_seconds,
                ): client
                for client in self._clients
            }
            done, not_done = concurrent.futures.wait(
                future_to_client,
                return_when=concurrent.futures.FIRST_EXCEPTION,
            )

            first_exc: BaseException | None = None
            for f in done:
                exc = f.exception()
                if exc is not None:
                    first_exc = exc
                    break

            if first_exc is not None:
                server_snapshots = self._collect_server_snapshot(
                    label=FORTIO_SERVER_LABEL,
                )
                self._cleanup_after_failure(
                    cast(
                        "set[concurrent.futures.Future[BenchmarkResult | LatencyResult]]",  # noqa: E501
                        not_done,
                    ),
                    first_exc,
                    FORTIO_CLIENT_LABEL,
                )
                raise BenchmarkFailure(
                    cause=first_exc,
                    attempt_diagnostics=_diagnostics_from(first_exc),
                    server_snapshots=server_snapshots,
                    stage=stage_label,
                    iteration=iteration,
                ) from first_exc

            return [f.result() for f in done]

    def _collect_server_snapshot(self, *, label: str) -> list[dict[str, Any]]:
        """Snapshot server pods matching ``label``; log a warning, return rows.

        Fires once per failed stage from the stage method's
        ``first_exc is not None`` branch -- no cross-thread
        coordination because both :meth:`_run_bandwidth_stage` and
        :meth:`_run_latency_stage` execute on the calling thread after
        the executor's ``wait(...)`` returns.

        Best-effort: every downstream call is wrapped so a missing
        server, vanished pod, events-listing failure, or log-read
        failure degrades to an empty row (or empty return) rather than
        raising. Diagnostics must never mask the primary failure they
        describe.

        Args:
            label: Label selector for the server pods, e.g.
                ``SERVER_LABEL`` or ``FORTIO_SERVER_LABEL``.

        Returns:
            One dict per server pod with ``name``, ``phase``,
            ``container_statuses`` (name, ready, restart_count,
            last_terminated.reason/exit_code), ``events``, and
            ``log_tail`` fields. Empty list when listing failed or no
            pods matched.
        """
        ns = self.node_pair.namespace
        try:
            pods = self.client.list_pods_by_label(label, ns)
        except Exception:
            logger.warning(
                "server snapshot [label=%s] list_pods_by_label failed",
                label,
                exc_info=True,
            )
            return []
        rows: list[dict[str, Any]] = []
        for pod in pods:
            name = str(getattr(getattr(pod, "metadata", None), "name", "") or "")
            phase = str(getattr(getattr(pod, "status", None), "phase", "") or "")
            container_statuses = _server_container_status_rows(pod)
            events: list[Any] = []
            log_tail = ""
            if name:
                try:
                    events = list(
                        self.client._recent_pod_events(name, ns, 10),  # noqa: SLF001
                    )
                except Exception:  # noqa: BLE001 - best-effort diagnostic, swallow silently
                    events = []
                try:
                    log_tail = self.client._read_pod_log_tail(name, ns, 100)  # noqa: SLF001
                except Exception:  # noqa: BLE001 - best-effort diagnostic, swallow silently
                    log_tail = ""
            rows.append({
                "name": name,
                "phase": phase,
                "container_statuses": container_statuses,
                "events": events,
                "log_tail": log_tail,
            })
        logger.warning(
            "server snapshot [label=%s] pods=%d rows=%s",
            label,
            len(rows),
            rows,
        )
        return rows

    def _cleanup_after_failure(
        self,
        not_done: set[concurrent.futures.Future[BenchmarkResult | LatencyResult]],
        first_exc: BaseException,
        label: str,
    ) -> None:
        """Cancel pending client Jobs, sweep by label, drain remaining futures.

        When the label sweep itself fails, the cleanup error is raised
        with ``first_exc`` as its explicit cause so the operator sees
        both the original client failure and the cleanup failure in one
        chained traceback.

        Args:
            not_done: Futures that had not completed when the first
                exception surfaced.
            first_exc: The primary failure to re-raise, attached as the
                cause of any cleanup failure.
            label: Selector (e.g. ``CLIENT_LABEL`` /
                ``FORTIO_CLIENT_LABEL``) used to sweep the matching
                client Jobs in the namespace.

        Raises:
            RuntimeError: The label-based cleanup API call returned
                non-zero. The original ``first_exc`` is attached as the
                exception's cause.
        """
        for f in not_done:
            f.cancel()
        try:
            self.client.delete_by_label(
                "job",
                label,
                self.node_pair.namespace,
            )
        except K8sApiError as cleanup_err:
            logger.warning(
                "Failed label-based cleanup of client jobs (label=%s)",
                label,
                exc_info=True,
            )
            msg = f"label-based cleanup failed after primary failure: {cleanup_err}"
            raise RuntimeError(msg) from first_exc
        # Drain remaining futures so their own finally blocks complete.
        # The primary failure is already captured in ``first_exc``; any
        # per-client cleanup error was handled inside the per-client
        # runner. Log at debug and move on.
        for f in not_done:
            try:
                f.result()
            except Exception:
                logger.debug("drained client future after failure", exc_info=True)

    def cleanup(self) -> None:
        """Remove iperf3 + fortio server/client resources by label."""
        ns = self.node_pair.namespace
        for label in (SERVER_LABEL, FORTIO_SERVER_LABEL):
            self.client.delete_by_label("deployment", label, ns)
            self.client.delete_by_label("service", label, ns)
        for label in (CLIENT_LABEL, FORTIO_CLIENT_LABEL):
            self.client.delete_by_label("job", label, ns)
        logger.info("iperf3 + fortio resources cleaned up")
