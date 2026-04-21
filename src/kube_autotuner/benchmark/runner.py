"""Concurrent iperf3 + fortio benchmark runner.

:class:`BenchmarkRunner` launches one iperf3 server Deployment and one
fortio server Deployment on the target node, then drives each iteration
as three sub-stages executed sequentially:

1. **bandwidth** (iperf3 fan-out).
2. **fortio-saturation** (fortio ``-qps 0`` fan-out; sole source of
   the ``rps`` metric).
3. **fortio-fixed-qps** (fortio at the configured ``fixed_qps``; sole
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
import queue
import threading
from typing import TYPE_CHECKING, Literal, cast

from kube_autotuner.benchmark.client_spec import build_client_yaml
from kube_autotuner.benchmark.fortio_client_spec import (
    build_fortio_client_yaml,
    fortio_client_job_name,
)
from kube_autotuner.benchmark.fortio_parser import (
    extract_fortio_result_json,
    parse_fortio_json,
)
from kube_autotuner.benchmark.fortio_server_spec import build_fortio_server_yaml
from kube_autotuner.benchmark.parser import parse_iperf_json, parse_k8s_memory
from kube_autotuner.benchmark.patch import apply_patches
from kube_autotuner.benchmark.server_spec import build_server_yaml
from kube_autotuner.experiment import CniSection, FortioSection, IperfSection
from kube_autotuner.k8s.client import K8sApiError, K8sClient
from kube_autotuner.models import IterationResults
from kube_autotuner.progress import NullObserver

if TYPE_CHECKING:
    from kube_autotuner.benchmark.fortio_client_spec import Workload
    from kube_autotuner.experiment import Patch
    from kube_autotuner.models import (
        BenchmarkConfig,
        BenchmarkResult,
        LatencyResult,
        NodePair,
    )
    from kube_autotuner.progress import ProgressObserver

logger = logging.getLogger(__name__)

CLIENT_LABEL = "app.kubernetes.io/name=iperf3-client"
SERVER_LABEL = "app.kubernetes.io/name=iperf3-server"
FORTIO_CLIENT_LABEL = "app.kubernetes.io/name=fortio-client"
FORTIO_SERVER_LABEL = "app.kubernetes.io/name=fortio-server"
_IPERF_BASE_PORT = 5201
_CLIENT_WAIT_TIMEOUT_SECONDS = 180
SAMPLE_INTERVAL_S = 5.0


class BenchmarkRunner:
    """Orchestrates iperf3 server/client lifecycle via the Kubernetes API."""

    def __init__(  # noqa: PLR0913
        self,
        node_pair: NodePair,
        config: BenchmarkConfig,
        client: K8sClient | None = None,
        iperf_args: IperfSection | None = None,
        patches: list[Patch] | None = None,
        cni: CniSection | None = None,
        *,
        fortio_args: FortioSection | None = None,
        observer: ProgressObserver | None = None,
    ) -> None:
        """Wire the runner to a node pair and benchmark config.

        Args:
            node_pair: Source/target nodes and namespace for this run.
            config: :class:`BenchmarkConfig` (duration, iterations, modes,
                parallel streams, TCP window).
            client: Injected :class:`K8sClient`. Defaults to a freshly
                constructed real client.
            iperf_args: Optional per-role ``extra_args`` for the iperf3
                client and server commands.
            patches: Optional kustomize patches applied to every rendered
                manifest (server Deployment/Service and every client
                Job) via :func:`kube_autotuner.benchmark.patch.apply_patches`.
            cni: Selector for CNI pods to track on the target node. When
                ``enabled`` is ``False``, CNI sampling is skipped.
            fortio_args: Optional fortio per-role ``extra_args`` plus
                ``fixed_qps`` / ``connections`` / ``duration`` for the
                latency sub-stages.
            observer: Optional progress callback. Defaults to
                :class:`~kube_autotuner.progress.NullObserver` so
                library consumers and tests see no output side-effects.
        """
        self.node_pair = node_pair
        self.config = config
        self.client = client or K8sClient()
        self.iperf_args = iperf_args or IperfSection()
        self.fortio_args = fortio_args or FortioSection()
        self.patches = patches or []
        self.cni = cni or CniSection()
        self.observer: ProgressObserver = observer or NullObserver()
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
    ) -> BenchmarkResult:
        """Run a single iperf3 client Job and return a parsed result.

        Returns:
            The parsed :class:`BenchmarkResult` for this client and
            iteration.

        Raises:
            RuntimeError: When the best-effort Job cleanup in
                ``finally`` fails; the original cleanup
                :class:`K8sApiError` is attached as the cause.
        """
        job_name = f"iperf3-client-{client}-p{port}"
        ns = self.node_pair.namespace

        client_yaml = build_client_yaml(
            node=client,
            target=self.node_pair.target,
            port=port,
            duration=self.config.duration,
            omit=self.config.omit,
            parallel=self.config.parallel,
            mode=mode,
            window=self.config.window,
            extra_args=self.iperf_args.client.extra_args,
        )
        client_yaml = apply_patches(client_yaml, self.patches)

        try:
            self.client.apply(client_yaml, ns)
            self.client.wait(
                "job",
                job_name,
                "condition=complete",
                ns,
                timeout=_CLIENT_WAIT_TIMEOUT_SECONDS,
            )
            output = self.client.logs("job", job_name, ns)
            raw = json.loads(output)
            return parse_iperf_json(
                raw,
                mode,
                client_node=client,
                iteration=iteration,
            )
        finally:
            try:
                self.client.delete("job", job_name, ns, ignore_not_found=True)
            except K8sApiError as cleanup_err:
                logger.warning(
                    "Best-effort delete of client job %s failed",
                    job_name,
                    exc_info=True,
                )
                # Don't swallow: surface the cleanup failure. If a primary
                # exception is in flight, Python chains it via
                # ``__context__``; we wrap the cleanup error explicitly so
                # the chain is visible in tracebacks.
                msg = f"cleanup of client job {job_name} failed"
                raise RuntimeError(msg) from cleanup_err

    def _run_one_fortio_client(
        self,
        client: str,
        iteration: int,
        workload: Workload,
    ) -> LatencyResult:
        """Run a single fortio client Job and return a parsed result.

        Returns:
            The parsed :class:`LatencyResult` for this client,
            iteration, and sub-stage workload.

        Raises:
            RuntimeError: When the best-effort Job cleanup in
                ``finally`` fails; the original cleanup
                :class:`K8sApiError` is attached as the cause.
        """
        job_name = fortio_client_job_name(client, workload, iteration)
        ns = self.node_pair.namespace
        qps = 0 if workload == "saturation" else self.fortio_args.fixed_qps

        client_yaml = build_fortio_client_yaml(
            node=client,
            target=self.node_pair.target,
            iteration=iteration,
            workload=workload,
            qps=qps,
            connections=self.fortio_args.connections,
            duration=self.fortio_args.duration,
            extra_args=self.fortio_args.client.extra_args,
        )
        client_yaml = apply_patches(client_yaml, self.patches)

        try:
            self.client.apply(client_yaml, ns)
            self.client.wait(
                "job",
                job_name,
                "condition=complete",
                ns,
                timeout=_CLIENT_WAIT_TIMEOUT_SECONDS,
            )
            output = self.client.logs("job", job_name, ns)
            raw = extract_fortio_result_json(output)
            return parse_fortio_json(
                raw,
                client_node=client,
                iteration=iteration,
                workload=workload,
            )
        finally:
            try:
                self.client.delete("job", job_name, ns, ignore_not_found=True)
            except K8sApiError as cleanup_err:
                logger.warning(
                    "Best-effort delete of fortio client job %s failed",
                    job_name,
                    exc_info=True,
                )
                msg = f"cleanup of fortio client job {job_name} failed"
                raise RuntimeError(msg) from cleanup_err

    def _sum_cni_memory(self, target: str) -> int | None:
        """Return summed CNI-pod memory on ``target``, or ``None`` when empty.

        Args:
            target: Node name whose CNI pods are totalled.

        Returns:
            Total memory in bytes across every container of every CNI
            pod on ``target``, or ``None`` when no container produced a
            reading.
        """
        pods = self.client.list_pods_by_selector_on_node(
            self.cni.label_selector,
            self.cni.namespace,
            target,
        )
        total = 0
        saw_any = False
        for pod in pods:
            for r in self.client.top_pod_containers(pod, self.cni.namespace):
                m = r.get("memory")
                if m:
                    total += parse_k8s_memory(m)
                    saw_any = True
        return total if saw_any else None

    def _sample_resources_loop(
        self,
        stop: threading.Event,
        sink: queue.Queue[dict[str, int]],
    ) -> None:
        """Poll metrics-server for node and CNI memory until ``stop`` is set.

        On every tick the loop polls:

        * Whole-node memory on ``self.node_pair.target`` via
          :meth:`K8sClient.top_node`.
        * When ``self.cni.enabled``, memory summed across the CNI pods
          on the target node matching ``self.cni.label_selector`` in
          ``self.cni.namespace``.

        Empty rows (no scrape yet) are silently skipped; poisoning the
        peak with ``0`` would be wrong. The loop body catches
        :class:`Exception` broadly and logs at ``warning``: this is a
        daemon thread, and an uncaught exception would terminate it
        silently and leave the operator with no sampling.

        Args:
            stop: Event signalling shutdown. The loop blocks on
                ``stop.wait(SAMPLE_INTERVAL_S)`` so shutdown is prompt.
            sink: Thread-safe queue receiving one dict per successful
                poll. Keys are a subset of ``{"node", "cni"}``.
        """
        target = self.node_pair.target
        while not stop.is_set():
            sample: dict[str, int] = {}
            try:
                node_mem = self.client.top_node(target).get("memory")
                if node_mem:
                    sample["node"] = parse_k8s_memory(node_mem)
                if self.cni.enabled:
                    cni_total = self._sum_cni_memory(target)
                    if cni_total is not None:
                        sample["cni"] = cni_total
            except Exception:
                logger.warning(
                    "Resource sampler poll failed; continuing",
                    exc_info=True,
                )
            if sample:
                sink.put(sample)
            stop.wait(SAMPLE_INTERVAL_S)

    def run(self) -> IterationResults:  # noqa: C901, PLR0912, PLR0915
        """Run all configured benchmark iterations and modes.

        Each iteration expands into three sub-stages (bandwidth,
        fortio saturation, fortio fixed-qps) executed sequentially so
        fortio never contends with iperf3 for NIC, CPU, or CNI state.
        The resource sampler straddles the whole iteration so peak
        node and CNI memory reflect all three sub-stages rather than
        any one phase.

        Fires :meth:`ProgressObserver.on_benchmark_start` once up
        front with the trial-wide iteration budget
        (``len(modes) * iterations``) so observers can size a
        trial-scoped progress bar before the per-mode callbacks start.

        Returns:
            An :class:`IterationResults` holding every
            :class:`BenchmarkResult` and :class:`LatencyResult`
            produced across iterations and modes, tagged with the
            peak node and CNI memory observed during each iteration
            when sampling produced any data.
        """
        self.observer.on_benchmark_start(
            len(self.config.modes) * self.config.iterations,
        )
        all_bench: list[BenchmarkResult] = []
        all_latency: list[LatencyResult] = []
        for mode in self.config.modes:
            self.observer.on_mode_start(mode, self.config.iterations)
            for i in range(self.config.iterations):
                logger.info(
                    "Running %s iteration %d/%d: %s -> %s",
                    mode,
                    i + 1,
                    self.config.iterations,
                    self._clients,
                    self.node_pair.target,
                )
                self.observer.on_iteration_start(mode, i)
                stop = threading.Event()
                sink: queue.Queue[dict[str, int]] = queue.Queue()
                sampler = threading.Thread(
                    target=self._sample_resources_loop,
                    args=(stop, sink),
                    name=f"mem-sampler-{mode}-{i}",
                    daemon=True,
                )
                sampler.start()
                bench: list[BenchmarkResult] = []
                sat: list[LatencyResult] = []
                fixed: list[LatencyResult] = []
                try:
                    self.observer.on_stage_start("bw", mode, i)
                    try:
                        bench = self._run_bandwidth_stage(mode, i)
                    finally:
                        self.observer.on_stage_end("bw", mode, i)
                    self.observer.on_stage_start("fortio-sat", mode, i)
                    try:
                        sat = self._run_latency_stage(i, workload="saturation")
                    finally:
                        self.observer.on_stage_end("fortio-sat", mode, i)
                    self.observer.on_stage_start("fortio-fixed", mode, i)
                    try:
                        fixed = self._run_latency_stage(i, workload="fixed_qps")
                    finally:
                        self.observer.on_stage_end("fortio-fixed", mode, i)
                finally:
                    stop.set()
                    sampler.join(timeout=SAMPLE_INTERVAL_S * 2)
                    self.observer.on_iteration_end(mode, i)
                samples: list[dict[str, int]] = []
                while True:
                    try:
                        samples.append(sink.get_nowait())
                    except queue.Empty:
                        break
                peaks: dict[str, int] = {}
                for sample in samples:
                    for k, v in sample.items():
                        cur = peaks.get(k)
                        peaks[k] = v if cur is None else max(cur, v)
                for r in bench:
                    if "node" in peaks:
                        r.node_memory_used_bytes = peaks["node"]
                    if "cni" in peaks:
                        r.cni_memory_used_bytes = peaks["cni"]
                latency_records = [*sat, *fixed]
                for lr in latency_records:
                    if "node" in peaks:
                        lr.node_memory_used_bytes = peaks["node"]
                    if "cni" in peaks:
                        lr.cni_memory_used_bytes = peaks["cni"]
                all_bench.extend(bench)
                all_latency.extend(latency_records)
        return IterationResults(bench=all_bench, latency=all_latency)

    def _run_bandwidth_stage(
        self,
        mode: Literal["tcp", "udp"],
        iteration: int,
    ) -> list[BenchmarkResult]:
        """Launch one iperf3 client Job per client concurrently.

        Returns:
            The list of :class:`BenchmarkResult`, one per client, for
            this iteration. On any client failure the first failure is
            re-raised and no partial results are returned.
        """
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
                self._cleanup_after_failure(
                    cast(
                        "set[concurrent.futures.Future[BenchmarkResult | LatencyResult]]",  # noqa: E501
                        not_done,
                    ),
                    first_exc,
                    CLIENT_LABEL,
                )
                raise first_exc

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
            this iteration and sub-stage. On any client failure the
            first failure is re-raised and no partial results are
            returned.
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self._clients),
        ) as executor:
            future_to_client = {
                executor.submit(
                    self._run_one_fortio_client,
                    client,
                    iteration,
                    workload,
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
                self._cleanup_after_failure(
                    cast(
                        "set[concurrent.futures.Future[BenchmarkResult | LatencyResult]]",  # noqa: E501
                        not_done,
                    ),
                    first_exc,
                    FORTIO_CLIENT_LABEL,
                )
                raise first_exc

            return [f.result() for f in done]

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
