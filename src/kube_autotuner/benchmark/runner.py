"""Concurrent iperf3 benchmark runner.

:class:`BenchmarkRunner` launches one iperf3 server Deployment on the
target node and one iperf3 client Job per source node. Client jobs run
concurrently through :class:`concurrent.futures.ThreadPoolExecutor`, and
the wait loop uses
:data:`concurrent.futures.FIRST_EXCEPTION` on purpose:

* On a clean run, ``wait(... FIRST_EXCEPTION)`` returns only once every
  future has either completed or raised, so it collapses to the usual
  "wait for all" semantics.
* When any client raises, ``FIRST_EXCEPTION`` returns early with that
  future in ``done`` and the still-running ones in ``not_done``. We then
  cancel the pending futures, fire a label-based cleanup against the
  namespace (belt-and-braces -- each ``_run_one_client`` also cleans up
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
from typing import TYPE_CHECKING, Literal

from kube_autotuner.benchmark.client_spec import build_client_yaml
from kube_autotuner.benchmark.parser import parse_iperf_json, parse_k8s_memory
from kube_autotuner.benchmark.patch import apply_patches
from kube_autotuner.benchmark.server_spec import build_server_yaml
from kube_autotuner.experiment import IperfSection
from kube_autotuner.k8s.client import K8sApiError, K8sClient

if TYPE_CHECKING:
    from kube_autotuner.experiment import Patch
    from kube_autotuner.models import BenchmarkConfig, BenchmarkResult, NodePair

logger = logging.getLogger(__name__)

CLIENT_LABEL = "app.kubernetes.io/name=iperf3-client"
_IPERF_BASE_PORT = 5201
_CLIENT_WAIT_TIMEOUT_SECONDS = 180
SAMPLE_INTERVAL_S = 5.0


class BenchmarkRunner:
    """Orchestrates iperf3 server/client lifecycle via the Kubernetes API."""

    def __init__(
        self,
        node_pair: NodePair,
        config: BenchmarkConfig,
        client: K8sClient | None = None,
        iperf_args: IperfSection | None = None,
        patches: list[Patch] | None = None,
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
        """
        self.node_pair = node_pair
        self.config = config
        self.client = client or K8sClient()
        self.iperf_args = iperf_args or IperfSection()
        self.patches = patches or []
        self._server_name = f"iperf3-server-{node_pair.target}"
        self._server_label = (
            f"app.kubernetes.io/instance=iperf3-server-{node_pair.target}"
        )
        self._clients = list(node_pair.all_sources)
        self._ports = {c: _IPERF_BASE_PORT + i for i, c in enumerate(self._clients)}

    def setup_server(self) -> None:
        """Deploy the iperf3 server on the target node and await rollout."""
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

    def _sample_server_memory_loop(
        self,
        stop: threading.Event,
        sink: queue.Queue[int],
    ) -> None:
        """Poll metrics-server for server-pod memory until ``stop`` is set.

        Re-resolves the server pod name on every tick so a pod restart
        or late readiness does not strand the sampler on a dead name.
        Empty rows (no scrape yet) and empty pod names are silently
        skipped; poisoning the peak with ``0`` would be wrong. The
        loop body catches :class:`Exception` broadly and logs at
        ``warning``: this is a daemon thread, and an uncaught exception
        would terminate it silently and leave the operator with no
        sampling and no signal. Parsing errors from
        :func:`parse_k8s_memory` (``ValueError``) or malformed row
        dicts (``KeyError``) are therefore explicitly tolerated.

        Args:
            stop: Event signalling shutdown. The loop blocks on
                ``stop.wait(SAMPLE_INTERVAL_S)`` so shutdown is prompt.
            sink: Thread-safe queue receiving one per-pod byte total
                per successful poll.
        """
        ns = self.node_pair.namespace
        while not stop.is_set():
            try:
                name = self.client.get_pod_name(self._server_label, ns)
                if name:
                    rows = self.client.top_pod_containers(name, ns)
                    if rows:
                        total = sum(
                            parse_k8s_memory(r["memory"])
                            for r in rows
                            if r.get("memory")
                        )
                        sink.put(total)
            except Exception:
                logger.warning(
                    "Memory sampler poll failed; continuing",
                    exc_info=True,
                )
            stop.wait(SAMPLE_INTERVAL_S)

    def run(self) -> list[BenchmarkResult]:
        """Run all configured benchmark iterations and modes.

        Returns:
            Every :class:`BenchmarkResult` recorded across iterations
            and modes, tagged with the peak server memory observed
            during each iteration when sampling produced any data.
        """
        results: list[BenchmarkResult] = []
        for mode in self.config.modes:
            for i in range(self.config.iterations):
                logger.info(
                    "Running %s iteration %d/%d: %s -> %s",
                    mode,
                    i + 1,
                    self.config.iterations,
                    self._clients,
                    self.node_pair.target,
                )
                stop = threading.Event()
                sink: queue.Queue[int] = queue.Queue()
                sampler = threading.Thread(
                    target=self._sample_server_memory_loop,
                    args=(stop, sink),
                    name=f"mem-sampler-{mode}-{i}",
                    daemon=True,
                )
                sampler.start()
                try:
                    iter_results = self._run_iteration(mode, i)
                finally:
                    stop.set()
                    sampler.join(timeout=SAMPLE_INTERVAL_S * 2)
                samples: list[int] = []
                while True:
                    try:
                        samples.append(sink.get_nowait())
                    except queue.Empty:
                        break
                if samples:
                    peak = max(samples)
                    for r in iter_results:
                        r.memory_used_bytes = peak
                results.extend(iter_results)
        return results

    def _run_iteration(
        self,
        mode: Literal["tcp", "udp"],
        iteration: int,
    ) -> list[BenchmarkResult]:
        """Launch one client Job per client concurrently; clean up on failure.

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
                self._cleanup_after_failure(not_done, first_exc)
                raise first_exc

            return [f.result() for f in done]

    def _cleanup_after_failure(
        self,
        not_done: set[concurrent.futures.Future[BenchmarkResult]],
        first_exc: BaseException,
    ) -> None:
        """Cancel pending client Jobs, sweep by label, drain remaining futures.

        When the label sweep itself fails, the cleanup error is raised
        with ``first_exc`` as its explicit cause so the operator sees
        both the original client failure and the cleanup failure in one
        chained traceback.

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
                CLIENT_LABEL,
                self.node_pair.namespace,
            )
        except K8sApiError as cleanup_err:
            logger.warning(
                "Failed label-based cleanup of client jobs",
                exc_info=True,
            )
            msg = f"label-based cleanup failed after primary failure: {cleanup_err}"
            raise RuntimeError(msg) from first_exc
        # Drain remaining futures so their own finally blocks complete.
        # The primary failure is already captured in ``first_exc``; any
        # per-client cleanup error was handled inside
        # ``_run_one_client``. Log at debug and move on.
        for f in not_done:
            try:
                f.result()
            except Exception:
                logger.debug("drained client future after failure", exc_info=True)

    def cleanup(self) -> None:
        """Remove iperf3 server/client resources by label."""
        ns = self.node_pair.namespace
        self.client.delete_by_label(
            "deployment", "app.kubernetes.io/name=iperf3-server", ns
        )
        self.client.delete_by_label(
            "service", "app.kubernetes.io/name=iperf3-server", ns
        )
        self.client.delete_by_label("job", CLIENT_LABEL, ns)
        logger.info("iperf3 resources cleaned up")
