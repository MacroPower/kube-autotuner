"""Baseline / trial / optimize orchestration entry points.

The three functions in this module -- :func:`run_baseline`,
:func:`run_trial`, and :func:`run_optimize` -- glue together the data
models, the kubectl/lease primitives, the sysctl backends, the
benchmark runner,
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

import contextlib
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, cast

from kube_autotuner.benchmark.runner import BenchmarkRunner
from kube_autotuner.k8s.lease import NodeLease
from kube_autotuner.models import TrialLog, TrialResult
from kube_autotuner.sysctl.params import PARAM_SPACE

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from kube_autotuner.experiment import ExperimentConfig
    from kube_autotuner.k8s.client import Kubectl
    from kube_autotuner.models import NodePair
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
        kubectl: Injected :class:`Kubectl` client shared with the
            lease, benchmark, and (indirectly) backend layers.
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
    """

    exp: ExperimentConfig
    kubectl: Kubectl
    backend: SysctlBackend
    output: Path


def _resolve_zones(node_pair: NodePair, kubectl: Kubectl) -> NodePair:
    """Populate zone fields on ``node_pair`` via ``kubectl`` node labels.

    Args:
        node_pair: Pair to enrich.
        kubectl: Client used to query
            ``topology.kubernetes.io/zone`` on each node.

    Returns:
        A copy of ``node_pair`` with ``source_zone``, ``target_zone``,
        and ``extra_source_zones`` filled in. Unresolvable nodes map
        to ``""`` rather than aborting the run -- topology metadata is
        informational.
    """

    def _zone(node: str) -> str:
        try:
            return kubectl.get_node_zone(node)
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
    kubectl: Kubectl,
) -> Iterator[None]:
    """Acquire a :class:`NodeLease` on every node in sorted order.

    Sorted entry order produces a total ordering across concurrent
    callers so two runs contending on the same node set cannot
    deadlock.

    Args:
        nodes: Node names to lock. Duplicates collapse.
        namespace: Namespace hosting the ``coordination.k8s.io/v1``
            Lease resources.
        kubectl: Client forwarded to :class:`NodeLease`.

    Yields:
        ``None`` once every lease is held. Leases release in reverse
        order on exit, via :class:`contextlib.ExitStack`.
    """
    with contextlib.ExitStack() as stack:
        for node in sorted(set(nodes)):
            stack.enter_context(
                NodeLease(node, namespace=namespace, kubectl=kubectl),
            )
        yield


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
    node_pair = _resolve_zones(exp.to_node_pair(), ctx.kubectl)
    config = exp.benchmark
    all_params = [*PARAM_SPACE.param_names(), "kernel.osrelease"]

    lease_nodes = [node_pair.target, *node_pair.all_sources]
    with _acquire_all_leases(lease_nodes, node_pair.namespace, ctx.kubectl):
        snapshot = ctx.backend.snapshot(all_params)
        kernel_version = snapshot.pop("kernel.osrelease", "")

        runner = BenchmarkRunner(
            node_pair,
            config,
            kubectl=ctx.kubectl,
            iperf_args=exp.iperf,
            patches=exp.patches,
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
        results=results,
    )
    TrialLog.append(ctx.output, trial)
    logger.info("Wrote %d results to %s", len(results), ctx.output)
    logger.info("Mean throughput: %.1f Mbps", trial.mean_throughput() / 1e6)
    logger.info("Mean CPU: %.1f%%", trial.mean_cpu())


def run_trial(ctx: RunContext) -> None:
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

    params: dict[str, str] = {k: str(v) for k, v in exp.trial.sysctls.items()}
    node_pair = _resolve_zones(exp.to_node_pair(), ctx.kubectl)
    config = exp.benchmark

    lease_nodes = [node_pair.target, *node_pair.all_sources]
    with _acquire_all_leases(lease_nodes, node_pair.namespace, ctx.kubectl):
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
            kubectl=ctx.kubectl,
            iperf_args=exp.iperf,
            patches=exp.patches,
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
        results=results,
    )
    TrialLog.append(ctx.output, trial_result)
    logger.info("Wrote %d results to %s", len(results), ctx.output)
    logger.info(
        "Mean throughput: %.1f Mbps",
        trial_result.mean_throughput() / 1e6,
    )
    logger.info("Mean CPU: %.1f%%", trial_result.mean_cpu())


def run_optimize(ctx: RunContext) -> None:
    """Drive the Ax Bayesian optimization loop.

    Per-trial lease acquisition lives inside
    :class:`~kube_autotuner.optimizer.OptimizationLoop` so Ctrl-C
    releases leases between trials; this function deliberately does
    *not* wrap the loop in an outer lease. ``ctx.backend`` is unused
    here -- the optimizer builds its own per-node setters (one per
    source node when ``apply_source=True``) because a single fan-out
    setter does not fit a single target-only backend.

    Args:
        ctx: Orchestration context. ``ctx.exp.optimize`` must be
            populated (guaranteed by
            :class:`~kube_autotuner.experiment.ExperimentConfig`'s
            mode validator).

    Raises:
        RuntimeError: ``ctx.exp.optimize`` is ``None``, or
            ``ax-platform`` is not installed (the ``optimize`` dep
            group provides it).
    """
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

    node_pair = _resolve_zones(exp.to_node_pair(), ctx.kubectl)
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
        patches=exp.patches,
    )
    trials = loop.run()
    logger.info("Completed %d trials. Results in %s", len(trials), ctx.output)
    if not trials:
        return

    try:
        pareto = loop.pareto_front()
    except Exception:
        logger.warning("Could not compute Pareto front", exc_info=True)
        return

    logger.info("=== Pareto-optimal configurations (%d) ===", len(pareto))
    for _params, metrics, trial_idx, _arm in pareto:
        tp = metrics.get("throughput", 0)
        cpu = metrics.get("cpu", 0)
        rt = metrics.get("retransmits", 0)
        tp_val = tp[0] if isinstance(tp, tuple) else tp
        cpu_val = cpu[0] if isinstance(cpu, tuple) else cpu
        rt_val = rt[0] if isinstance(rt, tuple) else rt
        logger.info(
            "  [%d] throughput=%.1f Mbps cpu=%.1f%% retransmits=%.0f",
            trial_idx,
            float(tp_val) / 1e6,
            float(cpu_val),
            float(rt_val),
        )
