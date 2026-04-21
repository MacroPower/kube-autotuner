"""Ax Bayesian optimization loop for sysctl tuning.

This module wraps `ax-platform`'s multi-objective Bayesian optimizer
around the :class:`~kube_autotuner.benchmark.runner.BenchmarkRunner`.
Each trial proposes a sysctl parameterization, applies it to the target
(and optionally every source) node, runs iperf3, and reports the
resulting throughput / CPU / retransmit rate / node memory / CNI
memory back to Ax.

``ax-platform`` is a heavyweight optional dependency that lives in the
``optimize`` dependency group, not in ``dev``. Every Ax symbol is
imported lazily so ``uv sync`` (base ``dev`` only) keeps the package
importable and ``task completions`` stays cheap. Three rules enforce
this:

* ``from __future__ import annotations`` at the top keeps every
  signature a string, so ``-> Client`` and ``list[ChoiceParameterConfig]``
  do not trigger an eager import.
* Annotation-only symbols live under ``if TYPE_CHECKING:``.
* Every runtime use of Ax imports inside the function body, wrapped
  in a ``try/except ImportError`` that raises a clear hint pointing
  at ``uv sync --group optimize``.
"""

from __future__ import annotations

from collections import defaultdict
import contextlib
import logging
import math
import statistics
from typing import TYPE_CHECKING
import warnings

from kube_autotuner.benchmark.runner import BenchmarkRunner
from kube_autotuner.k8s.client import K8sClient
from kube_autotuner.k8s.lease import NodeLease
from kube_autotuner.models import TrialLog, TrialResult, retransmit_rate_by_iteration
from kube_autotuner.progress import NullObserver
from kube_autotuner.sysctl.setter import make_sysctl_setter_from_env

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path

    from ax.api.client import Client
    from ax.api.configs import ChoiceParameterConfig

    from kube_autotuner.experiment import (
        CniSection,
        FortioSection,
        IperfSection,
        ObjectivesSection,
        Patch,
    )
    from kube_autotuner.models import (
        BenchmarkConfig,
        BenchmarkResult,
        LatencyResult,
        NodePair,
        ParamSpace,
    )
    from kube_autotuner.progress import ProgressObserver
    from kube_autotuner.sysctl.backend import SysctlBackend

logger = logging.getLogger(__name__)

_OPTIMIZE_HINT = "install optimize group: uv sync --group optimize"

# pyro.ops.stats has an unescaped \g in a docstring; silence on import.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"pyro\..*")


def _require_ax_client() -> type[Client]:
    """Return the Ax ``Client`` class, raising a hint when Ax is missing.

    Returns:
        The :class:`ax.api.client.Client` type.

    Raises:
        RuntimeError: If ``ax-platform`` is not installed.
    """
    try:
        from ax.api.client import Client  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(_OPTIMIZE_HINT) from e
    return Client


def _require_ax_choice_param() -> type[ChoiceParameterConfig]:
    """Return the Ax ``ChoiceParameterConfig`` class.

    Returns:
        The :class:`ax.api.configs.ChoiceParameterConfig` type.

    Raises:
        RuntimeError: If ``ax-platform`` is not installed.
    """
    try:
        from ax.api.configs import ChoiceParameterConfig  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(_OPTIMIZE_HINT) from e
    return ChoiceParameterConfig


def _encode_param_name(name: str) -> str:
    """Encode a dotted sysctl name into an Ax-safe parameter name.

    Ax parameter names cannot contain ``"."``. The replacement is
    reversible via :func:`_decode_param_name`.

    Args:
        name: Dotted sysctl key (e.g. ``"net.core.rmem_max"``).

    Returns:
        The name with every ``"."`` replaced by ``"__"``.
    """
    return name.replace(".", "__")


def _decode_param_name(name: str) -> str:
    """Decode an Ax parameter name back into its dotted sysctl form.

    Args:
        name: Name previously produced by :func:`_encode_param_name`.

    Returns:
        The original dotted sysctl key.
    """
    return name.replace("__", ".")


def build_ax_params(param_space: ParamSpace) -> list[ChoiceParameterConfig]:
    """Convert a :class:`ParamSpace` into Ax ``ChoiceParameterConfig`` objects.

    Each :class:`~kube_autotuner.models.SysctlParam` becomes a
    ``ChoiceParameterConfig`` with string-coerced values. Integer
    parameters are marked ``is_ordered=True`` so Ax can exploit the
    natural ordering; ``choice`` parameters are unordered. This helper
    lazy-imports Ax and raises :exc:`RuntimeError` when the ``optimize``
    group is missing.

    Args:
        param_space: Search space to translate.

    Returns:
        One :class:`ax.api.configs.ChoiceParameterConfig` per entry in
        ``param_space.params``, in the same order.
    """
    choice_cls = _require_ax_choice_param()
    return [
        choice_cls(
            name=_encode_param_name(p.name),
            values=[str(v) for v in p.values],
            parameter_type="str",
            is_ordered=p.param_type == "int",
        )
        for p in param_space.params
    ]


def _aggregate_by_iteration(
    results: list[BenchmarkResult],
    value_fn: Callable[[BenchmarkResult], float | None],
    reducer: Callable[[list[float]], float],
) -> list[float]:
    """Reduce per-record values into one value per iteration.

    The input is the raw list of :class:`BenchmarkResult` records
    produced by a *single* trial (potentially multiple clients across
    multiple iterations). Records are grouped by their
    :attr:`~BenchmarkResult.iteration` index, ``value_fn`` extracts a
    numeric value from each record (returning ``None`` to drop the
    record), and ``reducer`` collapses each iteration's values into one
    float. Iterations with no surviving values are dropped entirely.

    The output is therefore one float *per iteration* — not one value
    across all iterations, and not one value per record.

    Args:
        results: Raw benchmark records for a single trial.
        value_fn: Extractor returning the scalar to aggregate, or
            ``None`` to skip the record.
        reducer: Aggregator applied to each iteration's surviving
            values (e.g. :func:`sum` or :func:`statistics.mean`).

    Returns:
        One reduced float per iteration that had at least one
        non-``None`` value.
    """
    grouped: dict[int, list[float]] = defaultdict(list)
    for r in results:
        v = value_fn(r)
        if v is not None:
            grouped[r.iteration].append(v)
    return [reducer(vals) for vals in grouped.values() if vals]


def _mean_sem(vals: list[float]) -> tuple[float, float]:
    """Return ``(mean, standard error of the mean)`` for ``vals``.

    Args:
        vals: Numeric samples. Empty input returns ``(0.0, 0.0)``; a
            single-element input returns ``(mean, 0.0)``.

    Returns:
        The arithmetic mean and the standard error of the mean.
    """
    if not vals:
        return 0.0, 0.0
    mean = statistics.mean(vals)
    sem = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return mean, sem


def _cpu_value(r: BenchmarkResult) -> float | None:
    """Return the preferred CPU sample for ``r``.

    Prefers the server-side CPU (shared across clients) and falls back
    to the per-client host CPU.

    Args:
        r: Benchmark record.

    Returns:
        The CPU percentage, or ``None`` when neither field is set.
    """
    if r.cpu_server_percent is not None:
        return r.cpu_server_percent
    return r.cpu_utilization_percent


def _memory_mean_sem(
    results: list[BenchmarkResult],
    field: Callable[[BenchmarkResult], int | None],
) -> tuple[float, float]:
    """Return ``(mean, SEM)`` for a per-record memory field across iterations."""
    vals = _aggregate_by_iteration(
        results,
        lambda r: float(v) if (v := field(r)) else None,
        statistics.mean,
    )
    return _mean_sem(vals)


def _aggregate_latency_by_iteration(
    results: list[LatencyResult],
    value_fn: Callable[[LatencyResult], float | None],
    reducer: Callable[[list[float]], float],
) -> list[float]:
    """Reduce per-record fortio values into one value per iteration.

    Mirrors :func:`_aggregate_by_iteration` for
    :class:`~kube_autotuner.models.LatencyResult`: records are grouped
    by their ``iteration`` index, ``value_fn`` extracts a scalar or
    ``None``, and ``reducer`` collapses each iteration's values.

    Args:
        results: Fortio records for a single trial.
        value_fn: Extractor returning the scalar to aggregate, or
            ``None`` to skip the record.
        reducer: Aggregator applied to each iteration's surviving
            values (e.g. :func:`sum` or :func:`statistics.mean`).

    Returns:
        One reduced float per iteration that had at least one
        non-``None`` value.
    """
    grouped: dict[int, list[float]] = defaultdict(list)
    for r in results:
        v = value_fn(r)
        if v is not None:
            grouped[r.iteration].append(v)
    return [reducer(vals) for vals in grouped.values() if vals]


def _compute_metrics(  # noqa: PLR0914
    trial: TrialResult,
) -> dict[str, tuple[float, float]]:
    """Collapse raw trial results into per-metric ``(mean, SEM)`` pairs.

    Aggregation proceeds in two stages: :func:`_aggregate_by_iteration`
    (or its latency counterpart) folds multi-client samples into one
    value per iteration, then :func:`_mean_sem` computes the mean and
    SEM across iterations. The ``iterations=1`` multi-client corner
    case collapses to a single iteration with zero SEM; the fallback
    replaces it with a per-client SEM (best-effort — the samples share
    server state and are correlated).

    ``retransmit_rate`` is the per-iteration ratio
    ``sum(retransmits) / sum(bytes_sent)`` averaged across iterations.
    When no iteration produced both a ``retransmits`` reading and a
    non-zero ``bytes_sent`` total (UDP-only trials, failed TCP runs),
    the mean is ``NaN`` and callers are expected to drop the key
    before handing results to Ax.

    ``rps`` is sourced from the fortio saturation sub-stage only (it
    is meaningless under fixed-QPS load). The latency percentiles are
    sourced from the fortio fixed-QPS sub-stage only (they are
    meaningless under saturation). When the corresponding sub-stage
    produced no records, the mean is ``NaN`` and callers drop the
    key before handing results to Ax — exactly as they do for
    ``retransmit_rate``.

    Args:
        trial: The trial whose raw records are to be summarised.

    Returns:
        A dict keyed by ``"throughput"`` / ``"cpu"`` /
        ``"retransmit_rate"`` / ``"node_memory"`` / ``"cni_memory"`` /
        ``"rps"`` / ``"latency_p50"`` / ``"latency_p90"`` /
        ``"latency_p99"`` with ``(mean, SEM)`` values ready for
        :meth:`ax.api.client.Client.complete_trial`.
    """
    results = trial.results
    throughput_vals = _aggregate_by_iteration(
        results,
        lambda r: r.bits_per_second,
        sum,
    )
    cpu_vals = _aggregate_by_iteration(results, _cpu_value, statistics.mean)
    rate_vals = retransmit_rate_by_iteration(results)

    throughput_mean, throughput_sem = _mean_sem(throughput_vals)
    cpu_mean, cpu_sem = _mean_sem(cpu_vals)
    if rate_vals:
        rate_mean, rate_sem = _mean_sem(rate_vals)
    else:
        rate_mean, rate_sem = float("nan"), 0.0

    if len(throughput_vals) == 1 and len(results) > 1:
        raw = [r.bits_per_second for r in results]
        throughput_sem = statistics.stdev(raw) / math.sqrt(len(raw))
        logger.info(
            "iterations=1 with multiple clients: using per-client SEM "
            "approximation (samples are correlated via shared server)",
        )

    saturation = [r for r in trial.latency_results if r.workload == "saturation"]
    fixed_qps = [r for r in trial.latency_results if r.workload == "fixed_qps"]

    rps_vals = _aggregate_latency_by_iteration(
        saturation,
        lambda r: r.rps,
        sum,
    )
    if rps_vals:
        rps_mean, rps_sem = _mean_sem(rps_vals)
    else:
        rps_mean, rps_sem = float("nan"), 0.0

    def _pct_mean_sem(
        field: Callable[[LatencyResult], float | None],
    ) -> tuple[float, float]:
        vals = _aggregate_latency_by_iteration(fixed_qps, field, statistics.mean)
        if vals:
            return _mean_sem(vals)
        return float("nan"), 0.0

    return {
        "throughput": (throughput_mean, throughput_sem),
        "cpu": (cpu_mean, cpu_sem),
        "retransmit_rate": (rate_mean, rate_sem),
        "node_memory": _memory_mean_sem(results, lambda r: r.node_memory_used_bytes),
        "cni_memory": _memory_mean_sem(results, lambda r: r.cni_memory_used_bytes),
        "rps": (rps_mean, rps_sem),
        "latency_p50": _pct_mean_sem(lambda r: r.latency_p50_ms),
        "latency_p90": _pct_mean_sem(lambda r: r.latency_p90_ms),
        "latency_p99": _pct_mean_sem(lambda r: r.latency_p99_ms),
    }


def build_ax_objective(
    section: ObjectivesSection,
) -> tuple[str, list[str]]:
    """Render an :class:`ObjectivesSection` into Ax's configure args.

    Ax's ``configure_optimization`` accepts the multi-objective string
    form ``"metric, -metric, ..."`` where a leading minus flips a
    maximize direction to minimize. This helper emits that string plus
    the section's outcome constraints verbatim.

    Args:
        section: Validated :class:`ObjectivesSection`.

    Returns:
        A ``(objective_string, outcome_constraints)`` pair ready to
        pass to :meth:`ax.api.client.Client.configure_optimization`.
    """
    terms: list[str] = []
    for obj in section.pareto:
        if obj.direction == "maximize":
            terms.append(obj.metric)
        else:
            terms.append(f"-{obj.metric}")
    return ", ".join(terms), list(section.constraints)


def _constraint_metric(constraint: str) -> str | None:
    """Return the metric name referenced by an outcome constraint, or None.

    Parses with :data:`kube_autotuner.experiment._CONSTRAINT_RE`. Non-matching
    strings return ``None`` rather than raising; :class:`ObjectivesSection`
    has already validated the list at construction time.
    """
    from kube_autotuner.experiment import _CONSTRAINT_RE  # noqa: PLC0415

    match = _CONSTRAINT_RE.match(constraint)
    return match.group("metric") if match else None


def filter_objectives_for_observability(
    section: ObjectivesSection,
    unobservable: set[str],
) -> ObjectivesSection:
    """Drop ``unobservable`` metrics from every part of ``section``.

    Removes matching entries from ``pareto``,
    ``recommendation_weights``, and any outcome constraint that
    references one of the unobservable metrics. Used when the
    configured benchmark cannot produce a metric (e.g. UDP-only
    runs cannot observe ``retransmit_rate``). The fortio-sourced
    metrics (``rps``, ``latency_p50/p90/p99``) are always observable
    as long as the fortio sub-stages run, so they never feed this
    set; their NaN-on-empty handling lives in :func:`_compute_metrics`
    instead.

    Args:
        section: The objectives block loaded from user config.
        unobservable: Metric names that will never have observations.

    Returns:
        A new :class:`ObjectivesSection` with the unobservable
        metrics removed. The input is left unchanged.

    Raises:
        ValueError: Every Pareto objective referenced an unobservable
            metric, leaving no objective to optimize against.
    """
    if not unobservable:
        return section
    filtered_pareto = [obj for obj in section.pareto if obj.metric not in unobservable]
    if not filtered_pareto:
        msg = (
            f"every Pareto objective references an unobservable "
            f"metric {sorted(unobservable)}; add at least one "
            "observable objective to experiment.yaml"
        )
        raise ValueError(msg)
    filtered_weights = {
        k: v for k, v in section.recommendation_weights.items() if k not in unobservable
    }
    filtered_constraints = [
        c for c in section.constraints if _constraint_metric(c) not in unobservable
    ]
    return section.model_copy(
        update={
            "pareto": filtered_pareto,
            "recommendation_weights": filtered_weights,
            "constraints": filtered_constraints,
        },
    )


class OptimizationLoop:
    """Drive Ax trial proposals through benchmark execution.

    Each iteration asks Ax for a parameterization, applies it to the
    target node (and, when ``apply_source=True``, every source client),
    runs the benchmark, and reports the resulting metrics back. Failed
    trials are marked failed with Ax so the surrogate model skips them
    on future rounds.
    """

    def __init__(  # noqa: PLR0913
        self,
        node_pair: NodePair,
        config: BenchmarkConfig,
        param_space: ParamSpace,
        output: Path,
        n_trials: int = 50,
        n_sobol: int = 15,
        *,
        apply_source: bool = False,
        iperf_args: IperfSection | None = None,
        fortio_args: FortioSection | None = None,
        patches: list[Patch] | None = None,
        objectives: ObjectivesSection,
        cni: CniSection | None = None,
        observer: ProgressObserver | None = None,
    ) -> None:
        """Wire up the Ax client, benchmark runner, and sysctl setters.

        Args:
            node_pair: Source/target nodes for the benchmark.
            config: Benchmark session configuration.
            param_space: Sysctl search space to optimize over.
            output: JSONL path receiving one :class:`TrialResult` per
                completed trial.
            n_trials: Total number of trials to propose.
            n_sobol: Number of Sobol (quasi-random) initialization
                trials before Ax switches to its Bayesian surrogate.
            apply_source: If ``True``, apply the proposed sysctls to
                every source client in addition to the target.
            iperf_args: Optional extra iperf3 client/server flags
                threaded into :class:`BenchmarkRunner`.
            fortio_args: Optional fortio per-role ``extra_args`` plus
                ``fixed_qps`` / ``connections`` / ``duration`` threaded
                into :class:`BenchmarkRunner` for the latency
                sub-stages.
            patches: Optional kustomize patches applied to the
                rendered benchmark manifests.
            objectives: Pareto objectives and outcome constraints
                handed to Ax via :func:`build_ax_objective`.
            cni: Selector for CNI pods tracked by the benchmark
                runner's resource sampler on the target node.
            observer: Optional progress callback. Defaults to
                :class:`~kube_autotuner.progress.NullObserver`. The
                same instance is forwarded to the internal
                :class:`~kube_autotuner.benchmark.runner.BenchmarkRunner`
                so iteration-level hooks share the live display.
        """
        self.node_pair: NodePair = node_pair
        self.config: BenchmarkConfig = config
        self.param_space: ParamSpace = param_space
        self.output: Path = output
        self.n_trials: int = n_trials
        self.n_sobol: int = n_sobol
        self.apply_source: bool = apply_source
        self.iperf_args: IperfSection | None = iperf_args
        self.fortio_args: FortioSection | None = fortio_args
        self.patches: list[Patch] | None = patches
        self.cni: CniSection | None = cni
        self.observer: ProgressObserver = observer or NullObserver()

        unobservable: set[str] = set()
        if "tcp" not in config.modes:
            unobservable.add("retransmit_rate")
        references_unobservable = unobservable and (
            any(obj.metric in unobservable for obj in objectives.pareto)
            or any(
                _constraint_metric(c) in unobservable for c in objectives.constraints
            )
        )
        if references_unobservable:
            logger.warning(
                "benchmark modes=%s cannot observe %s; stripping from "
                "objectives / constraints",
                list(config.modes),
                sorted(unobservable),
            )
        objectives = filter_objectives_for_observability(objectives, unobservable)
        self.objectives: ObjectivesSection = objectives
        self._ax_metric_names: set[str] = {obj.metric for obj in objectives.pareto} | {
            metric
            for metric in (_constraint_metric(c) for c in objectives.constraints)
            if metric is not None
        }

        client_cls = _require_ax_client()
        self.client: Client = client_cls()
        self.client.configure_experiment(
            name=f"sysctl-tune-{node_pair.hardware_class}",
            parameters=build_ax_params(param_space),
        )
        objective_str, outcome_constraints = build_ax_objective(objectives)
        self.client.configure_optimization(
            objective=objective_str,
            outcome_constraints=outcome_constraints,
        )
        self.client.configure_generation_strategy(
            initialization_budget=n_sobol,
        )

        self.k8s_client: K8sClient = K8sClient()
        self.target_setter: SysctlBackend = make_sysctl_setter_from_env(
            node=node_pair.target,
            namespace=node_pair.namespace,
            client=self.k8s_client,
        )
        self.client_setters: dict[str, SysctlBackend] = {}
        if apply_source:
            for client_node in node_pair.all_sources:
                self.client_setters[client_node] = make_sysctl_setter_from_env(
                    node=client_node,
                    namespace=node_pair.namespace,
                    client=self.k8s_client,
                )
        self.runner: BenchmarkRunner = BenchmarkRunner(
            node_pair,
            config,
            client=self.k8s_client,
            iperf_args=iperf_args,
            patches=patches,
            cni=cni,
            fortio_args=fortio_args,
            observer=self.observer,
        )
        self._completed: list[TrialResult] = []

    def _snapshot_params(self) -> list[str]:
        """Return the sysctl keys to snapshot before each trial."""
        return [*self.param_space.param_names(), "kernel.osrelease"]

    @contextlib.contextmanager
    def _node_leases(self) -> Iterator[None]:
        """Acquire leases on every node involved in the trial.

        Leases are acquired in sorted node-name order to produce a
        total ordering across concurrent optimizers and prevent
        deadlocks.

        Yields:
            ``None`` once every lease is held.
        """
        nodes = sorted({self.node_pair.target, *self.node_pair.all_sources})
        with contextlib.ExitStack() as stack:
            for node in nodes:
                stack.enter_context(
                    NodeLease(
                        node,
                        namespace=self.node_pair.namespace,
                        client=self.k8s_client,
                    ),
                )
            yield

    def _evaluate(
        self,
        parameterization: dict[str, str],
    ) -> dict[str, tuple[float, float]]:
        """Run one trial end-to-end: lock, snapshot, apply, benchmark, restore.

        Args:
            parameterization: Ax-encoded parameter dict (keys use ``__``
                separators).

        Returns:
            The per-metric ``(mean, SEM)`` table returned by
            :func:`_compute_metrics`.
        """
        sysctl_params: dict[str, str | int] = {
            _decode_param_name(k): v for k, v in parameterization.items()
        }

        with self._node_leases():
            snap_keys = self._snapshot_params()
            original_target = self.target_setter.snapshot(snap_keys)
            kernel_version = original_target.pop("kernel.osrelease", "")
            original_clients: dict[str, dict[str, str]] = {}
            for name, setter in self.client_setters.items():
                snap = setter.snapshot(snap_keys)
                snap.pop("kernel.osrelease", None)
                original_clients[name] = snap

            try:
                self.target_setter.apply(sysctl_params)
                for setter in self.client_setters.values():
                    setter.apply(sysctl_params)
                iteration_results = self.runner.run()
            finally:
                self.target_setter.restore(original_target)
                for name, setter in self.client_setters.items():
                    setter.restore(original_clients[name])

        trial_result = TrialResult(
            node_pair=self.node_pair,
            sysctl_values=sysctl_params,
            kernel_version=kernel_version,
            config=self.config,
            results=iteration_results.bench,
            latency_results=iteration_results.latency,
        )
        TrialLog.append(self.output, trial_result)
        self._completed.append(trial_result)

        return _compute_metrics(trial_result)

    def _should_stop(self, completed_iterations: int) -> bool:
        """Return ``True`` when the main loop should terminate.

        Args:
            completed_iterations: Number of trial attempts (successful
                plus failed) already performed.

        Returns:
            ``True`` once ``completed_iterations`` reaches
            :attr:`n_trials`.
        """
        return completed_iterations >= self.n_trials

    def _suggest(self, i: int) -> tuple[int, dict[str, str], str]:
        """Ask Ax for the next trial's parameterization.

        Args:
            i: Zero-based iteration index.

        Returns:
            A ``(trial_index, parameterization, phase)`` tuple where
            ``phase`` is ``"sobol"`` during the initialization budget
            and ``"bayesian"`` afterwards. Parameter values are coerced
            to ``str`` to match the declared ``parameter_type="str"``
            schema and the downstream sysctl writer's signature.
        """
        phase = "sobol" if i < self.n_sobol else "bayesian"
        trials = self.client.get_next_trials(max_trials=1)
        trial_index, parameterization = next(iter(trials.items()))
        return trial_index, {k: str(v) for k, v in parameterization.items()}, phase

    def _record(
        self,
        i: int,
        phase: str,
        trial_index: int,
        metrics: dict[str, tuple[float, float]],
    ) -> None:
        """Report a successful trial's metrics back to Ax and log a summary.

        The raw metrics dict is narrowed to the metrics the configured
        objectives actually reference, and any entry with a NaN mean is
        dropped -- Ax's ``complete_trial`` rejects NaN outcomes, but it
        treats a missing key as "not reported this trial" and the MOO
        surrogate handles the partial observation.

        Args:
            i: Zero-based iteration index.
            phase: ``"sobol"`` or ``"bayesian"``.
            trial_index: Ax trial index returned by :meth:`_suggest`.
            metrics: Per-metric ``(mean, SEM)`` mapping returned by
                :meth:`_evaluate`.
        """
        raw_data: dict[str, tuple[float, float]] = {}
        for name in self._ax_metric_names:
            pair = metrics.get(name)
            if pair is None:
                continue
            mean, _sem = pair
            if math.isnan(mean):
                logger.debug(
                    "trial %d: dropping %s from raw_data (mean is NaN)",
                    trial_index,
                    name,
                )
                continue
            raw_data[name] = pair
        self.client.complete_trial(trial_index=trial_index, raw_data=raw_data)

        tp = metrics["throughput"][0]
        cpu = metrics["cpu"][0]
        rate = metrics["retransmit_rate"][0]
        rate_str = "NaN" if math.isnan(rate) else f"{rate * 1e6:.2f}"
        rps = metrics.get("rps", (float("nan"), 0.0))[0]
        rps_str = "NaN" if math.isnan(rps) else f"{rps:.1f}"
        p99 = metrics.get("latency_p99", (float("nan"), 0.0))[0]
        p99_str = "NaN" if math.isnan(p99) else f"{p99:.1f}"
        logger.info(
            "Trial %d/%d [%s] throughput=%.1f Mbps cpu=%.1f%% "
            "retransmit_rate=%s retx/MB rps=%s p99=%s ms",
            i + 1,
            self.n_trials,
            phase,
            tp / 1e6,
            cpu,
            rate_str,
            rps_str,
            p99_str,
        )
        self.observer.on_trial_complete(i, phase, metrics)

    def run(self) -> list[TrialResult]:
        """Execute the full optimization loop.

        Returns:
            The completed :class:`TrialResult` records, in execution
            order. Failed trials are omitted; they are marked failed
            with the Ax client instead.

        Raises:
            KeyboardInterrupt: Propagated from a trial if the user
                interrupts mid-benchmark; the loop still runs
                :meth:`cleanup` through the ``finally`` block before
                re-raising.
        """
        self.runner.setup_server()
        try:
            i = 0
            while not self._should_stop(i):
                trial_index, parameterization, phase = self._suggest(i)
                self.observer.on_trial_start(
                    i,
                    self.n_trials,
                    phase,
                    parameterization,
                )
                try:
                    metrics = self._evaluate(parameterization)
                    self._record(i, phase, trial_index, metrics)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.warning(
                        "Trial %d/%d failed, continuing",
                        i + 1,
                        self.n_trials,
                        exc_info=True,
                    )
                    self.client.mark_trial_failed(trial_index=trial_index)
                    self.observer.on_trial_failed(i, e)
                i += 1
        except KeyboardInterrupt:
            logger.info("Interrupted after %d trials", len(self._completed))
        finally:
            self.cleanup()
        return self._completed

    def pareto_front(
        self,
    ) -> list[
        tuple[
            Mapping[str, int | float | str],
            Mapping[str, int | float | tuple[int | float, int | float]],
            int,
            str,
        ]
    ]:
        """Return the Pareto-optimal parameters and their predicted metrics.

        Returns:
            The Ax-computed Pareto frontier, as returned by
            :meth:`ax.api.client.Client.get_pareto_frontier`. The value
            types match Ax's (mixed numeric/string parameters, mixed
            scalar/``(mean, SEM)`` metrics).
        """
        return self.client.get_pareto_frontier()

    def cleanup(self) -> None:
        """Remove benchmark server resources created by :meth:`run`."""
        self.runner.cleanup()
