"""Ax Bayesian optimization loop for sysctl tuning.

This module wraps `ax-platform`'s multi-objective Bayesian optimizer
around the :class:`~kube_autotuner.benchmark.runner.BenchmarkRunner`.
Each trial proposes a sysctl parameterization, applies it to the target
(and optionally every source) node, runs iperf3, and reports the
resulting throughput / retransmit rate / jitter / rps / latency
percentiles back to Ax.

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
import json
import logging
import math
import statistics
from typing import TYPE_CHECKING, Any
import warnings

from kube_autotuner.benchmark.errors import BenchmarkFailure
from kube_autotuner.benchmark.runner import BenchmarkRunner
from kube_autotuner.k8s.client import K8sClient
from kube_autotuner.k8s.lease import NodeLease
from kube_autotuner.models import (
    TrialResult,
    is_primary,
    tcp_retransmit_rate_by_iteration,
    udp_loss_rate_by_iteration,
)
from kube_autotuner.progress import NullObserver
from kube_autotuner.sysctl.params import RECOMMENDED_DEFAULTS
from kube_autotuner.sysctl.setter import make_sysctl_setter_from_env
from kube_autotuner.trial_log import TrialLog
from kube_autotuner.units import format_duration

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path

    from ax.api.client import Client
    from ax.api.configs import ChoiceParameterConfig

    from kube_autotuner.experiment import (
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


def _register_noise_filters() -> None:
    """Register ``warnings.filterwarnings`` entries for upstream noise.

    Called once at module import. Exposed as a standalone function so
    tests can re-register filters after pytest swaps
    :data:`warnings.filters` between tests.
    """
    # pyro.ops.stats has an unescaped ``\g`` in a docstring.
    warnings.filterwarnings(
        "ignore",
        category=SyntaxWarning,
        module=r"pyro\..*",
    )
    # torch's "To copy construct from a tensor..." UserWarning is
    # reissued on every Bayesian generate because
    # ``ax.generators.torch.botorch_moo_utils`` calls
    # ``torch.tensor(sourceTensor)``. The warning is emitted inside
    # torch internals, so a ``module=`` regex on ``ax\..*`` would not
    # match (the filter is compared against the frame that called
    # ``warnings.warn``, which is torch's). Match on message instead.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"To copy construct from a tensor.*",
    )
    # botorch ``InputDataWarning`` fires once per generate when any
    # outcome has near-zero std (typically because a Pareto metric has
    # collapsed; the ``_warn_on_collapsed_objectives`` helper surfaces
    # that as one kube-autotuner-owned warning instead).
    # ``BotorchWarning`` inherits from ``Warning`` directly (not
    # ``UserWarning``), so the category must be ``Warning``.
    warnings.filterwarnings(
        "ignore",
        category=Warning,
        module=r"botorch($|\.)",
    )
    # gpytorch ``NumericalWarning``: MLE noise parameter collapses to
    # ~0 during GP fit and is rounded up to 1e-6 for Cholesky
    # stability. Benign at our sample sizes. ``NumericalWarning``
    # subclasses ``RuntimeWarning``.
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module=r"gpytorch($|\.)",
    )


_register_noise_filters()


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


def _log_strandings(
    round_index: int,
    parent_ids: set[str],
    done: set[str],
    prev_round_topk: set[str],
) -> None:
    """Emit one INFO line per parent stranded by re-ranking.

    Round 1 surfaces parents that a prior session's resume completed
    but whose ids have since fallen out of the current top-K. Later
    rounds surface parents that drop out of top-K between rounds.

    Args:
        round_index: 1-indexed refinement round being entered.
        parent_ids: ``trial_id`` set picked for this round's top-K.
        done: ``trial_id`` set already sampled in this round (resume
            state).
        prev_round_topk: ``trial_id`` set picked in the previous round;
            empty on round 1.
    """
    if round_index == 1:
        for stranded in done - parent_ids:
            logger.info(
                "prior refinement of %s in round %d is stranded; "
                "current top-K has re-ranked",
                stranded,
                round_index,
            )
        return
    for stranded in prev_round_topk - parent_ids:
        logger.info("%s dropped out of top-K in round %d", stranded, round_index)


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

    The output is therefore one float *per iteration* -- not one value
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
    replaces it with a per-client SEM (best-effort -- the samples share
    server state and are correlated).

    ``tcp_throughput`` filters to ``mode == "tcp"`` records before
    aggregating, preserving the optimizer's historical TCP-only
    semantics. ``udp_throughput`` mirrors it but filters to
    ``mode == "udp"`` records.

    ``tcp_retransmit_rate`` is the per-iteration ratio
    ``sum(retransmits) * 1e9 / sum(bytes_sent)`` (retransmits per GB)
    averaged across iterations. When no iteration produced both a
    ``retransmits`` reading and a non-zero ``bytes_sent`` total (i.e.
    every ``bw-tcp`` stage failed), the mean is ``NaN`` and callers
    are expected to drop the key before handing results to Ax. ``udp_loss_rate`` is the
    UDP-side analog: per-iteration ``sum(lost_packets) / sum(packets)``
    averaged across iterations, with the same NaN-when-empty contract.

    ``udp_jitter`` is the mean of the per-iteration cross-client mean
    of ``jitter`` (seconds; only UDP records carry it). When every
    ``bw-udp``
    stage failed, the mean is ``NaN`` and callers drop the key
    before handing results to Ax.

    ``rps`` is sourced from the fortio saturation sub-stage only (it
    is meaningless under fixed-QPS load). The latency percentiles are
    sourced from the fortio fixed-QPS sub-stage only (they are
    meaningless under saturation). When the corresponding sub-stage
    produced no records, the mean is ``NaN`` and callers drop the
    key before handing results to Ax -- exactly as they do for
    ``tcp_retransmit_rate``.

    Args:
        trial: The trial whose raw records are to be summarised.

    Returns:
        A dict keyed by ``"tcp_throughput"`` / ``"udp_throughput"`` /
        ``"tcp_retransmit_rate"`` / ``"udp_loss_rate"`` /
        ``"udp_jitter"`` / ``"rps"`` / ``"latency_p50"`` /
        ``"latency_p90"`` / ``"latency_p99"`` with ``(mean, SEM)``
        values ready for
        :meth:`ax.api.client.Client.complete_trial`.
    """
    results = trial.results
    tcp_results = [r for r in results if r.mode == "tcp"]
    udp_results = [r for r in results if r.mode == "udp"]
    tcp_throughput_vals = _aggregate_by_iteration(
        tcp_results,
        lambda r: r.bits_per_second,
        sum,
    )
    udp_throughput_vals = _aggregate_by_iteration(
        udp_results,
        lambda r: r.bits_per_second,
        sum,
    )
    tcp_rate_vals = tcp_retransmit_rate_by_iteration(results)
    udp_loss_vals = udp_loss_rate_by_iteration(results)
    udp_jitter_vals = _aggregate_by_iteration(
        results,
        lambda r: r.jitter,
        statistics.mean,
    )

    tcp_throughput_mean, tcp_throughput_sem = _mean_sem(tcp_throughput_vals)
    udp_throughput_mean, udp_throughput_sem = _mean_sem(udp_throughput_vals)
    if tcp_rate_vals:
        tcp_rate_mean, tcp_rate_sem = _mean_sem(tcp_rate_vals)
    else:
        tcp_rate_mean, tcp_rate_sem = float("nan"), 0.0
    if udp_loss_vals:
        udp_loss_mean, udp_loss_sem = _mean_sem(udp_loss_vals)
    else:
        udp_loss_mean, udp_loss_sem = float("nan"), 0.0
    if udp_jitter_vals:
        udp_jitter_mean, udp_jitter_sem = _mean_sem(udp_jitter_vals)
    else:
        udp_jitter_mean, udp_jitter_sem = float("nan"), 0.0

    if len(tcp_throughput_vals) == 1 and len(tcp_results) > 1:
        raw = [r.bits_per_second for r in tcp_results]
        tcp_throughput_sem = statistics.stdev(raw) / math.sqrt(len(raw))
        logger.info(
            "iterations=1 with multiple clients: using per-client SEM "
            "approximation (samples are correlated via shared server)",
        )
    if len(udp_throughput_vals) == 1 and len(udp_results) > 1:
        raw_udp = [r.bits_per_second for r in udp_results]
        udp_throughput_sem = statistics.stdev(raw_udp) / math.sqrt(len(raw_udp))

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
        "tcp_throughput": (tcp_throughput_mean, tcp_throughput_sem),
        "udp_throughput": (udp_throughput_mean, udp_throughput_sem),
        "tcp_retransmit_rate": (tcp_rate_mean, tcp_rate_sem),
        "udp_loss_rate": (udp_loss_mean, udp_loss_sem),
        "udp_jitter": (udp_jitter_mean, udp_jitter_sem),
        "rps": (rps_mean, rps_sem),
        "latency_p50": _pct_mean_sem(lambda r: r.latency_p50),
        "latency_p90": _pct_mean_sem(lambda r: r.latency_p90),
        "latency_p99": _pct_mean_sem(lambda r: r.latency_p99),
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
        observer: ProgressObserver | None = None,
        prior_trials: list[TrialResult] | None = None,
        collect_host_state: bool = False,
    ) -> None:
        """Wire up the Ax client, benchmark runner, and sysctl setters.

        Args:
            node_pair: Source/target nodes for the benchmark.
            config: Benchmark session configuration.
            param_space: Sysctl search space to optimize over.
            output: Trial-log directory receiving one
                :class:`TrialResult` per completed trial (plus a
                ``failures/`` subdirectory for failed-trial dumps).
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
            observer: Optional progress callback. Defaults to
                :class:`~kube_autotuner.progress.NullObserver`. The
                same instance is forwarded to the internal
                :class:`~kube_autotuner.benchmark.runner.BenchmarkRunner`
                so iteration-level hooks share the live display.
            prior_trials: Prior successful :class:`TrialResult`
                records to seed Ax with before the live loop begins.
                Each is replayed through ``attach_trial`` +
                ``complete_trial`` so the surrogate sees the full
                history. ``n_trials`` remains the total budget across
                sessions; the live loop runs
                ``max(0, n_trials - len(prior_trials))`` more attempts.
            collect_host_state: Forwarded to
                :class:`~kube_autotuner.benchmark.runner.BenchmarkRunner`
                to opt into per-iteration host-state snapshots
                (target plus every client setter). Off by default.
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
        self.observer: ProgressObserver = observer or NullObserver()

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
        self.collect_host_state: bool = collect_host_state
        snapshot_backends: list[SysctlBackend] = [
            self.target_setter,
            *self.client_setters.values(),
        ]
        self.runner: BenchmarkRunner = BenchmarkRunner(
            node_pair,
            config,
            client=self.k8s_client,
            iperf_args=iperf_args,
            patches=patches,
            fortio_args=fortio_args,
            observer=self.observer,
            flush_backends=[self.target_setter, *self.client_setters.values()],
            snapshot_backends=snapshot_backends,
            collect_host_state=collect_host_state,
        )
        self._completed: list[TrialResult] = []
        self.prior_count: int = sum(1 for t in (prior_trials or []) if is_primary(t))
        self._first_bayesian_generate: bool = True
        self._collapse_warned: set[str] = set()
        self._seed_prior_trials(prior_trials or [])
        # Resumes skip the seed: the prior run already had its shot at
        # anchoring the surrogate, and re-attaching RECOMMENDED_DEFAULTS
        # here would consume budget a resuming user did not ask for.
        self._seed_attempts_remaining: int = 2 if self.prior_count == 0 else 0
        self._seed_trial_index: int | None = None

    def _seed_prior_trials(self, prior: list[TrialResult]) -> None:
        """Replay ``prior`` into the Ax client and the completion list.

        Primary rows (sobol / bayesian) are replayed through
        Ax via ``attach_trial`` + ``complete_trial`` so the surrogate
        sees the full history, using the same metric aggregation the
        live loop would produce (:func:`_compute_metrics`) and the
        NaN filter from :meth:`_record` so ``complete_trial`` cannot
        abort mid-seeding.

        Refinement rows (``phase == "refinement"``) are **not**
        attached: Ax cannot accept the same arm twice, and the
        surrogate has no use for a repeat observation of a parent it
        already saw. They are still appended to ``self._completed``
        so the aggregation and re-ranking paths in the live panel and
        the refinement summary see the full sample population.

        Parameter names are encoded with :func:`_encode_param_name` and
        values are string-coerced to match the Ax schema built by
        :func:`build_ax_params`; otherwise Ax's categorical dictionary
        keys would not line up with the live suggestions.

        Args:
            prior: Prior trial records in file order.
        """
        space_names = set(self.param_space.param_names())
        for tr in prior:
            if not is_primary(tr):
                self._completed.append(tr)
                continue
            params = {
                _encode_param_name(k): str(v)
                for k, v in tr.sysctl_values.items()
                if k in space_names
            }
            trial_index = self.client.attach_trial(parameters=params)
            metrics = _compute_metrics(tr)
            raw_data: dict[str, tuple[float, float]] = {}
            for name in self._ax_metric_names:
                pair = metrics.get(name)
                if pair is None or math.isnan(pair[0]):
                    continue
                raw_data[name] = pair
            self.client.complete_trial(
                trial_index=trial_index,
                raw_data=raw_data,
            )
            self._completed.append(tr)

    def _attach_recommended_defaults(self) -> tuple[int, dict[str, str]] | None:
        """Attach ``RECOMMENDED_DEFAULTS`` as the next Sobol trial.

        The search space contains several benchmark-flat /
        production-live knobs (``tcp_mtu_probing``, ``tcp_ecn``,
        ``tcp_max_tw_buckets``, ``nf_conntrack_max``,
        ``nf_conntrack_tcp_timeout_established``,
        ``tcp_slow_start_after_idle``, ``tcp_autocorking``) whose
        response is ~flat under the default fortio shape. Without
        an anchor the Sobol phase picks random values on those axes and
        the GP regresses noise; attaching the production-reasonable
        point once gives the surrogate a known-good observation.

        Called lazily from :meth:`_suggest` so a failed seed trial can
        be re-attached on the next iteration with a fresh trial_index.
        Returns ``None`` when the configured :attr:`param_space` is an
        override that does not line up with
        :data:`RECOMMENDED_DEFAULTS`, or when Ax itself rejects the
        parameterization (e.g. the override changed a knob's rungs).
        Either failure mode is non-fatal: the run proceeds without the
        anchor.

        Returns:
            ``(trial_index, parameterization)`` to return from
            :meth:`_suggest`, or ``None`` to skip seeding.
        """
        space_names = set(self.param_space.param_names())
        seed: dict[str, str] = {
            _encode_param_name(k): str(v)
            for k, v in RECOMMENDED_DEFAULTS.items()
            if k in space_names
        }
        if len(seed) != len(space_names):
            logger.debug(
                "RECOMMENDED_DEFAULTS does not cover the configured "
                "search space; skipping seeded prior",
            )
            return None
        try:
            trial_index = self.client.attach_trial(parameters=seed)
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(
                "Ax rejected the RECOMMENDED_DEFAULTS seed "
                "(likely a param_space override with mismatched rungs); "
                "continuing without anchor: %s",
                e,
            )
            return None
        return trial_index, seed

    def _snapshot_params(self) -> list[str]:
        """Return the sysctl keys to snapshot before each trial.

        Includes ``net.ipv4.tcp_no_metrics_save`` even though the knob
        is not part of the search space, because the per-trial loop
        pins it to 1 for methodology (trial independence) and the
        snapshot/restore path is what returns the node to its
        pre-experiment state.
        """
        return [
            *self.param_space.param_names(),
            "net.ipv4.tcp_no_metrics_save",
            "kernel.osrelease",
        ]

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
        *,
        phase: str,
        parent_trial_id: str | None,
        refinement_round: int | None = None,
    ) -> tuple[TrialResult, dict[str, tuple[float, float]]]:
        """Run one trial end-to-end: lock, snapshot, apply, benchmark, restore.

        The returned :class:`TrialResult` carries ``phase``,
        ``parent_trial_id`` and ``refinement_round`` so the observer
        and the persisted row agree on which population the sample
        belongs to. Primary call sites pass the Ax phase label with
        ``parent_trial_id=None`` and ``refinement_round=None``; the
        refinement pass passes ``phase="refinement"``, the primary's
        ``trial_id``, and the 1-indexed round number.

        Args:
            parameterization: Ax-encoded parameter dict (keys use ``__``
                separators).
            phase: Phase label to stamp on the resulting
                :class:`TrialResult` (``"sobol"`` / ``"bayesian"`` /
                ``"refinement"``).
            parent_trial_id: For refinement runs, the ``trial_id``
                of the primary being refined; ``None`` otherwise.
            refinement_round: 1-indexed refinement round number when
                ``phase == "refinement"``; ``None`` for primary trials.

        Returns:
            A ``(trial_result, metrics)`` pair. ``metrics`` is the
            per-metric ``(mean, SEM)`` table from
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

            trial_sysctls: dict[str, str | int] = {
                **sysctl_params,
                "net.ipv4.tcp_no_metrics_save": 1,
            }
            try:
                # Apply first so ``tcp_no_metrics_save=1`` is in effect
                # before ``runner.run()`` performs its per-iteration
                # ``flush_network_state()``; otherwise the kernel could
                # cache fresh entries between the flush and the pin
                # landing.
                self.target_setter.apply(trial_sysctls)
                for setter in self.client_setters.values():
                    setter.apply(trial_sysctls)
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
            host_state_snapshots=iteration_results.host_state_snapshots,
            phase=phase,  # ty: ignore[invalid-argument-type]
            parent_trial_id=parent_trial_id,
            refinement_round=refinement_round,
        )
        TrialLog.append(self.output, trial_result)
        self._completed.append(trial_result)

        return trial_result, _compute_metrics(trial_result)

    def _dump_failure(
        self,
        *,
        trial_index: int,
        phase: str,
        parameterization: dict[str, str],
        exc: BaseException,
    ) -> None:
        """Persist a per-trial failure dump inside :attr:`output`.

        Best-effort: wraps the write in ``try/except Exception`` so a
        disk failure never masks the primary trial failure the dump is
        describing.

        ``self.output`` is the trial-log directory; the dump lands at
        ``<output>/failures/trial-<idx>.json`` so it travels with the
        rest of the dataset (including through ``--fresh`` archive
        renames).

        Args:
            trial_index: Zero-based trial index as logged by the
                observer (``prior_count + i``).
            phase: Ax phase label for the trial (``"sobol"`` /
                ``"bayesian"`` / ``"refinement"``).
            parameterization: The Ax-encoded parameter dict proposed
                for this trial.
            exc: The exception that marked the trial failed.
        """
        try:
            payload: dict[str, Any] = {
                "trial_index": trial_index,
                "phase": phase,
                "parameterization": parameterization,
                "exception": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "repr": repr(exc),
                },
                "attempt_diagnostics": [],
                "server_snapshots": [],
            }
            if isinstance(exc, BenchmarkFailure):
                payload["stage"] = exc.stage
                payload["iteration"] = exc.iteration
                payload["attempt_diagnostics"] = exc.attempt_diagnostics
                payload["server_snapshots"] = exc.server_snapshots
            failures_dir = self.output / "failures"
            failures_dir.mkdir(parents=True, exist_ok=True)
            (failures_dir / f"trial-{trial_index}.json").write_text(
                json.dumps(payload, indent=2, default=repr),
            )
        except Exception:
            logger.warning(
                "Failed to write per-trial failure dump for trial %d",
                trial_index,
                exc_info=True,
            )

    def _should_stop(self, completed_iterations: int) -> bool:
        """Return ``True`` when the main loop should terminate.

        ``completed_iterations`` is the live-session attempt counter
        (success plus failure). :attr:`prior_count` folds in seeded
        trials from a prior session so the combined budget
        ``prior_count + completed_iterations >= n_trials`` is honoured
        across resumes.

        Args:
            completed_iterations: Number of live-session trial
                attempts (successful plus failed) already performed.

        Returns:
            ``True`` once the combined attempt count reaches
            :attr:`n_trials`.
        """
        return self.prior_count + completed_iterations >= self.n_trials

    def _suggest(self, i: int) -> tuple[int, dict[str, str], str]:
        """Ask Ax for the next trial's parameterization.

        The phase label reflects the *global* position across prior
        and live sessions (``prior_count + i``), so resumed runs pick
        up their phase continuously rather than restarting the
        ``"sobol"`` label counter at zero.

        Args:
            i: Zero-based live-loop iteration index.

        Returns:
            A ``(trial_index, parameterization, phase)`` tuple where
            ``phase`` is ``"sobol"`` during the initialization budget
            and ``"bayesian"`` afterwards. Parameter values are coerced
            to ``str`` to match the declared ``parameter_type="str"``
            schema and the downstream sysctl writer's signature.
        """
        phase = "sobol" if self.prior_count + i < self.n_sobol else "bayesian"
        if self._seed_attempts_remaining > 0:
            self._seed_attempts_remaining -= 1
            seed = self._attach_recommended_defaults()
            if seed is not None:
                self._seed_trial_index, parameterization = seed
                return self._seed_trial_index, parameterization, phase
            # _attach_recommended_defaults logged the reason; fall through
            # to a normal Sobol / Bayesian suggestion for this iteration.
            self._seed_attempts_remaining = 0
        trials = self.client.get_next_trials(max_trials=1)
        trial_index, parameterization = next(iter(trials.items()))
        return trial_index, {k: str(v) for k, v in parameterization.items()}, phase

    def _record(
        self,
        i: int,
        phase: str,
        trial_index: int,
        trial_result: TrialResult,
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
            trial_result: The :class:`TrialResult` persisted by
                :meth:`_evaluate`; forwarded to the observer so it can
                key aggregation / grouping by ``trial_id``.
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

        tp = metrics.get("tcp_throughput", (float("nan"), 0.0))[0]
        tp_str = "NaN" if math.isnan(tp) else f"{tp / 1e6:.1f}"
        rate = metrics.get("tcp_retransmit_rate", (float("nan"), 0.0))[0]
        rate_str = "NaN" if math.isnan(rate) else f"{rate:.2f}"
        rps = metrics.get("rps", (float("nan"), 0.0))[0]
        rps_str = "NaN" if math.isnan(rps) else f"{rps:.1f}"
        p99 = metrics.get("latency_p99", (float("nan"), 0.0))[0]
        p99_str = "NaN" if math.isnan(p99) else format_duration(p99)
        logger.info(
            "Trial %d/%d [%s] tcp_throughput=%s Mbps "
            "tcp_retransmit_rate=%s retx/GB rps=%s p99=%s",
            self.prior_count + i + 1,
            self.n_trials,
            phase,
            tp_str,
            rate_str,
            rps_str,
            p99_str,
        )
        self.observer.on_trial_complete(
            self.prior_count + i,
            trial_result,
            metrics,
        )

    def _warn_on_collapsed_objectives(self) -> list[str]:
        """Warn once per Pareto objective whose observed variance has collapsed.

        Ax cannot learn a gradient on a metric whose observed std is
        effectively zero; upstream responds with a cascade of
        ``standardize_y`` WARNINGs and botorch ``InputDataWarning``
        lines on every Bayesian generate. This helper surfaces one
        actionable kube-autotuner warning instead, pointing the user
        at ``objectives.pareto`` and ``objectives.recommendationWeights``.

        The check iterates ``objectives.pareto`` metric names only;
        constraint-only metrics are expected to be bounded and a
        zero-variance constraint is not noteworthy. A metric is
        "collapsed" when its sample std across ``self._completed`` is
        below ``1e-12`` (absolute floor for near-zero means) or below
        ``1e-9 * abs(mean)`` (relative floor otherwise). Fires once per
        metric per :class:`OptimizationLoop` instance via
        :attr:`_collapse_warned`.

        Returns:
            The metric names that triggered a warning on this call
            (may be empty). Exposed primarily for tests; callers in
            :meth:`run` discard it.
        """
        primary = [tr for tr in self._completed if is_primary(tr)]
        if len(primary) < 2:  # noqa: PLR2004 - stdev needs >= 2
            return []
        abs_floor = 1e-12
        rel_floor = 1e-9
        pareto_metrics = [obj.metric for obj in self.objectives.pareto]
        per_trial_metrics = [_compute_metrics(tr) for tr in primary]
        newly_warned: list[str] = []
        for name in pareto_metrics:
            if name in self._collapse_warned:
                continue
            vals = [
                m[name][0]
                for m in per_trial_metrics
                if name in m and not math.isnan(m[name][0])
            ]
            if len(vals) < 2:  # noqa: PLR2004 - stdev needs >= 2
                continue
            std = statistics.stdev(vals)
            mean = statistics.mean(vals)
            threshold = max(abs_floor, rel_floor * abs(mean))
            if std < threshold:
                logger.warning(
                    "Objective %r has collapsed to near-constant variance "
                    "(std=%.2e, n=%d). Ax cannot learn a gradient on this "
                    "metric; consider removing it from `objectives.pareto` "
                    "AND `objectives.recommendationWeights`. It will still "
                    "act as a feasibility gate via any matching entry in "
                    "`objectives.constraints`.",
                    name,
                    std,
                    len(vals),
                )
                self._collapse_warned.add(name)
                newly_warned.append(name)
        return newly_warned

    def run(self) -> list[TrialResult]:
        """Execute the full optimization loop.

        Returns:
            The completed :class:`TrialResult` records, in execution
            order.

        Raises:
            KeyboardInterrupt: Propagated from a trial if the user
                interrupts mid-benchmark; the loop still runs
                :meth:`cleanup` through the ``finally`` block before
                re-raising.
            Exception: Any trial-level exception (typically
                :class:`~kube_autotuner.benchmark.errors.BenchmarkFailure`
                once the per-client-job ``max_attempts`` are exhausted)
                is re-raised after :meth:`_dump_failure` writes a
                post-mortem JSON and :meth:`cleanup` runs through the
                ``finally`` block. The loop does not advance past the
                failing trial.
        """  # noqa: DOC502 - trial-level exceptions originate in _evaluate
        self.runner.setup_server()
        try:
            i = 0
            while not self._should_stop(i):
                trial_index, parameterization, phase = self._suggest(i)
                if phase == "bayesian" and self._first_bayesian_generate:
                    self._warn_on_collapsed_objectives()
                    self._first_bayesian_generate = False
                logger.info(
                    "Trial %d/%d [%s] starting: sysctls=%s",
                    self.prior_count + i + 1,
                    self.n_trials,
                    phase,
                    {_decode_param_name(k): v for k, v in parameterization.items()},
                )
                self.observer.on_trial_start(
                    self.prior_count + i,
                    self.n_trials,
                    phase,
                    parameterization,
                )
                try:
                    trial_result, metrics = self._evaluate(
                        parameterization,
                        phase=phase,
                        parent_trial_id=None,
                    )
                    self._record(i, phase, trial_index, trial_result, metrics)
                    if trial_index == self._seed_trial_index:
                        self._seed_attempts_remaining = 0
                        self._seed_trial_index = None
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.warning(
                        "Trial %d/%d failed, exiting",
                        self.prior_count + i + 1,
                        self.n_trials,
                        exc_info=True,
                    )
                    logger.warning(
                        "Trial %d parameterization: %s",
                        self.prior_count + i + 1,
                        parameterization,
                    )
                    self._dump_failure(
                        trial_index=self.prior_count + i,
                        phase=phase,
                        parameterization=parameterization,
                        exc=e,
                    )
                    self.observer.on_trial_failed(self.prior_count + i, e)
                    live = len(self._completed) - self.prior_count
                    logger.info(
                        "Trial %d aborted, exiting after %d completed trials",
                        self.prior_count + i + 1,
                        live,
                    )
                    raise
                i += 1
        except KeyboardInterrupt:
            live = len(self._completed) - self.prior_count
            logger.info(
                "Interrupted after %d live trials (%d total)",
                live,
                len(self._completed),
            )
        finally:
            self.cleanup()
        return self._completed

    def _run_refinement_trial(
        self,
        *,
        parent: TrialResult,
        round_index: int,
        obs_index: int,
        total_samples: int,
    ) -> TrialResult:
        """Run one refinement sample of ``parent`` and return its row.

        Wraps the same observer / logging / failure-dump dance the
        primary loop uses, but with ``phase="refinement"`` and the
        parent's sysctl arm replayed verbatim. Exceptions are dumped,
        reported to the observer, and re-raised so :meth:`run_refinement`
        can stop the round.

        Args:
            parent: The primary trial whose arm is being replayed.
            round_index: 1-indexed refinement round number.
            obs_index: Observer-space index for this sample (continues
                the primary counter; see :meth:`run_refinement`).
            total_samples: Total refinement samples expected this run
                (``top_k * rounds``); used to size the observer bar.

        Returns:
            The persisted refinement :class:`TrialResult`.

        Raises:
            KeyboardInterrupt: Propagated when the user interrupts the
                in-flight benchmark.
            Exception: Any trial-level exception is re-raised after
                :meth:`_dump_failure` writes a post-mortem JSON and the
                observer is notified via ``on_trial_failed``.
        """  # noqa: DOC502 - refinement-trial exceptions originate in _evaluate
        ax_params: dict[str, str] = {
            _encode_param_name(k): str(v) for k, v in parent.sysctl_values.items()
        }
        logger.info(
            "Trial %d [refinement round %d] starting: sysctls=%s",
            obs_index + 1,
            round_index,
            {_decode_param_name(k): v for k, v in ax_params.items()},
        )
        self.observer.on_trial_start(obs_index, total_samples, "refinement", ax_params)
        try:
            trial_result, metrics = self._evaluate(
                ax_params,
                phase="refinement",
                parent_trial_id=parent.trial_id,
                refinement_round=round_index,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.warning(
                "Refinement run for parent %s round %d failed, exiting",
                parent.trial_id,
                round_index,
                exc_info=True,
            )
            self._dump_failure(
                trial_index=obs_index,
                phase="refinement",
                parameterization=ax_params,
                exc=e,
            )
            self.observer.on_trial_failed(obs_index, e)
            raise
        logger.info(
            "Refinement [round %d] parent=%s tcp_throughput=%.1f Mbps",
            round_index,
            parent.trial_id,
            metrics["tcp_throughput"][0] / 1e6,
        )
        self.observer.on_trial_complete(obs_index, trial_result, metrics)
        return trial_result

    def run_refinement(
        self,
        top_k: int,
        rounds: int,
        completed_by_round: dict[int, set[str]] | None = None,
    ) -> list[TrialResult]:
        """Iteratively re-rank top-K and sample one extra trial per round.

        At the start of each round ``r in 1..rounds``, aggregates the
        current ``self._completed`` population by parent via
        :func:`kube_autotuner.scoring.aggregate_by_parent`, scores the
        rows with :func:`kube_autotuner.scoring.score_rows` (ties
        broken by ``trial_id`` ascending -- same key as
        :func:`kube_autotuner.report.analysis.recommend_configs`),
        picks the top-K, and runs one fresh benchmark per parent that
        is not already in ``completed_by_round.get(r, set())``.

        Refinement samples are persisted with ``phase="refinement"``,
        ``parent_trial_id`` pointing at the primary's id, and a
        1-indexed ``refinement_round``. They are **not** attached to
        the Ax client: Ax rejects duplicate arms and the surrogate has
        no use for a repeat observation. The top-K is recomputed every
        round, so a parent that regresses toward the mean on its first
        refinement sample drops out and stops accumulating, while a
        previously-unsampled parent that climbs into top-K starts
        accumulating from its primary baseline.

        Stranded-parent logging: a parent that was in top-K last round
        but is not in top-K this round is logged once at INFO. Resume
        stranded entries (any ``done`` parent missing from round R's
        top-K) are also logged once at the start.

        The observer sees one
        ``on_refinement_round_start(round_index, total_rounds, pending)``
        per round that has at least one pending parent, plus one
        ``on_trial_start`` / ``on_trial_complete`` pair per sample with
        ``phase="refinement"``. ``pending`` is the count of this
        round's top-K parents not already in ``done`` (a stranded
        ``done`` parent that fell out of top-K does not subtract), so
        a partially-resumed round still drives a progress bar that
        reaches 100% when the remaining samples land. Observer indices
        continue the primary counter
        (``prior_count + live_primary + total_prior_refinement +
        session_ordinal``) so
        :class:`~kube_autotuner.progress.RichProgressObserver`'s
        ``_all_rows`` can index every row unambiguously across resumes.

        Args:
            top_k: Number of top configs to refine each round.
            rounds: Number of refinement rounds. Total budget is
                ``top_k * rounds`` when at least ``top_k`` primaries
                exist; fewer primaries shrink each round's top-K.
            completed_by_round: Optional map of
                ``round_index -> {parent_trial_id, ...}`` from a prior
                partial run; parents already sampled in a given round
                are skipped on resume.

        Returns:
            The newly created refinement :class:`TrialResult` records,
            in execution order.

        Raises:
            KeyboardInterrupt: Propagated when the user interrupts
                mid-refinement; :meth:`cleanup` still runs through the
                ``finally`` block before re-raising.
            Exception: Any refinement-trial exception (typically
                :class:`~kube_autotuner.benchmark.errors.BenchmarkFailure`
                once the per-client-job ``max_attempts`` are exhausted)
                is re-raised after :meth:`_dump_failure` writes a
                post-mortem JSON and :meth:`cleanup` runs through the
                ``finally`` block. The loop does not advance past the
                failing sample.
        """  # noqa: DOC502 - refinement-trial exceptions originate in _evaluate
        if top_k <= 0 or rounds <= 0:
            return []
        primary = [tr for tr in self._completed if is_primary(tr)]
        if not primary:
            return []

        done_by_round: dict[int, set[str]] = {
            r: set(ids) for r, ids in (completed_by_round or {}).items()
        }

        from kube_autotuner.scoring import (  # noqa: PLC0415
            aggregate_by_parent,
            config_memory_cost,
            score_rows,
        )
        from kube_autotuner.sysctl.params import PARAM_SPACE  # noqa: PLC0415

        parent_by_id: dict[str, TrialResult] = {p.trial_id: p for p in primary}

        def _select_top_k() -> list[TrialResult]:
            usable: list[tuple[dict[str, float | int | str], TrialResult]] = [
                (row, parent_by_id[str(row["trial_id"])])
                for row in aggregate_by_parent(self._completed)
                if str(row["trial_id"]) in parent_by_id
            ]
            if not usable:
                return []
            rows = [row for row, _ in usable]
            parents = [parent for _, parent in usable]
            scores = score_rows(
                rows,
                self.objectives.pareto,
                self.objectives.recommendation_weights,
                memory_costs=[
                    config_memory_cost(p.sysctl_values, PARAM_SPACE) for p in parents
                ],
                memory_cost_weight=self.objectives.memory_cost_weight,
            )
            ranking = sorted(
                range(len(parents)),
                key=lambda i: (-scores[i], parents[i].trial_id),
            )
            return [parents[i] for i in ranking[:top_k]]

        prev_round_topk: set[str] = set()
        total_prior_refinement = sum(
            1 for tr in self._completed if tr.phase == "refinement"
        )
        live_primary = len(primary) - self.prior_count
        base_index = self.prior_count + live_primary + total_prior_refinement
        ordinal = 0
        created: list[TrialResult] = []

        self.runner.setup_server()
        try:
            for round_index in range(1, rounds + 1):
                parents = _select_top_k()
                parent_ids = {p.trial_id for p in parents}
                done = done_by_round.get(round_index, set())

                _log_strandings(round_index, parent_ids, done, prev_round_topk)

                pending = [p for p in parents if p.trial_id not in done]
                if not pending:
                    logger.info(
                        "refinement round %d: nothing to run (already complete)",
                        round_index,
                    )
                    prev_round_topk = parent_ids
                    continue
                self.observer.on_refinement_round_start(
                    round_index,
                    rounds,
                    len(pending),
                )

                for parent in pending:
                    obs_index = base_index + ordinal
                    created.append(
                        self._run_refinement_trial(
                            parent=parent,
                            round_index=round_index,
                            obs_index=obs_index,
                            total_samples=top_k * rounds,
                        ),
                    )
                    ordinal += 1

                prev_round_topk = parent_ids
        finally:
            self.cleanup()
        return created

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

        This call reads from the Ax client, which only sees primary
        trials (refinement samples are deliberately not attached,
        since Ax rejects duplicate arms). The post-primary refinement
        summary table emitted by :func:`kube_autotuner.runs.run_optimize`
        is the authoritative combined-mean view; this frontier is the
        Ax-model view over primary observations only.

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
