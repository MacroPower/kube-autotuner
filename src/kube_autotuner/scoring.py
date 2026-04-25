"""Unit-agnostic scalarization of Pareto objectives into a single score.

Factored out of :func:`kube_autotuner.analysis.recommend_configs` so the
same ranking formula drives both the post-hoc recommendation output and
the live ``Best so far`` panel in
:class:`kube_autotuner.progress.RichProgressObserver`. Pure-stdlib by
design: :mod:`kube_autotuner.progress` sits under an import ceiling
that forbids ``pandas`` / ``numpy`` / ``ax-platform`` and forbids
reaching into :mod:`kube_autotuner.analysis`, and this helper is
reachable from there.

Min-max normalization makes the score scale-invariant: the live panel
can hand in raw-domain floats (bits/sec, bytes) while
``recommend_configs`` passes values pulled from a pandas DataFrame
(``bits_per_second`` likewise), and the resulting ranks are identical.
Only the magnitude of the score differs, and score magnitudes are
never surfaced by the live panel.
"""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kube_autotuner.experiment import ParetoObjective
    from kube_autotuner.models import MemoryCost, ParamSpace, TrialResult


METRIC_TO_DF_COLUMN: dict[str, str] = {
    "tcp_throughput": "mean_tcp_throughput",
    "udp_throughput": "mean_udp_throughput",
    "tcp_retransmit_rate": "tcp_retransmit_rate",
    "udp_loss_rate": "udp_loss_rate",
    "udp_jitter": "mean_udp_jitter",
    "rps": "mean_rps",
    "latency_p50": "mean_latency_p50",
    "latency_p90": "mean_latency_p90",
    "latency_p99": "mean_latency_p99",
}

_DEGENERATE_NORM = 0.5


def _to_float_or_nan(v: object) -> float:
    """Coerce ``v`` to a finite float; map ``None`` or non-finite to NaN.

    Args:
        v: Raw cell value pulled from a caller-supplied row dict. May
            be a ``float``, ``int``, ``None``, or a pandas sentinel that
            :func:`float` accepts.

    Returns:
        A finite float or :data:`math.nan`. NaN covers three inputs
        that :func:`_normalize_column` treats identically:

        * ``None`` (``recommend_configs`` passes this for object-dtype
          NaN, e.g. ``mean_udp_jitter`` on a TCP-only trial).
        * ``float("nan")`` (both callers can pass this directly).
        * Any value that fails :func:`float` coercion. Defensive
          catch-all; no caller is known to trigger this path.
    """
    if v is None:
        return math.nan
    try:
        f = float(v)  # ty: ignore[invalid-argument-type]
    except TypeError, ValueError:
        return math.nan
    return f if math.isfinite(f) else math.nan


def _normalize_column(values: list[float]) -> list[float]:
    """Min-max normalize ``values``, mapping NaN and degenerate to ``0.5``.

    Canonical implementation: the analysis path consumes this through
    :func:`score_rows`. When the column has no finite values or
    ``min == max`` the entire column collapses to ``0.5``; finite
    values are mapped linearly to ``[0, 1]``; NaN rows fall back to
    ``0.5`` (the equivalent of ``.fillna(0.5)`` after division).

    Args:
        values: Per-row raw values for a single metric column.

    Returns:
        A list of normalized floats the same length as ``values``.
    """
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return [_DEGENERATE_NORM] * len(values)
    lo = min(finite)
    hi = max(finite)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi == lo:
        return [_DEGENERATE_NORM] * len(values)
    span = hi - lo
    return [_DEGENERATE_NORM if math.isnan(v) else (v - lo) / span for v in values]


def score_rows(
    rows: Sequence[Mapping[str, object]],
    objectives: Sequence[ParetoObjective],
    weights: Mapping[str, float],
    memory_costs: Sequence[float] | None = None,
    memory_cost_weight: float = 0.0,
) -> list[float]:
    """Return per-row weighted scores using the shared recommendation formula.

    Score formula:

        ``sum(+weights.get(m, 1.0) * norm(m) for m in maximize-direction) -
         sum(weights.get(m, 0.0) * norm(m) for m in minimize-direction)
         - memory_cost_weight * norm(memory_cost)``

    Each ``norm`` is a min-max normalization across the supplied
    ``rows``. ``weights`` applies to both directions, with
    direction-sensitive defaults: an omitted maximize-metric weight
    falls back to ``1.0`` (preserving that metric's full +norm
    contribution), while an omitted minimize-metric weight falls back
    to ``0.0`` (the metric participates in frontier selection upstream
    but does not bias the score).

    The optional ``memory_costs`` / ``memory_cost_weight`` pair adds a
    synthetic minimize term for static kernel/CNI memory footprint; see
    :func:`config_memory_cost`. When both are supplied, the cost column
    is min-max normalized across ``rows`` and
    ``memory_cost_weight * norm`` is subtracted from each row's score
    mirroring the minimize branch above. Single-row inputs collapse to
    ``norm = 0.5`` (uniform offset, no rank change); ``memory_costs``
    omitted or ``memory_cost_weight = 0.0`` yields the cost-free score.

    Both call sites pass their own idiomatic row shape and the helper
    is tolerant of both:

    * :mod:`kube_autotuner.progress` emits raw ``float`` values with
      :data:`math.nan` for missing readings.
    * :func:`kube_autotuner.analysis.recommend_configs` passes
      ``front[cols].to_dict(orient="records")``; pandas emits ``nan``
      for numeric columns and ``None`` for object-dtype columns.

    The lookup key for each objective is
    ``METRIC_TO_DF_COLUMN[objective.metric]`` -- the DataFrame column
    name, not the short metric label. Both call sites therefore key
    their rows by the DataFrame column naming.

    A row missing an objective's column, or an objective missing from
    the rows, resolves to NaN for that metric. NaN + the
    degenerate-column fallback in :func:`_normalize_column` collapses
    the column to ``0.5`` uniformly across rows, so the contribution
    is identical across rows and does not change the ranking.

    Ranking is scale-invariant under min-max normalization, so
    callers may feed values in their native units (bits/sec, bytes,
    Mbps, MiB) without affecting the rank order. Score magnitudes
    differ; ranks do not.

    Args:
        rows: Per-trial metric bundles keyed by DataFrame column name
            (see :data:`METRIC_TO_DF_COLUMN`).
        objectives: Pareto objectives driving the scoring formula.
        weights: Non-negative per-metric multipliers keyed by the
            short metric name (``"tcp_throughput"``,
            ``"tcp_retransmit_rate"``, ...). Missing maximize-metric
            keys default to ``1.0``; missing minimize-metric keys
            default to ``0.0``.
        memory_costs: Optional per-row static memory footprint in
            bytes aligned with ``rows``. Callers precompute this via
            :func:`config_memory_cost` so the helper stays
            stdlib-only.
        memory_cost_weight: Non-negative multiplier applied to the
            normalized memory-cost column. ``0.0`` (or ``None`` costs)
            disables the term entirely.

    Returns:
        A list of raw float scores in ``rows`` order. Rounding and
        sorting are caller concerns.
    """
    n = len(rows)
    if n == 0:
        return []

    scores = [0.0] * n
    for obj in objectives:
        col = METRIC_TO_DF_COLUMN[obj.metric]
        raw = [_to_float_or_nan(row.get(col)) for row in rows]
        norm = _normalize_column(raw)
        if obj.direction == "maximize":
            weight = weights.get(obj.metric, 1.0)
            for i, value in enumerate(norm):
                scores[i] += weight * value
        else:
            weight = weights.get(obj.metric, 0.0)
            for i, value in enumerate(norm):
                scores[i] -= weight * value

    if memory_costs is not None and memory_cost_weight > 0.0:
        cost_norm = _normalize_column([_to_float_or_nan(c) for c in memory_costs])
        for i, value in enumerate(cost_norm):
            scores[i] -= memory_cost_weight * value
    return scores


def _apply_memory_cost_rule(rule: MemoryCost, value: int | str) -> int:
    """Evaluate a :class:`MemoryCost` rule against a selected rung.

    Args:
        rule: Derivation kind attached to the sysctl.
        value: The rung the trial selected. Mandatory argument kinds:
            ``identity``, ``kib``, and ``per_entry`` require a numeric
            value; ``triple_max`` and ``triple_max_pages`` require a
            space-separated triple whose last field parses as an int.

    Returns:
        The estimated bytes consumed by this rung. ``0`` when the
        value cannot be coerced under the chosen rule, so a bad
        annotation degrades to "no cost" rather than crashing the
        scorer mid-run.
    """
    kind = rule.kind
    try:
        if kind == "identity":
            return int(value)
        if kind == "kib":
            return int(value) * 1024
        if kind == "per_entry":
            return int(value) * rule.per_entry_bytes
        if kind in {"triple_max", "triple_max_pages"}:
            max_field = int(str(value).split()[-1])
            return max_field * 4096 if kind == "triple_max_pages" else max_field
    except TypeError, ValueError:
        return 0
    return 0


def config_memory_cost(
    sysctl_values: Mapping[str, int | str],
    param_space: ParamSpace,
) -> float:
    """Return the static kernel/CNI memory cost for a sysctl configuration.

    Evaluates each annotated :class:`~kube_autotuner.models.SysctlParam`'s
    :class:`~kube_autotuner.models.MemoryCost` rule against the selected
    rung and sums the result. Unannotated params contribute ``0``, as
    do rungs whose shape does not match the rule (see
    :func:`_apply_memory_cost_rule`).

    Pure-stdlib so :mod:`kube_autotuner.progress` can call it without
    breaking the live-panel import ceiling.

    Args:
        sysctl_values: Mapping from sysctl name to the selected rung
            value (``TrialResult.sysctl_values`` shape).
        param_space: The search space the values were drawn from.

    Returns:
        Total estimated bytes across all costed sysctls. ``0.0`` when
        nothing in the configuration touches a costed knob.
    """
    rules = {
        p.name: p.memory_cost for p in param_space.params if p.memory_cost is not None
    }
    total = 0
    for name, value in sysctl_values.items():
        rule = rules.get(name)
        if rule is None:
            continue
        total += _apply_memory_cost_rule(rule, value)
    return float(total)


def _per_trial_metric_means(t: TrialResult) -> dict[str, float]:
    """Project one :class:`TrialResult` into METRIC_TO_DF_COLUMN means.

    Uses the canonical per-trial aggregation methods on
    :class:`~kube_autotuner.models.TrialResult` so the arithmetic
    matches both the live panel and ``_compute_metrics``.

    Args:
        t: Trial record whose metrics should be projected.

    Returns:
        A dict keyed by the DataFrame column names in
        :data:`METRIC_TO_DF_COLUMN`, with :data:`math.nan` in place of
        any unreported metric.
    """
    rate = t.tcp_retransmit_rate()
    return {
        METRIC_TO_DF_COLUMN["tcp_throughput"]: t.mean_tcp_throughput(),
        METRIC_TO_DF_COLUMN["udp_throughput"]: t.mean_udp_throughput(),
        METRIC_TO_DF_COLUMN["tcp_retransmit_rate"]: (
            math.nan if rate is None else rate
        ),
        METRIC_TO_DF_COLUMN["udp_loss_rate"]: t.udp_loss_rate(),
        METRIC_TO_DF_COLUMN["udp_jitter"]: t.mean_udp_jitter(),
        METRIC_TO_DF_COLUMN["rps"]: t.mean_rps(),
        METRIC_TO_DF_COLUMN["latency_p50"]: t.mean_latency_p50(),
        METRIC_TO_DF_COLUMN["latency_p90"]: t.mean_latency_p90(),
        METRIC_TO_DF_COLUMN["latency_p99"]: t.mean_latency_p99(),
    }


def _mean_sem_of(values: list[float]) -> tuple[float, float]:
    """Return ``(mean, SEM)`` over ``values`` dropping NaN entries.

    SEM is ``stdev / sqrt(n)`` with ``n`` counting the finite samples;
    fewer than two finite samples yields ``(mean, 0.0)``. An
    all-NaN input yields ``(nan, 0.0)`` so callers can propagate the
    "not measured" sentinel through the aggregated row.
    """
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return math.nan, 0.0
    mean = statistics.mean(finite)
    if len(finite) < 2:  # noqa: PLR2004 - stdev needs >= 2
        return mean, 0.0
    sem = statistics.stdev(finite) / math.sqrt(len(finite))
    return mean, sem


def aggregate_verification(
    trials: list[TrialResult],
) -> list[dict[str, float | int | str]]:
    """Return one row per parent config, metrics meaned across all samples.

    Groups ``trials`` by ``parent_trial_id or trial_id`` so every
    verification repeat folds back into its parent's bucket and a
    primary trial without any verification repeats still produces a
    single-sample row. Per-metric aggregation mirrors
    :func:`kube_autotuner.optimizer._compute_metrics`:

    * ``tcp_throughput`` / ``udp_throughput`` / ``rps``: mean of
      per-trial means.
    * ``udp_jitter`` / ``latency_p50`` / ``latency_p90`` /
      ``latency_p99``: mean of per-trial means; NaN (not measured)
      drops out of the mean.
    * ``tcp_retransmit_rate`` / ``udp_loss_rate``: mean of per-trial
      rates; NaN drops out.

    Each ``<metric>_sem`` is the standard error of the mean over the
    parent's finite samples (``stdev / sqrt(n)``); ``0.0`` when fewer
    than two samples survive.

    Verification rows whose ``parent_trial_id`` does not match any
    primary ``trial_id`` in ``trials`` still form their own group
    keyed by that id, which is harmless because downstream callers
    filter by the primary ``trial_id`` they care about.

    Args:
        trials: The combined primary + verification population.

    Returns:
        One row per aggregation group, keyed by the DataFrame column
        names in :data:`METRIC_TO_DF_COLUMN` plus ``trial_id``,
        ``sample_count``, and ``<metric>_sem`` per metric. Rows are
        returned in first-seen group order so callers get a stable,
        file-order-ish iteration.
    """
    groups: dict[str, list[TrialResult]] = defaultdict(list)
    order: list[str] = []
    for t in trials:
        key = t.parent_trial_id or t.trial_id
        if key not in groups:
            order.append(key)
        groups[key].append(t)

    rows: list[dict[str, float | int | str]] = []
    for key in order:
        samples = groups[key]
        per_trial = [_per_trial_metric_means(t) for t in samples]
        row: dict[str, float | int | str] = {
            "trial_id": key,
            "sample_count": len(samples),
        }
        for col in METRIC_TO_DF_COLUMN.values():
            values = [m[col] for m in per_trial]
            mean, sem = _mean_sem_of(values)
            row[col] = mean
            row[f"{col}_sem"] = sem
        rows.append(row)
    return rows
