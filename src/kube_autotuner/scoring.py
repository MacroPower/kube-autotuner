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
    from kube_autotuner.models import TrialResult


METRIC_TO_DF_COLUMN: dict[str, str] = {
    "throughput": "mean_throughput",
    "cpu": "mean_cpu",
    "node_memory": "mean_node_memory",
    "cni_memory": "mean_cni_memory",
    "retransmit_rate": "retransmit_rate",
    "jitter": "mean_jitter_ms",
    "rps": "mean_rps",
    "latency_p50": "mean_latency_p50_ms",
    "latency_p90": "mean_latency_p90_ms",
    "latency_p99": "mean_latency_p99_ms",
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
          NaN, e.g. ``mean_cni_memory`` on a CNI-disabled trial).
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

    Mirrors ``kube_autotuner.analysis._norm``: when the column has no
    finite values or ``min == max`` the entire column collapses to
    ``0.5``; finite values are mapped linearly to ``[0, 1]``; NaN rows
    fall back to ``0.5`` (pandas ``.fillna(0.5)`` after division).

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
) -> list[float]:
    """Return per-row weighted scores using the shared recommendation formula.

    Score formula, matching the derivation documented on
    :func:`kube_autotuner.analysis.recommend_configs`:

        ``sum(+norm(metric)     for objectives with direction="maximize") -
         sum(weights[metric] * norm(metric) for objectives with direction="minimize")``

    Each ``norm`` is a min-max normalization across the supplied
    ``rows``. Maximize-direction objectives always contribute
    ``+1.0 * norm``; ``weights`` are only consulted for
    minimize-direction metrics (``weights.get(metric, 0.0)``).

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
        weights: Non-negative multipliers for minimize-direction
            metrics, keyed by the short metric name (``"cpu"``,
            ``"latency_p99"``, ...). Missing keys default to ``0.0``.

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
            for i, value in enumerate(norm):
                scores[i] += value
        else:
            weight = weights.get(obj.metric, 0.0)
            for i, value in enumerate(norm):
                scores[i] -= weight * value
    return scores


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
    nmem = t.mean_node_memory()
    cmem = t.mean_cni_memory()
    rate = t.retransmit_rate()
    return {
        METRIC_TO_DF_COLUMN["throughput"]: t.mean_throughput(),
        METRIC_TO_DF_COLUMN["cpu"]: t.mean_cpu(),
        METRIC_TO_DF_COLUMN["node_memory"]: (math.nan if nmem is None else nmem),
        METRIC_TO_DF_COLUMN["cni_memory"]: (math.nan if cmem is None else cmem),
        METRIC_TO_DF_COLUMN["retransmit_rate"]: (math.nan if rate is None else rate),
        METRIC_TO_DF_COLUMN["jitter"]: t.mean_jitter_ms(),
        METRIC_TO_DF_COLUMN["rps"]: t.mean_rps(),
        METRIC_TO_DF_COLUMN["latency_p50"]: t.mean_latency_p50_ms(),
        METRIC_TO_DF_COLUMN["latency_p90"]: t.mean_latency_p90_ms(),
        METRIC_TO_DF_COLUMN["latency_p99"]: t.mean_latency_p99_ms(),
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

    * ``throughput`` / ``rps``: mean of per-trial means.
    * ``cpu`` / ``jitter`` / ``node_memory`` / ``cni_memory`` /
      ``latency_p50`` / ``latency_p90`` / ``latency_p99``: mean of
      per-trial means; NaN (not measured) drops out of the mean.
    * ``retransmit_rate``: mean of per-trial rates; NaN drops out.

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
