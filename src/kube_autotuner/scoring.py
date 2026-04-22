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

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kube_autotuner.experiment import ParetoObjective


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
