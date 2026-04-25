"""Offline analysis of trial data: Pareto frontier, importance, recommendations.

The heavy dependencies -- ``pandas``, ``numpy``, ``scikit-learn`` -- live in
the optional ``analysis`` dependency group. They are lazy-imported inside
function bodies so that ``import kube_autotuner.report.analysis`` succeeds
under the base ``dev`` sync (``task completions`` and the analyze Typer
command both rely on this). Three rules enforce the discipline:

* ``from __future__ import annotations`` at the top keeps every
  signature a string, so ``-> pd.DataFrame`` does not trigger an eager
  import.
* Annotation-only symbols live under ``if TYPE_CHECKING:``.
* Every runtime use imports inside the function body, wrapped in
  ``try/except ImportError`` and raising :class:`RuntimeError` with the
  ``uv sync --group analysis`` hint.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC
import logging
import math
from operator import itemgetter
import statistics
from typing import TYPE_CHECKING, Any

from kube_autotuner.models import (
    compute_sysctl_hash,
    is_primary,
    tcp_retransmit_rate_by_iteration,
    udp_loss_rate_by_iteration,
)
from kube_autotuner.scoring import (
    METRIC_TO_DF_COLUMN,
    aggregate_verification,
    config_memory_cost,
    score_rows,
)
from kube_autotuner.sysctl.params import (
    PARAM_SPACE,
    PARAM_TO_CATEGORY,
    RECOMMENDED_DEFAULTS,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    from kube_autotuner.experiment import ParetoObjective
    from kube_autotuner.models import ResumeMetadata, TrialResult

_ANALYSIS_HINT = "install analysis group: uv sync --group analysis"

_PARAM_TYPE_LOOKUP: dict[str, str] = {p.name: p.param_type for p in PARAM_SPACE.params}
_SYSCTL_COLUMNS: list[str] = PARAM_SPACE.param_names()

DEFAULT_OBJECTIVES: list[tuple[str, str]] = [
    (METRIC_TO_DF_COLUMN["tcp_throughput"], "maximize"),
    (METRIC_TO_DF_COLUMN["udp_throughput"], "maximize"),
    (METRIC_TO_DF_COLUMN["tcp_retransmit_rate"], "minimize"),
    (METRIC_TO_DF_COLUMN["udp_loss_rate"], "minimize"),
    (METRIC_TO_DF_COLUMN["udp_jitter"], "minimize"),
    (METRIC_TO_DF_COLUMN["rps"], "maximize"),
    (METRIC_TO_DF_COLUMN["latency_p50"], "minimize"),
    (METRIC_TO_DF_COLUMN["latency_p90"], "minimize"),
    (METRIC_TO_DF_COLUMN["latency_p99"], "minimize"),
]

_FRAME_BASE_COLUMNS: list[str] = [
    "trial_id",
    "hardware_class",
    "topology",
    "source_zone",
    "target_zone",
    "mean_tcp_throughput",
    "mean_udp_throughput",
    "tcp_retransmit_rate",
    "udp_loss_rate",
    "mean_udp_jitter",
    "mean_rps",
    "mean_latency_p50",
    "mean_latency_p90",
    "mean_latency_p99",
]

_MIN_SPEARMAN_SAMPLES = 3

_STABILITY_GREEN_MAX = 0.05
_STABILITY_AMBER_MAX = 0.15
_ZERO_MEAN_EPSILON = 1e-12
_HOST_STATE_ERROR_MAX_LEN = 240

_METRIC_COL_TO_TRIAL_ATTR: dict[str, str] = {
    "mean_tcp_throughput": "mean_tcp_throughput",
    "mean_udp_throughput": "mean_udp_throughput",
    "tcp_retransmit_rate": "tcp_retransmit_rate",
    "udp_loss_rate": "udp_loss_rate",
    "mean_udp_jitter": "mean_udp_jitter",
    "mean_rps": "mean_rps",
    "mean_latency_p50": "mean_latency_p50",
    "mean_latency_p90": "mean_latency_p90",
    "mean_latency_p99": "mean_latency_p99",
}


def _trial_metric_value(t: TrialResult, col: str) -> float | None:
    """Return a trial's aggregated per-trial value for the given metric column.

    Returns ``None`` when the metric is unmeasured (the trial accessor
    returned ``None``) or non-finite.
    """
    attr = _METRIC_COL_TO_TRIAL_ATTR.get(col)
    if attr is None:
        return None
    raw = getattr(t, attr)()
    if raw is None:
        return None
    try:
        v = float(raw)
    except TypeError, ValueError:
        return None
    return v if math.isfinite(v) else None


def _per_iteration_throughput(t: TrialResult, mode: str) -> list[float]:
    """Return per-iteration summed ``bits_per_second`` for a TCP/UDP mode."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for r in t.results:
        if r.mode == mode:
            grouped[r.iteration].append(r.bits_per_second)
    return [sum(vs) for vs in grouped.values() if vs]


def _per_iteration_jitter(t: TrialResult) -> list[float]:
    """Return per-iteration mean UDP jitter across clients."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for r in t.results:
        if r.jitter is not None:
            grouped[r.iteration].append(r.jitter)
    return [sum(vs) / len(vs) for vs in grouped.values() if vs]


def _per_iteration_rps(t: TrialResult) -> list[float]:
    """Return per-iteration summed RPS from fortio saturation."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for r in t.latency_results:
        if r.workload == "saturation":
            grouped[r.iteration].append(r.rps)
    return [sum(vs) for vs in grouped.values() if vs]


def _per_iteration_latency(t: TrialResult, attr: str) -> list[float]:
    """Return per-iteration mean of fortio fixed-QPS ``attr`` across clients."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for r in t.latency_results:
        if r.workload == "fixed_qps":
            v = getattr(r, attr)
            if v is not None:
                grouped[r.iteration].append(v)
    return [sum(vs) / len(vs) for vs in grouped.values() if vs]


_PER_ITERATION_DISPATCH: dict[str, Any] = {
    "mean_tcp_throughput": lambda t: _per_iteration_throughput(t, "tcp"),
    "mean_udp_throughput": lambda t: _per_iteration_throughput(t, "udp"),
    "tcp_retransmit_rate": lambda t: list(tcp_retransmit_rate_by_iteration(t.results)),
    "udp_loss_rate": lambda t: list(udp_loss_rate_by_iteration(t.results)),
    "mean_udp_jitter": _per_iteration_jitter,
    "mean_rps": _per_iteration_rps,
    "mean_latency_p50": lambda t: _per_iteration_latency(t, "latency_p50"),
    "mean_latency_p90": lambda t: _per_iteration_latency(t, "latency_p90"),
    "mean_latency_p99": lambda t: _per_iteration_latency(t, "latency_p99"),
}


def _per_iteration_metric_values(t: TrialResult, col: str) -> list[float]:
    """Return per-iteration values for a metric column.

    Mirrors the aggregation semantics used by
    :class:`~kube_autotuner.models.TrialResult` accessors: throughput
    and RPS are summed across clients within an iteration, jitter and
    latency percentiles are averaged across clients, and rate metrics
    are per-iteration ratios-of-sums. Iterations with no contributing
    record are dropped; every surviving value is finite.
    """
    dispatcher = _PER_ITERATION_DISPATCH.get(col)
    if dispatcher is None:
        return []
    return dispatcher(t)


def _trial_metric_std(t: TrialResult, col: str) -> float | None:
    """Return the per-iteration stdev for a metric column, or ``None``.

    ``None`` when fewer than two finite per-iteration samples exist or
    when the computed stdev is non-finite.
    """
    values = [v for v in _per_iteration_metric_values(t, col) if math.isfinite(v)]
    if len(values) < 2:  # noqa: PLR2004 - stdev needs >= 2
        return None
    stdev = statistics.stdev(values)
    return stdev if math.isfinite(stdev) else None


def _finite_or_none(v: object) -> float | None:
    """Coerce a numeric value to a finite float or ``None``.

    Mirrors the guard in ``_build_axis_payload`` so new payload fields
    never contain ``NaN`` / ``Infinity`` tokens that would break
    ``json.dumps(allow_nan=False)``.

    Args:
        v: Any value; ``None``, non-numeric, and non-finite inputs all
            map to ``None``.

    Returns:
        A finite ``float`` or ``None``.
    """
    if v is None:
        return None
    try:
        f = float(v)  # ty: ignore[invalid-argument-type]
    except TypeError, ValueError:
        return None
    return f if math.isfinite(f) else None


def _require_pandas() -> Any:  # noqa: ANN401
    """Return the ``pandas`` module, raising a hint when it is missing.

    Returns:
        The imported ``pandas`` module.

    Raises:
        RuntimeError: ``pandas`` is not installed.
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(_ANALYSIS_HINT) from e
    return pd


def _require_numpy() -> Any:  # noqa: ANN401
    """Return the ``numpy`` module, raising a hint when it is missing.

    Returns:
        The imported ``numpy`` module.

    Raises:
        RuntimeError: ``numpy`` is not installed (``pandas`` pulls it
            in, so in practice this fires only when the ``analysis``
            group is missing entirely).
    """
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(_ANALYSIS_HINT) from e
    return np


def _require_sklearn() -> tuple[Any, Any]:
    """Return the scikit-learn symbols the analysis module uses.

    Returns:
        A ``(RandomForestRegressor, LabelEncoder)`` tuple.

    Raises:
        RuntimeError: ``scikit-learn`` is not installed.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415
        from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(_ANALYSIS_HINT) from e
    return RandomForestRegressor, LabelEncoder


def split_trials_by_hardware_class(
    trials: list[TrialResult],
) -> dict[str, list[TrialResult]]:
    """Group ``trials`` by ``node_pair.hardware_class`` in a stable order.

    Args:
        trials: Trial records to partition.

    Returns:
        A dict keyed by hardware-class label (e.g. ``"1g"``, ``"10g"``)
        mapping to the trials belonging to that class, in input order.
        Hardware-class keys are sorted alphabetically.
    """
    by_class: dict[str, list[TrialResult]] = {}
    for t in trials:
        by_class.setdefault(t.node_pair.hardware_class, []).append(t)
    return {hw: by_class[hw] for hw in sorted(by_class)}


def trials_to_dataframe(
    trials: list[TrialResult],
    hardware_class: str | None = None,
    topology: str | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Convert trial results into a DataFrame suitable for analysis.

    When ``hardware_class`` is ``None`` and the input spans more than
    one hardware class, the function raises -- callers must either pass
    a filter or split the input via
    :func:`split_trials_by_hardware_class` first. This makes the
    mixed-class branch explicit rather than silently analysing all
    classes as one population.

    Parameters with ``param_type="int"`` in the search space are
    coerced to numeric. Choice parameters with non-numeric string
    values are label-encoded; the returned ``encoders`` dict maps
    param name to :class:`sklearn.preprocessing.LabelEncoder` for
    reverse lookup in plots. This helper lazy-imports ``pandas`` and
    ``scikit-learn`` and raises :exc:`RuntimeError` with the
    ``uv sync --group analysis`` hint when the group is missing.

    Args:
        trials: Input trial records.
        hardware_class: If set, only trials with this hardware class
            are retained.
        topology: If set, only trials with this topology label
            (``"intra-az"`` / ``"inter-az"`` / ``"unknown"``) are
            retained.

    Returns:
        A ``(DataFrame, encoders)`` pair.

    Raises:
        ValueError: ``hardware_class`` is ``None`` and the input
            contains more than one hardware class.
    """
    pd = _require_pandas()
    _, label_encoder_cls = _require_sklearn()

    if hardware_class is None:
        distinct = {t.node_pair.hardware_class for t in trials}
        if len(distinct) > 1:
            classes = sorted(distinct)
            msg = (
                f"trials span multiple hardware classes {classes}; "
                "pass hardware_class=<label> or call "
                "split_trials_by_hardware_class first"
            )
            raise ValueError(msg)

    if hardware_class is not None:
        trials = [t for t in trials if t.node_pair.hardware_class == hardware_class]
    if topology is not None:
        trials = [t for t in trials if t.topology == topology]

    rows: list[dict[str, Any]] = []
    for t in trials:
        row: dict[str, Any] = {
            "trial_id": t.trial_id,
            "hardware_class": t.node_pair.hardware_class,
            "topology": t.topology,
            "source_zone": t.node_pair.source_zone,
            "target_zone": t.node_pair.target_zone,
            "mean_tcp_throughput": t.mean_tcp_throughput(),
            "mean_udp_throughput": t.mean_udp_throughput(),
            "tcp_retransmit_rate": t.tcp_retransmit_rate(),
            "udp_loss_rate": t.udp_loss_rate(),
            "mean_udp_jitter": t.mean_udp_jitter(),
            "mean_rps": t.mean_rps(),
            "mean_latency_p50": t.mean_latency_p50(),
            "mean_latency_p90": t.mean_latency_p90(),
            "mean_latency_p99": t.mean_latency_p99(),
        }
        for metric_col in METRIC_TO_DF_COLUMN.values():
            row[f"{metric_col}_std"] = _trial_metric_std(t, metric_col)
        for key in _SYSCTL_COLUMNS:
            row[key] = t.sysctl_values.get(key)
        rows.append(row)

    if not rows:
        std_cols = [f"{c}_std" for c in METRIC_TO_DF_COLUMN.values()]
        return (
            pd.DataFrame(columns=_FRAME_BASE_COLUMNS + std_cols + _SYSCTL_COLUMNS),
            {},
        )

    df = pd.DataFrame(rows)
    for col in METRIC_TO_DF_COLUMN.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    encoders = _encode_sysctl_columns(df, pd, label_encoder_cls)
    return df, encoders


def _encode_sysctl_columns(
    df: pd.DataFrame,
    pd_module: Any,  # noqa: ANN401
    label_encoder_cls: Any,  # noqa: ANN401
) -> dict[str, LabelEncoder]:
    """Coerce int params to numeric and label-encode non-numeric choices.

    Args:
        df: Frame containing one row per trial; mutated in place.
        pd_module: The already-imported ``pandas`` module (passed in to
            avoid re-importing in a hot loop).
        label_encoder_cls: The
            :class:`sklearn.preprocessing.LabelEncoder` class.

    Returns:
        A mapping from column name to the fitted
        :class:`LabelEncoder`, populated only for columns that needed
        label encoding.
    """
    encoders: dict[str, LabelEncoder] = {}
    for col in _SYSCTL_COLUMNS:
        if col not in df.columns:
            continue
        ptype = _PARAM_TYPE_LOOKUP.get(col, "choice")
        if ptype == "int":
            df[col] = pd_module.to_numeric(df[col], errors="coerce")
            continue
        numeric = pd_module.to_numeric(df[col], errors="coerce")
        if numeric.notna().all():
            df[col] = numeric
        else:
            enc = label_encoder_cls()
            df[col] = enc.fit_transform(df[col].astype(str))
            encoders[col] = enc
    return encoders


def _objectives_with_data(
    df: pd.DataFrame,
    objectives: list[tuple[str, str]],
    *,
    log: bool = True,
) -> list[tuple[str, str]]:
    """Drop objectives whose column is missing or all-NaN in ``df``.

    A metric is excluded when its column is absent from ``df`` entirely
    or when every value in that column is NaN. Both cases mean "not
    measured for this experiment".

    Args:
        df: Frame produced by :func:`trials_to_dataframe`.
        objectives: List of ``(column, "maximize"|"minimize")`` tuples.
        log: When ``True`` (default), emit one INFO line per dropped
            objective so operators can see why a column disappeared
            from reports and plots. Callers that re-filter an
            already-filtered list pass ``log=False`` to avoid
            duplicate output.

    Returns:
        The subset of ``objectives`` whose column has at least one
        non-null value in ``df``, preserving input order.
    """
    kept: list[tuple[str, str]] = []
    for col, direction in objectives:
        if col in df.columns and df[col].notna().any():
            kept.append((col, direction))
        elif log:
            logger.info(
                "objective %r excluded: no trial reported this metric",
                col,
            )
    return kept


def pareto_front(
    df: pd.DataFrame,
    objectives: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Return the non-dominated rows from ``df``.

    Objectives whose column is absent from ``df`` or entirely NaN are
    dropped first (see :func:`_objectives_with_data`) so a
    disabled-metric column does not reduce the frontier to the empty
    set. Rows with a NaN value on any surviving objective column are
    then dropped before the dominance scan -- numpy comparisons against
    NaN are always ``False``, so a NaN-bearing row is neither
    dominated nor dominates, and would otherwise survive the frontier
    and poison downstream normalization.

    Lazy-imports ``numpy`` and raises :exc:`RuntimeError` with the
    ``uv sync --group analysis`` hint when the group is missing.

    Args:
        df: Frame produced by :func:`trials_to_dataframe`.
        objectives: List of ``(column, "maximize"|"minimize")`` tuples.
            Defaults to :data:`DEFAULT_OBJECTIVES`.

    Returns:
        A new DataFrame containing only Pareto-optimal rows, indexed
        from ``0``.
    """
    np = _require_numpy()

    if objectives is None:
        objectives = DEFAULT_OBJECTIVES

    if df.empty:
        return df

    objectives = _objectives_with_data(df, objectives)
    if not objectives:
        return df.iloc[0:0].reset_index(drop=True)

    cols = [col for col, _ in objectives]
    finite_mask = df[cols].notna().all(axis=1)
    dropped = int((~finite_mask).sum())
    if dropped:
        dropped_ids = df.loc[~finite_mask, "trial_id"].tolist()
        preview = dropped_ids[:5]
        ellipsis = ", ..." if len(dropped_ids) > len(preview) else ""
        logger.warning(
            "pareto_front: dropping %d trial(s) with NaN on %s: %s%s",
            dropped,
            cols,
            preview,
            ellipsis,
        )
    finite_df = df.loc[finite_mask]
    if finite_df.empty:
        return finite_df.reset_index(drop=True)

    vals = finite_df[cols].to_numpy(dtype=float)

    signs = np.array([1.0 if d == "minimize" else -1.0 for _, d in objectives])
    minimized = vals * signs

    n = len(minimized)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if np.all(minimized[j] <= minimized[i]) and np.any(
                minimized[j] < minimized[i],
            ):
                is_dominated[i] = True
                break

    return finite_df.loc[~is_dominated].reset_index(drop=True)


def parameter_importance(
    df: pd.DataFrame,
    target: str = "mean_tcp_throughput",
) -> pd.DataFrame:
    """Rank sysctl parameters by importance for the ``target`` metric.

    Lazy-imports ``pandas`` and ``scikit-learn`` and raises
    :exc:`RuntimeError` with the ``uv sync --group analysis`` hint
    when the group is missing.

    Args:
        df: Frame produced by :func:`trials_to_dataframe`.
        target: Metric column name to score against (defaults to
            ``"mean_tcp_throughput"``).

    Returns:
        A DataFrame with columns ``param``, ``category``,
        ``spearman_r``, ``rf_importance``, sorted by ``rf_importance``
        descending. Returns an empty frame with the right columns when
        no sysctl parameter varies in ``df``.
    """
    pd = _require_pandas()
    rf_cls, _ = _require_sklearn()

    sysctl_cols = [c for c in _SYSCTL_COLUMNS if c in df.columns]
    sysctl_cols = [c for c in sysctl_cols if len(df[c].unique()) > 1]

    empty = pd.DataFrame(
        columns=["param", "category", "spearman_r", "rf_importance"],
    )
    if not sysctl_cols:
        return empty

    # Target-column variance is required for both correlation and RF
    # fitting to be meaningful; constant or all-NaN targets yield no
    # information and would make the RF regressor fail.
    if target not in df.columns:
        return empty
    target_series = pd.to_numeric(df[target], errors="coerce")
    finite_mask = target_series.notna()
    if int(finite_mask.sum()) < _MIN_SPEARMAN_SAMPLES:
        return empty
    fit_df = df.loc[finite_mask].copy()
    fit_df[target] = target_series.loc[finite_mask]
    if fit_df[target].nunique() < 2:  # noqa: PLR2004 - RF needs variance
        return empty

    spearman = _spearman_scores(fit_df, sysctl_cols, target, pd)
    rf_imp = _rf_importance_scores(fit_df, sysctl_cols, target, rf_cls, pd)

    records = [
        {
            "param": col,
            "category": PARAM_TO_CATEGORY.get(col, "unknown"),
            "spearman_r": spearman.get(col, 0.0),
            "rf_importance": rf_imp.get(col, 0.0),
        }
        for col in sysctl_cols
    ]
    return (
        pd
        .DataFrame(records)
        .sort_values("rf_importance", ascending=False)
        .reset_index(drop=True)
    )


def _spearman_scores(
    df: pd.DataFrame,
    sysctl_cols: list[str],
    target: str,
    pd_module: Any,  # noqa: ANN401
) -> dict[str, float]:
    """Return Spearman rank correlations for each column in ``sysctl_cols``.

    Args:
        df: Frame produced by :func:`trials_to_dataframe`.
        sysctl_cols: Columns to score.
        target: Target metric column name.
        pd_module: The already-imported ``pandas`` module.

    Returns:
        A dict mapping column name to Spearman ``r``. Columns with
        fewer than :data:`_MIN_SPEARMAN_SAMPLES` non-null values map to
        ``0.0``.
    """
    scores: dict[str, float] = {}
    for col in sysctl_cols:
        series = pd_module.to_numeric(df[col], errors="coerce")
        if series.notna().sum() >= _MIN_SPEARMAN_SAMPLES:
            scores[col] = series.corr(df[target], method="spearman")
        else:
            scores[col] = 0.0
    return scores


def _rf_importance_scores(
    df: pd.DataFrame,
    sysctl_cols: list[str],
    target: str,
    rf_cls: Any,  # noqa: ANN401
    pd_module: Any,  # noqa: ANN401
) -> dict[str, float]:
    """Fit a random forest regressor and return per-feature importances.

    Args:
        df: Frame produced by :func:`trials_to_dataframe`.
        sysctl_cols: Feature columns; each is coerced to numeric and
            NaN-filled with ``0`` before fitting.
        target: Target metric column name.
        rf_cls: :class:`sklearn.ensemble.RandomForestRegressor` class.
        pd_module: The already-imported ``pandas`` module.

    Returns:
        A dict mapping column name to its normalized Random Forest
        feature importance.
    """
    features = df[sysctl_cols].apply(pd_module.to_numeric, errors="coerce").fillna(0)
    # Writable copy: pandas may hand back a read-only view here, and we
    # mutate below. Scale y to unit std so MSE-based split improvements
    # stay above sklearn's tree-splitter epsilon (~2.2e-15). Positive
    # scaling preserves both ranking and the normalized per-feature shares.
    y = df[target].to_numpy(dtype=float, copy=True)
    std = y.std()
    if std > 0:
        y /= std
    rf = rf_cls(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(features, y)
    return dict(zip(sysctl_cols, rf.feature_importances_, strict=True))


def pareto_recommendation_rows(  # noqa: PLR0914 - one-pass build over many intermediate frames
    trials: list[TrialResult],
    hardware_class: str,
    topology: str | None = None,
    *,
    objectives: list[ParetoObjective] | None = None,
    weights: dict[str, float] | None = None,
    memory_cost_weight: float | None = None,
) -> list[dict[str, Any]]:
    """Return every Pareto-frontier row for a class, scored and sorted.

    The full-frontier counterpart to :func:`recommend_configs`.
    Aggregates verification samples back into their parents, computes
    the Pareto frontier, scores each row via
    :func:`kube_autotuner.scoring.score_rows`, and sorts by
    ``(score desc, trial_id asc)`` with a stable mergesort.

    Every unmeasured metric is returned as ``None`` rather than
    ``float('nan')`` or a pandas sentinel, so the result is safe to
    serialize with :func:`json.dumps` without needing
    ``allow_nan=False``. The browser-side interactive report relies on
    that: ``JSON.parse`` rejects the ``NaN`` / ``Infinity`` tokens that
    Python emits by default.

    Args:
        trials: Input trial records (any number of hardware classes).
        hardware_class: Hardware-class label to filter on.
        topology: Optional topology filter.
        objectives: Pareto objectives driving frontier selection and
            scoring. Defaults to the seven built-in metrics.
        weights: Per-metric non-negative multipliers applied to both
            maximize and minimize objectives. Missing maximize-metric
            keys default to ``1.0`` (full +norm contribution); missing
            minimize-metric keys default to ``0.0``.
        memory_cost_weight: Non-negative multiplier on the static
            memory-footprint term fed into
            :func:`kube_autotuner.scoring.score_rows`. ``None`` picks
            up the :class:`ObjectivesSection` default (``0.1``); set
            ``0.0`` to disable.

    Lazy-imports ``pandas`` and raises :exc:`RuntimeError` with the
    ``uv sync --group analysis`` hint when the group is missing.

    Returns:
        One dict per Pareto-frontier row in rank order. Each dict
        contains ``trial_id``, ``sysctl_values``, every key in
        :data:`~kube_autotuner.scoring.METRIC_TO_DF_COLUMN` (value
        ``None`` for unmeasured metrics), a ``memory_cost`` float
        estimating total kernel/CNI bytes, and an *unrounded* ``score``
        float. Callers that need a fixed-precision score round at the
        surface; rounding in the helper would erase the mergesort
        tiebreak stability that :mod:`kube_autotuner.progress` relies
        on. Returns an empty list when no trials match or the frontier
        is empty.
    """
    pd = _require_pandas()
    from kube_autotuner.experiment import ObjectivesSection  # noqa: PLC0415

    filtered = [
        t
        for t in trials
        if t.node_pair.hardware_class == hardware_class
        and (topology is None or t.topology == topology)
    ]
    if not filtered:
        return []

    defaults = ObjectivesSection()
    if objectives is None:
        objectives = defaults.pareto
    if weights is None:
        weights = defaults.recommendation_weights
    if memory_cost_weight is None:
        memory_cost_weight = defaults.memory_cost_weight

    agg_rows = aggregate_verification(filtered)
    if not agg_rows:
        return []

    agg_df = pd.DataFrame(agg_rows)
    for col in METRIC_TO_DF_COLUMN.values():
        if col in agg_df.columns:
            agg_df[col] = pd.to_numeric(agg_df[col], errors="coerce")

    tuple_objectives: list[tuple[str, str]] = [
        (METRIC_TO_DF_COLUMN[obj.metric], obj.direction) for obj in objectives
    ]
    # Suppress the log here so the CLI's direct pareto_front call is
    # the single source of truth for "excluded" INFO lines.
    tuple_objectives = _objectives_with_data(agg_df, tuple_objectives, log=False)
    front = pareto_front(agg_df, objectives=tuple_objectives)
    if front.empty:
        return []

    # Memory cost: computed once per primary trial off the filtered list
    # (aggregate_verification drops sysctl_values, so we keep a parent-
    # keyed lookup aligned with the aggregation key
    # ``parent_trial_id or trial_id``).
    cost_by_trial: dict[str, float] = {
        t.trial_id: config_memory_cost(t.sysctl_values, PARAM_SPACE) for t in filtered
    }
    memory_costs = [cost_by_trial.get(tid, 0.0) for tid in front["trial_id"].tolist()]
    records = front[list(METRIC_TO_DF_COLUMN.values())].to_dict(orient="records")
    raw_scores = score_rows(
        records,
        objectives,
        weights,
        memory_costs=memory_costs,
        memory_cost_weight=memory_cost_weight,
    )
    front = front.assign(score=raw_scores, memory_cost=memory_costs)
    front = front.sort_values(
        by=["score", "trial_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    def _maybe(row: pd.Series, col: str) -> float | None:
        v = row[col]
        return None if pd.isna(v) else float(v)

    rows: list[dict[str, Any]] = []
    for _, row in front.iterrows():
        # row["trial_id"] is the aggregation key (``parent_trial_id
        # or trial_id``) so it always resolves to the primary trial,
        # not a verification repeat.
        trial = next(t for t in filtered if t.trial_id == row["trial_id"])
        rows.append(
            {
                "trial_id": row["trial_id"],
                "sysctl_values": trial.sysctl_values,
                "mean_tcp_throughput": _maybe(row, "mean_tcp_throughput"),
                "mean_udp_throughput": _maybe(row, "mean_udp_throughput"),
                "tcp_retransmit_rate": _maybe(row, "tcp_retransmit_rate"),
                "udp_loss_rate": _maybe(row, "udp_loss_rate"),
                "mean_udp_jitter": _maybe(row, "mean_udp_jitter"),
                "mean_rps": _maybe(row, "mean_rps"),
                "mean_latency_p50": _maybe(row, "mean_latency_p50"),
                "mean_latency_p90": _maybe(row, "mean_latency_p90"),
                "mean_latency_p99": _maybe(row, "mean_latency_p99"),
                "memory_cost": float(row["memory_cost"]),
                "score": float(row["score"]),
            },
        )
    return rows


def recommend_configs(
    trials: list[TrialResult],
    hardware_class: str,
    n: int = 3,
    topology: str | None = None,
    *,
    objectives: list[ParetoObjective] | None = None,
    weights: dict[str, float] | None = None,
    memory_cost_weight: float | None = None,
) -> list[dict[str, Any]]:
    """Return the top ``n`` recommended sysctl configurations for a class.

    Thin wrapper over :func:`pareto_recommendation_rows`: slices the
    first ``n`` rows, prepends a ``rank`` field, and rounds ``score``
    to 4 decimals for display. The underlying sort uses the unrounded
    score so near-ties that round to equal 4-decimal values still sort
    by their true ordering before falling back to the ``trial_id asc``
    secondary key.

    Default weights come from
    :class:`~kube_autotuner.experiment.ObjectivesSection` --
    ``{tcp_retransmit_rate: 0.3, udp_loss_rate: 0.3, udp_jitter: 0.1,
    latency_p90: 0.1, latency_p99: 0.15}`` at the time of writing --
    so the live ``Best so far`` panel and this recommendation output
    rank trials identically.

    Args:
        trials: Input trial records (any number of hardware classes).
        hardware_class: Hardware-class label to filter on.
        n: Maximum number of recommendations to return.
        topology: Optional topology filter.
        objectives: Pareto objectives driving both frontier selection
            and scoring. Defaults to the seven built-in metrics.
        weights: Per-metric non-negative multipliers applied to both
            maximize and minimize objectives. Missing maximize-metric
            keys default to ``1.0``; missing minimize-metric keys
            default to ``0.0`` (i.e. they do not influence the score).
        memory_cost_weight: Non-negative multiplier on the static
            memory-footprint term. ``None`` picks up the
            :class:`ObjectivesSection` default (``0.1``); set
            ``0.0`` to disable.

    Lazy-imports ``pandas`` (via :func:`trials_to_dataframe`) and
    raises :exc:`RuntimeError` with the ``uv sync --group analysis``
    hint when the group is missing.

    Returns:
        A list of recommendation dicts. Each dict always contains the
        same keys (``rank``, ``trial_id``, ``sysctl_values``, the nine
        base metric names -- ``mean_tcp_throughput``,
        ``mean_udp_throughput``, ``tcp_retransmit_rate``,
        ``udp_loss_rate``, ``mean_udp_jitter``, ``mean_rps``,
        ``mean_latency_p50``, ``mean_latency_p90``,
        ``mean_latency_p99`` -- and a ``score``). A metric value is
        ``None`` when the trial produced no reading for it. Returns
        an empty list when no trials match.
    """
    rows = pareto_recommendation_rows(
        trials,
        hardware_class,
        topology,
        objectives=objectives,
        weights=weights,
        memory_cost_weight=memory_cost_weight,
    )
    results: list[dict[str, Any]] = []
    for rank, row in enumerate(rows[:n], start=1):
        results.append(
            {
                "rank": rank,
                "trial_id": row["trial_id"],
                "sysctl_values": row["sysctl_values"],
                "mean_tcp_throughput": row["mean_tcp_throughput"],
                "mean_udp_throughput": row["mean_udp_throughput"],
                "tcp_retransmit_rate": row["tcp_retransmit_rate"],
                "udp_loss_rate": row["udp_loss_rate"],
                "mean_udp_jitter": row["mean_udp_jitter"],
                "mean_rps": row["mean_rps"],
                "mean_latency_p50": row["mean_latency_p50"],
                "mean_latency_p90": row["mean_latency_p90"],
                "mean_latency_p99": row["mean_latency_p99"],
                "memory_cost": row["memory_cost"],
                "score": round(row["score"], 4),
            },
        )
    return results


def host_state_series(
    trials: list[TrialResult],
    hardware_class: str,
    topology: str | None,
) -> dict[str, Any] | None:
    """Shape per-trial host-state snapshots for the analysis report.

    Flattens :class:`~kube_autotuner.models.HostStateSnapshot` entries
    from every matching :class:`~kube_autotuner.models.TrialResult`
    into a single timestamp-ordered list so the browser-side chart
    can render one Plotly trace per metric across the whole tuning
    session. The shape is:

    .. code-block:: python

        {
            "metrics": [sorted union of metric keys],
            "points": [
                {
                    "timestamp": str,  # snap.timestamp.isoformat()
                    "trial_id": str,
                    "sysctl_hash": str,
                    "iteration": int | None,
                    "phase": str,
                    "metrics": {str: int},
                },
                ...
            ],
        }

    Points are sorted on the raw ``datetime`` before stringification
    so the ordering is correct even if a replayed results JSON carries
    a naive timestamp. ``HostStateSnapshot.errors`` is intentionally
    dropped -- it would bloat the embedded JSON without aiding the
    visualization.

    Args:
        trials: The trial set to scan (caller passes the pre-filtered
            hardware-class list in practice, but this function also
            filters defensively so stray topology-mixed inputs do not
            silently leak).
        hardware_class: The hardware-class label to keep.
        topology: When set, keep only trials whose ``topology``
            matches; when ``None``, keep all topologies.

    Returns:
        The payload described above, or ``None`` when no trial in
        the filtered set carries any host-state snapshots, or when
        every snapshot's ``metrics`` dict is empty (so the metric
        union is empty and the multi-select would render with zero
        options).
    """
    filtered = [
        t
        for t in trials
        if t.node_pair.hardware_class == hardware_class
        and (topology is None or t.topology == topology)
        and t.host_state_snapshots
    ]
    if not filtered:
        return None

    metric_union: set[str] = set()
    tagged: list[tuple[datetime, dict[str, Any]]] = []
    for trial in filtered:
        for snap in trial.host_state_snapshots:
            metrics: dict[str, int] = dict(snap.metrics)
            metric_union.update(metrics.keys())
            # Normalize naive timestamps to UTC so a replayed results
            # JSON that mixes aware and naive entries does not raise
            # TypeError at sort time.
            ts = snap.timestamp
            key = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
            tagged.append(
                (
                    key,
                    {
                        "timestamp": snap.timestamp.isoformat(),
                        "trial_id": trial.trial_id,
                        "sysctl_hash": trial.sysctl_hash,
                        "iteration": snap.iteration,
                        "phase": snap.phase,
                        "metrics": metrics,
                    },
                ),
            )
    if not metric_union:
        return None
    points = [p for _, p in sorted(tagged, key=itemgetter(0))]
    return {
        "metrics": sorted(metric_union),
        "points": points,
    }


def baseline_comparison(  # noqa: PLR0914 - one-pass aggregation over many intermediates
    trials: list[TrialResult],
    objectives: list[dict[str, str]],
    top_row: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Return a per-objective delta vs the :data:`RECOMMENDED_DEFAULTS` baseline.

    The baseline is the set of primary trials whose sysctl hash matches
    :data:`kube_autotuner.sysctl.params.RECOMMENDED_DEFAULTS`. Matches
    are aggregated with verification children via
    :func:`kube_autotuner.scoring.aggregate_verification`; the
    per-metric mean across matches forms the baseline value. Returns
    ``None`` when no primary trial matches the defaults.

    Args:
        trials: The per-hardware-class trial set.
        objectives: Pareto objectives as
            ``ParetoObjective.model_dump(mode="json")`` dicts (keys
            ``metric`` and ``direction``).
        top_row: The top recommendation row (first row returned by
            :func:`pareto_recommendation_rows`), or ``None`` when no
            recommendation is available. Metric keys are DataFrame
            column names.

    Returns:
        One dict per objective with ``metric``, ``direction``,
        ``baseline``, ``recommended``, ``abs_delta``, ``pct_delta``;
        or ``None`` when the baseline card should be suppressed
        entirely (no matching trial).
    """
    defaults_hash = compute_sysctl_hash(RECOMMENDED_DEFAULTS)
    primary_matches = [
        t
        for t in trials
        if is_primary(t) and compute_sysctl_hash(t.sysctl_values) == defaults_hash
    ]
    if not primary_matches:
        return None
    match_ids = {t.trial_id for t in primary_matches}
    related = [
        t
        for t in trials
        if (is_primary(t) and t.trial_id in match_ids)
        or (not is_primary(t) and t.parent_trial_id in match_ids)
    ]
    agg_rows = aggregate_verification(related)
    if not agg_rows:
        return None

    per_metric_baseline: dict[str, float | None] = {}
    for col in METRIC_TO_DF_COLUMN.values():
        raw_values: list[float] = []
        for row in agg_rows:
            v = row.get(col)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                raw_values.append(float(v))
        per_metric_baseline[col] = (
            sum(raw_values) / len(raw_values) if raw_values else None
        )

    out: list[dict[str, Any]] = []
    for obj in objectives:
        metric = obj["metric"]
        direction = obj["direction"]
        col = METRIC_TO_DF_COLUMN[metric]
        baseline = per_metric_baseline.get(col)
        recommended = _finite_or_none(
            top_row.get(col) if top_row is not None else None,
        )
        if baseline is None or recommended is None:
            abs_delta: float | None = None
            pct_delta: float | None = None
        else:
            abs_delta = recommended - baseline
            pct_delta = (
                abs_delta / baseline if abs(baseline) >= _ZERO_MEAN_EPSILON else None
            )
        out.append(
            {
                "metric": metric,
                "direction": direction,
                "baseline": _finite_or_none(baseline),
                "recommended": recommended,
                "abs_delta": _finite_or_none(abs_delta),
                "pct_delta": _finite_or_none(pct_delta),
            },
        )
    return out


def verification_stats(
    trials: list[TrialResult],
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Return per-parent mean/stdev/cv across verification children.

    Groups verification rows by ``parent_trial_id`` and projects each
    child's per-trial metric means (from
    :meth:`TrialResult.mean_tcp_throughput` and friends). Per-metric
    ``mean`` is the mean across finite samples, ``stdev`` is
    :func:`statistics.stdev` across finite samples (skipped when
    fewer than two remain), and ``cv`` is ``stdev / abs(mean)`` with
    ``None`` when ``abs(mean) < 1e-12``.

    Args:
        trials: The combined primary + verification population.

    Returns:
        A dict keyed by parent trial id, mapping to a dict keyed by
        metric column name, mapping to a ``{mean, stdev, cv}`` dict.
        Parents with no qualifying metric are omitted.
    """
    groups: dict[str, list[TrialResult]] = defaultdict(list)
    for t in trials:
        if is_primary(t):
            continue
        if t.parent_trial_id is None:
            continue
        groups[t.parent_trial_id].append(t)

    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for parent_id, children in groups.items():
        per_metric: dict[str, dict[str, float | None]] = {}
        for col in METRIC_TO_DF_COLUMN.values():
            samples = [_trial_metric_value(c, col) for c in children]
            finite = [s for s in samples if s is not None]
            if len(finite) < 2:  # noqa: PLR2004 - stdev needs >= 2
                continue
            mean = statistics.mean(finite)
            stdev = statistics.stdev(finite)
            if not math.isfinite(mean) or not math.isfinite(stdev):
                continue
            if abs(mean) < _ZERO_MEAN_EPSILON:
                cv: float | None = None
            else:
                cv = stdev / abs(mean)
                if not math.isfinite(cv):
                    cv = None
            per_metric[col] = {
                "mean": _finite_or_none(mean),
                "stdev": _finite_or_none(stdev),
                "cv": cv,
            }
        if per_metric:
            out[parent_id] = per_metric
    return out


def _per_iteration_cells(  # noqa: PLR0914 - one-pass projection over many independent metrics
    t: TrialResult,
) -> dict[int, dict[str, float | None]]:
    """Return one cell-dict per observed iteration index for ``t``.

    Walks ``t.results`` (iperf3) and ``t.latency_results`` (fortio)
    once each, bucketing by the record's ``iteration`` field. Per-metric
    aggregation mirrors the canonical
    :class:`~kube_autotuner.models.TrialResult` accessors so a mean of
    the per-iteration values equals the parent's
    :func:`~kube_autotuner.scoring._per_trial_metric_means` value:

    * Throughput / RPS: summed across clients within an iteration.
    * Jitter / latency percentiles: meaned across clients.
    * ``tcp_retransmit_rate`` / ``udp_loss_rate``: per-iteration
      ratio-of-sums with the same drop conditions as
      :func:`~kube_autotuner.models.tcp_retransmit_rate_by_iteration`
      and
      :func:`~kube_autotuner.models.udp_loss_rate_by_iteration`.

    Iterations contribute one entry to the returned mapping when at
    least one record (iperf3 or fortio) carries that iteration index;
    metrics absent at that iteration are emitted as ``None`` so the
    iteration index is preserved without shifting other columns.

    Args:
        t: Trial whose records should be projected.

    Returns:
        A dict keyed by the original ``iteration`` index, mapping to a
        cell dict whose keys are the values of
        :data:`~kube_autotuner.scoring.METRIC_TO_DF_COLUMN`. Each numeric
        cell is a finite ``float`` or ``None``.
    """
    bench_by_iter: dict[int, list[Any]] = defaultdict(list)
    for r in t.results:
        bench_by_iter[r.iteration].append(r)
    latency_by_iter: dict[int, list[Any]] = defaultdict(list)
    for r in t.latency_results:
        latency_by_iter[r.iteration].append(r)

    out: dict[int, dict[str, float | None]] = {}
    for it in sorted(set(bench_by_iter) | set(latency_by_iter)):
        bench = bench_by_iter.get(it, [])
        latency = latency_by_iter.get(it, [])

        tcp_records = [r for r in bench if r.mode == "tcp"]
        udp_records = [r for r in bench if r.mode == "udp"]

        tcp_throughput: float | None = (
            sum(r.bits_per_second for r in tcp_records) if tcp_records else None
        )
        udp_throughput: float | None = (
            sum(r.bits_per_second for r in udp_records) if udp_records else None
        )

        retx_total = sum(r.retransmits for r in bench if r.retransmits is not None)
        bytes_total = sum(
            r.bytes_sent for r in bench if r.bytes_sent is not None and r.bytes_sent > 0
        )
        saw_retx = any(r.retransmits is not None for r in bench)
        tcp_retx_rate: float | None = (
            retx_total * 1e9 / bytes_total if saw_retx and bytes_total > 0 else None
        )

        lost_total = sum(r.lost_packets for r in bench if r.lost_packets is not None)
        packets_total = sum(
            r.packets for r in bench if r.packets is not None and r.packets > 0
        )
        saw_lost = any(r.lost_packets is not None for r in bench)
        udp_loss: float | None = (
            lost_total / packets_total if saw_lost and packets_total > 0 else None
        )

        jitter_vals = [r.jitter for r in bench if r.jitter is not None]
        jitter_mean: float | None = (
            sum(jitter_vals) / len(jitter_vals) if jitter_vals else None
        )

        saturation = [r for r in latency if r.workload == "saturation"]
        rps_sum: float | None = sum(r.rps for r in saturation) if saturation else None

        fixed = [r for r in latency if r.workload == "fixed_qps"]

        def _latency_mean(records: list[Any], attr: str) -> float | None:
            vals = [getattr(r, attr) for r in records if getattr(r, attr) is not None]
            return sum(vals) / len(vals) if vals else None

        out[it] = {
            "mean_tcp_throughput": _finite_or_none(tcp_throughput),
            "mean_udp_throughput": _finite_or_none(udp_throughput),
            "tcp_retransmit_rate": _finite_or_none(tcp_retx_rate),
            "udp_loss_rate": _finite_or_none(udp_loss),
            "mean_udp_jitter": _finite_or_none(jitter_mean),
            "mean_rps": _finite_or_none(rps_sum),
            "mean_latency_p50": _finite_or_none(_latency_mean(fixed, "latency_p50")),
            "mean_latency_p90": _finite_or_none(_latency_mean(fixed, "latency_p90")),
            "mean_latency_p99": _finite_or_none(_latency_mean(fixed, "latency_p99")),
        }
    return out


def per_iteration_samples(
    trials: list[TrialResult],
) -> dict[str, list[dict[str, int | str | float | None]]]:
    """Return per-iteration measurement rows grouped by recommendation key.

    Group key matches :func:`~kube_autotuner.scoring.aggregate_verification`:
    ``parent_trial_id or trial_id``. Within each group, every contributing
    :class:`~kube_autotuner.models.TrialResult` emits one row per
    iteration index observed in its records. The original
    ``BenchmarkResult.iteration`` / ``LatencyResult.iteration`` values
    are preserved verbatim so a row can be correlated with the on-disk
    record. Numeric cells are finite or ``None`` -- never ``nan`` -- so
    the payload survives ``json.dumps(allow_nan=False)`` in
    ``render._embed_json``.

    Args:
        trials: The combined primary + verification population for one
            hardware-class section.

    Returns:
        A dict keyed by recommendation/parent ``trial_id``, mapping to
        a list of per-iteration row dicts. Each row carries the original
        ``iteration`` index, the source ``trial_id``, and one
        finite-or-``None`` cell per metric column in
        :data:`~kube_autotuner.scoring.METRIC_TO_DF_COLUMN`.
    """
    groups: dict[str, list[TrialResult]] = defaultdict(list)
    order: list[str] = []
    for t in trials:
        key = t.parent_trial_id or t.trial_id
        if key not in groups:
            order.append(key)
        groups[key].append(t)

    out: dict[str, list[dict[str, int | str | float | None]]] = {}
    for key in order:
        members = sorted(groups[key], key=lambda x: (x.created_at, x.trial_id))
        rows: list[dict[str, int | str | float | None]] = []
        for trial in members:
            cells = _per_iteration_cells(trial)
            for it in sorted(cells):
                row: dict[str, int | str | float | None] = {
                    "iteration": it,
                    "trial_id": trial.trial_id,
                }
                row.update(cells[it])
                rows.append(row)
        if rows:
            out[key] = rows
    return out


def stability_badge(
    verif_row: dict[str, dict[str, float | None]] | None,
) -> str:
    """Classify a recommendation row's stability from its verification CVs.

    Thresholds match the plan's interpretation aid:

    * ``"green"``   — max CV < 5%
    * ``"amber"``   — 5% ≤ max CV < 15%
    * ``"red"``     — max CV ≥ 15%
    * ``"unverified"`` — no verification children, or every CV is
      ``None`` (zero-mean guard).

    Args:
        verif_row: One value of :func:`verification_stats`, i.e. the
            per-metric ``{mean, stdev, cv}`` dict for a single parent;
            ``None`` when the parent has no verification children.

    Returns:
        The badge label.
    """
    if not verif_row:
        return "unverified"
    cvs = [
        entry["cv"]
        for entry in verif_row.values()
        if entry.get("cv") is not None and math.isfinite(float(entry["cv"] or 0.0))
    ]
    if not cvs:
        return "unverified"
    max_cv = max(abs(cv) for cv in cvs if cv is not None)
    if max_cv < _STABILITY_GREEN_MAX:
        return "green"
    if max_cv < _STABILITY_AMBER_MAX:
        return "amber"
    return "red"


def trajectory_rows(
    trials: list[TrialResult],
    objectives: list[dict[str, str]],
    resume_metadata: ResumeMetadata | None,
) -> list[dict[str, Any]]:
    """Return a per-primary-trial running-best trajectory.

    Primary trials (``is_primary``) are ordered by ``created_at`` and
    emitted one row at a time. Each row carries the trial's phase plus
    per-objective ``<col>_best_so_far`` values updated monotonically
    (``cummax`` for ``maximize``, ``cummin`` for ``minimize``).
    ``t.phase`` is used verbatim with ``None`` bucketed as
    ``"unknown"``.

    Args:
        trials: The full trial population (any hardware class/phase).
        objectives: Pareto objectives as
            ``ParetoObjective.model_dump(mode="json")`` dicts.
        resume_metadata: Unused; kept for call-site symmetry with
            :func:`section_metadata`.

    Returns:
        One row per primary trial, in ``created_at`` order.
    """
    _ = resume_metadata
    primaries = sorted(
        (t for t in trials if is_primary(t)),
        key=lambda t: t.created_at,
    )
    bests: dict[str, float | None] = {obj["metric"]: None for obj in objectives}
    rows: list[dict[str, Any]] = []
    for index, t in enumerate(primaries):
        phase = t.phase if t.phase is not None else "unknown"
        row: dict[str, Any] = {
            "trial_index": index,
            "trial_id": t.trial_id,
            "created_at_iso": t.created_at.isoformat(),
            "phase_effective": phase,
        }
        for obj in objectives:
            metric = obj["metric"]
            direction = obj["direction"]
            col = METRIC_TO_DF_COLUMN[metric]
            v = _trial_metric_value(t, col)
            current = bests[metric]
            if v is not None:
                if current is None:
                    bests[metric] = v
                elif direction == "maximize":
                    bests[metric] = max(current, v)
                else:
                    bests[metric] = min(current, v)
            row[f"{col}_best_so_far"] = _finite_or_none(bests[metric])
        rows.append(row)
    return rows


def section_metadata(
    trials: list[TrialResult],
    resume_metadata: ResumeMetadata | None,
) -> dict[str, Any]:
    """Return an experimental-setup summary for a hardware-class section.

    Fields that agree across every primary trial render that value;
    otherwise render the literal string ``"mixed"``. An all-empty
    ``kernel_version`` is coerced to ``None`` so the header can omit
    the field rather than print a blank. ``stages`` is rendered as a
    sorted list when agreed.

    Phase counts include every trial (primary + verification);
    primary phases use ``t.phase`` verbatim (``None`` → ``"unknown"``).

    Args:
        trials: The per-hardware-class trial set.
        resume_metadata: Source for the iperf/fortio durations;
            ``None`` leaves both ``iperf_duration`` and
            ``fortio_duration`` ``None``.

    Returns:
        A dict with keys ``trial_count``, ``phase_counts``,
        ``kernel_version``, ``iperf_duration``, ``fortio_duration``,
        ``iterations``, ``stages``, ``first_created_at_iso``,
        ``last_created_at_iso``.
    """
    primaries = sorted(
        (t for t in trials if is_primary(t)),
        key=lambda t: t.created_at,
    )
    phase_counts: dict[str, int] = {
        "sobol": 0,
        "bayesian": 0,
        "verification": 0,
        "unknown": 0,
    }
    for t in primaries:
        label = t.phase if t.phase is not None else "unknown"
        phase_counts[label] = phase_counts.get(label, 0) + 1
    phase_counts["verification"] += sum(1 for t in trials if not is_primary(t))

    def _unique_or_mixed(values: list[Any]) -> Any:  # noqa: ANN401
        unique = []
        seen: set[Any] = set()
        for v in values:
            marker = v
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(v)
        if len(unique) == 1:
            return unique[0]
        return "mixed"

    kernel_vals = [t.kernel_version for t in primaries]
    resolved_kernel = _unique_or_mixed(kernel_vals)
    kernel_version: str | None = resolved_kernel or None

    iperf_duration: int | None = (
        resume_metadata.iperf.duration if resume_metadata is not None else None
    )
    fortio_duration: int | None = (
        resume_metadata.fortio.duration if resume_metadata is not None else None
    )
    iterations_vals = [t.config.iterations for t in primaries]
    iterations = _unique_or_mixed(iterations_vals)

    stages_vals = [frozenset(t.config.stages) for t in primaries]
    stages_raw = _unique_or_mixed(stages_vals)
    stages: list[str] | str = (
        sorted(stages_raw) if isinstance(stages_raw, frozenset) else stages_raw
    )

    if primaries:
        first_iso: str | None = min(t.created_at for t in primaries).isoformat()
        last_iso: str | None = max(t.created_at for t in primaries).isoformat()
    else:
        first_iso = None
        last_iso = None

    return {
        "trial_count": len(primaries),
        "phase_counts": phase_counts,
        "kernel_version": kernel_version,
        "iperf_duration": iperf_duration,
        "fortio_duration": fortio_duration,
        "iterations": iterations,
        "stages": stages,
        "first_created_at_iso": first_iso,
        "last_created_at_iso": last_iso,
    }


def sysctl_correlation_matrix(
    df: pd.DataFrame,
    importance_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """Return a pairwise Spearman matrix over the importance-block params.

    Restricts the column set to the union of params mentioned in any
    ``importance_by_target`` frame, drops constant columns, and
    computes pairwise Spearman. Cells with fewer than
    :data:`_MIN_SPEARMAN_SAMPLES` non-null paired samples are marked
    as ``NaN`` so the serialization layer renders them as ``None``.

    Args:
        df: Per-trial frame from :func:`trials_to_dataframe` with
            sysctl columns already numerically encoded.
        importance_frames: Per-target importance frames; a column is
            included when any frame lists it.

    Returns:
        A square :class:`pandas.DataFrame` indexed and columned by
        parameter name, or ``None`` when fewer than two qualifying
        columns remain after constant-column pruning.
    """
    pd = _require_pandas()

    wanted: set[str] = set()
    for frame in importance_frames.values():
        if frame is None or frame.empty:
            continue
        wanted.update(frame["param"].astype(str).tolist())
    if len(wanted) < 2:  # noqa: PLR2004 - matrix needs at least 2 axes
        return None
    present = [c for c in wanted if c in df.columns]
    if len(present) < 2:  # noqa: PLR2004 - matrix needs at least 2 axes
        return None

    numeric = df[present].apply(pd.to_numeric, errors="coerce")
    varied = [c for c in present if len(numeric[c].dropna().unique()) > 1]
    if len(varied) < 2:  # noqa: PLR2004 - matrix needs at least 2 axes
        return None

    matrix = pd.DataFrame(
        float("nan"),
        index=varied,
        columns=varied,
        dtype=float,
    )
    for i, a in enumerate(varied):
        matrix.loc[a, a] = 1.0
        for b in varied[i + 1 :]:
            pair = numeric[[a, b]].dropna()
            if len(pair) < _MIN_SPEARMAN_SAMPLES:
                continue
            r = pair[a].corr(pair[b], method="spearman")
            if r is None or not math.isfinite(float(r)):
                continue
            matrix.loc[a, b] = float(r)
            matrix.loc[b, a] = float(r)
    return matrix


def host_state_issues(trials: list[TrialResult]) -> list[dict[str, Any]]:
    """Flatten :attr:`HostStateSnapshot.errors` across every trial.

    Error strings longer than :data:`_HOST_STATE_ERROR_MAX_LEN`
    characters are truncated with a trailing ellipsis so the rendered
    ``<details>`` does not blow up horizontally.

    Args:
        trials: The per-hardware-class trial set.

    Returns:
        One row per snapshot error, in trial + snapshot order. Empty
        list when no snapshot carries any errors.
    """
    out: list[dict[str, Any]] = []
    for t in trials:
        for snap in t.host_state_snapshots:
            for err in snap.errors:
                if len(err) > _HOST_STATE_ERROR_MAX_LEN:
                    text = err[:_HOST_STATE_ERROR_MAX_LEN] + "…"
                else:
                    text = err
                out.append(
                    {
                        "trial_id": t.trial_id,
                        "node": snap.node,
                        "phase": snap.phase,
                        "iteration": snap.iteration,
                        "error_text": text,
                    },
                )
    return out


def category_importance_rollup(
    importance_frames: dict[str, pd.DataFrame],
) -> dict[str, list[dict[str, Any]]]:
    """Sum Random Forest importances by category per target metric.

    Args:
        importance_frames: Per-target frames from
            :func:`parameter_importance`; each frame contains
            ``param``, ``category``, ``spearman_r``, ``rf_importance``.

    Returns:
        ``{target: [{category, rf_sum}, ...]}`` with categories sorted
        by ``rf_sum`` descending. Empty or ``None`` frames are
        skipped.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for target, frame in importance_frames.items():
        if frame is None or frame.empty:
            continue
        rollup = frame.groupby("category", sort=False)["rf_importance"].sum()
        pairs = [(str(cat), float(val)) for cat, val in rollup.items()]
        pairs.sort(key=lambda kv: (-kv[1], kv[0]))
        out[target] = [
            {"category": cat, "rf_sum": _finite_or_none(val) or 0.0}
            for cat, val in pairs
        ]
    return out
