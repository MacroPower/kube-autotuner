"""Offline analysis of trial data: Pareto frontier, importance, recommendations.

The heavy dependencies -- ``pandas``, ``numpy``, ``scikit-learn`` -- live in
the optional ``analysis`` dependency group. They are lazy-imported inside
function bodies so that ``import kube_autotuner.analysis`` succeeds under
the base ``dev`` sync (``task completions`` and the analyze Typer command
both rely on this). Three rules enforce the discipline:

* ``from __future__ import annotations`` at the top keeps every
  signature a string, so ``-> pd.DataFrame`` does not trigger an eager
  import.
* Annotation-only symbols live under ``if TYPE_CHECKING:``.
* Every runtime use imports inside the function body, wrapped in
  ``try/except ImportError`` and raising :class:`RuntimeError` with the
  ``uv sync --group analysis`` hint.
"""

from __future__ import annotations

from datetime import UTC
import logging
from operator import itemgetter
from typing import TYPE_CHECKING, Any

from kube_autotuner.scoring import (
    METRIC_TO_DF_COLUMN,
    aggregate_verification,
    config_memory_cost,
    score_rows,
)
from kube_autotuner.sysctl.params import PARAM_SPACE, PARAM_TO_CATEGORY

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    from kube_autotuner.experiment import ParetoObjective
    from kube_autotuner.models import TrialResult

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
        for key in _SYSCTL_COLUMNS:
            row[key] = t.sysctl_values.get(key)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=_FRAME_BASE_COLUMNS + _SYSCTL_COLUMNS), {}

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
    y = df[target].to_numpy()
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
