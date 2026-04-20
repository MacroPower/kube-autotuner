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

from typing import TYPE_CHECKING, Any

from kube_autotuner.sysctl.params import PARAM_SPACE, PARAM_TO_CATEGORY

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    from kube_autotuner.models import TrialResult

_ANALYSIS_HINT = "install analysis group: uv sync --group analysis"

_PARAM_TYPE_LOOKUP: dict[str, str] = {p.name: p.param_type for p in PARAM_SPACE.params}
_SYSCTL_COLUMNS: list[str] = PARAM_SPACE.param_names()

DEFAULT_OBJECTIVES: list[tuple[str, str]] = [
    ("mean_throughput", "maximize"),
    ("mean_cpu", "minimize"),
    ("mean_memory", "minimize"),
    ("total_retransmits", "minimize"),
]

_FRAME_BASE_COLUMNS: list[str] = [
    "trial_id",
    "hardware_class",
    "topology",
    "source_zone",
    "target_zone",
    "mean_throughput",
    "mean_cpu",
    "mean_memory",
    "total_retransmits",
]

_MIN_SPEARMAN_SAMPLES = 3
_RECOMMEND_CPU_WEIGHT = 0.15
_RECOMMEND_MEM_WEIGHT = 0.15
_RECOMMEND_RETX_WEIGHT = 0.3


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
            "mean_throughput": t.mean_throughput(),
            "mean_cpu": t.mean_cpu(),
            "mean_memory": t.mean_memory(),
            "total_retransmits": t.total_retransmits(),
        }
        for key in _SYSCTL_COLUMNS:
            row[key] = t.sysctl_values.get(key)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=_FRAME_BASE_COLUMNS + _SYSCTL_COLUMNS), {}

    df = pd.DataFrame(rows)
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


def pareto_front(
    df: pd.DataFrame,
    objectives: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Return the non-dominated rows from ``df``.

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

    cols = [col for col, _ in objectives]
    vals = df[cols].to_numpy(dtype=float)

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

    return df.loc[~is_dominated].reset_index(drop=True)


def parameter_importance(
    df: pd.DataFrame,
    target: str = "mean_throughput",
) -> pd.DataFrame:
    """Rank sysctl parameters by importance for the ``target`` metric.

    Lazy-imports ``pandas`` and ``scikit-learn`` and raises
    :exc:`RuntimeError` with the ``uv sync --group analysis`` hint
    when the group is missing.

    Args:
        df: Frame produced by :func:`trials_to_dataframe`.
        target: Metric column name to score against (defaults to
            ``"mean_throughput"``).

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

    if not sysctl_cols:
        return pd.DataFrame(
            columns=["param", "category", "spearman_r", "rf_importance"],
        )

    spearman = _spearman_scores(df, sysctl_cols, target, pd)
    rf_imp = _rf_importance_scores(df, sysctl_cols, target, rf_cls, pd)

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


def recommend_configs(
    trials: list[TrialResult],
    hardware_class: str,
    n: int = 3,
    topology: str | None = None,
) -> list[dict[str, Any]]:
    """Return the top ``n`` recommended sysctl configurations for a class.

    The ranking picks from the Pareto frontier of the filtered trials
    and scores each candidate as
    ``tp_norm - 0.15 * cpu_norm - 0.15 * mem_norm - 0.3 * retx_norm``
    where each term is min-max normalized across the Pareto set. CPU
    and memory split the cost weight evenly because higher CPU is
    typically a symptom of better-tuned parameters enabling more
    useful work rather than a cost in its own right.

    Args:
        trials: Input trial records (any number of hardware classes).
        hardware_class: Hardware-class label to filter on.
        n: Maximum number of recommendations to return.
        topology: Optional topology filter.

    Lazy-imports ``pandas`` (via :func:`trials_to_dataframe`) and
    raises :exc:`RuntimeError` with the ``uv sync --group analysis``
    hint when the group is missing.

    Returns:
        A list of recommendation dicts, each containing ``rank``,
        ``trial_id``, ``sysctl_values``, the three metric values, and a
        ``score``. Returns an empty list when no trials match.
    """
    pd = _require_pandas()

    df, _ = trials_to_dataframe(
        trials,
        hardware_class=hardware_class,
        topology=topology,
    )
    if df.empty:
        return []

    front = pareto_front(df)
    if front.empty:
        return []

    def _norm(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series(0.5, index=series.index)
        return (series - lo) / (hi - lo)

    tp_norm = _norm(front["mean_throughput"])
    cpu_norm = _norm(front["mean_cpu"])
    mem_norm = _norm(front["mean_memory"])
    rt_norm = _norm(front["total_retransmits"])
    front = front.copy()
    front["score"] = (
        tp_norm
        - _RECOMMEND_CPU_WEIGHT * cpu_norm
        - _RECOMMEND_MEM_WEIGHT * mem_norm
        - _RECOMMEND_RETX_WEIGHT * rt_norm
    )
    front = front.sort_values("score", ascending=False).reset_index(drop=True)

    results: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(front.head(n).iterrows(), start=1):
        trial = next(t for t in trials if t.trial_id == row["trial_id"])
        results.append(
            {
                "rank": rank,
                "trial_id": row["trial_id"],
                "sysctl_values": trial.sysctl_values,
                "mean_throughput": row["mean_throughput"],
                "mean_cpu": row["mean_cpu"],
                "mean_memory": row["mean_memory"],
                "total_retransmits": int(row["total_retransmits"]),
                "score": round(row["score"], 4),
            },
        )
    return results
