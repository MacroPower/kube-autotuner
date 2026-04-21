"""Tests for :mod:`kube_autotuner.scoring`."""

from __future__ import annotations

import math

import pytest

from kube_autotuner.experiment import ParetoObjective
from kube_autotuner.scoring import METRIC_TO_DF_COLUMN, score_rows


def _col(metric: str) -> str:
    return METRIC_TO_DF_COLUMN[metric]


def test_score_rows_matches_recommend_configs_formula() -> None:
    """Hand-computed scores match the helper on a minimal three-row fixture.

    Throughput (maximize) normalizes to (1.0, 0.5, 0.0); CPU
    (minimize, weight 0.15) normalizes to (1.0, 0.5, 0.0); no other
    objectives participate. The expected scores are:

    - row A: 1.0 - 0.15 * 1.0 = 0.85
    - row B: 0.5 - 0.15 * 0.5 = 0.425
    - row C: 0.0 - 0.15 * 0.0 = 0.0
    """
    objectives = [
        ParetoObjective(metric="throughput", direction="maximize"),
        ParetoObjective(metric="cpu", direction="minimize"),
    ]
    rows = [
        {_col("throughput"): 2.0e9, _col("cpu"): 80.0},
        {_col("throughput"): 1.5e9, _col("cpu"): 50.0},
        {_col("throughput"): 1.0e9, _col("cpu"): 20.0},
    ]
    scores = score_rows(rows, objectives, {"cpu": 0.15})
    assert scores[0] == pytest.approx(0.85)
    assert scores[1] == pytest.approx(0.425)
    assert scores[2] == pytest.approx(0.0)


def test_score_rows_handles_nan_and_degenerate_columns() -> None:
    """NaN rows, constant columns, and all-NaN columns resolve to ``0.5``."""
    objectives = [
        ParetoObjective(metric="throughput", direction="maximize"),
        ParetoObjective(metric="cpu", direction="minimize"),
        ParetoObjective(metric="node_memory", direction="minimize"),
    ]
    rows = [
        # One all-NaN column (node_memory), one constant column (cpu),
        # one row with a NaN throughput.
        {
            _col("throughput"): 1.0e9,
            _col("cpu"): 10.0,
            _col("node_memory"): math.nan,
        },
        {
            _col("throughput"): math.nan,
            _col("cpu"): 10.0,
            _col("node_memory"): math.nan,
        },
        {
            _col("throughput"): 2.0e9,
            _col("cpu"): 10.0,
            _col("node_memory"): math.nan,
        },
    ]
    scores = score_rows(
        rows,
        objectives,
        {"cpu": 0.15, "node_memory": 0.15},
    )
    # throughput norms: row0 -> 0.0, row1 -> NaN-fallback 0.5, row2 -> 1.0.
    # cpu norms: all 0.5 (constant column); contribution = -0.15 * 0.5 = -0.075 each.
    # node_memory: all-NaN column -> 0.5 each; contribution = -0.15 * 0.5 = -0.075 each.
    assert scores[0] == pytest.approx(0.0 - 0.075 - 0.075)
    assert scores[1] == pytest.approx(0.5 - 0.075 - 0.075)
    assert scores[2] == pytest.approx(1.0 - 0.075 - 0.075)


def test_score_rows_accepts_none_and_float_nan() -> None:
    """``None`` and ``float("nan")`` yield identical scores.

    ``progress.py`` emits ``math.nan`` for missing readings;
    ``recommend_configs`` emits ``None`` (object-dtype NaN from
    pandas). The helper must treat both as NaN so the live panel
    and the post-hoc ranking agree on the same inputs.
    """
    objectives = [
        ParetoObjective(metric="throughput", direction="maximize"),
        ParetoObjective(metric="cpu", direction="minimize"),
    ]
    rows_nan = [
        {_col("throughput"): 1e9, _col("cpu"): 20.0},
        {_col("throughput"): float("nan"), _col("cpu"): float("nan")},
        {_col("throughput"): 2e9, _col("cpu"): 40.0},
    ]
    rows_none = [
        {_col("throughput"): 1e9, _col("cpu"): 20.0},
        {_col("throughput"): None, _col("cpu"): None},
        {_col("throughput"): 2e9, _col("cpu"): 40.0},
    ]
    scores_nan = score_rows(rows_nan, objectives, {"cpu": 0.15})
    scores_none = score_rows(rows_none, objectives, {"cpu": 0.15})
    for a, b in zip(scores_nan, scores_none, strict=True):
        assert a == pytest.approx(b)


def test_score_rows_empty_input() -> None:
    """Zero rows return an empty list, not an error."""
    objectives = [ParetoObjective(metric="throughput", direction="maximize")]
    assert score_rows([], objectives, {}) == []


def test_score_rows_missing_metric_column_is_neutral() -> None:
    """A metric absent from every row collapses to a flat 0.5 contribution.

    The objective is ``node_memory`` (minimize, weight 0.15) but no
    row supplies that column. The column's ``_normalize_column``
    falls into the no-finite-values branch and maps every row to
    ``0.5``, so the contribution is ``-0.075`` for every row --
    identical across rows, no effect on the ranking relative to the
    maximize-throughput term.
    """
    objectives = [
        ParetoObjective(metric="throughput", direction="maximize"),
        ParetoObjective(metric="node_memory", direction="minimize"),
    ]
    rows = [
        {_col("throughput"): 1e9},
        {_col("throughput"): 2e9},
    ]
    scores = score_rows(rows, objectives, {"node_memory": 0.15})
    # throughput norms: 0.0 and 1.0; node_memory norms: 0.5 each.
    assert scores[0] == pytest.approx(0.0 - 0.075)
    assert scores[1] == pytest.approx(1.0 - 0.075)
