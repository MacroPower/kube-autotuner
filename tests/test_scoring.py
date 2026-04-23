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

    Throughput (maximize) normalizes to (1.0, 0.5, 0.0); jitter
    (minimize, weight 0.15) normalizes to (1.0, 0.5, 0.0); no other
    objectives participate. The expected scores are:

    - row A: 1.0 - 0.15 * 1.0 = 0.85
    - row B: 0.5 - 0.15 * 0.5 = 0.425
    - row C: 0.0 - 0.15 * 0.0 = 0.0
    """
    objectives = [
        ParetoObjective(metric="tcp_throughput", direction="maximize"),
        ParetoObjective(metric="udp_jitter", direction="minimize"),
    ]
    rows = [
        {_col("tcp_throughput"): 2.0e9, _col("udp_jitter"): 8.0},
        {_col("tcp_throughput"): 1.5e9, _col("udp_jitter"): 5.0},
        {_col("tcp_throughput"): 1.0e9, _col("udp_jitter"): 2.0},
    ]
    scores = score_rows(rows, objectives, {"udp_jitter": 0.15})
    assert scores[0] == pytest.approx(0.85)
    assert scores[1] == pytest.approx(0.425)
    assert scores[2] == pytest.approx(0.0)


def test_score_rows_handles_nan_and_degenerate_columns() -> None:
    """NaN rows, constant columns, and all-NaN columns resolve to ``0.5``."""
    objectives = [
        ParetoObjective(metric="tcp_throughput", direction="maximize"),
        ParetoObjective(metric="udp_jitter", direction="minimize"),
        ParetoObjective(metric="latency_p99", direction="minimize"),
    ]
    rows = [
        # One all-NaN column (latency_p99), one constant column (jitter),
        # one row with a NaN throughput.
        {
            _col("tcp_throughput"): 1.0e9,
            _col("udp_jitter"): 1.0,
            _col("latency_p99"): math.nan,
        },
        {
            _col("tcp_throughput"): math.nan,
            _col("udp_jitter"): 1.0,
            _col("latency_p99"): math.nan,
        },
        {
            _col("tcp_throughput"): 2.0e9,
            _col("udp_jitter"): 1.0,
            _col("latency_p99"): math.nan,
        },
    ]
    scores = score_rows(
        rows,
        objectives,
        {"udp_jitter": 0.15, "latency_p99": 0.15},
    )
    # throughput norms: row0 -> 0.0, row1 -> NaN-fallback 0.5, row2 -> 1.0.
    # jitter norms: all 0.5 (constant column); contribution = -0.15 * 0.5 = -0.075.
    # latency_p99: all-NaN column -> 0.5 each; contribution = -0.15 * 0.5 = -0.075.
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
        ParetoObjective(metric="tcp_throughput", direction="maximize"),
        ParetoObjective(metric="udp_jitter", direction="minimize"),
    ]
    rows_nan = [
        {_col("tcp_throughput"): 1e9, _col("udp_jitter"): 2.0},
        {_col("tcp_throughput"): float("nan"), _col("udp_jitter"): float("nan")},
        {_col("tcp_throughput"): 2e9, _col("udp_jitter"): 4.0},
    ]
    rows_none = [
        {_col("tcp_throughput"): 1e9, _col("udp_jitter"): 2.0},
        {_col("tcp_throughput"): None, _col("udp_jitter"): None},
        {_col("tcp_throughput"): 2e9, _col("udp_jitter"): 4.0},
    ]
    scores_nan = score_rows(rows_nan, objectives, {"udp_jitter": 0.15})
    scores_none = score_rows(rows_none, objectives, {"udp_jitter": 0.15})
    for a, b in zip(scores_nan, scores_none, strict=True):
        assert a == pytest.approx(b)


def test_score_rows_empty_input() -> None:
    """Zero rows return an empty list, not an error."""
    objectives = [ParetoObjective(metric="tcp_throughput", direction="maximize")]
    assert score_rows([], objectives, {}) == []


def test_score_rows_missing_metric_column_is_neutral() -> None:
    """A metric absent from every row collapses to a flat 0.5 contribution.

    The objective is ``latency_p99`` (minimize, weight 0.15) but no
    row supplies that column. The column's ``_normalize_column``
    falls into the no-finite-values branch and maps every row to
    ``0.5``, so the contribution is ``-0.075`` for every row --
    identical across rows, no effect on the ranking relative to the
    maximize-throughput term.
    """
    objectives = [
        ParetoObjective(metric="tcp_throughput", direction="maximize"),
        ParetoObjective(metric="latency_p99", direction="minimize"),
    ]
    rows = [
        {_col("tcp_throughput"): 1e9},
        {_col("tcp_throughput"): 2e9},
    ]
    scores = score_rows(rows, objectives, {"latency_p99": 0.15})
    # throughput norms: 0.0 and 1.0; latency_p99 norms: 0.5 each.
    assert scores[0] == pytest.approx(0.0 - 0.075)
    assert scores[1] == pytest.approx(1.0 - 0.075)


def test_score_rows_maximize_weight_scales_contribution() -> None:
    """An explicit maximize weight multiplies the +norm contribution.

    Throughput (maximize, weight 2.0) normalizes to (1.0, 0.5, 0.0);
    jitter (minimize, weight 0.15) normalizes to (1.0, 0.5, 0.0).
    Composed score per row is ``2.0 * norm_tp - 0.15 * norm_j``.
    """
    objectives = [
        ParetoObjective(metric="tcp_throughput", direction="maximize"),
        ParetoObjective(metric="udp_jitter", direction="minimize"),
    ]
    rows = [
        {_col("tcp_throughput"): 2.0e9, _col("udp_jitter"): 8.0},
        {_col("tcp_throughput"): 1.5e9, _col("udp_jitter"): 5.0},
        {_col("tcp_throughput"): 1.0e9, _col("udp_jitter"): 2.0},
    ]
    scores = score_rows(
        rows,
        objectives,
        {"tcp_throughput": 2.0, "udp_jitter": 0.15},
    )
    assert scores[0] == pytest.approx(2.0 * 1.0 - 0.15 * 1.0)
    assert scores[1] == pytest.approx(2.0 * 0.5 - 0.15 * 0.5)
    assert scores[2] == pytest.approx(2.0 * 0.0 - 0.15 * 0.0)


def test_score_rows_missing_maximize_weight_defaults_to_one() -> None:
    """Omitting a maximize weight reproduces the historical ``+norm``.

    Regression guard for the behavior-preservation claim: callers
    that never set a maximize weight keep the same per-row scores
    they had before weights were extensible to maximize metrics.
    """
    objectives = [
        ParetoObjective(metric="tcp_throughput", direction="maximize"),
        ParetoObjective(metric="udp_jitter", direction="minimize"),
    ]
    rows = [
        {_col("tcp_throughput"): 2.0e9, _col("udp_jitter"): 8.0},
        {_col("tcp_throughput"): 1.0e9, _col("udp_jitter"): 2.0},
    ]
    implicit = score_rows(rows, objectives, {"udp_jitter": 0.15})
    explicit = score_rows(
        rows,
        objectives,
        {"tcp_throughput": 1.0, "udp_jitter": 0.15},
    )
    for a, b in zip(implicit, explicit, strict=True):
        assert a == pytest.approx(b)


def test_score_rows_zero_weight_on_maximize_disables_contribution() -> None:
    """Weight ``0.0`` on a maximize metric zeroes its contribution.

    With throughput disabled, the score reduces to just the
    minimize-direction terms. All three rows then share the same
    jitter-only ranking.
    """
    objectives = [
        ParetoObjective(metric="tcp_throughput", direction="maximize"),
        ParetoObjective(metric="udp_jitter", direction="minimize"),
    ]
    rows = [
        {_col("tcp_throughput"): 2.0e9, _col("udp_jitter"): 8.0},
        {_col("tcp_throughput"): 1.5e9, _col("udp_jitter"): 5.0},
        {_col("tcp_throughput"): 1.0e9, _col("udp_jitter"): 2.0},
    ]
    scores = score_rows(
        rows,
        objectives,
        {"tcp_throughput": 0.0, "udp_jitter": 0.15},
    )
    assert scores[0] == pytest.approx(-0.15 * 1.0)
    assert scores[1] == pytest.approx(-0.15 * 0.5)
    assert scores[2] == pytest.approx(-0.15 * 0.0)
