"""Tests for :mod:`kube_autotuner.scoring`."""

from __future__ import annotations

from datetime import UTC, datetime
import math
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from kube_autotuner.experiment import ObjectivesSection, ParetoObjective
from kube_autotuner.models import (
    BenchmarkConfig,
    BenchmarkResult,
    MemoryCost,
    NodePair,
    ParamSpace,
    SysctlParam,
    TrialResult,
)
from kube_autotuner.scoring import (
    METRIC_TO_DF_COLUMN,
    config_memory_cost,
    score_rows,
)


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


def test_score_rows_memory_cost_disabled_matches_cost_free_call() -> None:
    """With weight ``0`` or missing costs, the extra term is a no-op."""
    objectives = [
        ParetoObjective(metric="tcp_throughput", direction="maximize"),
        ParetoObjective(metric="udp_jitter", direction="minimize"),
    ]
    rows = [
        {_col("tcp_throughput"): 2.0e9, _col("udp_jitter"): 8.0},
        {_col("tcp_throughput"): 1.5e9, _col("udp_jitter"): 5.0},
        {_col("tcp_throughput"): 1.0e9, _col("udp_jitter"): 2.0},
    ]
    weights = {"udp_jitter": 0.15}
    cost_free = score_rows(rows, objectives, weights)
    with_costs_zero_weight = score_rows(
        rows,
        objectives,
        weights,
        memory_costs=[1.0, 100.0, 5000.0],
        memory_cost_weight=0.0,
    )
    without_costs = score_rows(
        rows,
        objectives,
        weights,
        memory_costs=None,
        memory_cost_weight=0.15,
    )
    for a, b in zip(cost_free, with_costs_zero_weight, strict=True):
        assert a == pytest.approx(b)
    for a, b in zip(cost_free, without_costs, strict=True):
        assert a == pytest.approx(b)


def test_score_rows_memory_cost_flips_tied_rows() -> None:
    """Identical perf metrics + non-zero weight: lower-memory row wins."""
    objectives = [
        ParetoObjective(metric="tcp_throughput", direction="maximize"),
    ]
    rows = [
        {_col("tcp_throughput"): 1.0e9},
        {_col("tcp_throughput"): 1.0e9},
    ]
    scores = score_rows(
        rows,
        objectives,
        {},
        memory_costs=[200.0, 100.0],
        memory_cost_weight=0.1,
    )
    # Tied throughput (constant column -> 0.5 each, uniform).
    # Memory cost norms: 1.0, 0.0. Expect row 1 to outrank row 0.
    assert scores[1] > scores[0]


def test_score_rows_memory_cost_respects_dominant_performance() -> None:
    """A clearly better throughput still wins under a gentle cost weight."""
    objectives = [
        ParetoObjective(metric="tcp_throughput", direction="maximize"),
    ]
    rows = [
        {_col("tcp_throughput"): 2.0e9},  # big winner, big buffer
        {_col("tcp_throughput"): 1.0e9},  # loser, tiny buffer
    ]
    scores = score_rows(
        rows,
        objectives,
        {},
        memory_costs=[1.0e9, 1.0],
        memory_cost_weight=0.1,
    )
    # Row 0 gains 1.0 on throughput, loses 0.1 on cost -> 0.9.
    # Row 1 gains 0.0 on throughput, loses 0.0 on cost -> 0.0.
    assert scores[0] > scores[1]


def test_score_rows_within_tolerance_metric_is_neutral() -> None:
    """Two rows differing only on retx by less than tolerance score equally.

    With ``tolerances={"tcp_retransmit_rate": 0.10}``, a 1% retx gap
    snaps both rows to the same endpoint on that axis, so the only
    remaining axis (throughput) drives the score.
    """
    objectives = [
        ParetoObjective(metric="tcp_throughput", direction="maximize"),
        ParetoObjective(metric="tcp_retransmit_rate", direction="minimize"),
    ]
    rows = [
        {_col("tcp_throughput"): 1.0e9, _col("tcp_retransmit_rate"): 1.0},
        {_col("tcp_throughput"): 1.0e9, _col("tcp_retransmit_rate"): 1.005},
    ]
    scores = score_rows(
        rows,
        objectives,
        {"tcp_retransmit_rate": 0.3},
        tolerances={"tcp_retransmit_rate": 0.10},
    )
    assert scores[0] == pytest.approx(scores[1])


def test_score_rows_user_example_throughput_beats_noise_regression() -> None:
    """1000 vs 1200 Mbps throughput with a 1% retx blip: B should win.

    Captures the user's complaint: under defaults a real 20%
    throughput improvement was being cancelled by a sub-noise retx
    "regression". With the default tolerances both rows tie on retx
    and the throughput winner ranks higher.
    """
    objectives = [
        ParetoObjective(metric="tcp_throughput", direction="maximize"),
        ParetoObjective(metric="tcp_retransmit_rate", direction="minimize"),
    ]
    rows = [
        {_col("tcp_throughput"): 1.0e3, _col("tcp_retransmit_rate"): 1.0},
        {_col("tcp_throughput"): 1.2e3, _col("tcp_retransmit_rate"): 1.01},
    ]
    section = ObjectivesSection()
    scores = score_rows(
        rows,
        objectives,
        section.recommendation_weights,
        tolerances=section.tolerances,
    )
    assert scores[1] > scores[0]


def test_score_rows_zero_tolerance_zero_sem_matches_legacy() -> None:
    """Regression guard: empty tolerances + None SEM reduces bit-for-bit.

    Reuses the fixture from
    :func:`test_score_rows_matches_recommend_configs_formula`; calls
    with ``tolerances={}`` and ``sems=None``; asserts per-row scores
    match the hand-computed legacy values to ``abs=1e-15``.
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
        {"udp_jitter": 0.15},
        tolerances={},
        sems=None,
    )
    assert scores[0] == pytest.approx(0.85, abs=1e-15)
    assert scores[1] == pytest.approx(0.425, abs=1e-15)
    assert scores[2] == pytest.approx(0.0, abs=1e-15)


def test_score_rows_sem_widens_threshold_above_relative() -> None:
    """SEM dominates a small relative tolerance and ties the rows.

    Two rows 1000 vs 1010 throughput with
    ``tolerances={"tcp_throughput": 0.005}``: the relative term is
    ``0.005 * 1010 = 5.05``, far below the 10-unit gap. With
    ``sems=[50, 50]`` (column endpoints have SEM 0 by definition of
    where the snap happens, but the *row* SEM is what's used) the
    SEM term ``1.0 * 50 = 50`` covers the gap; both rows snap and
    tie.
    """
    objectives = [ParetoObjective(metric="tcp_throughput", direction="maximize")]
    rows = [
        {_col("tcp_throughput"): 1000.0},
        {_col("tcp_throughput"): 1010.0},
    ]
    sems = [
        {f"{_col('tcp_throughput')}_sem": 50.0},
        {f"{_col('tcp_throughput')}_sem": 50.0},
    ]
    scores_no_sem = score_rows(
        rows,
        objectives,
        {},
        tolerances={"tcp_throughput": 0.005},
    )
    # Without SEM the gap is decisive; row 1 outscores row 0.
    assert scores_no_sem[1] > scores_no_sem[0]
    scores_with_sem = score_rows(
        rows,
        objectives,
        {},
        tolerances={"tcp_throughput": 0.005},
        sems=sems,
    )
    assert scores_with_sem[0] == pytest.approx(scores_with_sem[1])


def test_score_rows_memory_cost_tolerance_snaps() -> None:
    """Memory cost tolerance ties tightly-bunched costs on the cost axis.

    Two rows with identical perf metrics and memory costs 100 vs 105:
    a 10% tolerance on ``memory_cost`` snaps the 5-unit gap, leaving
    the rows tied. The existing
    :func:`test_score_rows_memory_cost_flips_tied_rows` (without a
    tolerance) keeps its expected behavior.
    """
    objectives = [ParetoObjective(metric="tcp_throughput", direction="maximize")]
    rows = [
        {_col("tcp_throughput"): 1.0e9},
        {_col("tcp_throughput"): 1.0e9},
    ]
    scores = score_rows(
        rows,
        objectives,
        {},
        memory_costs=[100.0, 105.0],
        memory_cost_weight=0.1,
        tolerances={"memory_cost": 0.10},
    )
    assert scores[0] == pytest.approx(scores[1])


def test_config_memory_cost_rules_derive_per_rung() -> None:
    """Every rule kind derives the expected bytes from the selected rung."""
    space = ParamSpace(
        params=[
            SysctlParam(
                name="rmem",
                values=[1024, 2048],
                param_type="int",
                memory_cost=MemoryCost(kind="identity"),
            ),
            SysctlParam(
                name="triple",
                values=["1 2 3", "10 20 30"],
                param_type="choice",
                memory_cost=MemoryCost(kind="triple_max"),
            ),
            SysctlParam(
                name="pages",
                values=["1 2 3"],
                param_type="choice",
                memory_cost=MemoryCost(kind="triple_max_pages"),
            ),
            SysctlParam(
                name="kib",
                values=[1, 2],
                param_type="int",
                memory_cost=MemoryCost(kind="kib"),
            ),
            SysctlParam(
                name="entries",
                values=[100, 200],
                param_type="int",
                memory_cost=MemoryCost(kind="per_entry", per_entry_bytes=10),
            ),
            SysctlParam(
                name="uncosted",
                values=[0, 1],
                param_type="choice",
            ),
        ],
    )
    total = config_memory_cost(
        {
            "rmem": 2048,
            "triple": "10 20 30",
            "pages": "1 2 3",
            "kib": 2,
            "entries": 200,
            "uncosted": 1,
        },
        space,
    )
    expected = 2048 + 30 + 3 * 4096 + 2 * 1024 + 200 * 10
    assert total == pytest.approx(expected)


def test_config_memory_cost_zero_for_all_uncosted() -> None:
    """A config touching only uncosted sysctls sums to 0."""
    space = ParamSpace(
        params=[
            SysctlParam(name="a", values=[0, 1], param_type="choice"),
        ],
    )
    assert config_memory_cost({"a": 1}, space) == pytest.approx(0.0)


def _trial(bps: float, retransmits: int) -> TrialResult:
    """One-iteration TCP-only primary trial parametrized by throughput and retx.

    Local copy of the helper in
    :mod:`tests.test_aggregate_by_parent` so this test stays
    self-contained (the flat ``tests/`` tree has no ``__init__.py``,
    so sibling test modules are not importable by name).

    Returns:
        A primary :class:`TrialResult` carrying a single TCP iperf3
        record with the requested throughput, byte total, and
        retransmit count.
    """
    return TrialResult(
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values={"net.core.rmem_max": int(bps)},
        config=BenchmarkConfig(),
        results=[
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=bps,
                bytes_sent=int(bps),
                retransmits=retransmits,
                iteration=0,
                client_node="a",
            ),
        ],
        phase="bayesian",
    )


def test_pareto_recommendation_rows_score_matches_score_rows() -> None:
    """The analysis path's ``score`` field equals a direct ``score_rows`` call.

    Regression guard against the analysis path silently bypassing
    :func:`score_rows` -- e.g. an inline normalization or
    post-multiplier sneaking into
    :func:`kube_autotuner.report.analysis.pareto_recommendation_rows`
    between the ``score_rows`` call and the returned ``score`` field.
    A change inside ``score_rows`` itself would shift both sides
    equally and would *not* be caught here; that contract is held
    by the unit tests above.
    """
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from kube_autotuner.report.analysis import (  # noqa: PLC0415
        pareto_recommendation_rows,
    )

    # Three trials chosen so multiple sit on the Pareto frontier (each
    # wins on at least one of the two surviving objectives,
    # ``tcp_throughput`` and ``tcp_retransmit_rate``). The other seven
    # default objectives have all-NaN columns and are filtered out by
    # ``_objectives_with_data``.
    trials = [
        _trial(bps=2.0e9, retransmits=10),  # high tp, high retx
        _trial(bps=1.5e9, retransmits=2),  # mid tp, mid retx
        _trial(bps=1.0e9, retransmits=0),  # low tp, low retx
    ]

    analysis_rows = pareto_recommendation_rows(trials, "10g")
    assert len(analysis_rows) >= 2  # the test only has teeth with >1 row

    section = ObjectivesSection()
    records = [
        {col: row[col] for col in METRIC_TO_DF_COLUMN.values()} for row in analysis_rows
    ]
    memory_costs = [row["memory_cost"] for row in analysis_rows]
    sem_records = [
        {f"{col}_sem": row.get(f"{col}_sem") for col in METRIC_TO_DF_COLUMN.values()}
        for row in analysis_rows
    ]
    direct_scores = score_rows(
        records,
        section.pareto,
        section.recommendation_weights,
        memory_costs=memory_costs,
        memory_cost_weight=section.memory_cost_weight,
        sems=sem_records,
        tolerances=section.tolerances,
    )

    direct_by_id = {
        row["trial_id"]: s for row, s in zip(analysis_rows, direct_scores, strict=True)
    }
    analysis_by_id = {row["trial_id"]: row["score"] for row in analysis_rows}
    assert direct_by_id == pytest.approx(analysis_by_id, abs=1e-12)


def test_cli_analyze_one_class_threads_tolerances(tmp_path: Path) -> None:
    """Reverting ``cli.py``'s ``tolerances=`` kwarg must break this test.

    Regression guard for ``cli.analyze``: ``_analyze_one_class`` calls
    :func:`pareto_recommendation_rows` and must thread the user's
    :attr:`ObjectivesSection.tolerances` through, including the
    ``tolerances: {}`` opt-out. The fixture pins A barely below B on
    throughput (within the 3% default tolerance) and far above B on
    retx. Default tolerances snap throughput to a tie so A dominates B
    by retx alone; ``tolerances={}`` leaves both mutually
    non-dominated. The recommendation list length then differs
    precisely when the CLI honors the user's tolerances.
    """
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from kube_autotuner.cli import _analyze_one_class  # noqa: PLC0415, PLC2701
    from kube_autotuner.models import ALL_STAGES  # noqa: PLC0415
    from kube_autotuner.report import analysis as analysis_mod  # noqa: PLC0415

    trials = [
        _trial(bps=1.0e9, retransmits=100),
        _trial(bps=1.029e9, retransmits=130),
    ]

    def _run(objectives: ObjectivesSection, sub: str) -> int:
        sub_out = tmp_path / sub
        sub_out.mkdir()
        section = _analyze_one_class(
            trials,
            hardware_class="10g",
            topology=None,
            top_n=3,
            output_dir=sub_out,
            analysis=analysis_mod,
            explicit_class=True,
            objectives=objectives,
            stages=ALL_STAGES,
        )
        assert section is not None
        return len(section["pareto_rows"])

    with_defaults = _run(ObjectivesSection(), "defaults")
    without = _run(ObjectivesSection(tolerances={}), "empty")
    assert with_defaults != without
