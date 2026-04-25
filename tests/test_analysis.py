"""Tests for the offline trial analysis module."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import math
from typing import TYPE_CHECKING, Any, Literal

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from typer.testing import CliRunner  # noqa: E402

from kube_autotuner.cli import _build_axis_payload, app  # noqa: E402, PLC2701
from kube_autotuner.experiment import (  # noqa: E402
    FortioSection,
    IperfSection,
    ObjectivesSection,
    ParetoObjective,
)
from kube_autotuner.models import (  # noqa: E402
    ALL_STAGES,
    BenchmarkConfig,
    BenchmarkResult,
    HostStateSnapshot,
    LatencyResult,
    NodePair,
    ParamSpace,
    ResumeMetadata,
    TrialResult,
)
from kube_autotuner.report.analysis import (  # noqa: E402
    DEFAULT_OBJECTIVES,
    METRIC_TO_DF_COLUMN,
    baseline_comparison,
    category_importance_rollup,
    host_state_issues,
    host_state_series,
    parameter_importance,
    pareto_front,
    per_iteration_samples,
    recommend_configs,
    refinement_stats,
    section_metadata,
    split_trials_by_hardware_class,
    stability_badge,
    sysctl_correlation_matrix,
    trajectory_rows,
    trials_to_dataframe,
)
from kube_autotuner.scoring import _per_trial_metric_means  # noqa: E402, PLC2701
from kube_autotuner.trial_log import TrialLog  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


_DEFAULT_BYTES_SENT = 1_000_000_000


def _result(
    bps: float,
    retransmits: int = 0,
    bytes_sent: int | None = _DEFAULT_BYTES_SENT,
) -> BenchmarkResult:
    return BenchmarkResult(
        timestamp=datetime.now(UTC),
        mode="tcp",
        bits_per_second=bps,
        retransmits=retransmits,
        bytes_sent=bytes_sent,
    )


def _trial(  # noqa: PLR0913, PLR0917
    hw: str = "10g",
    bps: float = 5e9,
    retransmits: int = 5,
    rmem_max: int = 212992,
    congestion: str = "cubic",
    trial_id: str | None = None,
    source_zone: str = "",
    target_zone: str = "",
    bytes_sent: int | None = _DEFAULT_BYTES_SENT,
) -> TrialResult:
    kw: dict[str, Any] = {}
    if trial_id:
        kw["trial_id"] = trial_id
    return TrialResult(
        node_pair=NodePair(
            source="a",
            target="b",
            hardware_class=hw,
            source_zone=source_zone,
            target_zone=target_zone,
        ),
        sysctl_values={
            "net.core.rmem_max": rmem_max,
            "net.ipv4.tcp_congestion_control": congestion,
        },
        config=BenchmarkConfig(iterations=1),
        results=[_result(bps, retransmits, bytes_sent)],
        **kw,
    )


def _build_10g_trials() -> list[TrialResult]:
    trials: list[TrialResult] = []
    for i in range(15):
        rmem = [212992, 4194304, 16777216, 67108864][i % 4]
        cc = "bbr" if i % 3 == 0 else "cubic"
        bps = 1e9 + rmem * 1.0 + (50e6 if cc == "bbr" else 0)
        trials.append(
            _trial(
                hw="10g",
                bps=bps,
                retransmits=max(0, 20 - i),
                rmem_max=rmem,
                congestion=cc,
            ),
        )
    return trials


def _build_1g_trials() -> list[TrialResult]:
    return [
        _trial(
            hw="1g",
            bps=5e8 + i * 1e7,
            retransmits=10 + i,
        )
        for i in range(5)
    ]


@pytest.fixture
def mixed_trials() -> list[TrialResult]:
    """Return 20 trials: 15 x 10g, 5 x 1g with varied params and metrics.

    Returns:
        A hand-tuned 20-trial list where ``net.core.rmem_max`` is the
        dominant throughput driver on the 10g class.
    """
    return [*_build_10g_trials(), *_build_1g_trials()]


# --- trials_to_dataframe -------------------------------------------------


class TestTrialsToDataframe:
    def test_shape(self, mixed_trials: list[TrialResult]) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
        assert len(df) == 15
        assert "trial_id" in df.columns
        assert "mean_tcp_throughput" in df.columns
        assert "net.core.rmem_max" in df.columns

    def test_hardware_filter(self, mixed_trials: list[TrialResult]) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
        assert len(df) == 15
        assert (df["hardware_class"] == "10g").all()

    def test_empty_filter(self, mixed_trials: list[TrialResult]) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="1g")
        assert len(df) == 5

    def test_unknown_filter_returns_empty_with_columns(
        self,
        mixed_trials: list[TrialResult],
    ) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="99g")
        assert df.empty
        assert "mean_tcp_throughput" in df.columns

    def test_int_params_numeric(self, mixed_trials: list[TrialResult]) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
        assert pd.api.types.is_numeric_dtype(df["net.core.rmem_max"])

    def test_choice_params_encoded(self, mixed_trials: list[TrialResult]) -> None:
        df, encoders = trials_to_dataframe(mixed_trials, hardware_class="10g")
        assert pd.api.types.is_numeric_dtype(df["net.ipv4.tcp_congestion_control"])
        assert "net.ipv4.tcp_congestion_control" in encoders

    def test_topology_columns_present(self, mixed_trials: list[TrialResult]) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
        assert "topology" in df.columns
        assert "source_zone" in df.columns
        assert "target_zone" in df.columns

    def test_topology_filter(self) -> None:
        trials = [
            _trial(bps=5e9, source_zone="az01", target_zone="az01"),
            _trial(bps=6e9, source_zone="az01", target_zone="az02"),
            _trial(bps=7e9, source_zone="az02", target_zone="az02"),
        ]
        df_intra, _ = trials_to_dataframe(trials, topology="intra-az")
        assert len(df_intra) == 2
        assert (df_intra["topology"] == "intra-az").all()

        df_inter, _ = trials_to_dataframe(trials, topology="inter-az")
        assert len(df_inter) == 1
        assert (df_inter["topology"] == "inter-az").all()

    def test_mixed_hardware_class_without_filter_raises(
        self,
        mixed_trials: list[TrialResult],
    ) -> None:
        with pytest.raises(ValueError, match="multiple hardware classes"):
            trials_to_dataframe(mixed_trials)

    def test_single_hardware_class_without_filter_ok(self) -> None:
        trials = [_trial(hw="10g", bps=5e9), _trial(hw="10g", bps=6e9)]
        df, _ = trials_to_dataframe(trials)
        assert len(df) == 2


# --- split_trials_by_hardware_class -------------------------------------


class TestSplitByHardwareClass:
    def test_partitions_trials(self, mixed_trials: list[TrialResult]) -> None:
        groups = split_trials_by_hardware_class(mixed_trials)
        assert set(groups.keys()) == {"10g", "1g"}
        assert len(groups["10g"]) == 15
        assert len(groups["1g"]) == 5

    def test_sorted_keys(self, mixed_trials: list[TrialResult]) -> None:
        groups = split_trials_by_hardware_class(mixed_trials)
        assert list(groups.keys()) == sorted(groups.keys())


# --- pareto_front --------------------------------------------------------


def _pad_latency_cols(rows: int) -> dict[str, list[float]]:
    """Pad a synthetic pareto DataFrame with constant latency/RPS/jitter columns.

    The default :data:`DEFAULT_OBJECTIVES` spans seven axes; filling
    the latency/RPS/jitter columns with constants keeps the dominance
    scan deterministic so the throughput/retransmit_rate tests still
    exercise the exact trials they were designed for.

    Returns:
        A mapping of column name to a list of ``rows`` constant values
        suitable for splatting into a DataFrame constructor.
    """
    return {
        "mean_udp_jitter": [0.1] * rows,
        "mean_rps": [1000.0] * rows,
        "mean_latency_p50": [1.0] * rows,
        "mean_latency_p90": [5.0] * rows,
        "mean_latency_p99": [10.0] * rows,
    }


class TestParetoFront:
    def test_dominated_removed(self) -> None:
        df = pd.DataFrame(
            {
                "trial_id": ["A", "B", "C", "D"],
                "mean_tcp_throughput": [100, 80, 60, 50],
                "tcp_retransmit_rate": [1, 2, 0, 5],
                **_pad_latency_cols(4),
            },
        )
        front = pareto_front(df)
        ids = set(front["trial_id"])
        assert "D" not in ids
        assert "A" in ids

    def test_all_nondominated(self) -> None:
        df = pd.DataFrame(
            {
                "trial_id": ["A", "B"],
                "mean_tcp_throughput": [100, 50],
                "tcp_retransmit_rate": [5, 1],
                **_pad_latency_cols(2),
            },
        )
        front = pareto_front(df)
        assert len(front) == 2

    def test_single_trial(self) -> None:
        df = pd.DataFrame(
            {
                "trial_id": ["A"],
                "mean_tcp_throughput": [100],
                "tcp_retransmit_rate": [1],
                **_pad_latency_cols(1),
            },
        )
        assert len(pareto_front(df)) == 1

    def test_empty(self) -> None:
        df = pd.DataFrame(
            columns=[
                "trial_id",
                "mean_tcp_throughput",
                "tcp_retransmit_rate",
                "mean_udp_jitter",
                "mean_rps",
                "mean_latency_p50",
                "mean_latency_p90",
                "mean_latency_p99",
            ],
        )
        assert pareto_front(df).empty

    def test_default_objectives_shape(self) -> None:
        assert len(DEFAULT_OBJECTIVES) == 9
        names = [n for n, _ in DEFAULT_OBJECTIVES]
        assert "mean_tcp_throughput" in names
        assert "mean_udp_throughput" in names
        assert "mean_udp_jitter" in names
        assert "mean_rps" in names
        assert "mean_latency_p99" in names
        assert "tcp_retransmit_rate" in names
        assert "udp_loss_rate" in names

    def test_drops_nan_rows_with_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        df = pd.DataFrame(
            {
                "trial_id": ["A", "B"],
                "mean_tcp_throughput": [100, 50],
                "tcp_retransmit_rate": [0.1, float("nan")],
                **_pad_latency_cols(2),
            },
        )
        with caplog.at_level("WARNING", logger="kube_autotuner.report.analysis"):
            front = pareto_front(df)
        assert list(front["trial_id"]) == ["A"]
        assert any("NaN" in rec.message for rec in caplog.records)

    def test_all_nan_column_dropped_from_objectives(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A metric whose column is entirely NaN is silently excluded.

        Rather than dropping every trial (which would empty the
        frontier) or poisoning dominance with ``0.0``, the objective
        is removed and the frontier is computed from the remaining
        axes.
        """
        df = pd.DataFrame(
            {
                "trial_id": ["A", "B"],
                "mean_tcp_throughput": [100, 50],
                "tcp_retransmit_rate": [0.2, 0.1],
                "mean_udp_jitter": [float("nan"), float("nan")],
                "mean_rps": [1000.0] * 2,
                "mean_latency_p50": [1.0] * 2,
                "mean_latency_p90": [5.0] * 2,
                "mean_latency_p99": [10.0] * 2,
            },
        )
        with caplog.at_level("INFO", logger="kube_autotuner.report.analysis"):
            front = pareto_front(df)
        assert set(front["trial_id"]) == {"A", "B"}
        assert any(
            "mean_udp_jitter" in rec.message and "excluded" in rec.message
            for rec in caplog.records
        )

    def test_all_objectives_dropped_returns_empty(self) -> None:
        """If every objective column is NaN, the frontier is empty."""
        df = pd.DataFrame(
            {
                "trial_id": ["A", "B"],
                "mean_tcp_throughput": [float("nan"), float("nan")],
                "tcp_retransmit_rate": [float("nan"), float("nan")],
                "mean_udp_jitter": [float("nan"), float("nan")],
                "mean_rps": [float("nan"), float("nan")],
                "mean_latency_p50": [float("nan"), float("nan")],
                "mean_latency_p90": [float("nan"), float("nan")],
                "mean_latency_p99": [float("nan"), float("nan")],
            },
        )
        assert pareto_front(df).empty


# --- parameter_importance -----------------------------------------------


class TestParameterImportance:
    def test_identifies_top_param(self, mixed_trials: list[TrialResult]) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
        imp = parameter_importance(df, target="mean_tcp_throughput")
        assert not imp.empty
        assert imp.iloc[0]["param"] == "net.core.rmem_max"

    def test_has_category(self, mixed_trials: list[TrialResult]) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
        imp = parameter_importance(df)
        assert "category" in imp.columns
        assert imp.iloc[0]["category"] != "unknown"

    def test_returns_all_varied_params(
        self,
        mixed_trials: list[TrialResult],
    ) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
        imp = parameter_importance(df)
        assert set(imp["param"]) == {
            "net.core.rmem_max",
            "net.ipv4.tcp_congestion_control",
        }

    def test_rf_importance_survives_tiny_target_magnitude(self) -> None:
        """RF importance must not collapse to zero on ~1e-10 targets.

        ``RandomForestRegressor`` uses MSE-based impurity, which scales
        as ``y**2``. For ``y ~ 1e-10`` the per-split improvement is
        ``~1e-20``, below sklearn's tree-splitter epsilon; without the
        std-scale step in ``_rf_importance_scores`` every tree degenerates
        into a single leaf and ``feature_importances_`` sums to zero. This
        is not hypothetical: ``tcp_retransmit_rate`` (when stored as
        retransmits per byte) lives in exactly this regime.
        """
        n = 40
        driver = [i / (n - 1) for i in range(n)]
        noise1 = [((i * 7) % n) / (n - 1) for i in range(n)]
        noise2 = [((i * 13) % n) / (n - 1) for i in range(n)]
        df = pd.DataFrame(
            {
                "trial_id": list(range(n)),
                "hardware_class": ["1g"] * n,
                "net.core.rmem_max": driver,
                "net.core.wmem_max": noise1,
                "net.core.netdev_max_backlog": noise2,
                "tcp_retransmit_rate": [3e-10 + 5e-10 * d for d in driver],
            },
        )
        imp = parameter_importance(df, target="tcp_retransmit_rate")
        assert not imp.empty
        assert imp["rf_importance"].sum() == pytest.approx(1.0)
        assert imp.iloc[0]["param"] == "net.core.rmem_max"


# --- recommend_configs ---------------------------------------------------


class TestRecommendConfigs:
    def test_count(self, mixed_trials: list[TrialResult]) -> None:
        recs = recommend_configs(mixed_trials, "10g", n=3)
        assert 1 <= len(recs) <= 3

    def test_pareto_optimal(self, mixed_trials: list[TrialResult]) -> None:
        recs = recommend_configs(mixed_trials, "10g", n=5)
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
        front = pareto_front(df)
        front_ids = set(front["trial_id"])
        for r in recs:
            assert r["trial_id"] in front_ids

    def test_sorted_by_score(self, mixed_trials: list[TrialResult]) -> None:
        recs = recommend_configs(mixed_trials, "10g", n=5)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_empty_for_missing_class(
        self,
        mixed_trials: list[TrialResult],
    ) -> None:
        assert recommend_configs(mixed_trials, "99g") == []  # type: ignore[arg-type]

    def test_topology_filter(self) -> None:
        trials = [
            _trial(
                bps=5e9,
                source_zone="az01",
                target_zone="az01",
                rmem_max=212992,
            ),
            _trial(
                bps=8e9,
                source_zone="az01",
                target_zone="az02",
                rmem_max=67108864,
            ),
            _trial(
                bps=6e9,
                source_zone="az02",
                target_zone="az02",
                rmem_max=4194304,
            ),
        ]
        recs = recommend_configs(trials, "10g", topology="intra-az")
        assert len(recs) >= 1
        rec_ids = {r["trial_id"] for r in recs}
        for t in trials:
            if t.trial_id in rec_ids:
                assert t.topology == "intra-az"

    def test_output_includes_surviving_metric_fields(
        self,
        mixed_trials: list[TrialResult],
    ) -> None:
        recs = recommend_configs(mixed_trials, "10g", n=5)
        assert recs, "expected at least one recommendation"
        for r in recs:
            assert "mean_tcp_throughput" in r
            assert "tcp_retransmit_rate" in r

    def test_rate_metric_reranks_over_absolute_count(self) -> None:
        """Per-GB rate reorders candidates vs. absolute retransmit count.

        Trial A: 10 Gbps, 1000 retransmits over 37.5 GB -> rate ~26.67/GB.
        Trial B: 5 Gbps, 100 retransmits over 18.75 GB -> rate ~5.33/GB.

        On absolute count A looks bad (1000 > 100), but per GB A is
        5x cleaner than B. Under the default (rate) objectives A wins
        on throughput AND ties/beats B on rate, so A is ranked first.
        """

        def _mk(
            tp_gbps: float,
            retx: int,
            bytes_: int,
            tid: str,
        ) -> TrialResult:
            return TrialResult(
                trial_id=tid,
                node_pair=NodePair(source="a", target="b", hardware_class="10g"),
                sysctl_values={"net.core.rmem_max": 212992 + (1 if tid == "a" else 2)},
                config=BenchmarkConfig(iterations=1),
                results=[
                    BenchmarkResult(
                        timestamp=datetime.now(UTC),
                        mode="tcp",
                        bits_per_second=tp_gbps * 1e9,
                        retransmits=retx,
                        bytes_sent=bytes_,
                    ),
                ],
            )

        trials = [
            _mk(10.0, 1000, 37_500_000_000, "a"),
            _mk(5.0, 100, 18_750_000_000, "b"),
        ]
        recs = recommend_configs(trials, "10g", n=2)
        assert [r["trial_id"] for r in recs] == ["a", "b"]
        assert recs[0]["tcp_retransmit_rate"] == pytest.approx(
            1000 * 1e9 / 37_500_000_000,
        )

    def test_metric_to_df_column_covers_default_objectives(self) -> None:
        expected = {
            METRIC_TO_DF_COLUMN[m]
            for m in (
                "tcp_throughput",
                "udp_throughput",
                "tcp_retransmit_rate",
                "udp_loss_rate",
                "udp_jitter",
                "rps",
                "latency_p50",
                "latency_p90",
                "latency_p99",
            )
        }
        default_cols = {col for col, _ in DEFAULT_OBJECTIVES}
        assert default_cols == expected

    def test_custom_weights_reorders_recommendations(
        self,
        mixed_trials: list[TrialResult],
    ) -> None:
        default_recs = recommend_configs(mixed_trials, "10g", n=5)
        heavy_rate_recs = recommend_configs(
            mixed_trials,
            "10g",
            n=5,
            weights={
                "tcp_retransmit_rate": 5.0,
                "udp_jitter": 0.1,
                "latency_p99": 0.15,
            },
        )
        assert default_recs != heavy_rate_recs or all(
            a["score"] != b["score"]
            for a, b in zip(default_recs, heavy_rate_recs, strict=False)
        )

    def test_reduced_pareto_drops_rate_from_scoring(
        self,
        mixed_trials: list[TrialResult],
    ) -> None:
        reduced = [
            ParetoObjective(metric="tcp_throughput", direction="maximize"),
            ParetoObjective(metric="udp_jitter", direction="minimize"),
        ]
        recs = recommend_configs(
            mixed_trials,
            "10g",
            n=5,
            objectives=reduced,
            weights={"udp_jitter": 0.15},
        )
        assert recs
        for r in recs:
            assert set(r.keys()) >= {
                "mean_tcp_throughput",
                "tcp_retransmit_rate",
                "mean_udp_jitter",
            }

    def test_matches_pareto_recommendation_rows_wrapper(
        self,
        mixed_trials: list[TrialResult],
    ) -> None:
        """``recommend_configs`` is a top-N slice of ``pareto_recommendation_rows``."""
        from kube_autotuner.report.analysis import (  # noqa: PLC0415
            pareto_recommendation_rows,
        )

        rows = pareto_recommendation_rows(mixed_trials, "10g")
        recs = recommend_configs(mixed_trials, "10g", n=3)
        assert len(recs) == min(3, len(rows))

        # trial ordering identical
        assert [r["trial_id"] for r in recs] == [r["trial_id"] for r in rows[:3]]

        # unrounded helper scores round to the wrapper's 4-decimal output
        for i, (rec, row) in enumerate(zip(recs, rows[:3], strict=True)):
            assert rec["score"] == round(row["score"], 4)
            assert rec["rank"] == i + 1
            # every metric key carries through unchanged
            for key in (
                "mean_tcp_throughput",
                "tcp_retransmit_rate",
                "mean_udp_jitter",
                "mean_rps",
                "mean_latency_p50",
                "mean_latency_p90",
                "mean_latency_p99",
            ):
                assert rec[key] == row[key]


# --- _build_axis_payload -------------------------------------------------


class TestBuildAxisPayload:
    def test_coerces_nan_inf_and_none_to_none(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "trial_id": 1,
                    "mean_tcp_throughput": 4.2e10,
                    "tcp_retransmit_rate": 1.0,
                    "mean_udp_jitter": 0.001,
                },
                {
                    "trial_id": 2,
                    "mean_tcp_throughput": float("nan"),
                    "tcp_retransmit_rate": math.inf,
                    "mean_udp_jitter": -math.inf,
                },
                {
                    "trial_id": 3,
                    "mean_tcp_throughput": None,
                    "tcp_retransmit_rate": 2.0,
                    "mean_udp_jitter": None,
                },
            ],
        )
        pareto_mask = pd.Series([True, False, True])

        rows, axis_columns = _build_axis_payload(df, pareto_mask, stages=ALL_STAGES)

        assert axis_columns == [
            "mean_tcp_throughput",
            "tcp_retransmit_rate",
            "mean_udp_jitter",
        ]
        assert rows[0] == {
            "trial_id": "1",
            "pareto": True,
            "phase": "unknown",
            "mean_tcp_throughput": 4.2e10,
            "mean_tcp_throughput_std": None,
            "tcp_retransmit_rate": 1.0,
            "tcp_retransmit_rate_std": None,
            "mean_udp_jitter": 0.001,
            "mean_udp_jitter_std": None,
        }
        assert rows[1] == {
            "trial_id": "2",
            "pareto": False,
            "phase": "unknown",
            "mean_tcp_throughput": None,
            "mean_tcp_throughput_std": None,
            "tcp_retransmit_rate": None,
            "tcp_retransmit_rate_std": None,
            "mean_udp_jitter": None,
            "mean_udp_jitter_std": None,
        }
        assert rows[2] == {
            "trial_id": "3",
            "pareto": True,
            "phase": "unknown",
            "mean_tcp_throughput": None,
            "mean_tcp_throughput_std": None,
            "tcp_retransmit_rate": 2.0,
            "tcp_retransmit_rate_std": None,
            "mean_udp_jitter": None,
            "mean_udp_jitter_std": None,
        }

    def test_drops_all_null_columns_from_axis_columns(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "trial_id": "t1",
                    "mean_tcp_throughput": 1e10,
                    "udp_loss_rate": None,
                },
            ],
        )
        rows, axis_columns = _build_axis_payload(
            df,
            pd.Series([True]),
            stages=ALL_STAGES,
        )

        assert axis_columns == ["mean_tcp_throughput"]
        assert "udp_loss_rate" not in rows[0]

    def test_filters_by_stages(self) -> None:
        """Metric columns from disabled stages drop out of axis_columns."""
        df = pd.DataFrame(
            [
                {
                    "trial_id": "t1",
                    "mean_tcp_throughput": 1.0e10,
                    "tcp_retransmit_rate": 1.0,
                    "mean_udp_jitter": 1e-4,
                    "udp_loss_rate": 1e-3,
                    "mean_udp_throughput": 5e9,
                    "mean_rps": 1000.0,
                    "mean_latency_p50": 1.0,
                    "mean_latency_p90": 2.0,
                    "mean_latency_p99": 4.0,
                },
            ],
        )
        _, axis_columns = _build_axis_payload(
            df,
            pd.Series([True]),
            stages=frozenset({"bw-tcp"}),
        )
        assert axis_columns == ["mean_tcp_throughput", "tcp_retransmit_rate"]

    def test_json_dumps_allow_nan_false_roundtrips(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "trial_id": "t1",
                    "mean_tcp_throughput": float("nan"),
                    "tcp_retransmit_rate": math.inf,
                },
            ],
        )
        rows, _ = _build_axis_payload(df, pd.Series([True]), stages=ALL_STAGES)

        # The whole point of the scrubbing: json.dumps with allow_nan=False
        # must not raise ValueError on the output.
        encoded = json.dumps(rows, allow_nan=False)
        assert json.loads(encoded) == rows


# --- CLI analyze command -------------------------------------------------


class TestCLIAnalyze:
    def test_analyze_writes_output(
        self,
        mixed_trials: list[TrialResult],
        tmp_path: Path,
    ) -> None:
        dataset = tmp_path / "trials"
        for t in mixed_trials:
            TrialLog.append(dataset, t)

        out_dir = tmp_path / "out"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "analyze",
                str(dataset),
                "-o",
                str(out_dir),
                "--hardware-class",
                "10g",
            ],
        )
        assert result.exit_code == 0, result.output

        hw_dir = out_dir / "10g"
        assert (hw_dir / "recommendations.json").exists()
        assert (hw_dir / "importance.json").exists()

        recs = json.loads((hw_dir / "recommendations.json").read_text())
        assert len(recs) >= 1

    def test_analyze_auto_detects_classes(
        self,
        mixed_trials: list[TrialResult],
        tmp_path: Path,
    ) -> None:
        dataset = tmp_path / "trials"
        for t in mixed_trials:
            TrialLog.append(dataset, t)

        out_dir = tmp_path / "out"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["analyze", str(dataset), "-o", str(out_dir)],
        )
        assert result.exit_code == 0, result.output

        assert (out_dir / "10g" / "recommendations.json").exists()
        assert (out_dir / "1g" / "recommendations.json").exists()

    def test_analyze_with_topology_filter(self, tmp_path: Path) -> None:
        trials = [
            _trial(
                bps=5e9,
                source_zone="az01",
                target_zone="az01",
                rmem_max=212992,
            ),
            _trial(
                bps=6e9,
                source_zone="az01",
                target_zone="az02",
                rmem_max=4194304,
            ),
            _trial(
                bps=7e9,
                source_zone="az02",
                target_zone="az02",
                rmem_max=16777216,
            ),
        ]
        dataset = tmp_path / "trials"
        for t in trials:
            TrialLog.append(dataset, t)

        out_dir = tmp_path / "out"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "analyze",
                str(dataset),
                "-o",
                str(out_dir),
                "--hardware-class",
                "10g",
                "--topology",
                "intra-az",
            ],
        )
        assert result.exit_code == 0, result.output

        hw_dir = out_dir / "10g"
        recs = json.loads((hw_dir / "recommendations.json").read_text())
        assert len(recs) >= 1

    def test_analyze_reads_sidecar_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        """End-to-end: dataset + sidecar → analyze → custom weights applied.

        Two non-dominated trials differ in throughput and retransmit
        rate. With a heavy retransmit_rate weight the lower-retx trial
        wins over the higher-throughput one.
        """

        def _mk(tp_gbps: float, retx: int, tid: str) -> TrialResult:
            return TrialResult(
                trial_id=tid,
                node_pair=NodePair(source="a", target="b", hardware_class="10g"),
                sysctl_values={"net.core.rmem_max": 212992 + retx},
                config=BenchmarkConfig(iterations=1),
                results=[
                    BenchmarkResult(
                        timestamp=datetime.now(UTC),
                        mode="tcp",
                        bits_per_second=tp_gbps * 1e9,
                        retransmits=retx,
                        bytes_sent=1_000_000_000,
                    ),
                ],
            )

        trials = [_mk(10.0, 500, "hi-tp"), _mk(5.0, 0, "lo-retx")]
        dataset = tmp_path / "trials"
        for t in trials:
            TrialLog.append(dataset, t)

        section = ObjectivesSection(
            pareto=[
                ParetoObjective(metric="tcp_throughput", direction="maximize"),
                ParetoObjective(metric="tcp_retransmit_rate", direction="minimize"),
            ],
            constraints=[],
            recommendation_weights={"tcp_retransmit_rate": 5.0},
        )
        TrialLog.write_resume_metadata(
            dataset,
            ResumeMetadata(
                objectives=section,
                param_space=ParamSpace(params=[]),
                benchmark=BenchmarkConfig(iterations=1),
                iperf=IperfSection(duration=10),
                fortio=FortioSection(),
            ),
        )

        out_dir = tmp_path / "out"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "analyze",
                str(dataset),
                "-o",
                str(out_dir),
                "--hardware-class",
                "10g",
            ],
        )
        assert result.exit_code == 0, result.output

        recs = json.loads((out_dir / "10g" / "recommendations.json").read_text())
        assert recs[0]["trial_id"] == "lo-retx"

    def test_analyze_empty_dataset(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "analyze",
                str(empty),
                "-o",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0


# --- memory cost recommendation preference -------------------------------


def test_pareto_recommendation_rows_flips_tied_configs_by_memory_cost() -> None:
    """Two trials tied on performance; lower-memory rmem_max wins at 0.1."""
    from kube_autotuner.report.analysis import (  # noqa: PLC0415
        pareto_recommendation_rows,
    )

    big = _trial(
        hw="10g",
        bps=1.0e9,
        retransmits=0,
        rmem_max=67108864,  # 64 MiB
        trial_id="big",
    )
    small = _trial(
        hw="10g",
        bps=1.0e9,
        retransmits=0,
        rmem_max=212992,  # 208 KiB
        trial_id="small",
    )

    # weight=0.0 disables the memory-cost term; lexicographic tiebreak
    # on trial_id puts "big" ahead of "small" (ascending).
    rows_disabled = pareto_recommendation_rows(
        [big, small],
        "10g",
        memory_cost_weight=0.0,
    )
    assert [r["trial_id"] for r in rows_disabled] == ["big", "small"]

    # weight=0.1 makes the cheaper config win despite trial_id order.
    rows_enabled = pareto_recommendation_rows(
        [big, small],
        "10g",
        memory_cost_weight=0.1,
    )
    assert [r["trial_id"] for r in rows_enabled] == ["small", "big"]
    # Every row carries a ``memory_cost`` field for downstream consumers.
    assert rows_enabled[0]["memory_cost"] == pytest.approx(212992)
    assert rows_enabled[1]["memory_cost"] == pytest.approx(67108864)


# --- host_state_series ---------------------------------------------------


def _snap(
    iteration: int | None,
    phase: Literal["baseline", "post-flush", "post-iteration"],
    metrics: dict[str, int],
    timestamp: datetime | None = None,
) -> HostStateSnapshot:
    kwargs: dict[str, Any] = {
        "node": "node-a",
        "iteration": iteration,
        "phase": phase,
        "metrics": metrics,
    }
    if timestamp is not None:
        kwargs["timestamp"] = timestamp
    return HostStateSnapshot(**kwargs)


def _trial_with_snapshots(
    *,
    trial_id: str,
    hw: str,
    snapshots: list[HostStateSnapshot],
    topology: Literal["intra-az", "inter-az", "unknown"] = "unknown",
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        node_pair=NodePair(source="a", target="b", hardware_class=hw),
        sysctl_values={"net.core.rmem_max": 212992},
        topology=topology,
        config=BenchmarkConfig(iterations=1),
        results=[_result(1e9)],
        host_state_snapshots=snapshots,
    )


def test_host_state_series_returns_none_when_no_snapshots() -> None:
    trials = [_trial(hw="10g", trial_id="t1")]
    assert host_state_series(trials, "10g", topology=None) is None


def test_host_state_series_returns_none_when_metric_union_is_empty() -> None:
    trial = _trial_with_snapshots(
        trial_id="empty",
        hw="10g",
        snapshots=[_snap(None, "baseline", {})],
    )
    assert host_state_series([trial], "10g", topology=None) is None


def test_host_state_series_filters_by_hardware_class_and_topology() -> None:
    snaps = [_snap(None, "baseline", {"conntrack_count": 10})]
    in_class = _trial_with_snapshots(
        trial_id="keep",
        hw="10g",
        snapshots=snaps,
        topology="intra-az",
    )
    other_class = _trial_with_snapshots(
        trial_id="drop-hw",
        hw="1g",
        snapshots=snaps,
    )
    other_topo = _trial_with_snapshots(
        trial_id="drop-topo",
        hw="10g",
        snapshots=snaps,
        topology="inter-az",
    )
    payload = host_state_series(
        [in_class, other_class, other_topo],
        "10g",
        topology="intra-az",
    )
    assert payload is not None
    trial_ids = {p["trial_id"] for p in payload["points"]}
    assert trial_ids == {"keep"}


def test_host_state_series_preserves_baseline_iteration_none() -> None:
    trial = _trial_with_snapshots(
        trial_id="t1",
        hw="10g",
        snapshots=[
            _snap(
                None,
                "baseline",
                {"conntrack_count": 10},
                timestamp=datetime(2026, 4, 24, 10, 0, 0, tzinfo=UTC),
            ),
            _snap(
                0,
                "post-flush",
                {"conntrack_count": 11},
                timestamp=datetime(2026, 4, 24, 10, 0, 1, tzinfo=UTC),
            ),
            _snap(
                0,
                "post-iteration",
                {"conntrack_count": 13},
                timestamp=datetime(2026, 4, 24, 10, 0, 2, tzinfo=UTC),
            ),
        ],
    )
    payload = host_state_series([trial], "10g", topology=None)
    assert payload is not None
    points = payload["points"]
    assert points[0]["iteration"] is None
    assert points[0]["phase"] == "baseline"
    assert points[1]["iteration"] == 0
    # JSON round-trip keeps the None as null and survives allow_nan=False.
    dumped = json.dumps(payload, allow_nan=False)
    assert '"iteration": null' in dumped or '"iteration":null' in dumped


def test_host_state_series_metric_union_is_sorted() -> None:
    trial_a = _trial_with_snapshots(
        trial_id="a",
        hw="10g",
        snapshots=[_snap(None, "baseline", {"zzz": 1, "aaa": 2})],
    )
    trial_b = _trial_with_snapshots(
        trial_id="b",
        hw="10g",
        snapshots=[_snap(0, "post-flush", {"mmm": 3, "aaa": 4})],
    )
    payload = host_state_series([trial_a, trial_b], "10g", topology=None)
    assert payload is not None
    assert payload["metrics"] == ["aaa", "mmm", "zzz"]


def test_host_state_series_points_sorted_by_timestamp() -> None:
    # Two trials with deliberately interleaved capture times: trial_a's
    # second snapshot lands between trial_b's two snapshots. The flat
    # output must be monotonic in timestamp regardless of trial order.
    trial_a = _trial_with_snapshots(
        trial_id="a",
        hw="10g",
        snapshots=[
            _snap(
                None,
                "baseline",
                {"conntrack_count": 1},
                timestamp=datetime(2026, 4, 24, 10, 0, 0, tzinfo=UTC),
            ),
            _snap(
                0,
                "post-iteration",
                {"conntrack_count": 3},
                timestamp=datetime(2026, 4, 24, 10, 0, 10, tzinfo=UTC),
            ),
        ],
    )
    trial_b = _trial_with_snapshots(
        trial_id="b",
        hw="10g",
        snapshots=[
            _snap(
                None,
                "baseline",
                {"conntrack_count": 2},
                timestamp=datetime(2026, 4, 24, 10, 0, 5, tzinfo=UTC),
            ),
            _snap(
                0,
                "post-iteration",
                {"conntrack_count": 4},
                timestamp=datetime(2026, 4, 24, 10, 0, 15, tzinfo=UTC),
            ),
        ],
    )
    # Reverse the trial order to prove sorting is on timestamp, not input.
    payload = host_state_series([trial_b, trial_a], "10g", topology=None)
    assert payload is not None
    points = payload["points"]
    timestamps = [p["timestamp"] for p in points]
    assert timestamps == sorted(timestamps)
    trial_ids = [p["trial_id"] for p in points]
    assert trial_ids == ["a", "b", "a", "b"]


def test_host_state_series_sorts_when_timestamps_mix_naive_and_aware() -> None:
    # Replayed older JSON may carry a naive timestamp alongside the
    # aware default. Raw `sorted` would raise TypeError; the function
    # must normalize so the chronological order still holds.
    trial = _trial_with_snapshots(
        trial_id="t1",
        hw="10g",
        snapshots=[
            _snap(
                None,
                "baseline",
                {"conntrack_count": 1},
                timestamp=datetime(2026, 4, 24, 10, 0, 0),  # naive  # noqa: DTZ001
            ),
            _snap(
                0,
                "post-iteration",
                {"conntrack_count": 3},
                timestamp=datetime(2026, 4, 24, 10, 0, 5, tzinfo=UTC),  # aware
            ),
        ],
    )
    payload = host_state_series([trial], "10g", topology=None)
    assert payload is not None
    phases = [p["phase"] for p in payload["points"]]
    assert phases == ["baseline", "post-iteration"]


def test_host_state_series_timestamp_is_iso_string() -> None:
    source = datetime(2026, 4, 24, 17, 32, 10, 123456, tzinfo=UTC)
    trial = _trial_with_snapshots(
        trial_id="t1",
        hw="10g",
        snapshots=[_snap(None, "baseline", {"conntrack_count": 1}, timestamp=source)],
    )
    payload = host_state_series([trial], "10g", topology=None)
    assert payload is not None
    ts = payload["points"][0]["timestamp"]
    assert isinstance(ts, str)
    assert datetime.fromisoformat(ts) == source
    # allow_nan=False is the report's serialization gate; a string
    # timestamp must round-trip cleanly.
    json.dumps(payload, allow_nan=False)


# --- baseline_comparison -------------------------------------------------


def _defaults_trial(
    *,
    bps: float = 1e10,
    trial_id: str = "baseline",
    phase: Literal["sobol", "bayesian", "refinement"] | None = None,
    parent_trial_id: str | None = None,
) -> TrialResult:
    from kube_autotuner.sysctl.params import RECOMMENDED_DEFAULTS  # noqa: PLC0415

    kwargs: dict[str, Any] = {
        "trial_id": trial_id,
        "node_pair": NodePair(source="a", target="b", hardware_class="10g"),
        "sysctl_values": dict(RECOMMENDED_DEFAULTS),
        "config": BenchmarkConfig(iterations=2),
        "results": [_result(bps)],
    }
    if phase is not None:
        kwargs["phase"] = phase
    if parent_trial_id is not None:
        kwargs["parent_trial_id"] = parent_trial_id
    return TrialResult(**kwargs)


_SIMPLE_OBJECTIVES: list[dict[str, str]] = [
    {"metric": "tcp_throughput", "direction": "maximize"},
    {"metric": "tcp_retransmit_rate", "direction": "minimize"},
]


def test_baseline_comparison_returns_none_when_no_match() -> None:
    trials = [_trial(hw="10g", bps=5e9, rmem_max=212992)]
    top_row = {"mean_tcp_throughput": 6e9, "tcp_retransmit_rate": 0.1}
    assert baseline_comparison(trials, _SIMPLE_OBJECTIVES, top_row) is None


def test_baseline_comparison_single_match_emits_deltas() -> None:
    baseline = _defaults_trial(bps=5e9, trial_id="b1")
    candidate = _trial(hw="10g", bps=1e10, rmem_max=212992)
    top_row = {"mean_tcp_throughput": 1e10, "tcp_retransmit_rate": 0.0}
    entries = baseline_comparison(
        [baseline, candidate],
        _SIMPLE_OBJECTIVES,
        top_row,
    )
    assert entries is not None
    by_metric = {e["metric"]: e for e in entries}
    assert by_metric["tcp_throughput"]["baseline"] == pytest.approx(5e9)
    assert by_metric["tcp_throughput"]["recommended"] == pytest.approx(1e10)
    assert by_metric["tcp_throughput"]["abs_delta"] == pytest.approx(5e9)
    assert by_metric["tcp_throughput"]["pct_delta"] == pytest.approx(1.0)


def test_baseline_comparison_multi_match_means_across_primaries() -> None:
    b1 = _defaults_trial(bps=4e9, trial_id="b1")
    b2 = _defaults_trial(bps=6e9, trial_id="b2")
    top_row = {"mean_tcp_throughput": 1e10, "tcp_retransmit_rate": 0.0}
    entries = baseline_comparison([b1, b2], _SIMPLE_OBJECTIVES, top_row)
    assert entries is not None
    assert entries[0]["metric"] == "tcp_throughput"
    assert entries[0]["baseline"] == pytest.approx(5e9)


def test_baseline_comparison_folds_refinement_children() -> None:
    b1 = _defaults_trial(bps=4e9, trial_id="b1", phase="bayesian")
    v1 = _defaults_trial(
        bps=6e9,
        trial_id="b1-v1",
        phase="refinement",
        parent_trial_id="b1",
    )
    v2 = _defaults_trial(
        bps=8e9,
        trial_id="b1-v2",
        phase="refinement",
        parent_trial_id="b1",
    )
    top_row = {"mean_tcp_throughput": 1e10, "tcp_retransmit_rate": 0.0}
    entries = baseline_comparison([b1, v1, v2], _SIMPLE_OBJECTIVES, top_row)
    assert entries is not None
    # Aggregated refinement-adjusted baseline: mean(4e9, 6e9, 8e9) = 6e9
    assert entries[0]["baseline"] == pytest.approx(6e9)


# --- refinement_stats ----------------------------------------------------


def test_refinement_stats_skips_groups_under_two_children() -> None:
    parent = _trial(hw="10g", bps=5e9, trial_id="p1")
    child = _trial(
        hw="10g",
        bps=6e9,
        trial_id="c1",
    )
    # Manually mark child as refinement of p1
    child = child.model_copy(
        update={"phase": "refinement", "parent_trial_id": "p1"},
    )
    stats = refinement_stats([parent, child])
    assert stats == {}


def test_refinement_stats_computes_mean_stdev_cv() -> None:
    parent = _trial(hw="10g", bps=5e9, trial_id="p1")
    children = [
        _trial(hw="10g", bps=bps, trial_id=f"c{i}").model_copy(
            update={"phase": "refinement", "parent_trial_id": "p1"},
        )
        for i, bps in enumerate([6e9, 8e9, 10e9])
    ]
    stats = refinement_stats([parent, *children])
    assert "p1" in stats
    tp = stats["p1"]["mean_tcp_throughput"]
    assert tp["mean"] == pytest.approx(8e9)
    # stdev of [6e9, 8e9, 10e9] == 2e9
    assert tp["stdev"] == pytest.approx(2e9)
    assert tp["cv"] == pytest.approx(0.25)


def test_refinement_stats_zero_mean_cv_is_none() -> None:
    parent = _trial(hw="10g", bps=5e9, trial_id="p1")
    # Retransmit rate is 0 in all children -> mean 0, cv should be None.
    children = [
        _trial(
            hw="10g",
            bps=5e9,
            retransmits=0,
            trial_id=f"c{i}",
            bytes_sent=_DEFAULT_BYTES_SENT,
        ).model_copy(update={"phase": "refinement", "parent_trial_id": "p1"})
        for i in range(3)
    ]
    stats = refinement_stats([parent, *children])
    assert stats["p1"]["tcp_retransmit_rate"]["mean"] == pytest.approx(0.0)
    assert stats["p1"]["tcp_retransmit_rate"]["cv"] is None


# --- stability_badge -----------------------------------------------------


def test_stability_badge_unverified_when_no_stats() -> None:
    assert stability_badge(None) == "unverified"
    assert stability_badge({}) == "unverified"


def test_stability_badge_unverified_when_all_cvs_none() -> None:
    entry = {"tcp_throughput": {"mean": 0.0, "stdev": 0.0, "cv": None}}
    assert stability_badge(entry) == "unverified"


@pytest.mark.parametrize(
    ("cv", "expected"),
    [
        (0.04, "green"),
        (0.0499, "green"),
        (0.05, "amber"),
        (0.149, "amber"),
        (0.15, "red"),
        (0.5, "red"),
    ],
)
def test_stability_badge_thresholds(cv: float, expected: str) -> None:
    entry: dict[str, dict[str, float | None]] = {
        "tcp_throughput": {"mean": 1.0, "stdev": cv, "cv": cv},
    }
    assert stability_badge(entry) == expected


# --- trajectory_rows -----------------------------------------------------


def test_trajectory_rows_running_best_cummax() -> None:
    t1 = _trial(hw="10g", bps=5e9, trial_id="t1")
    t2 = _trial(hw="10g", bps=3e9, trial_id="t2")
    t3 = _trial(hw="10g", bps=7e9, trial_id="t3")
    t1 = t1.model_copy(update={"created_at": datetime(2026, 1, 1, tzinfo=UTC)})
    t2 = t2.model_copy(update={"created_at": datetime(2026, 1, 2, tzinfo=UTC)})
    t3 = t3.model_copy(update={"created_at": datetime(2026, 1, 3, tzinfo=UTC)})
    rows = trajectory_rows(
        [t2, t1, t3],
        [{"metric": "tcp_throughput", "direction": "maximize"}],
        resume_metadata=None,
    )
    assert [r["trial_id"] for r in rows] == ["t1", "t2", "t3"]
    bests = [r["mean_tcp_throughput_best_so_far"] for r in rows]
    assert bests == pytest.approx([5e9, 5e9, 7e9])


def test_trajectory_rows_phase_fallback_to_unknown_without_metadata() -> None:
    t = _trial(hw="10g", bps=5e9, trial_id="t1")
    rows = trajectory_rows(
        [t],
        [{"metric": "tcp_throughput", "direction": "maximize"}],
        resume_metadata=None,
    )
    assert rows[0]["phase_effective"] == "unknown"


def test_trajectory_rows_skips_refinement_repeats() -> None:
    primary = _trial(hw="10g", bps=5e9, trial_id="p1")
    verif = _trial(hw="10g", bps=6e9, trial_id="v1").model_copy(
        update={"phase": "refinement", "parent_trial_id": "p1"},
    )
    rows = trajectory_rows(
        [primary, verif],
        [{"metric": "tcp_throughput", "direction": "maximize"}],
        resume_metadata=None,
    )
    assert [r["trial_id"] for r in rows] == ["p1"]


# --- section_metadata ----------------------------------------------------


def _resume_metadata(
    *,
    iperf_duration: int = 10,
    fortio_duration: int = 7,
) -> ResumeMetadata:
    return ResumeMetadata(
        objectives=ObjectivesSection(
            pareto=[ParetoObjective(metric="tcp_throughput", direction="maximize")],
            constraints=[],
            recommendation_weights={},
        ),
        param_space=ParamSpace(params=[]),
        benchmark=BenchmarkConfig(),
        iperf=IperfSection(duration=iperf_duration),
        fortio=FortioSection(duration=fortio_duration),
    )


def test_section_metadata_uniform_fields() -> None:
    t1 = _trial(hw="10g", bps=5e9, trial_id="t1")
    t2 = _trial(hw="10g", bps=6e9, trial_id="t2")
    md = section_metadata([t1, t2], resume_metadata=_resume_metadata())
    assert md["trial_count"] == 2
    assert md["iperf_duration"] == 10
    assert md["fortio_duration"] == 7
    assert md["iterations"] == 1
    assert isinstance(md["stages"], list)
    assert md["kernel_version"] is None  # default "" -> None


def test_section_metadata_durations_none_without_resume_metadata() -> None:
    t1 = _trial(hw="10g", bps=5e9, trial_id="t1")
    md = section_metadata([t1], resume_metadata=None)
    assert md["iperf_duration"] is None
    assert md["fortio_duration"] is None


def test_section_metadata_mixed_fields_render_mixed() -> None:
    t1 = _trial(hw="10g", bps=5e9, trial_id="t1")
    t2 = _trial(hw="10g", bps=6e9, trial_id="t2")
    t2 = t2.model_copy(
        update={
            "config": BenchmarkConfig(iterations=5),
        },
    )
    md = section_metadata([t1, t2], resume_metadata=None)
    assert md["iterations"] == "mixed"


def test_section_metadata_mixed_kernel_renders_mixed() -> None:
    t1 = _trial(hw="10g", bps=5e9, trial_id="t1").model_copy(
        update={"kernel_version": "6.1.0"},
    )
    t2 = _trial(hw="10g", bps=6e9, trial_id="t2").model_copy(
        update={"kernel_version": "6.2.0"},
    )
    md = section_metadata([t1, t2], resume_metadata=None)
    assert md["kernel_version"] == "mixed"


def test_section_metadata_all_empty_kernel_is_none() -> None:
    t1 = _trial(hw="10g", bps=5e9, trial_id="t1")
    md = section_metadata([t1], resume_metadata=None)
    assert md["kernel_version"] is None


def test_section_metadata_counts_refinement_separately() -> None:
    p1 = _trial(hw="10g", bps=5e9, trial_id="p1")
    v1 = _trial(hw="10g", bps=6e9, trial_id="v1").model_copy(
        update={"phase": "refinement", "parent_trial_id": "p1"},
    )
    md = section_metadata([p1, v1], resume_metadata=None)
    assert md["trial_count"] == 1
    assert md["phase_counts"]["refinement"] == 1


# --- sysctl_correlation_matrix ------------------------------------------


def test_sysctl_correlation_matrix_returns_none_below_two_cols(
    mixed_trials: list[TrialResult],
) -> None:
    df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
    # Empty importance frames -> None
    assert sysctl_correlation_matrix(df, {}) is None


def test_sysctl_correlation_matrix_drops_constant_cols(
    mixed_trials: list[TrialResult],
) -> None:
    df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
    # Make one column constant
    df = df.copy()
    df["net.core.rmem_max"] = 212992
    imp_frame = pd.DataFrame(
        [
            {
                "param": "net.core.rmem_max",
                "category": "tcp_buffer",
                "spearman_r": 0.0,
                "rf_importance": 0.0,
            },
            {
                "param": "net.ipv4.tcp_congestion_control",
                "category": "congestion",
                "spearman_r": 0.0,
                "rf_importance": 0.0,
            },
        ],
    )
    assert sysctl_correlation_matrix(df, {"mean_tcp_throughput": imp_frame}) is None


def test_sysctl_correlation_matrix_shape(
    mixed_trials: list[TrialResult],
) -> None:
    df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
    imp_frame = pd.DataFrame(
        [
            {
                "param": "net.core.rmem_max",
                "category": "tcp_buffer",
                "spearman_r": 0.0,
                "rf_importance": 0.0,
            },
            {
                "param": "net.ipv4.tcp_congestion_control",
                "category": "congestion",
                "spearman_r": 0.0,
                "rf_importance": 0.0,
            },
        ],
    )
    matrix = sysctl_correlation_matrix(df, {"mean_tcp_throughput": imp_frame})
    assert matrix is not None
    assert list(matrix.columns) == list(matrix.index)
    # Diagonal is 1.0
    for c in matrix.columns:
        assert matrix.loc[c, c] == pytest.approx(1.0)


# --- host_state_issues ---------------------------------------------------


def test_host_state_issues_empty_when_no_errors() -> None:
    t = _trial_with_snapshots(
        trial_id="t1",
        hw="10g",
        snapshots=[_snap(None, "baseline", {"conntrack_count": 1})],
    )
    assert host_state_issues([t]) == []


def test_host_state_issues_flattens_errors() -> None:
    snap = HostStateSnapshot(
        node="node-a",
        iteration=0,
        phase="post-iteration",
        metrics={},
        errors=["conntrack parse failed", "sockstat missing key"],
    )
    t = _trial_with_snapshots(trial_id="t1", hw="10g", snapshots=[snap])
    issues = host_state_issues([t])
    assert len(issues) == 2
    assert issues[0]["trial_id"] == "t1"
    assert issues[0]["phase"] == "post-iteration"
    assert issues[0]["error_text"] == "conntrack parse failed"


def test_host_state_issues_truncates_long_lines() -> None:
    long_err = "x" * 500
    snap = HostStateSnapshot(
        node="node-a",
        iteration=0,
        phase="post-iteration",
        metrics={},
        errors=[long_err],
    )
    t = _trial_with_snapshots(trial_id="t1", hw="10g", snapshots=[snap])
    issues = host_state_issues([t])
    assert len(issues[0]["error_text"]) == 241
    assert issues[0]["error_text"].endswith("…")


# --- category_importance_rollup -----------------------------------------


def test_category_importance_rollup_sums_by_category() -> None:
    frame = pd.DataFrame(
        [
            {
                "param": "net.core.rmem_max",
                "category": "tcp_buffer",
                "spearman_r": 0.5,
                "rf_importance": 0.4,
            },
            {
                "param": "net.ipv4.tcp_wmem",
                "category": "tcp_buffer",
                "spearman_r": 0.3,
                "rf_importance": 0.2,
            },
            {
                "param": "net.ipv4.tcp_congestion_control",
                "category": "congestion",
                "spearman_r": 0.1,
                "rf_importance": 0.05,
            },
        ],
    )
    rollup = category_importance_rollup({"mean_tcp_throughput": frame})
    assert "mean_tcp_throughput" in rollup
    entries = rollup["mean_tcp_throughput"]
    assert entries[0]["category"] == "tcp_buffer"
    assert entries[0]["rf_sum"] == pytest.approx(0.6)
    assert entries[1]["category"] == "congestion"
    assert entries[1]["rf_sum"] == pytest.approx(0.05)


def test_category_importance_rollup_skips_empty_frames() -> None:
    frame = pd.DataFrame()
    rollup = category_importance_rollup({"mean_tcp_throughput": frame})
    assert rollup == {}


# --- trials_to_dataframe std columns ------------------------------------


def test_trials_to_dataframe_emits_std_columns_when_multi_iteration() -> None:
    trial = TrialResult(
        trial_id="t1",
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values={"net.core.rmem_max": 212992},
        config=BenchmarkConfig(iterations=3),
        results=[
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=bps,
                retransmits=0,
                bytes_sent=1_000_000_000,
                iteration=i,
            )
            for i, bps in enumerate([5e9, 7e9, 9e9])
        ],
    )
    df, _ = trials_to_dataframe([trial], hardware_class="10g")
    assert "mean_tcp_throughput_std" in df.columns
    # stdev of [5e9, 7e9, 9e9] is 2e9
    assert df.iloc[0]["mean_tcp_throughput_std"] == pytest.approx(2e9)


def test_trials_to_dataframe_single_iteration_std_is_none() -> None:
    trial = _trial(hw="10g", bps=5e9, trial_id="t1")  # iterations=1
    df, _ = trials_to_dataframe([trial], hardware_class="10g")
    assert df.iloc[0]["mean_tcp_throughput_std"] is None or math.isnan(
        df.iloc[0]["mean_tcp_throughput_std"],
    )


# --- per_iteration_samples ----------------------------------------------


def _bench(  # noqa: PLR0913 - per-record fields land here as kwargs
    *,
    mode: Literal["tcp", "udp"],
    iteration: int,
    bps: float,
    retransmits: int | None = None,
    bytes_sent: int | None = None,
    packets: int | None = None,
    lost_packets: int | None = None,
    jitter: float | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        timestamp=datetime.now(UTC),
        mode=mode,
        bits_per_second=bps,
        iteration=iteration,
        retransmits=retransmits,
        bytes_sent=bytes_sent,
        packets=packets,
        lost_packets=lost_packets,
        jitter=jitter,
    )


def _multi_iter_trial(
    *,
    trial_id: str,
    bps_per_iter: list[float],
    retx_per_iter: list[int] | None = None,
    bytes_sent: int = _DEFAULT_BYTES_SENT,
    phase: Literal["sobol", "bayesian", "refinement"] | None = None,
    parent_trial_id: str | None = None,
) -> TrialResult:
    retx_vals = retx_per_iter if retx_per_iter is not None else [0] * len(bps_per_iter)
    kwargs: dict[str, Any] = {
        "trial_id": trial_id,
        "node_pair": NodePair(source="a", target="b", hardware_class="10g"),
        "sysctl_values": {"net.core.rmem_max": 212992},
        "config": BenchmarkConfig(iterations=len(bps_per_iter)),
        "results": [
            _bench(
                mode="tcp",
                iteration=i,
                bps=bps,
                retransmits=retx,
                bytes_sent=bytes_sent,
            )
            for i, (bps, retx) in enumerate(zip(bps_per_iter, retx_vals, strict=True))
        ],
    }
    if phase is not None:
        kwargs["phase"] = phase
    if parent_trial_id is not None:
        kwargs["parent_trial_id"] = parent_trial_id
    return TrialResult(**kwargs)


def test_per_iteration_samples_primary_only_emits_one_row_per_iteration() -> None:
    trial = _multi_iter_trial(
        trial_id="p1",
        bps_per_iter=[5e9, 7e9, 9e9],
    )
    out = per_iteration_samples([trial])
    assert list(out.keys()) == ["p1"]
    rows = out["p1"]
    assert [r["iteration"] for r in rows] == [0, 1, 2]
    assert [r["trial_id"] for r in rows] == ["p1", "p1", "p1"]
    assert [r["mean_tcp_throughput"] for r in rows] == pytest.approx([5e9, 7e9, 9e9])


def test_per_iteration_samples_preserves_iteration_index_across_children() -> None:
    primary = _multi_iter_trial(
        trial_id="p1",
        bps_per_iter=[5e9, 6e9, 7e9],
        phase="bayesian",
    )
    primary = primary.model_copy(
        update={"created_at": datetime(2026, 1, 1, tzinfo=UTC)},
    )
    children = []
    for i, base in enumerate([8e9, 9e9]):
        child = _multi_iter_trial(
            trial_id=f"p1-v{i + 1}",
            bps_per_iter=[base, base + 1e9, base + 2e9],
            phase="refinement",
            parent_trial_id="p1",
        )
        child = child.model_copy(
            update={"created_at": datetime(2026, 1, 2 + i, tzinfo=UTC)},
        )
        children.append(child)

    out = per_iteration_samples([primary, *children])
    assert list(out.keys()) == ["p1"]
    rows = out["p1"]
    assert len(rows) == 9
    # Iteration index is preserved verbatim across children, not globally
    # renumbered: 0,1,2,0,1,2,0,1,2.
    assert [r["iteration"] for r in rows] == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    # Trial id ordering follows (created_at, trial_id) so primary first,
    # then refinement children chronologically.
    assert [r["trial_id"] for r in rows] == [
        "p1",
        "p1",
        "p1",
        "p1-v1",
        "p1-v1",
        "p1-v1",
        "p1-v2",
        "p1-v2",
        "p1-v2",
    ]


def test_per_iteration_samples_sparse_metric_uses_none_without_shifting() -> None:
    """Iterations with no UDP record produce ``None`` in udp_loss_rate.

    Regression test for the zip-by-position bug: TCP records exist for
    iterations 0/1/2, UDP records exist only for iteration 1. Per
    iteration alignment must put udp_loss_rate=None for iter 0 and 2,
    not shift iter 1's value into iter 0's row.
    """
    trial = TrialResult(
        trial_id="p1",
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values={"net.core.rmem_max": 212992},
        config=BenchmarkConfig(iterations=3),
        results=[
            _bench(
                mode="tcp",
                iteration=0,
                bps=5e9,
                retransmits=0,
                bytes_sent=_DEFAULT_BYTES_SENT,
            ),
            _bench(
                mode="tcp",
                iteration=1,
                bps=5e9,
                retransmits=0,
                bytes_sent=_DEFAULT_BYTES_SENT,
            ),
            _bench(
                mode="udp",
                iteration=1,
                bps=4e9,
                packets=1000,
                lost_packets=10,
                jitter=1e-3,
            ),
            _bench(
                mode="tcp",
                iteration=2,
                bps=5e9,
                retransmits=0,
                bytes_sent=_DEFAULT_BYTES_SENT,
            ),
        ],
    )
    rows = per_iteration_samples([trial])["p1"]
    by_iter = {r["iteration"]: r for r in rows}
    assert by_iter[0]["udp_loss_rate"] is None
    assert by_iter[2]["udp_loss_rate"] is None
    assert by_iter[1]["udp_loss_rate"] == pytest.approx(0.01)
    # mean_udp_throughput on iters 0/2 also stays None.
    assert by_iter[0]["mean_udp_throughput"] is None
    assert by_iter[2]["mean_udp_throughput"] is None
    assert by_iter[1]["mean_udp_throughput"] == pytest.approx(4e9)
    # TCP cells stay populated on every iteration.
    for it in (0, 1, 2):
        assert by_iter[it]["mean_tcp_throughput"] == pytest.approx(5e9)


def test_per_iteration_samples_no_nan_in_payload() -> None:
    """Every numeric cell is a finite float or None -- never nan."""
    trial = _multi_iter_trial(
        trial_id="p1",
        bps_per_iter=[5e9, 7e9, 9e9],
    )
    out = per_iteration_samples([trial])
    # json.dumps(allow_nan=False) is the report's serialization gate.
    json.dumps(out, allow_nan=False)
    for rows in out.values():
        for row in rows:
            for col in METRIC_TO_DF_COLUMN.values():
                cell = row.get(col)
                assert cell is None or (isinstance(cell, float) and math.isfinite(cell))


def test_per_iteration_samples_consistent_with_per_trial_metric_means() -> None:
    """Per-iteration mean equals the canonical per-trial metric mean.

    Locks the reuse-of-by-iteration semantics so the parent's mean
    surfaced via :func:`_per_trial_metric_means` and the per-iteration
    drill-down view never drift.
    """
    trial = TrialResult(
        trial_id="p1",
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values={"net.core.rmem_max": 212992},
        config=BenchmarkConfig(iterations=3),
        results=[
            _bench(
                mode="tcp",
                iteration=i,
                bps=tcp_bps,
                retransmits=retx,
                bytes_sent=_DEFAULT_BYTES_SENT,
            )
            for i, (tcp_bps, retx) in enumerate([
                (5e9, 5),
                (7e9, 3),
                (9e9, 1),
            ])
        ]
        + [
            _bench(
                mode="udp",
                iteration=i,
                bps=udp_bps,
                packets=1000,
                lost_packets=lost,
                jitter=jitter,
            )
            for i, (udp_bps, lost, jitter) in enumerate([
                (4e9, 20, 1e-3),
                (4.5e9, 10, 1.2e-3),
                (5e9, 5, 1.5e-3),
            ])
        ],
        latency_results=[
            LatencyResult(
                timestamp=datetime.now(UTC),
                workload="saturation",
                iteration=i,
                rps=rps,
            )
            for i, rps in enumerate([1000.0, 1100.0, 1200.0])
        ]
        + [
            LatencyResult(
                timestamp=datetime.now(UTC),
                workload="fixed_qps",
                iteration=i,
                rps=500.0,
                latency_p50=p50,
                latency_p90=p90,
                latency_p99=p99,
            )
            for i, (p50, p90, p99) in enumerate([
                (1e-3, 2e-3, 3e-3),
                (1.5e-3, 2.5e-3, 3.5e-3),
                (2e-3, 3e-3, 4e-3),
            ])
        ],
    )
    out = per_iteration_samples([trial])["p1"]
    parent_means = _per_trial_metric_means(trial)
    for col in METRIC_TO_DF_COLUMN.values():
        per_iter = [r[col] for r in out if r[col] is not None]
        assert per_iter, f"expected per-iteration values for {col}"
        actual_mean = sum(per_iter) / len(per_iter)
        expected = parent_means[col]
        assert math.isfinite(expected), f"unexpected nan parent mean for {col}"
        assert actual_mean == pytest.approx(expected), col


def test_per_iteration_samples_empty_when_no_trials() -> None:
    assert per_iteration_samples([]) == {}


def test_per_iteration_samples_orphan_refinement_keys_on_parent_id() -> None:
    """A refinement row whose parent_trial_id does not match groups by that id."""
    orphan = _multi_iter_trial(
        trial_id="v1",
        bps_per_iter=[5e9, 6e9],
        phase="refinement",
        parent_trial_id="missing-parent",
    )
    out = per_iteration_samples([orphan])
    assert list(out.keys()) == ["missing-parent"]
    assert [r["trial_id"] for r in out["missing-parent"]] == ["v1", "v1"]
