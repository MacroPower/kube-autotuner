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

from kube_autotuner.analysis import (  # noqa: E402
    DEFAULT_OBJECTIVES,
    METRIC_TO_DF_COLUMN,
    host_state_series,
    parameter_importance,
    pareto_front,
    recommend_configs,
    split_trials_by_hardware_class,
    trials_to_dataframe,
)
from kube_autotuner.cli import _build_axis_payload, app  # noqa: E402, PLC2701
from kube_autotuner.experiment import ObjectivesSection, ParetoObjective  # noqa: E402
from kube_autotuner.models import (  # noqa: E402
    ALL_STAGES,
    BenchmarkConfig,
    BenchmarkResult,
    HostStateSnapshot,
    NodePair,
    ParamSpace,
    ResumeMetadata,
    TrialLog,
    TrialResult,
)

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
        config=BenchmarkConfig(duration=10, iterations=1),
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
                "tcp_retransmit_rate": [1e-7, float("nan")],
                **_pad_latency_cols(2),
            },
        )
        with caplog.at_level("WARNING", logger="kube_autotuner.analysis"):
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
                "tcp_retransmit_rate": [2e-7, 1e-7],
                "mean_udp_jitter": [float("nan"), float("nan")],
                "mean_rps": [1000.0] * 2,
                "mean_latency_p50": [1.0] * 2,
                "mean_latency_p90": [5.0] * 2,
                "mean_latency_p99": [10.0] * 2,
            },
        )
        with caplog.at_level("INFO", logger="kube_autotuner.analysis"):
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
        """Per-byte rate reorders candidates vs. absolute retransmit count.

        Trial A: 10 Gbps, 1000 retransmits over 37.5 GB -> rate ~2.67e-8.
        Trial B: 5 Gbps, 100 retransmits over 18.75 GB -> rate ~5.33e-9.

        On absolute count A looks bad (1000 > 100), but per byte A is
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
                config=BenchmarkConfig(duration=10, iterations=1),
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
            1000 / 37_500_000_000,
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
        from kube_autotuner.analysis import (  # noqa: PLC0415
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
                    "tcp_retransmit_rate": 1e-6,
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
                    "tcp_retransmit_rate": 2e-6,
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
            "mean_tcp_throughput": 4.2e10,
            "tcp_retransmit_rate": 1e-6,
            "mean_udp_jitter": 0.001,
        }
        assert rows[1] == {
            "trial_id": "2",
            "pareto": False,
            "mean_tcp_throughput": None,
            "tcp_retransmit_rate": None,
            "mean_udp_jitter": None,
        }
        assert rows[2] == {
            "trial_id": "3",
            "pareto": True,
            "mean_tcp_throughput": None,
            "tcp_retransmit_rate": 2e-6,
            "mean_udp_jitter": None,
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
                    "tcp_retransmit_rate": 1e-6,
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
        jsonl = tmp_path / "trials.jsonl"
        for t in mixed_trials:
            TrialLog.append(jsonl, t)

        out_dir = tmp_path / "out"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "analyze",
                "-i",
                str(jsonl),
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
        jsonl = tmp_path / "trials.jsonl"
        for t in mixed_trials:
            TrialLog.append(jsonl, t)

        out_dir = tmp_path / "out"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["analyze", "-i", str(jsonl), "-o", str(out_dir)],
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
        jsonl = tmp_path / "trials.jsonl"
        for t in trials:
            TrialLog.append(jsonl, t)

        out_dir = tmp_path / "out"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "analyze",
                "-i",
                str(jsonl),
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
        """End-to-end: JSONL + sidecar → analyze → custom weights applied.

        Two non-dominated trials differ in throughput and retransmit
        rate. With a heavy retransmit_rate weight the lower-retx trial
        wins over the higher-throughput one.
        """

        def _mk(tp_gbps: float, retx: int, tid: str) -> TrialResult:
            return TrialResult(
                trial_id=tid,
                node_pair=NodePair(source="a", target="b", hardware_class="10g"),
                sysctl_values={"net.core.rmem_max": 212992 + retx},
                config=BenchmarkConfig(duration=10, iterations=1),
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
        jsonl = tmp_path / "trials.jsonl"
        for t in trials:
            TrialLog.append(jsonl, t)

        section = ObjectivesSection(
            pareto=[
                ParetoObjective(metric="tcp_throughput", direction="maximize"),
                ParetoObjective(metric="tcp_retransmit_rate", direction="minimize"),
            ],
            constraints=[],
            recommendation_weights={"tcp_retransmit_rate": 5.0},
        )
        TrialLog.write_resume_metadata(
            jsonl,
            ResumeMetadata(
                objectives=section,
                param_space=ParamSpace(params=[]),
                benchmark=BenchmarkConfig(duration=10, iterations=1),
            ),
        )

        out_dir = tmp_path / "out"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "analyze",
                "-i",
                str(jsonl),
                "-o",
                str(out_dir),
                "--hardware-class",
                "10g",
            ],
        )
        assert result.exit_code == 0, result.output

        recs = json.loads((out_dir / "10g" / "recommendations.json").read_text())
        assert recs[0]["trial_id"] == "lo-retx"

    def test_analyze_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "analyze",
                "-i",
                str(empty),
                "-o",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0


# --- memory cost recommendation preference -------------------------------


def test_pareto_recommendation_rows_flips_tied_configs_by_memory_cost() -> None:
    """Two trials tied on performance; lower-memory rmem_max wins at 0.1."""
    from kube_autotuner.analysis import pareto_recommendation_rows  # noqa: PLC0415

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
) -> HostStateSnapshot:
    return HostStateSnapshot(
        node="node-a",
        iteration=iteration,
        phase=phase,
        metrics=metrics,
    )


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
        config=BenchmarkConfig(duration=1, iterations=1),
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
    trial_ids = [t["trial_id"] for t in payload["trials"]]
    assert trial_ids == ["keep"]


def test_host_state_series_preserves_baseline_iteration_none() -> None:
    trial = _trial_with_snapshots(
        trial_id="t1",
        hw="10g",
        snapshots=[
            _snap(None, "baseline", {"conntrack_count": 10}),
            _snap(0, "post-flush", {"conntrack_count": 11}),
            _snap(0, "post-iteration", {"conntrack_count": 13}),
        ],
    )
    payload = host_state_series([trial], "10g", topology=None)
    assert payload is not None
    points = payload["trials"][0]["points"]
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
