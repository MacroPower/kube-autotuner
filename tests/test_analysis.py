"""Tests for the offline trial analysis module."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import TYPE_CHECKING, Any, Literal, cast

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from typer.testing import CliRunner  # noqa: E402

from kube_autotuner.analysis import (  # noqa: E402
    DEFAULT_OBJECTIVES,
    parameter_importance,
    pareto_front,
    recommend_configs,
    split_trials_by_hardware_class,
    trials_to_dataframe,
)
from kube_autotuner.cli import app  # noqa: E402
from kube_autotuner.models import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkResult,
    NodePair,
    TrialLog,
    TrialResult,
)

if TYPE_CHECKING:
    from pathlib import Path


def _result(
    bps: float,
    retransmits: int = 0,
    cpu: float = 10.0,
) -> BenchmarkResult:
    return BenchmarkResult(
        timestamp=datetime.now(UTC),
        mode="tcp",
        bits_per_second=bps,
        retransmits=retransmits,
        cpu_utilization_percent=cpu,
    )


def _trial(  # noqa: PLR0913, PLR0917
    hw: str = "10g",
    bps: float = 5e9,
    retransmits: int = 5,
    cpu: float = 25.0,
    rmem_max: int = 212992,
    congestion: str = "cubic",
    trial_id: str | None = None,
    source_zone: str = "",
    target_zone: str = "",
) -> TrialResult:
    kw: dict[str, Any] = {}
    if trial_id:
        kw["trial_id"] = trial_id
    return TrialResult(
        node_pair=NodePair(
            source="a",
            target="b",
            hardware_class=cast("Literal['1g', '10g']", hw),
            source_zone=source_zone,
            target_zone=target_zone,
        ),
        sysctl_values={
            "net.core.rmem_max": rmem_max,
            "net.ipv4.tcp_congestion_control": congestion,
        },
        config=BenchmarkConfig(duration=10, iterations=1),
        results=[_result(bps, retransmits, cpu)],
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
                cpu=20 + i * 2,
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
            cpu=15 + i,
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
        assert "mean_throughput" in df.columns
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
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="99g")  # type: ignore[arg-type]
        assert df.empty
        assert "mean_throughput" in df.columns

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


class TestParetoFront:
    def test_dominated_removed(self) -> None:
        df = pd.DataFrame(
            {
                "trial_id": ["A", "B", "C", "D"],
                "mean_throughput": [100, 80, 60, 50],
                "mean_cpu": [10, 5, 30, 20],
                "mean_memory": [1e8, 2e8, 3e8, 5e8],
                "total_retransmits": [1, 2, 0, 5],
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
                "mean_throughput": [100, 50],
                "mean_cpu": [50, 10],
                "mean_memory": [1e8, 2e8],
                "total_retransmits": [5, 5],
            },
        )
        front = pareto_front(df)
        assert len(front) == 2

    def test_single_trial(self) -> None:
        df = pd.DataFrame(
            {
                "trial_id": ["A"],
                "mean_throughput": [100],
                "mean_cpu": [10],
                "mean_memory": [1e8],
                "total_retransmits": [1],
            },
        )
        assert len(pareto_front(df)) == 1

    def test_memory_makes_trial_nondominated(self) -> None:
        df = pd.DataFrame(
            {
                "trial_id": ["A", "B"],
                "mean_throughput": [100, 100],
                "mean_cpu": [10, 10],
                "mean_memory": [1e8, 5e7],
                "total_retransmits": [1, 1],
            },
        )
        ids = set(pareto_front(df)["trial_id"])
        assert "B" in ids

    def test_empty(self) -> None:
        df = pd.DataFrame(
            columns=[
                "trial_id",
                "mean_throughput",
                "mean_cpu",
                "mean_memory",
                "total_retransmits",
            ],
        )
        assert pareto_front(df).empty

    def test_default_objectives_shape(self) -> None:
        assert len(DEFAULT_OBJECTIVES) == 4
        names = [n for n, _ in DEFAULT_OBJECTIVES]
        assert "mean_memory" in names


# --- parameter_importance -----------------------------------------------


class TestParameterImportance:
    def test_identifies_top_param(self, mixed_trials: list[TrialResult]) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
        imp = parameter_importance(df, target="mean_throughput")
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

    def test_output_includes_mean_memory(
        self,
        mixed_trials: list[TrialResult],
    ) -> None:
        recs = recommend_configs(mixed_trials, "10g", n=5)
        assert recs, "expected at least one recommendation"
        for r in recs:
            assert "mean_memory" in r

    def test_lower_memory_outranks_higher(self) -> None:
        """Trials identical except memory: the lower-memory one wins.

        Memory is a Pareto minimize-objective, so the higher-memory
        twin is strictly dominated and drops out of the frontier before
        scoring runs, leaving only the lower-memory trial.
        """

        def _trial_with_mem(mem: int, tid: str) -> TrialResult:
            return TrialResult(
                trial_id=tid,
                node_pair=NodePair(source="a", target="b", hardware_class="10g"),
                sysctl_values={"net.core.rmem_max": 212992 + mem},
                config=BenchmarkConfig(duration=10, iterations=1),
                results=[
                    BenchmarkResult(
                        timestamp=datetime.now(UTC),
                        mode="tcp",
                        bits_per_second=5e9,
                        retransmits=3,
                        cpu_utilization_percent=25.0,
                        memory_used_bytes=mem,
                    ),
                ],
            )

        trials = [
            _trial_with_mem(100_000_000, "hi-mem"),
            _trial_with_mem(50_000_000, "lo-mem"),
        ]
        recs = recommend_configs(trials, "10g", n=2)
        assert [r["trial_id"] for r in recs] == ["lo-mem"]


# --- plots ---------------------------------------------------------------


pytest.importorskip("plotly")

from kube_autotuner.plots import (  # noqa: E402
    plot_importance,
    plot_param_heatmap,
    plot_pareto_2d,
    plot_pareto_scatter_matrix,
)


class TestPlots:
    def test_scatter_matrix(self, mixed_trials: list[TrialResult]) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
        front = pareto_front(df)
        mask = df["trial_id"].isin(front["trial_id"])
        fig = plot_pareto_scatter_matrix(df, mask)
        assert fig is not None

    def test_2d(self, mixed_trials: list[TrialResult]) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
        front = pareto_front(df)
        fig = plot_pareto_2d(df, front, "mean_throughput", "mean_cpu")
        assert fig is not None

    def test_2d_memory_axis(self, mixed_trials: list[TrialResult]) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
        front = pareto_front(df)
        fig = plot_pareto_2d(df, front, "mean_throughput", "mean_memory")
        assert fig is not None

    def test_importance(self, mixed_trials: list[TrialResult]) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
        imp = parameter_importance(df)
        fig = plot_importance(imp)
        assert fig is not None

    def test_heatmap(self, mixed_trials: list[TrialResult]) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
        front = pareto_front(df)
        imp = parameter_importance(df)
        fig = plot_param_heatmap(df, front, imp)
        assert fig is not None


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
        assert (hw_dir / "pareto_scatter_matrix.html").exists()

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
