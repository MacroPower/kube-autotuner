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
    METRIC_TO_DF_COLUMN,
    parameter_importance,
    pareto_front,
    recommend_configs,
    split_trials_by_hardware_class,
    trials_to_dataframe,
)
from kube_autotuner.cli import app  # noqa: E402
from kube_autotuner.experiment import ObjectivesSection, ParetoObjective  # noqa: E402
from kube_autotuner.models import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkResult,
    NodePair,
    TrialLog,
    TrialResult,
)

if TYPE_CHECKING:
    from pathlib import Path


_DEFAULT_BYTES_SENT = 1_000_000_000


def _result(
    bps: float,
    retransmits: int = 0,
    cpu: float = 10.0,
    bytes_sent: int | None = _DEFAULT_BYTES_SENT,
) -> BenchmarkResult:
    return BenchmarkResult(
        timestamp=datetime.now(UTC),
        mode="tcp",
        bits_per_second=bps,
        retransmits=retransmits,
        bytes_sent=bytes_sent,
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
    bytes_sent: int | None = _DEFAULT_BYTES_SENT,
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
        results=[_result(bps, retransmits, cpu, bytes_sent)],
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


def _pad_latency_cols(rows: int) -> dict[str, list[float]]:
    """Pad a synthetic pareto DataFrame with constant latency/RPS columns.

    The new default :data:`DEFAULT_OBJECTIVES` spans nine axes; filling
    the latency/RPS columns with constants keeps the dominance scan
    deterministic so the pre-existing throughput/CPU/memory tests still
    exercise the exact trials they were designed for.

    Returns:
        A mapping of latency/RPS column name to a list of ``rows``
        constant values suitable for splatting into a DataFrame
        constructor.
    """
    return {
        "mean_rps": [1000.0] * rows,
        "mean_latency_p50_ms": [1.0] * rows,
        "mean_latency_p90_ms": [5.0] * rows,
        "mean_latency_p99_ms": [10.0] * rows,
    }


class TestParetoFront:
    def test_dominated_removed(self) -> None:
        df = pd.DataFrame(
            {
                "trial_id": ["A", "B", "C", "D"],
                "mean_throughput": [100, 80, 60, 50],
                "mean_cpu": [10, 5, 30, 20],
                "mean_node_memory": [1e8, 2e8, 3e8, 5e8],
                "mean_cni_memory": [1e7, 2e7, 3e7, 5e7],
                "retransmit_rate": [1, 2, 0, 5],
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
                "mean_throughput": [100, 50],
                "mean_cpu": [50, 10],
                "mean_node_memory": [1e8, 2e8],
                "mean_cni_memory": [1e7, 2e7],
                "retransmit_rate": [5, 5],
                **_pad_latency_cols(2),
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
                "mean_node_memory": [1e8],
                "mean_cni_memory": [1e7],
                "retransmit_rate": [1],
                **_pad_latency_cols(1),
            },
        )
        assert len(pareto_front(df)) == 1

    def test_memory_makes_trial_nondominated(self) -> None:
        df = pd.DataFrame(
            {
                "trial_id": ["A", "B"],
                "mean_throughput": [100, 100],
                "mean_cpu": [10, 10],
                "mean_node_memory": [1e8, 5e7],
                "mean_cni_memory": [1e7, 1e7],
                "retransmit_rate": [1, 1],
                **_pad_latency_cols(2),
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
                "mean_node_memory",
                "mean_cni_memory",
                "retransmit_rate",
                "mean_rps",
                "mean_latency_p50_ms",
                "mean_latency_p90_ms",
                "mean_latency_p99_ms",
            ],
        )
        assert pareto_front(df).empty

    def test_default_objectives_shape(self) -> None:
        assert len(DEFAULT_OBJECTIVES) == 9
        names = [n for n, _ in DEFAULT_OBJECTIVES]
        assert "mean_node_memory" in names
        assert "mean_cni_memory" in names
        assert "mean_rps" in names
        assert "mean_latency_p99_ms" in names

    def test_drops_nan_rows_with_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        df = pd.DataFrame(
            {
                "trial_id": ["A", "B"],
                "mean_throughput": [100, 50],
                "mean_cpu": [10, 20],
                "mean_node_memory": [1e8, 2e8],
                "mean_cni_memory": [1e7, 2e7],
                "retransmit_rate": [1e-7, float("nan")],
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

        The canonical case is a trial log produced with
        ``cni.enabled=false``: every ``mean_cni_memory`` cell is NaN.
        Rather than dropping every trial (the previous behavior, which
        emptied the frontier) or poisoning dominance with ``0.0``, the
        objective is removed and the frontier is computed from the
        remaining axes.
        """
        df = pd.DataFrame(
            {
                "trial_id": ["A", "B"],
                "mean_throughput": [100, 50],
                "mean_cpu": [50, 10],
                "mean_node_memory": [1e8, 2e8],
                "mean_cni_memory": [float("nan"), float("nan")],
                "retransmit_rate": [1e-7, 2e-7],
                **_pad_latency_cols(2),
            },
        )
        with caplog.at_level("INFO", logger="kube_autotuner.analysis"):
            front = pareto_front(df)
        assert set(front["trial_id"]) == {"A", "B"}
        assert any(
            "mean_cni_memory" in rec.message and "excluded" in rec.message
            for rec in caplog.records
        )

    def test_all_objectives_dropped_returns_empty(self) -> None:
        """If every objective column is NaN, the frontier is empty."""
        df = pd.DataFrame(
            {
                "trial_id": ["A", "B"],
                "mean_throughput": [float("nan"), float("nan")],
                "mean_cpu": [float("nan"), float("nan")],
                "mean_node_memory": [float("nan"), float("nan")],
                "mean_cni_memory": [float("nan"), float("nan")],
                "retransmit_rate": [float("nan"), float("nan")],
                "mean_rps": [float("nan"), float("nan")],
                "mean_latency_p50_ms": [float("nan"), float("nan")],
                "mean_latency_p90_ms": [float("nan"), float("nan")],
                "mean_latency_p99_ms": [float("nan"), float("nan")],
            },
        )
        assert pareto_front(df).empty


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

    def test_output_includes_memory_fields(
        self,
        mixed_trials: list[TrialResult],
    ) -> None:
        recs = recommend_configs(mixed_trials, "10g", n=5)
        assert recs, "expected at least one recommendation"
        for r in recs:
            assert "mean_node_memory" in r
            assert "mean_cni_memory" in r

    def test_default_scoring_snapshot(self) -> None:
        """Pin default scores against the baseline formula.

        With two non-dominated trials whose ``cpu``, ``retransmit_rate``,
        ``rps``, and every latency percentile are constant, ``_norm``
        clamps those columns to ``0.5``. ``cni_memory`` is all-NaN
        (never set on these trials) so
        :func:`_objectives_with_data` excludes it -- but the exclusion
        is score-neutral here because the default weights do not
        include ``cni_memory`` and ``_norm`` of a constant would have
        multiplied through by zero anyway. For the surviving
        ``throughput`` and ``node_memory`` axes, trial A (higher
        throughput, higher memory) normalizes to ``(1.0, 1.0)`` and
        trial B (lower throughput, lower memory) to ``(0.0, 0.0)``.
        The default formula with the ``rps`` maximize objective
        (unweighted ``+0.5`` for both) is:

        - score_A = 1.0 + 0.5 - 0.15 * 0.5 - 0.15 * 1.0 - 0.3 * 0.5 = 1.125
        - score_B = 0.0 + 0.5 - 0.15 * 0.5 - 0.15 * 0.0 - 0.3 * 0.5 = 0.275
        """

        def _mk(
            tp_gbps: float,
            mem_mib: int,
            tid: str,
        ) -> TrialResult:
            return TrialResult(
                trial_id=tid,
                node_pair=NodePair(source="a", target="b", hardware_class="10g"),
                sysctl_values={"net.core.rmem_max": 212992},
                config=BenchmarkConfig(duration=10, iterations=1),
                results=[
                    BenchmarkResult(
                        timestamp=datetime.now(UTC),
                        mode="tcp",
                        bits_per_second=tp_gbps * 1e9,
                        retransmits=5,
                        bytes_sent=1_000_000_000,
                        cpu_utilization_percent=20.0,
                        node_memory_used_bytes=mem_mib * 1024 * 1024,
                    ),
                ],
            )

        trials = [_mk(10.0, 100, "a"), _mk(5.0, 50, "b")]
        recs = recommend_configs(trials, "10g", n=2)
        assert [r["trial_id"] for r in recs] == ["a", "b"]
        assert recs[0]["score"] == pytest.approx(1.125)
        assert recs[1]["score"] == pytest.approx(0.275)

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
                        cpu_utilization_percent=20.0,
                        node_memory_used_bytes=100 * 1024 * 1024,
                    ),
                ],
            )

        trials = [
            _mk(10.0, 1000, 37_500_000_000, "a"),
            _mk(5.0, 100, 18_750_000_000, "b"),
        ]
        recs = recommend_configs(trials, "10g", n=2)
        assert [r["trial_id"] for r in recs] == ["a", "b"]
        assert recs[0]["retransmit_rate"] == pytest.approx(
            1000 / 37_500_000_000,
        )

    def test_metric_to_df_column_covers_default_objectives(self) -> None:
        expected = {
            METRIC_TO_DF_COLUMN[m]
            for m in (
                "throughput",
                "cpu",
                "node_memory",
                "cni_memory",
                "retransmit_rate",
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
        heavy_cpu_recs = recommend_configs(
            mixed_trials,
            "10g",
            n=5,
            weights={"cpu": 5.0, "node_memory": 0.15, "retransmit_rate": 0.3},
        )
        assert default_recs != heavy_cpu_recs or all(
            a["score"] != b["score"]
            for a, b in zip(default_recs, heavy_cpu_recs, strict=False)
        )

    def test_reduced_pareto_drops_rate_from_scoring(
        self,
        mixed_trials: list[TrialResult],
    ) -> None:
        reduced = [
            ParetoObjective(metric="throughput", direction="maximize"),
            ParetoObjective(metric="node_memory", direction="minimize"),
        ]
        recs = recommend_configs(
            mixed_trials,
            "10g",
            n=5,
            objectives=reduced,
            weights={"node_memory": 0.15},
        )
        assert recs
        for r in recs:
            assert set(r.keys()) >= {
                "mean_throughput",
                "mean_cpu",
                "mean_node_memory",
                "mean_cni_memory",
                "retransmit_rate",
            }

    def test_lower_memory_outranks_higher(self) -> None:
        """Trials identical except node memory: the lower-memory one wins.

        node_memory is a Pareto minimize-objective, so the higher-memory
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
                        bytes_sent=1_000_000_000,
                        cpu_utilization_percent=25.0,
                        node_memory_used_bytes=mem,
                    ),
                ],
            )

        trials = [
            _trial_with_mem(100_000_000, "hi-mem"),
            _trial_with_mem(50_000_000, "lo-mem"),
        ]
        recs = recommend_configs(trials, "10g", n=2)
        assert [r["trial_id"] for r in recs] == ["lo-mem"]

    def test_cni_disabled_returns_none_value_for_that_key(self) -> None:
        """CNI-disabled trials keep the key present with a ``None`` value.

        Every trial sets ``node_memory`` but leaves ``cni_memory`` unset,
        mirroring a run with ``cni.enabled=false``. The recommendation
        dict still has ``"mean_cni_memory"`` (contract preserved for
        downstream consumers) but the value is ``None`` rather than a
        misleading ``0.0``.
        """

        def _cni_disabled_trial(bps: float, tid: str) -> TrialResult:
            return TrialResult(
                trial_id=tid,
                node_pair=NodePair(source="a", target="b", hardware_class="10g"),
                sysctl_values={"net.core.rmem_max": 212992 + int(bps)},
                config=BenchmarkConfig(duration=10, iterations=1),
                results=[
                    BenchmarkResult(
                        timestamp=datetime.now(UTC),
                        mode="tcp",
                        bits_per_second=bps,
                        retransmits=5,
                        bytes_sent=1_000_000_000,
                        cpu_utilization_percent=20.0,
                        node_memory_used_bytes=100 * 1024 * 1024,
                    ),
                ],
            )

        trials = [_cni_disabled_trial(10e9, "a"), _cni_disabled_trial(5e9, "b")]
        recs = recommend_configs(trials, "10g", n=2)
        assert recs
        for r in recs:
            assert "mean_cni_memory" in r
            assert r["mean_cni_memory"] is None
            assert r["mean_node_memory"] is not None


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

    def test_2d_node_memory_axis(self, mixed_trials: list[TrialResult]) -> None:
        df, _ = trials_to_dataframe(mixed_trials, hardware_class="10g")
        front = pareto_front(df)
        fig = plot_pareto_2d(df, front, "mean_throughput", "mean_node_memory")
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

    def test_analyze_reads_sidecar_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        """End-to-end: JSONL + sidecar → analyze → custom weights applied.

        Two non-dominated trials differ only in memory. With the
        default memory weight (0.15) the high-throughput trial wins;
        with a heavy memory weight the low-memory trial wins.
        """

        def _mk(tp_gbps: float, mem_mib: int, tid: str) -> TrialResult:
            return TrialResult(
                trial_id=tid,
                node_pair=NodePair(source="a", target="b", hardware_class="10g"),
                sysctl_values={"net.core.rmem_max": 212992 + mem_mib},
                config=BenchmarkConfig(duration=10, iterations=1),
                results=[
                    BenchmarkResult(
                        timestamp=datetime.now(UTC),
                        mode="tcp",
                        bits_per_second=tp_gbps * 1e9,
                        retransmits=5,
                        bytes_sent=1_000_000_000,
                        cpu_utilization_percent=20.0,
                        node_memory_used_bytes=mem_mib * 1024 * 1024,
                    ),
                ],
            )

        trials = [_mk(10.0, 500, "hi-tp"), _mk(5.0, 50, "lo-mem")]
        jsonl = tmp_path / "trials.jsonl"
        for t in trials:
            TrialLog.append(jsonl, t)

        section = ObjectivesSection(
            pareto=[
                ParetoObjective(metric="throughput", direction="maximize"),
                ParetoObjective(metric="node_memory", direction="minimize"),
            ],
            constraints=[],
            recommendation_weights={"node_memory": 5.0},
        )
        TrialLog.write_metadata(jsonl, section)

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
        assert recs[0]["trial_id"] == "lo-mem"

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
