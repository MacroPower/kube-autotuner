"""Tests for :mod:`kube_autotuner.optimizer`.

Ax is an optional dependency. The module-level ``pytest.importorskip``
call below skips the whole file when ``ax-platform`` is not installed,
so the base ``dev`` sync keeps ``task test`` green without pulling the
``optimize`` group. Install it with ``uv sync --group optimize`` to run
these tests for real.
"""

from __future__ import annotations

from contextlib import ExitStack
from datetime import UTC, datetime
import math
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("ax")

from kube_autotuner.experiment import ObjectivesSection, ParetoObjective
from kube_autotuner.k8s.lease import LeaseHeldError
from kube_autotuner.models import (
    BenchmarkConfig,
    BenchmarkResult,
    HostStateSnapshot,
    IterationResults,
    LatencyResult,
    NodePair,
    TrialResult,
)
from kube_autotuner.optimizer import (
    OptimizationLoop,
    _aggregate_by_iteration,  # noqa: PLC2701
    _compute_metrics,  # noqa: PLC2701
    _decode_param_name,  # noqa: PLC2701
    _encode_param_name,  # noqa: PLC2701
    build_ax_objective,
    build_ax_params,
)
from kube_autotuner.sysctl.params import PARAM_SPACE, RECOMMENDED_DEFAULTS


def _make_results(n: int = 3) -> list[BenchmarkResult]:
    return [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=9_000_000_000 + i * 100_000,
            retransmits=5 + i,
            bytes_sent=1_000_000_000,
            client_node="kmain07",
            iteration=i,
        )
        for i in range(n)
    ]


def _make_latency_results(n: int = 3) -> list[LatencyResult]:
    records: list[LatencyResult] = []
    for i in range(n):
        records.extend([
            LatencyResult(
                timestamp=datetime.now(UTC),
                workload="saturation",
                client_node="kmain07",
                iteration=i,
                rps=1000.0 + i,
                latency_p50=None,
                latency_p90=None,
                latency_p99=None,
            ),
            LatencyResult(
                timestamp=datetime.now(UTC),
                workload="fixed_qps",
                client_node="kmain07",
                iteration=i,
                rps=1000.0,
                latency_p50=(1.0 + i) / 1000.0,
                latency_p90=(5.0 + i) / 1000.0,
                latency_p99=(10.0 + i) / 1000.0,
            ),
        ])
    return records


def _trial_from(
    results: list[BenchmarkResult],
    latency_results: list[LatencyResult] | None = None,
) -> TrialResult:
    return TrialResult(
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
        latency_results=latency_results or [],
    )


def _mock_snapshot(param_names):
    values = {}
    for name in param_names:
        if name == "kernel.osrelease":
            values[name] = "6.1.0-talos"
        else:
            values[name] = "212992"
    return values


class TestBuildAxParams:
    def test_count_matches_param_space(self):
        params = build_ax_params(PARAM_SPACE)
        assert len(params) == len(PARAM_SPACE.params)

    def test_values_are_strings(self):
        for p in build_ax_params(PARAM_SPACE):
            for v in p.values:
                assert isinstance(v, str)

    def test_int_params_ordered(self):
        by_name = {p.name: p for p in build_ax_params(PARAM_SPACE)}
        assert by_name["net__core__rmem_max"].is_ordered is True
        assert by_name["net__ipv4__tcp_congestion_control"].is_ordered is False

    def test_no_dots_in_names(self):
        for p in build_ax_params(PARAM_SPACE):
            assert "." not in p.name


class TestParamNameEncoding:
    def test_encode(self):
        assert _encode_param_name("net.core.rmem_max") == "net__core__rmem_max"

    def test_decode(self):
        assert _decode_param_name("net__core__rmem_max") == "net.core.rmem_max"

    def test_roundtrip(self):
        for p in PARAM_SPACE.params:
            assert _decode_param_name(_encode_param_name(p.name)) == p.name


class TestOptimizationLoop:
    @pytest.fixture
    def node_pair(self):
        return NodePair(
            source="kmain07",
            target="kmain08",
            hardware_class="10g",
            namespace="default",
        )

    @pytest.fixture
    def config(self):
        return BenchmarkConfig(duration=10, iterations=3)

    @patch("kube_autotuner.optimizer.NodeLease")
    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    def test_trial_execution(
        self,
        mock_setter_cls,
        mock_runner_cls,
        mock_lease_cls,
        node_pair,
        config,
        tmp_path,
    ):
        output = tmp_path / "results.jsonl"

        mock_setter = MagicMock()
        mock_setter.snapshot.side_effect = _mock_snapshot
        mock_setter_cls.return_value = mock_setter

        mock_runner = MagicMock()
        mock_runner.run.return_value = IterationResults(
            bench=_make_results(),
            latency=_make_latency_results(),
        )
        mock_runner_cls.return_value = mock_runner

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=output,
            n_trials=3,
            n_sobol=3,
            objectives=ObjectivesSection(),
        )
        trials = loop.run()

        assert len(trials) == 3
        mock_runner.setup_server.assert_called_once()
        mock_runner.cleanup.assert_called_once()
        assert mock_setter.snapshot.call_count == 3
        assert mock_setter.apply.call_count == 3
        assert mock_setter.restore.call_count == 3

        snap_args = mock_setter.snapshot.call_args_list[0][0][0]
        assert "kernel.osrelease" in snap_args
        # tcp_no_metrics_save is pinned to 1 per-trial (methodology) but
        # not part of the search space, so the snapshot keys cover the
        # space plus ``tcp_no_metrics_save`` plus ``kernel.osrelease``.
        assert "net.ipv4.tcp_no_metrics_save" in snap_args
        assert len(snap_args) == len(PARAM_SPACE.params) + 2

        assert mock_lease_cls.call_count == 6

        lines = output.read_text().strip().splitlines()
        assert len(lines) == 3

    @patch("kube_autotuner.optimizer.NodeLease")
    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    def test_host_state_snapshots_propagate_onto_trial_result(
        self,
        mock_setter_cls,
        mock_runner_cls,
        mock_lease_cls,  # noqa: ARG002
        node_pair,
        config,
        tmp_path,
    ):
        """``_evaluate`` must thread ``host_state_snapshots`` onto the TrialResult."""
        output = tmp_path / "results.jsonl"
        mock_setter = MagicMock()
        mock_setter.snapshot.side_effect = _mock_snapshot
        mock_setter_cls.return_value = mock_setter

        sample_snapshots = [
            HostStateSnapshot(
                node="b",
                iteration=None,
                phase="baseline",
                metrics={"conntrack_count": 0},
            ),
            HostStateSnapshot(
                node="b",
                iteration=0,
                phase="post-flush",
                metrics={"conntrack_count": 0},
            ),
        ]
        mock_runner = MagicMock()
        mock_runner.run.return_value = IterationResults(
            bench=_make_results(),
            latency=_make_latency_results(),
            host_state_snapshots=sample_snapshots,
        )
        mock_runner_cls.return_value = mock_runner

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=output,
            n_trials=1,
            n_sobol=1,
            objectives=ObjectivesSection(),
            collect_host_state=True,
        )
        trials = loop.run()
        assert len(trials) == 1

        # Runner received the flag + backend list.
        kwargs = mock_runner_cls.call_args.kwargs
        assert kwargs["collect_host_state"] is True
        assert kwargs["snapshot_backends"] == [mock_setter]

        # Snapshots landed on the TrialResult and round-tripped through JSONL.
        assert trials[0].host_state_snapshots == sample_snapshots
        import json  # noqa: PLC0415

        row = json.loads(output.read_text().strip())
        assert len(row["host_state_snapshots"]) == 2
        assert row["host_state_snapshots"][0]["phase"] == "baseline"

    @patch("kube_autotuner.optimizer.NodeLease")
    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    def test_handles_trial_failure(
        self,
        mock_setter_cls,
        mock_runner_cls,
        mock_lease_cls,  # noqa: ARG002
        node_pair,
        config,
        tmp_path,
    ):
        output = tmp_path / "results.jsonl"

        mock_setter = MagicMock()
        mock_setter.snapshot.side_effect = _mock_snapshot
        mock_setter_cls.return_value = mock_setter

        mock_runner = MagicMock()
        call_count = 0

        def run_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("benchmark failed")  # noqa: TRY003
            return IterationResults(
                bench=_make_results(),
                latency=_make_latency_results(),
            )

        mock_runner.run.side_effect = run_side_effect
        mock_runner_cls.return_value = mock_runner

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=output,
            n_trials=3,
            n_sobol=3,
            objectives=ObjectivesSection(),
        )
        trials = loop.run()

        assert len(trials) == 2
        assert mock_setter.restore.call_count == 3
        mock_runner.cleanup.assert_called_once()

    @patch("kube_autotuner.optimizer.NodeLease")
    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    def test_cleanup_on_unexpected_exception(
        self,
        mock_setter_cls,
        mock_runner_cls,
        mock_lease_cls,  # noqa: ARG002
        node_pair,
        config,
        tmp_path,
    ):
        mock_setter = MagicMock()
        mock_setter.snapshot.side_effect = _mock_snapshot
        mock_setter_cls.return_value = mock_setter

        mock_runner = MagicMock()
        mock_runner.run.side_effect = SystemExit(1)
        mock_runner_cls.return_value = mock_runner

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=tmp_path / "results.jsonl",
            n_trials=1,
            n_sobol=1,
            objectives=ObjectivesSection(),
        )
        with pytest.raises(SystemExit):
            loop.run()

        mock_runner.cleanup.assert_called_once()

    @patch("kube_autotuner.optimizer.NodeLease")
    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    def test_multi_node_lease_sorted_order(
        self,
        mock_setter_cls,
        mock_runner_cls,
        mock_lease_cls,
        config,
        tmp_path,
    ):
        """Leases for target + every client are acquired in sorted hostname order."""
        node_pair = NodePair(
            source="kmain09",
            target="kmain07",
            hardware_class="10g",
            namespace="default",
            extra_sources=["kmain08"],
        )

        mock_setter = MagicMock()
        mock_setter.snapshot.side_effect = _mock_snapshot
        mock_setter_cls.return_value = mock_setter

        mock_runner = MagicMock()
        mock_runner.run.return_value = IterationResults(
            bench=_make_results(),
            latency=_make_latency_results(),
        )
        mock_runner_cls.return_value = mock_runner

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=tmp_path / "results.jsonl",
            n_trials=1,
            n_sobol=1,
            apply_source=True,
            objectives=ObjectivesSection(),
        )
        trials = loop.run()

        assert len(trials) == 1
        lease_calls = mock_lease_cls.call_args_list
        trial_nodes = [c[0][0] for c in lease_calls]
        assert trial_nodes == ["kmain07", "kmain08", "kmain09"]

    @patch("kube_autotuner.optimizer.NodeLease")
    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    def test_apply_source_applies_to_every_client(
        self,
        mock_setter_cls,
        mock_runner_cls,
        mock_lease_cls,  # noqa: ARG002
        config,
        tmp_path,
    ):
        node_pair = NodePair(
            source="kmain07",
            target="kmain08",
            hardware_class="10g",
            namespace="default",
            extra_sources=["kmain09"],
        )

        setters_by_node: dict[str, MagicMock] = {}

        def _setter_factory(*, node, namespace, **_kwargs):
            del namespace
            s = MagicMock()
            s.snapshot.side_effect = _mock_snapshot
            setters_by_node[node] = s
            return s

        mock_setter_cls.side_effect = _setter_factory

        mock_runner = MagicMock()
        mock_runner.run.return_value = IterationResults(
            bench=_make_results(),
            latency=_make_latency_results(),
        )
        mock_runner_cls.return_value = mock_runner

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=tmp_path / "results.jsonl",
            n_trials=1,
            n_sobol=1,
            apply_source=True,
            objectives=ObjectivesSection(),
        )
        loop.run()

        assert set(setters_by_node.keys()) >= {"kmain07", "kmain08", "kmain09"}
        for node in ("kmain07", "kmain09"):
            assert setters_by_node[node].apply.called
            assert setters_by_node[node].restore.called


class TestNodeLeaseCleanup:
    def test_multi_node_partial_failure_cleanup(self):
        """If the second lease acquire fails, the first lease must be released."""
        first_lease = MagicMock()
        second_lease = MagicMock()
        second_lease.__enter__ = MagicMock(
            side_effect=LeaseHeldError(
                "kube-autotuner-lock-kmain08",
                "other",
                "2026-04-17T00:00:00Z",
            ),
        )
        lease_factory = MagicMock(side_effect=[first_lease, second_lease])
        client = MagicMock()
        nodes = sorted({"kmain07", "kmain08"})

        def _acquire_all():
            with ExitStack() as stack:
                for node in nodes:
                    lease = lease_factory(node, namespace="default", client=client)
                    stack.enter_context(lease)

        with pytest.raises(LeaseHeldError):
            _acquire_all()

        first_lease.__exit__.assert_called()


class TestAggregateByIteration:
    def test_single_client_per_iteration(self):
        results = [
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=1e9,
                retransmits=1,
                client_node="c1",
                iteration=0,
            ),
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=2e9,
                retransmits=2,
                client_node="c1",
                iteration=1,
            ),
        ]
        vals = _aggregate_by_iteration(
            results,
            lambda r: r.bits_per_second,
            sum,
        )
        assert sorted(vals) == [1e9, 2e9]

    def test_multi_client_groups_by_iteration(self):
        def _r(bps, client, itr):
            return BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=bps,
                retransmits=1,
                client_node=client,
                iteration=itr,
            )

        results = [
            _r(4e9, "c1", 0),
            _r(5e9, "c2", 0),
            _r(3e9, "c1", 1),
            _r(6e9, "c2", 1),
        ]
        vals = _aggregate_by_iteration(
            results,
            lambda r: r.bits_per_second,
            sum,
        )
        assert sorted(vals) == [9e9, 9e9]


class TestIterationsOneSEMFallback:
    def test_multi_client_iterations_one_uses_per_client_sem(self):
        def _r(bps, client):
            return BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=bps,
                retransmits=1,
                client_node=client,
                iteration=0,
            )

        results = [_r(4e9, "c1"), _r(6e9, "c2")]
        metrics = _compute_metrics(_trial_from(results))
        assert metrics["tcp_throughput"][0] == pytest.approx(1e10)
        assert metrics["tcp_throughput"][1] > 0.0


class TestBuildAxObjective:
    def test_default_section(self) -> None:
        objective, constraints = build_ax_objective(ObjectivesSection())
        assert objective == (
            "tcp_throughput, udp_throughput, -tcp_retransmit_rate, "
            "-udp_loss_rate, -udp_jitter, rps, "
            "-latency_p50, -latency_p90, -latency_p99"
        )
        assert constraints == [
            "tcp_throughput >= 1000000",
            "udp_throughput >= 1000000",
            "tcp_retransmit_rate <= 1e-06",
            "udp_loss_rate <= 0.05",
            "rps >= 100",
            "latency_p99 <= 1",
            "udp_jitter <= 0.01",
            "latency_p50 <= 0.1",
            "latency_p90 <= 0.5",
        ]

    def test_reduced_two_metric_section(self) -> None:
        section = ObjectivesSection(
            pareto=[
                ParetoObjective(metric="tcp_throughput", direction="maximize"),
                ParetoObjective(metric="udp_jitter", direction="minimize"),
            ],
            constraints=["tcp_throughput >= 1e6"],
            recommendation_weights={"udp_jitter": 0.5},
        )
        objective, constraints = build_ax_objective(section)
        assert objective == "tcp_throughput, -udp_jitter"
        assert constraints == ["tcp_throughput >= 1000000"]


class TestComputeMetricsJitter:
    def test_jitter_averaged_across_udp_records(self) -> None:
        results = [
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=1e9,
                retransmits=0,
                bytes_sent=1_000_000_000,
                iteration=0,
            ),
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="udp",
                bits_per_second=1e9,
                jitter=0.0002,
                iteration=0,
            ),
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="udp",
                bits_per_second=1e9,
                jitter=0.0004,
                iteration=1,
            ),
        ]
        metrics = _compute_metrics(_trial_from(results))
        # Iter 0 jitter mean 0.0002, iter 1 jitter mean 0.0004 -> 0.0003.
        assert metrics["udp_jitter"][0] == pytest.approx(0.0003)

    def test_jitter_nan_when_no_udp_records(self) -> None:
        results = [
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=1e9,
                retransmits=0,
                bytes_sent=1_000_000_000,
                iteration=0,
            ),
        ]
        metrics = _compute_metrics(_trial_from(results))
        assert math.isnan(metrics["udp_jitter"][0])


class TestComputeMetricsRate:
    def test_rate_nan_when_bytes_missing(self) -> None:
        results = [
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="udp",
                bits_per_second=1e9,
                retransmits=None,
                bytes_sent=None,
                iteration=0,
            ),
        ]
        metrics = _compute_metrics(_trial_from(results))
        assert math.isnan(metrics["tcp_retransmit_rate"][0])

    def test_rate_zero_when_no_retransmits(self) -> None:
        results = [
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=1e9,
                retransmits=0,
                bytes_sent=1_000_000_000,
                iteration=0,
            ),
        ]
        metrics = _compute_metrics(_trial_from(results))
        assert metrics["tcp_retransmit_rate"][0] == pytest.approx(0.0)
        assert not math.isnan(metrics["tcp_retransmit_rate"][0])


class TestComputeMetricsUdp:
    def test_udp_throughput_and_loss_rate_populated(self) -> None:
        """UDP aggregation produces non-NaN means across multiple iterations."""
        packets = 10_000
        results = [
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="udp",
                bits_per_second=1e9,
                packets=packets,
                lost_packets=100,  # iter 0: 1% loss
                iteration=0,
            ),
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="udp",
                bits_per_second=3e9,
                packets=packets,
                lost_packets=500,  # iter 1: 5% loss
                iteration=1,
            ),
        ]
        metrics = _compute_metrics(_trial_from(results))
        # iter 0 = 1e9, iter 1 = 3e9 -> mean 2e9.
        assert metrics["udp_throughput"][0] == pytest.approx(2e9)
        assert metrics["udp_throughput"][1] > 0.0
        # iter 0 = 0.01, iter 1 = 0.05 -> mean 0.03.
        assert metrics["udp_loss_rate"][0] == pytest.approx(0.03)
        assert metrics["udp_loss_rate"][1] > 0.0

    def test_udp_loss_rate_nan_when_no_udp_records(self) -> None:
        """TCP-only trial: udp_loss_rate is NaN so callers drop it."""
        results = [
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=1e9,
                retransmits=0,
                bytes_sent=1_000_000_000,
                iteration=0,
            ),
        ]
        metrics = _compute_metrics(_trial_from(results))
        assert math.isnan(metrics["udp_loss_rate"][0])

    def test_udp_throughput_single_iteration_multi_client_sem(self) -> None:
        """With iterations=1 and N UDP clients, fall back to per-client SEM."""
        results = [
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="udp",
                bits_per_second=1e9,
                packets=1000,
                lost_packets=0,
                client_node="c1",
                iteration=0,
            ),
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="udp",
                bits_per_second=3e9,
                packets=1000,
                lost_packets=0,
                client_node="c2",
                iteration=0,
            ),
        ]
        metrics = _compute_metrics(_trial_from(results))
        # iter 0 sum across clients = 4e9 -> mean 4e9.
        assert metrics["udp_throughput"][0] == pytest.approx(4e9)
        # Single-iteration fallback: per-client stdev / sqrt(n).
        assert metrics["udp_throughput"][1] > 0.0


class TestComputeMetricsLatency:
    def test_rps_and_latency_keys_present(self) -> None:
        metrics = _compute_metrics(
            _trial_from(_make_results(2), _make_latency_results(2)),
        )
        for key in ("rps", "latency_p50", "latency_p90", "latency_p99"):
            assert key in metrics
        # rps: per-iteration sum across clients, averaged across
        # iterations. Single client per iter: 1000, 1001 -> 1000.5.
        assert metrics["rps"][0] == pytest.approx(1000.5)
        # p99 from fixed_qps only: iter0=0.010, iter1=0.011 -> 0.0105.
        assert metrics["latency_p99"][0] == pytest.approx(0.0105)

    def test_rps_nan_when_no_saturation(self) -> None:
        bench = _make_results(1)
        # Only fixed_qps, no saturation.
        latency = [
            LatencyResult(
                timestamp=datetime.now(UTC),
                workload="fixed_qps",
                client_node="c1",
                iteration=0,
                rps=1000.0,
                latency_p99=0.005,
            ),
        ]
        metrics = _compute_metrics(_trial_from(bench, latency))
        assert math.isnan(metrics["rps"][0])
        assert metrics["latency_p99"][0] == pytest.approx(0.005)

    def test_latency_nan_when_no_fixed_qps(self) -> None:
        bench = _make_results(1)
        latency = [
            LatencyResult(
                timestamp=datetime.now(UTC),
                workload="saturation",
                client_node="c1",
                iteration=0,
                rps=1500.0,
            ),
        ]
        metrics = _compute_metrics(_trial_from(bench, latency))
        assert metrics["rps"][0] == pytest.approx(1500.0)
        for key in ("latency_p50", "latency_p90", "latency_p99"):
            assert math.isnan(metrics[key][0])


class TestSeedPriorTrials:
    @pytest.fixture
    def node_pair(self) -> NodePair:
        return NodePair(
            source="a",
            target="b",
            hardware_class="10g",
            namespace="default",
        )

    @pytest.fixture
    def config(self) -> BenchmarkConfig:
        return BenchmarkConfig(duration=10, iterations=3)

    def _prior(
        self,
        bps: float = 9e9,
        *,
        rate_nan: bool = False,
    ) -> TrialResult:
        return TrialResult(
            node_pair=NodePair(source="a", target="b", hardware_class="10g"),
            sysctl_values={"net.core.rmem_max": 1048576},
            config=BenchmarkConfig(),
            results=[
                BenchmarkResult(
                    timestamp=datetime.now(UTC),
                    mode="tcp",
                    bits_per_second=bps,
                    retransmits=None if rate_nan else 5,
                    bytes_sent=None if rate_nan else 1_000_000_000,
                    iteration=0,
                ),
            ],
        )

    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    @patch("kube_autotuner.optimizer._require_ax_client")
    def test_seed_prior_trials_attaches_and_completes(
        self,
        mock_require_client,
        mock_setter_cls,  # noqa: ARG002
        mock_runner_cls,  # noqa: ARG002
        node_pair,
        config,
        tmp_path,
    ):
        fake_client = MagicMock()
        fake_client.attach_trial.side_effect = [11, 12, 13]
        mock_require_client.return_value = lambda: fake_client
        priors = [self._prior(9e9), self._prior(8e9), self._prior(7e9)]

        from kube_autotuner.optimizer import OptimizationLoop  # noqa: PLC0415

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=tmp_path / "r.jsonl",
            n_trials=5,
            n_sobol=5,
            objectives=ObjectivesSection(),
            prior_trials=priors,
        )
        assert loop.prior_count == 3
        assert len(loop._completed) == 3

        attach_calls = fake_client.attach_trial.call_args_list
        assert len(attach_calls) == 3
        # Each call uses encoded names (no dots) and string values.
        for call in attach_calls:
            params = call.kwargs["parameters"]
            assert all("." not in k for k in params)
            assert all(isinstance(v, str) for v in params.values())

        complete_calls = fake_client.complete_trial.call_args_list
        assert len(complete_calls) == 3
        for call in complete_calls:
            raw = call.kwargs["raw_data"]
            assert "tcp_throughput" in raw

    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    @patch("kube_autotuner.optimizer._require_ax_client")
    def test_seed_drops_nan_metrics(
        self,
        mock_require_client,
        mock_setter_cls,  # noqa: ARG002
        mock_runner_cls,  # noqa: ARG002
        node_pair,
        config,
        tmp_path,
    ):
        fake_client = MagicMock()
        fake_client.attach_trial.return_value = 42
        mock_require_client.return_value = lambda: fake_client
        # UDP-only trial produces NaN tcp_retransmit_rate.
        nan_prior = self._prior(rate_nan=True)

        from kube_autotuner.optimizer import OptimizationLoop  # noqa: PLC0415

        OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=tmp_path / "r.jsonl",
            n_trials=5,
            n_sobol=5,
            objectives=ObjectivesSection(),
            prior_trials=[nan_prior],
        )
        assert fake_client.complete_trial.call_count == 1
        raw = fake_client.complete_trial.call_args.kwargs["raw_data"]
        assert "tcp_retransmit_rate" not in raw
        assert "tcp_throughput" in raw

    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    @patch("kube_autotuner.optimizer._require_ax_client")
    def test_should_stop_budget_counts_priors(
        self,
        mock_require_client,
        mock_setter_cls,  # noqa: ARG002
        mock_runner_cls,  # noqa: ARG002
        node_pair,
        config,
        tmp_path,
    ):
        fake_client = MagicMock()
        fake_client.attach_trial.side_effect = [1, 2, 3]
        mock_require_client.return_value = lambda: fake_client
        priors = [self._prior(), self._prior(), self._prior()]

        from kube_autotuner.optimizer import OptimizationLoop  # noqa: PLC0415

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=tmp_path / "r.jsonl",
            n_trials=5,
            n_sobol=5,
            objectives=ObjectivesSection(),
            prior_trials=priors,
        )
        assert loop._should_stop(0) is False
        assert loop._should_stop(1) is False
        assert loop._should_stop(2) is True

    @patch("kube_autotuner.optimizer.NodeLease")
    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    @patch("kube_autotuner.optimizer._require_ax_client")
    def test_run_keyboardinterrupt_log_reports_split_counts(  # noqa: PLR0913, PLR0917
        self,
        mock_require_client,
        mock_setter_cls,
        mock_runner_cls,
        mock_lease_cls,  # noqa: ARG002
        node_pair,
        config,
        tmp_path,
        caplog,
    ):
        fake_client = MagicMock()
        fake_client.attach_trial.side_effect = [1, 2]
        fake_client.get_next_trials.return_value = {
            99: {"net__core__rmem_max": "1048576"},
        }
        mock_require_client.return_value = lambda: fake_client

        mock_setter = MagicMock()
        mock_setter.snapshot.side_effect = _mock_snapshot
        mock_setter_cls.return_value = mock_setter

        call_state = {"count": 0}

        def _run_side_effect():
            call_state["count"] += 1
            if call_state["count"] == 2:
                raise KeyboardInterrupt
            return IterationResults(
                bench=_make_results(),
                latency=_make_latency_results(),
            )

        mock_runner = MagicMock()
        mock_runner.run.side_effect = _run_side_effect
        mock_runner_cls.return_value = mock_runner

        priors = [self._prior(), self._prior()]

        from kube_autotuner.optimizer import OptimizationLoop  # noqa: PLC0415

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=tmp_path / "r.jsonl",
            n_trials=10,
            n_sobol=5,
            objectives=ObjectivesSection(),
            prior_trials=priors,
        )
        caplog.set_level("INFO")
        loop.run()
        messages = [rec.message for rec in caplog.records]
        assert any("Interrupted after 1 live trials" in m for m in messages)
        assert any("(3 total)" in m for m in messages)

    @patch("kube_autotuner.optimizer.NodeLease")
    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    @patch("kube_autotuner.optimizer._require_ax_client")
    def test_record_log_numerator_counts_priors(  # noqa: PLR0913, PLR0917
        self,
        mock_require_client,
        mock_setter_cls,
        mock_runner_cls,
        mock_lease_cls,  # noqa: ARG002
        node_pair,
        config,
        tmp_path,
        caplog,
    ):
        fake_client = MagicMock()
        fake_client.attach_trial.side_effect = [1, 2]
        fake_client.get_next_trials.return_value = {
            55: {"net__core__rmem_max": "1048576"},
        }
        mock_require_client.return_value = lambda: fake_client

        mock_setter = MagicMock()
        mock_setter.snapshot.side_effect = _mock_snapshot
        mock_setter_cls.return_value = mock_setter

        mock_runner = MagicMock()
        mock_runner.run.return_value = IterationResults(
            bench=_make_results(),
            latency=_make_latency_results(),
        )
        mock_runner_cls.return_value = mock_runner

        priors = [self._prior(), self._prior()]

        from kube_autotuner.optimizer import OptimizationLoop  # noqa: PLC0415

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=tmp_path / "r.jsonl",
            n_trials=3,
            n_sobol=3,
            objectives=ObjectivesSection(),
            prior_trials=priors,
        )
        caplog.set_level("INFO")
        loop.run()
        # Expect log numerator = prior_count + i + 1 = 2 + 0 + 1 = 3, total = 3.
        assert any("Trial 3/3" in rec.message for rec in caplog.records), [
            rec.message for rec in caplog.records
        ]
        # New trial-start log must precede the existing trial-complete log.
        start_idx = next(
            (
                i
                for i, rec in enumerate(caplog.records)
                if "Trial 3/3 [sobol] starting" in rec.message
            ),
            -1,
        )
        complete_idx = next(
            (
                i
                for i, rec in enumerate(caplog.records)
                if "Trial 3/3 [sobol] tcp_throughput" in rec.message
            ),
            -1,
        )
        assert start_idx >= 0, [rec.message for rec in caplog.records]
        assert complete_idx >= 0, [rec.message for rec in caplog.records]
        assert start_idx < complete_idx


class TestComputeMetricsTcpFilter:
    """_compute_metrics must aggregate throughput over TCP only."""

    def test_mixed_mode_trial_aggregates_tcp_records(self) -> None:
        tcp_results = [
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=1e9,
                retransmits=0,
                bytes_sent=1_000_000_000,
                iteration=0,
            ),
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=3e9,
                retransmits=0,
                bytes_sent=1_000_000_000,
                iteration=1,
            ),
        ]
        udp_results = [
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="udp",
                bits_per_second=9e9,
                jitter=0.0001,
                iteration=0,
            ),
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="udp",
                bits_per_second=9e9,
                jitter=0.0002,
                iteration=1,
            ),
        ]
        metrics = _compute_metrics(_trial_from([*tcp_results, *udp_results]))
        # Throughput: TCP-only mean of 1e9 and 3e9.
        assert metrics["tcp_throughput"][0] == pytest.approx(2e9)

    def test_single_iteration_multi_tcp_client_uses_tcp_sem(self) -> None:
        """SEM fallback must base both the gate and the sample on TCP records."""
        tcp_results = [
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=4e9,
                retransmits=0,
                bytes_sent=1_000_000_000,
                client_node="c1",
                iteration=0,
            ),
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=6e9,
                retransmits=0,
                bytes_sent=1_000_000_000,
                client_node="c2",
                iteration=0,
            ),
        ]
        udp_results = [
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="udp",
                bits_per_second=9e9,
                jitter=0.0001,
                client_node="c1",
                iteration=0,
            ),
        ]
        metrics = _compute_metrics(_trial_from([*tcp_results, *udp_results]))
        # TCP-summed: 4e9 + 6e9 = 10e9.
        assert metrics["tcp_throughput"][0] == pytest.approx(1e10)
        # With 2 TCP samples (single iteration), SEM must be non-zero
        # and computed over TCP samples only (not mixed with UDP).
        assert metrics["tcp_throughput"][1] > 0.0


class TestWarnOnCollapsedObjectives:
    """``_warn_on_collapsed_objectives`` flags zero-variance Pareto metrics."""

    @pytest.fixture
    def node_pair(self) -> NodePair:
        return NodePair(
            source="a",
            target="b",
            hardware_class="10g",
            namespace="default",
        )

    @pytest.fixture
    def config(self) -> BenchmarkConfig:
        return BenchmarkConfig(duration=10, iterations=3)

    @staticmethod
    def _trial(bps: float, retransmits: int = 5) -> TrialResult:
        return TrialResult(
            node_pair=NodePair(source="a", target="b", hardware_class="10g"),
            sysctl_values={"net.core.rmem_max": 1048576},
            config=BenchmarkConfig(),
            results=[
                BenchmarkResult(
                    timestamp=datetime.now(UTC),
                    mode="tcp",
                    bits_per_second=bps,
                    retransmits=retransmits,
                    bytes_sent=1_000_000_000,
                    iteration=0,
                ),
            ],
        )

    @staticmethod
    def _reduced_objectives() -> ObjectivesSection:
        """Two-metric Pareto set that matches the synthetic fixtures' signal.

        The ``_trial`` fixture only varies ``throughput`` and
        ``tcp_retransmit_rate``; every other metric is constant or unset. A
        reduced Pareto set keeps the tests from tripping the helper on
        unrelated collapsed axes.

        Returns:
            An :class:`ObjectivesSection` whose Pareto set contains only
            ``throughput`` and ``tcp_retransmit_rate``.
        """
        return ObjectivesSection(
            pareto=[
                ParetoObjective(metric="tcp_throughput", direction="maximize"),
                ParetoObjective(metric="tcp_retransmit_rate", direction="minimize"),
            ],
            constraints=["tcp_throughput >= 1e6"],
            recommendation_weights={"tcp_retransmit_rate": 0.5},
        )

    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    @patch("kube_autotuner.optimizer._require_ax_client")
    def test_detects_collapsed_metric_and_dedups(
        self,
        mock_require_client,
        mock_setter_cls,  # noqa: ARG002
        mock_runner_cls,  # noqa: ARG002
        node_pair,
        config,
        tmp_path,
        caplog,
    ):
        fake_client = MagicMock()
        fake_client.attach_trial.side_effect = [1, 2, 3, 4]
        mock_require_client.return_value = lambda: fake_client

        # Four trials with varying throughput but identical retransmit
        # counts AND identical bytes_sent => tcp_retransmit_rate is constant
        # across trials, throughput is not.
        priors = [
            self._trial(bps=8e9, retransmits=5),
            self._trial(bps=9e9, retransmits=5),
            self._trial(bps=9.5e9, retransmits=5),
            self._trial(bps=8.5e9, retransmits=5),
        ]

        from kube_autotuner.optimizer import OptimizationLoop  # noqa: PLC0415

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=tmp_path / "r.jsonl",
            n_trials=10,
            n_sobol=1,
            objectives=self._reduced_objectives(),
            prior_trials=priors,
        )

        caplog.set_level("WARNING", logger="kube_autotuner.optimizer")
        warned = loop._warn_on_collapsed_objectives()

        assert warned == ["tcp_retransmit_rate"]
        collapse_records = [
            rec
            for rec in caplog.records
            if "collapsed to near-constant variance" in rec.message
        ]
        assert len(collapse_records) == 1
        assert "tcp_retransmit_rate" in collapse_records[0].message

        caplog.clear()
        warned_again = loop._warn_on_collapsed_objectives()
        assert warned_again == []
        assert not [
            rec
            for rec in caplog.records
            if "collapsed to near-constant variance" in rec.message
        ]

    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    @patch("kube_autotuner.optimizer._require_ax_client")
    def test_no_warning_when_variance_healthy(
        self,
        mock_require_client,
        mock_setter_cls,  # noqa: ARG002
        mock_runner_cls,  # noqa: ARG002
        node_pair,
        config,
        tmp_path,
        caplog,
    ):
        fake_client = MagicMock()
        fake_client.attach_trial.side_effect = [1, 2, 3]
        mock_require_client.return_value = lambda: fake_client

        priors = [
            self._trial(bps=8e9, retransmits=3),
            self._trial(bps=9e9, retransmits=5),
            self._trial(bps=9.5e9, retransmits=7),
        ]

        from kube_autotuner.optimizer import OptimizationLoop  # noqa: PLC0415

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=tmp_path / "r.jsonl",
            n_trials=10,
            n_sobol=1,
            objectives=self._reduced_objectives(),
            prior_trials=priors,
        )

        caplog.set_level("WARNING", logger="kube_autotuner.optimizer")
        warned = loop._warn_on_collapsed_objectives()
        assert warned == []


class TestUpstreamNoiseFilters:
    """``_register_noise_filters`` registers the four expected entries."""

    def test_filter_entries_registered(self) -> None:
        import warnings  # noqa: PLC0415

        from kube_autotuner.optimizer import (  # noqa: PLC0415
            _register_noise_filters,  # noqa: PLC2701
        )

        def _pat(obj) -> str | None:
            if obj is None:
                return None
            pat = getattr(obj, "pattern", obj)
            return pat or None

        def _canonical(filt: tuple) -> tuple:
            action, msg, cat, mod, lineno = filt
            return (action, _pat(msg), cat, _pat(mod), lineno)

        # ``catch_warnings`` saves and restores ``warnings.filters`` so
        # re-registering here does not pollute other tests.
        with warnings.catch_warnings():
            warnings.resetwarnings()
            _register_noise_filters()
            canonical = [_canonical(f) for f in warnings.filters]

        expected = [
            ("ignore", None, SyntaxWarning, r"pyro\..*", 0),
            (
                "ignore",
                r"To copy construct from a tensor.*",
                UserWarning,
                None,
                0,
            ),
            ("ignore", None, Warning, r"botorch($|\.)", 0),
            ("ignore", None, RuntimeWarning, r"gpytorch($|\.)", 0),
        ]
        for entry in expected:
            assert entry in canonical, (entry, canonical)


class TestSeededPriorAndPin:
    """Seeded prior + per-trial methodology pin behaviour."""

    @pytest.fixture
    def node_pair(self) -> NodePair:
        return NodePair(source="kmain07", target="kmain08", hardware_class="10g")

    @pytest.fixture
    def config(self) -> BenchmarkConfig:
        return BenchmarkConfig()

    @patch("kube_autotuner.optimizer.NodeLease")
    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    def test_first_trial_applies_recommended_defaults(
        self,
        mock_setter_cls,
        mock_runner_cls,
        mock_lease_cls,  # noqa: ARG002
        node_pair,
        config,
        tmp_path,
    ):
        """First trial on a fresh run must evaluate RECOMMENDED_DEFAULTS."""
        mock_setter = MagicMock()
        mock_setter.snapshot.side_effect = _mock_snapshot
        mock_setter_cls.return_value = mock_setter

        mock_runner = MagicMock()
        mock_runner.run.return_value = IterationResults(
            bench=_make_results(),
            latency=_make_latency_results(),
        )
        mock_runner_cls.return_value = mock_runner

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=tmp_path / "results.jsonl",
            n_trials=1,
            n_sobol=1,
            objectives=ObjectivesSection(),
        )
        trials = loop.run()

        assert len(trials) == 1
        # Every declared knob appears in the recorded parameterization
        # at its RECOMMENDED_DEFAULTS value.
        recorded = trials[0].sysctl_values
        for name, value in RECOMMENDED_DEFAULTS.items():
            if name in PARAM_SPACE.param_names():
                assert str(recorded[name]) == str(value), name

    @patch("kube_autotuner.optimizer.NodeLease")
    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    def test_apply_pins_tcp_no_metrics_save_and_wires_flush(
        self,
        mock_setter_cls,
        mock_runner_cls,
        mock_lease_cls,  # noqa: ARG002
        node_pair,
        config,
        tmp_path,
    ):
        """Apply pins ``tcp_no_metrics_save=1`` and flush is wired into the runner."""
        mock_setter = MagicMock()
        mock_setter.snapshot.side_effect = _mock_snapshot
        mock_setter_cls.return_value = mock_setter

        mock_runner = MagicMock()
        mock_runner.run.return_value = IterationResults(
            bench=_make_results(),
            latency=_make_latency_results(),
        )
        mock_runner_cls.return_value = mock_runner

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=tmp_path / "results.jsonl",
            n_trials=2,
            n_sobol=2,
            objectives=ObjectivesSection(),
        )
        loop.run()

        # Each of the 2 trials calls apply() once on the target.
        assert mock_setter.apply.call_count == 2
        for call in mock_setter.apply.call_args_list:
            applied = call[0][0]
            assert applied["net.ipv4.tcp_no_metrics_save"] == 1

        # Flushing moved from ``_evaluate`` into ``BenchmarkRunner.run()``,
        # so the optimizer's contract is to pass the target setter as a
        # flush backend when constructing the runner. With
        # ``apply_source`` defaulting to ``False`` there are no client
        # setters, so the list contains only the target.
        mock_runner_cls.assert_called_once()
        flush_backends = mock_runner_cls.call_args.kwargs["flush_backends"]
        assert flush_backends == [mock_setter]

    @patch("kube_autotuner.optimizer.NodeLease")
    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    def test_flush_backends_include_client_setters_when_apply_source(
        self,
        mock_setter_cls,
        mock_runner_cls,
        mock_lease_cls,  # noqa: ARG002
        config,
        tmp_path,
    ):
        """``apply_source=True`` wires target + each client setter for flushing."""
        node_pair = NodePair(
            source="kmain07",
            target="kmain08",
            hardware_class="10g",
            namespace="default",
            extra_sources=["kmain09"],
        )

        setters_by_node: dict[str, MagicMock] = {}

        def _setter_factory(*, node, namespace, **_kwargs):
            del namespace
            s = MagicMock()
            s.snapshot.side_effect = _mock_snapshot
            setters_by_node[node] = s
            return s

        mock_setter_cls.side_effect = _setter_factory

        mock_runner = MagicMock()
        mock_runner.run.return_value = IterationResults(
            bench=_make_results(),
            latency=_make_latency_results(),
        )
        mock_runner_cls.return_value = mock_runner

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=tmp_path / "results.jsonl",
            n_trials=1,
            n_sobol=1,
            apply_source=True,
            objectives=ObjectivesSection(),
        )
        loop.run()

        mock_runner_cls.assert_called_once()
        flush_backends = mock_runner_cls.call_args.kwargs["flush_backends"]
        target_setter = setters_by_node["kmain08"]
        client_setters = [setters_by_node[n] for n in ("kmain07", "kmain09")]
        assert flush_backends == [target_setter, *client_setters]

    @patch("kube_autotuner.optimizer.NodeLease")
    @patch("kube_autotuner.optimizer.BenchmarkRunner")
    @patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
    def test_seed_retries_on_failed_first_trial(
        self,
        mock_setter_cls,
        mock_runner_cls,
        mock_lease_cls,  # noqa: ARG002
        node_pair,
        config,
        tmp_path,
    ):
        """A failed seed trial gets re-attached on the next iteration."""
        mock_setter = MagicMock()
        mock_setter.snapshot.side_effect = _mock_snapshot
        mock_setter_cls.return_value = mock_setter

        mock_runner = MagicMock()
        call_count = 0

        def run_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("seed trial crashed")  # noqa: TRY003
            return IterationResults(
                bench=_make_results(),
                latency=_make_latency_results(),
            )

        mock_runner.run.side_effect = run_side_effect
        mock_runner_cls.return_value = mock_runner

        loop = OptimizationLoop(
            node_pair=node_pair,
            config=config,
            param_space=PARAM_SPACE,
            output=tmp_path / "results.jsonl",
            n_trials=2,
            n_sobol=2,
            objectives=ObjectivesSection(),
        )
        trials = loop.run()

        # One failure + one success => one recorded trial whose
        # parameterization must still be RECOMMENDED_DEFAULTS (the
        # second attempt re-attached the seed).
        assert len(trials) == 1
        recorded = trials[0].sysctl_values
        for name, value in RECOMMENDED_DEFAULTS.items():
            if name in PARAM_SPACE.param_names():
                assert str(recorded[name]) == str(value), name


def test_seed_prior_trials_filters_stale_keys(
    tmp_path,
):
    """Historical JSONL containing a now-removed knob must still attach."""
    node_pair = NodePair(source="a", target="b", hardware_class="10g")
    stale_sysctls: dict[str, str | int] = {
        p.name: p.values[0] for p in PARAM_SPACE.params
    }
    # Simulate a JSONL row written before ``tcp_no_metrics_save`` was
    # removed from the space.
    stale_sysctls["net.ipv4.tcp_no_metrics_save"] = 0
    prior = TrialResult(
        node_pair=node_pair,
        sysctl_values=stale_sysctls,
        config=BenchmarkConfig(),
        results=[
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=9e9,
                retransmits=5,
                bytes_sent=10**9,
            ),
        ],
        phase="sobol",
    )

    with (
        patch("kube_autotuner.optimizer.NodeLease"),
        patch("kube_autotuner.optimizer.BenchmarkRunner"),
        patch("kube_autotuner.optimizer.make_sysctl_setter_from_env"),
    ):
        # Constructor invokes ``_seed_prior_trials``; if the stale key
        # is passed through to Ax, ``attach_trial`` raises.
        OptimizationLoop(
            node_pair=node_pair,
            config=BenchmarkConfig(),
            param_space=PARAM_SPACE,
            output=tmp_path / "out.jsonl",
            n_trials=2,
            n_sobol=1,
            objectives=ObjectivesSection(),
            prior_trials=[prior],
        )
