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
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("ax")

from kube_autotuner.experiment import ObjectivesSection, ParetoObjective
from kube_autotuner.k8s.lease import LeaseHeldError
from kube_autotuner.models import BenchmarkConfig, BenchmarkResult, NodePair
from kube_autotuner.optimizer import (
    OptimizationLoop,
    _aggregate_by_iteration,  # noqa: PLC2701
    _compute_metrics,  # noqa: PLC2701
    _decode_param_name,  # noqa: PLC2701
    _encode_param_name,  # noqa: PLC2701
    build_ax_objective,
    build_ax_params,
)
from kube_autotuner.sysctl.params import PARAM_SPACE


def _make_results(n: int = 3) -> list[BenchmarkResult]:
    return [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=9_000_000_000 + i * 100_000,
            retransmits=5 + i,
            cpu_utilization_percent=30.0 + i,
            memory_used_bytes=100_000_000,
            client_node="kmain07",
            iteration=i,
        )
        for i in range(n)
    ]


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
        return BenchmarkConfig(duration=10, iterations=3, modes=["tcp"])

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
        mock_runner.run.return_value = _make_results()
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
        assert len(snap_args) == len(PARAM_SPACE.params) + 1

        assert mock_lease_cls.call_count == 6

        lines = output.read_text().strip().splitlines()
        assert len(lines) == 3

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
            return _make_results()

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
        mock_runner.run.return_value = _make_results()
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
        mock_runner.run.return_value = _make_results()
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
                cpu_utilization_percent=10.0,
                client_node="c1",
                iteration=0,
            ),
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=2e9,
                retransmits=2,
                cpu_utilization_percent=20.0,
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
                cpu_utilization_percent=10.0,
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
                cpu_utilization_percent=10.0,
                client_node=client,
                iteration=0,
            )

        results = [_r(4e9, "c1"), _r(6e9, "c2")]
        metrics = _compute_metrics(results)
        assert metrics["throughput"][0] == pytest.approx(1e10)
        assert metrics["throughput"][1] > 0.0


class TestBuildAxObjective:
    def test_default_section_matches_legacy_literal(self) -> None:
        objective, constraints = build_ax_objective(ObjectivesSection())
        assert objective == "throughput, -cpu, -retransmits, -memory"
        assert constraints == [
            "throughput >= 1e6",
            "cpu <= 200",
            "retransmits <= 1e6",
            "memory <= 1e10",
        ]

    def test_reduced_two_metric_section(self) -> None:
        section = ObjectivesSection(
            pareto=[
                ParetoObjective(metric="throughput", direction="maximize"),
                ParetoObjective(metric="memory", direction="minimize"),
            ],
            constraints=["throughput >= 1e6"],
            recommendation_weights={"memory": 0.5},
        )
        objective, constraints = build_ax_objective(section)
        assert objective == "throughput, -memory"
        assert constraints == ["throughput >= 1e6"]
