"""Tests for ``kube_autotuner.models``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from kube_autotuner.experiment import ObjectivesSection, ParetoObjective
from kube_autotuner.models import (
    BenchmarkConfig,
    BenchmarkResult,
    IterationResults,
    LatencyResult,
    NodePair,
    TrialLog,
    TrialResult,
    compute_sysctl_hash,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_sysctl_hash_deterministic():
    params = {"net.core.rmem_max": 67108864, "net.core.wmem_max": 67108864}
    h1 = compute_sysctl_hash(params)
    h2 = compute_sysctl_hash(params)
    assert h1 == h2
    assert len(h1) == 16


def test_sysctl_hash_order_independent():
    h1 = compute_sysctl_hash({"a": 1, "b": 2})
    h2 = compute_sysctl_hash({"b": 2, "a": 1})
    assert h1 == h2


def test_trial_result_auto_hash():
    trial = TrialResult(
        node_pair=NodePair(source="n1", target="n2", hardware_class="10g"),
        sysctl_values={"net.core.rmem_max": 67108864},
        config=BenchmarkConfig(),
    )
    assert trial.sysctl_hash
    assert trial.sysctl_hash == compute_sysctl_hash(trial.sysctl_values)


def test_trial_result_mean_throughput():
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1_000_000_000,
            cpu_utilization_percent=10.0,
            client_node="n1",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=2_000_000_000,
            cpu_utilization_percent=20.0,
            client_node="n1",
            iteration=1,
        ),
    ]
    trial = TrialResult(
        node_pair=NodePair(source="n1", target="n2", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
    )
    assert trial.mean_throughput() == pytest.approx(1_500_000_000)
    assert trial.mean_cpu() == pytest.approx(15.0)


def test_trial_log_round_trip(tmp_path: Path):
    path = tmp_path / "trials.jsonl"
    trial = TrialResult(
        node_pair=NodePair(source="n1", target="n2", hardware_class="1g"),
        sysctl_values={"net.core.rmem_max": 212992},
        kernel_version="6.1.0",
        config=BenchmarkConfig(duration=10, iterations=1),
        results=[
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=500_000_000,
                retransmits=5,
                cpu_utilization_percent=8.0,
            ),
        ],
    )

    TrialLog.append(path, trial)
    TrialLog.append(path, trial)

    loaded = TrialLog.load(path)
    assert len(loaded) == 2
    assert loaded[0].sysctl_values == trial.sysctl_values
    assert loaded[0].results[0].bits_per_second == pytest.approx(500_000_000)
    assert loaded[0].kernel_version == "6.1.0"


def test_trial_log_load_empty(tmp_path: Path):
    path = tmp_path / "nonexistent.jsonl"
    assert TrialLog.load(path) == []


def test_benchmark_config_defaults():
    config = BenchmarkConfig()
    assert config.duration == 30
    assert config.omit == 5
    assert config.iterations == 3
    assert config.parallel == 16
    assert config.modes == ["tcp"]
    assert config.window is None


def test_node_pair_defaults():
    pair = NodePair(source="a", target="b", hardware_class="10g")
    assert pair.namespace == "default"
    assert not pair.source_zone
    assert not pair.target_zone


def test_node_pair_topology_intra_az():
    pair = NodePair(
        source="a",
        target="b",
        hardware_class="10g",
        source_zone="az01",
        target_zone="az01",
    )
    assert pair.topology == "intra-az"


def test_node_pair_topology_inter_az():
    pair = NodePair(
        source="a",
        target="b",
        hardware_class="10g",
        source_zone="az01",
        target_zone="az02",
    )
    assert pair.topology == "inter-az"


def test_node_pair_topology_unknown_when_empty():
    pair = NodePair(source="a", target="b", hardware_class="10g")
    assert pair.topology == "unknown"
    # Both empty should still be unknown, not intra-az.
    pair2 = NodePair(
        source="a",
        target="b",
        hardware_class="10g",
        source_zone="",
        target_zone="",
    )
    assert pair2.topology == "unknown"


def test_node_pair_topology_unknown_when_partial():
    pair = NodePair(
        source="a",
        target="b",
        hardware_class="10g",
        source_zone="az01",
        target_zone="",
    )
    assert pair.topology == "unknown"


def test_trial_result_auto_topology():
    trial = TrialResult(
        node_pair=NodePair(
            source="n1",
            target="n2",
            hardware_class="10g",
            source_zone="az01",
            target_zone="az02",
        ),
        sysctl_values={},
        config=BenchmarkConfig(),
    )
    assert trial.topology == "inter-az"


def test_trial_result_topology_defaults_unknown():
    trial = TrialResult(
        node_pair=NodePair(source="n1", target="n2", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
    )
    assert trial.topology == "unknown"


def test_trial_log_round_trip_with_zones(tmp_path: Path):
    path = tmp_path / "trials.jsonl"
    trial = TrialResult(
        node_pair=NodePair(
            source="n1",
            target="n2",
            hardware_class="10g",
            source_zone="az01",
            target_zone="az02",
        ),
        sysctl_values={"net.core.rmem_max": 212992},
        config=BenchmarkConfig(duration=10, iterations=1),
        results=[
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=500_000_000,
            ),
        ],
    )
    TrialLog.append(path, trial)
    loaded = TrialLog.load(path)
    assert loaded[0].node_pair.source_zone == "az01"
    assert loaded[0].node_pair.target_zone == "az02"
    assert loaded[0].topology == "inter-az"


def test_benchmark_result_with_memory():
    result = BenchmarkResult(
        timestamp=datetime.now(UTC),
        mode="tcp",
        bits_per_second=9_000_000_000,
        node_memory_used_bytes=47185920,
        cni_memory_used_bytes=12582912,
    )
    assert result.node_memory_used_bytes == 47185920
    assert result.cni_memory_used_bytes == 12582912
    # Round-trip through JSON.
    data = result.model_dump_json()
    restored = BenchmarkResult.model_validate_json(data)
    assert restored.node_memory_used_bytes == 47185920
    assert restored.cni_memory_used_bytes == 12582912


def test_node_pair_all_sources_and_zone_for():
    pair = NodePair(
        source="n1",
        target="t",
        hardware_class="10g",
        source_zone="az01",
        extra_sources=["n2", "n3"],
        extra_source_zones={"n2": "az02", "n3": ""},
    )
    assert pair.all_sources == ["n1", "n2", "n3"]
    assert pair.zone_for("n1") == "az01"
    assert pair.zone_for("n2") == "az02"
    assert not pair.zone_for("n3")
    assert not pair.zone_for("unknown")


def test_trial_result_single_client_multiple_iterations():
    """Per-iteration sum equals the single record, so mean == mean of records."""
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            retransmits=5,
            bytes_sent=1_000_000_000,
            cpu_utilization_percent=10.0,
            client_node="n1",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=3e9,
            retransmits=2,
            bytes_sent=1_000_000_000,
            cpu_utilization_percent=30.0,
            client_node="n1",
            iteration=1,
        ),
    ]
    trial = TrialResult(
        node_pair=NodePair(source="n1", target="n2", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
    )
    assert trial.mean_throughput() == pytest.approx(2e9)
    assert trial.mean_cpu() == pytest.approx(20.0)
    # rates: iter0=5/1e9, iter1=2/1e9; mean=3.5/1e9.
    assert trial.retransmit_rate() == pytest.approx(3.5e-9)
    assert trial.total_bytes_sent() == 2_000_000_000


def test_trial_result_multi_client_per_iteration_grouping():
    """Multi-client records sum throughput within iteration, then mean."""

    def _r(bps, cpu, client, itr):
        return BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=bps,
            retransmits=1,
            bytes_sent=1_000_000_000,
            cpu_utilization_percent=cpu,
            cpu_server_percent=cpu,
            client_node=client,
            iteration=itr,
        )

    results = [
        _r(4e9, 10.0, "c1", 0),
        _r(5e9, 20.0, "c2", 0),
        _r(3e9, 15.0, "c1", 1),
        _r(6e9, 25.0, "c2", 1),
    ]
    trial = TrialResult(
        node_pair=NodePair(
            source="c1",
            target="t",
            hardware_class="10g",
            extra_sources=["c2"],
        ),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
    )
    # Iter 0 sum = 9e9, Iter 1 sum = 9e9 -> mean 9e9.
    assert trial.mean_throughput() == pytest.approx(9e9)
    # CPU uses cpu_server_percent when all present: iter0=15, iter1=20 -> mean 17.5.
    assert trial.mean_cpu() == pytest.approx(17.5)
    # Rate: each iteration has 2 retx over 2GB; per-iter rate = 1e-9.
    assert trial.retransmit_rate() == pytest.approx(1e-9)
    assert trial.total_bytes_sent() == 4_000_000_000


def test_retransmit_rate_returns_none_when_no_bytes():
    """UDP-only or bytes-less TCP trials have no observable rate."""
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=1e9,
            retransmits=None,
            bytes_sent=None,
            client_node="n1",
            iteration=0,
        ),
    ]
    trial = TrialResult(
        node_pair=NodePair(source="n1", target="n2", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
    )
    assert trial.retransmit_rate() is None
    assert trial.total_bytes_sent() == 0


def test_retransmit_rate_drops_iterations_missing_retx_record():
    """Iter with bytes but no retx record (mixed TCP/UDP) is dropped."""
    results = [
        # Iter 0: TCP record with both retx and bytes -> valid rate.
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            retransmits=10,
            bytes_sent=1_000_000_000,
            iteration=0,
        ),
        # Iter 1: UDP-only -> no retransmits, but some bytes_sent would
        # still arrive if bytes were attached. Here we set bytes=None so
        # the iter is dropped by the bytes>0 filter anyway. Construct a
        # deliberately mixed case: record has bytes but no retransmits.
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=1e9,
            retransmits=None,
            bytes_sent=1_000_000_000,
            iteration=1,
        ),
    ]
    trial = TrialResult(
        node_pair=NodePair(source="n1", target="n2", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
    )
    # Only iter 0 contributes (bytes present AND retx recorded).
    assert trial.retransmit_rate() == pytest.approx(1e-8)


def test_trial_result_multi_client_falls_back_to_host_cpu():
    """If any record lacks cpu_server_percent, fall back to host CPU."""
    a = BenchmarkResult(
        timestamp=datetime.now(UTC),
        mode="tcp",
        bits_per_second=1e9,
        cpu_utilization_percent=10.0,
        cpu_server_percent=40.0,
        client_node="c1",
        iteration=0,
    )
    b = BenchmarkResult(
        timestamp=datetime.now(UTC),
        mode="tcp",
        bits_per_second=1e9,
        cpu_utilization_percent=20.0,
        cpu_server_percent=None,
        client_node="c2",
        iteration=0,
    )
    trial = TrialResult(
        node_pair=NodePair(
            source="c1",
            target="t",
            hardware_class="10g",
            extra_sources=["c2"],
        ),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=[a, b],
    )
    # Falls back to host CPU mean: (10+20)/2 = 15.
    assert trial.mean_cpu() == pytest.approx(15.0)


def test_trial_result_mean_jitter_ms_single_client():
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=1e9,
            jitter_ms=0.1,
            client_node="n",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=1e9,
            jitter_ms=0.3,
            client_node="n",
            iteration=1,
        ),
    ]
    trial = TrialResult(
        node_pair=NodePair(source="n", target="t", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
    )
    assert trial.mean_jitter_ms() == pytest.approx(0.2)


def test_trial_result_mean_jitter_ms_multi_client():
    def _r(j, client, itr):
        return BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=1e9,
            jitter_ms=j,
            client_node=client,
            iteration=itr,
        )

    results = [
        _r(0.1, "c1", 0),
        _r(0.3, "c2", 0),
        _r(0.2, "c1", 1),
        _r(0.4, "c2", 1),
    ]
    trial = TrialResult(
        node_pair=NodePair(
            source="c1",
            target="t",
            hardware_class="10g",
            extra_sources=["c2"],
        ),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
    )
    # Iter 0 mean = 0.2, Iter 1 mean = 0.3 -> overall mean 0.25.
    assert trial.mean_jitter_ms() == pytest.approx(0.25)


def test_trial_result_mean_node_memory_single_client():
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            node_memory_used_bytes=100_000_000,
            client_node="n",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            node_memory_used_bytes=200_000_000,
            client_node="n",
            iteration=1,
        ),
    ]
    trial = TrialResult(
        node_pair=NodePair(source="n", target="t", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
    )
    assert trial.mean_node_memory() == pytest.approx(150_000_000)


def test_trial_result_mean_node_memory_multi_client():
    def _r(mem, client, itr):
        return BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            node_memory_used_bytes=mem,
            client_node=client,
            iteration=itr,
        )

    results = [
        _r(100_000_000, "c1", 0),
        _r(300_000_000, "c2", 0),
        _r(200_000_000, "c1", 1),
        _r(400_000_000, "c2", 1),
    ]
    trial = TrialResult(
        node_pair=NodePair(
            source="c1",
            target="t",
            hardware_class="10g",
            extra_sources=["c2"],
        ),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
    )
    # Iter 0 mean = 2e8, Iter 1 mean = 3e8 -> overall 2.5e8.
    assert trial.mean_node_memory() == pytest.approx(250_000_000)


def test_trial_result_mean_node_memory_skips_none_per_group():
    """Records with None node memory are skipped within each iteration."""
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            node_memory_used_bytes=None,
            client_node="c1",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            node_memory_used_bytes=500_000_000,
            client_node="c2",
            iteration=0,
        ),
    ]
    trial = TrialResult(
        node_pair=NodePair(
            source="c1",
            target="t",
            hardware_class="10g",
            extra_sources=["c2"],
        ),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
    )
    assert trial.mean_node_memory() == pytest.approx(500_000_000)


def test_trial_result_mean_node_memory_all_none_returns_none():
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            node_memory_used_bytes=None,
            client_node="n",
            iteration=0,
        ),
    ]
    trial = TrialResult(
        node_pair=NodePair(source="n", target="t", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
    )
    assert trial.mean_node_memory() is None


def test_trial_result_mean_cni_memory_single_client():
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            cni_memory_used_bytes=10_000_000,
            client_node="n",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            cni_memory_used_bytes=30_000_000,
            client_node="n",
            iteration=1,
        ),
    ]
    trial = TrialResult(
        node_pair=NodePair(source="n", target="t", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
    )
    assert trial.mean_cni_memory() == pytest.approx(20_000_000)


def test_trial_result_mean_cni_memory_multi_client():
    def _r(mem, client, itr):
        return BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            cni_memory_used_bytes=mem,
            client_node=client,
            iteration=itr,
        )

    results = [
        _r(10_000_000, "c1", 0),
        _r(30_000_000, "c2", 0),
        _r(20_000_000, "c1", 1),
        _r(40_000_000, "c2", 1),
    ]
    trial = TrialResult(
        node_pair=NodePair(
            source="c1",
            target="t",
            hardware_class="10g",
            extra_sources=["c2"],
        ),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
    )
    # Iter 0 mean = 2e7, Iter 1 mean = 3e7 -> overall 2.5e7.
    assert trial.mean_cni_memory() == pytest.approx(25_000_000)


def test_trial_result_mean_cni_memory_skips_none_per_group():
    """Records with None CNI memory are skipped within each iteration."""
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            cni_memory_used_bytes=None,
            client_node="c1",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            cni_memory_used_bytes=50_000_000,
            client_node="c2",
            iteration=0,
        ),
    ]
    trial = TrialResult(
        node_pair=NodePair(
            source="c1",
            target="t",
            hardware_class="10g",
            extra_sources=["c2"],
        ),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
    )
    assert trial.mean_cni_memory() == pytest.approx(50_000_000)


def test_trial_result_mean_cni_memory_all_none_returns_none():
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            cni_memory_used_bytes=None,
            client_node="n",
            iteration=0,
        ),
    ]
    trial = TrialResult(
        node_pair=NodePair(source="n", target="t", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
        results=results,
    )
    assert trial.mean_cni_memory() is None


def test_benchmark_result_new_fields_round_trip():
    result = BenchmarkResult(
        timestamp=datetime.now(UTC),
        mode="tcp",
        bits_per_second=9e9,
        cpu_server_percent=42.0,
        client_node="c7",
        iteration=2,
    )
    dumped = result.model_dump_json()
    restored = BenchmarkResult.model_validate_json(dumped)
    assert restored.cpu_server_percent == pytest.approx(42.0)
    assert restored.client_node == "c7"
    assert restored.iteration == 2


class TestLatencyAggregation:
    def _latency(
        self,
        *,
        workload,
        iteration: int,
        client_node: str,
        rps: float = 0.0,
        p50: float | None = None,
        p90: float | None = None,
        p99: float | None = None,
    ) -> LatencyResult:
        return LatencyResult(
            timestamp=datetime.now(UTC),
            workload=workload,
            client_node=client_node,
            iteration=iteration,
            rps=rps,
            latency_p50_ms=p50,
            latency_p90_ms=p90,
            latency_p99_ms=p99,
        )

    def test_mean_rps_sums_across_clients_per_iteration(self) -> None:
        trial = TrialResult(
            node_pair=NodePair(
                source="c1",
                target="t",
                hardware_class="10g",
                extra_sources=["c2"],
            ),
            sysctl_values={},
            config=BenchmarkConfig(),
            latency_results=[
                self._latency(
                    workload="saturation",
                    iteration=0,
                    client_node="c1",
                    rps=1000.0,
                ),
                self._latency(
                    workload="saturation",
                    iteration=0,
                    client_node="c2",
                    rps=2000.0,
                ),
                self._latency(
                    workload="saturation",
                    iteration=1,
                    client_node="c1",
                    rps=1500.0,
                ),
                self._latency(
                    workload="saturation",
                    iteration=1,
                    client_node="c2",
                    rps=1500.0,
                ),
                # fixed_qps must be ignored by mean_rps.
                self._latency(
                    workload="fixed_qps",
                    iteration=0,
                    client_node="c1",
                    rps=10_000.0,
                ),
            ],
        )
        # Iter 0 sum = 3000, iter 1 sum = 3000 -> mean 3000.
        assert trial.mean_rps() == pytest.approx(3000.0)

    def test_mean_rps_zero_without_saturation_records(self) -> None:
        trial = TrialResult(
            node_pair=NodePair(source="n", target="t", hardware_class="10g"),
            sysctl_values={},
            config=BenchmarkConfig(),
            latency_results=[
                self._latency(
                    workload="fixed_qps",
                    iteration=0,
                    client_node="n",
                    rps=1000.0,
                ),
            ],
        )
        assert trial.mean_rps() == pytest.approx(0.0)

    def test_mean_latency_p99_ignores_saturation_records(self) -> None:
        trial = TrialResult(
            node_pair=NodePair(source="n", target="t", hardware_class="10g"),
            sysctl_values={},
            config=BenchmarkConfig(),
            latency_results=[
                # fixed_qps records: p99 values 10 and 20 -> mean 15.
                self._latency(
                    workload="fixed_qps",
                    iteration=0,
                    client_node="n",
                    p99=10.0,
                ),
                self._latency(
                    workload="fixed_qps",
                    iteration=1,
                    client_node="n",
                    p99=20.0,
                ),
                # saturation records with nonsense p99 to confirm they are
                # not included in the mean.
                self._latency(
                    workload="saturation",
                    iteration=0,
                    client_node="n",
                    p99=999.0,
                ),
            ],
        )
        assert trial.mean_latency_p99_ms() == pytest.approx(15.0)

    def test_mean_latency_percentiles_per_client_mean(self) -> None:
        trial = TrialResult(
            node_pair=NodePair(
                source="c1",
                target="t",
                hardware_class="10g",
                extra_sources=["c2"],
            ),
            sysctl_values={},
            config=BenchmarkConfig(),
            latency_results=[
                self._latency(
                    workload="fixed_qps",
                    iteration=0,
                    client_node="c1",
                    p50=1.0,
                    p90=2.0,
                    p99=3.0,
                ),
                self._latency(
                    workload="fixed_qps",
                    iteration=0,
                    client_node="c2",
                    p50=3.0,
                    p90=4.0,
                    p99=5.0,
                ),
            ],
        )
        # Single iteration; per-client mean -> (1+3)/2 = 2 for p50.
        assert trial.mean_latency_p50_ms() == pytest.approx(2.0)
        assert trial.mean_latency_p90_ms() == pytest.approx(3.0)
        assert trial.mean_latency_p99_ms() == pytest.approx(4.0)


class TestIterationResults:
    def test_round_trip(self) -> None:
        ir = IterationResults(
            bench=[
                BenchmarkResult(
                    timestamp=datetime.now(UTC),
                    mode="tcp",
                    bits_per_second=1e9,
                ),
            ],
            latency=[
                LatencyResult(
                    timestamp=datetime.now(UTC),
                    workload="saturation",
                    client_node="c1",
                    iteration=0,
                    rps=100.0,
                ),
            ],
        )
        loaded = IterationResults.model_validate_json(ir.model_dump_json())
        assert len(loaded.bench) == 1
        assert len(loaded.latency) == 1
        assert loaded.latency[0].workload == "saturation"


def test_trial_result_latency_results_defaults_empty() -> None:
    trial = TrialResult(
        node_pair=NodePair(source="n1", target="n2", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
    )
    assert trial.latency_results == []


def test_trial_result_round_trip_preserves_latency_results(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    trial = TrialResult(
        node_pair=NodePair(source="n1", target="n2", hardware_class="10g"),
        sysctl_values={},
        config=BenchmarkConfig(),
        latency_results=[
            LatencyResult(
                timestamp=datetime.now(UTC),
                workload="fixed_qps",
                client_node="c1",
                iteration=0,
                rps=1000.0,
                latency_p99_ms=5.0,
            ),
        ],
    )
    TrialLog.append(path, trial)
    loaded = TrialLog.load(path)
    assert len(loaded) == 1
    assert len(loaded[0].latency_results) == 1
    assert loaded[0].latency_results[0].workload == "fixed_qps"


class TestTrialLogMetadata:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "results.jsonl"
        section = ObjectivesSection(
            pareto=[
                ParetoObjective(metric="throughput", direction="maximize"),
                ParetoObjective(metric="node_memory", direction="minimize"),
            ],
            constraints=["throughput >= 1e6"],
            recommendation_weights={"node_memory": 0.5},
        )
        TrialLog.write_metadata(path, section)
        loaded = TrialLog.load_metadata(path)
        assert loaded is not None
        assert loaded.model_dump() == section.model_dump()

    def test_missing_sidecar_returns_none(self, tmp_path: Path) -> None:
        assert TrialLog.load_metadata(tmp_path / "absent.jsonl") is None

    def test_drift_warning_on_overwrite(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = tmp_path / "results.jsonl"
        TrialLog.write_metadata(path, ObjectivesSection())
        capsys.readouterr()
        drifted = ObjectivesSection(
            recommendation_weights={
                "cpu": 0.4,
                "node_memory": 0.15,
                "retransmit_rate": 0.3,
            },
        )
        TrialLog.write_metadata(path, drifted)
        captured = capsys.readouterr()
        assert "overwriting" in captured.err
        assert "different" in captured.err
        assert TrialLog.load_metadata(path) == drifted

    def test_same_section_overwrite_is_silent(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = tmp_path / "results.jsonl"
        section = ObjectivesSection()
        TrialLog.write_metadata(path, section)
        capsys.readouterr()
        TrialLog.write_metadata(path, section)
        captured = capsys.readouterr()
        assert captured.err == ""
