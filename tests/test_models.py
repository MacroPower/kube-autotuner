"""Tests for ``kube_autotuner.models``."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import TYPE_CHECKING

from pydantic import ValidationError
import pytest

from kube_autotuner.experiment import ObjectivesSection, ParetoObjective
from kube_autotuner.models import (
    BenchmarkConfig,
    BenchmarkResult,
    IterationResults,
    LatencyResult,
    NodePair,
    ParamSpace,
    ResumeMetadata,
    SysctlParam,
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


def test_trial_result_mean_tcp_throughput():
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1_000_000_000,
            client_node="n1",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=2_000_000_000,
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
    assert trial.mean_tcp_throughput() == pytest.approx(1_500_000_000)


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
    assert config.window is None


def test_benchmark_config_silently_drops_legacy_modes_key():
    # Pre-change YAML embedded `modes: [tcp]`; Pydantic's default
    # extra="ignore" must drop the stale key so resume-sidecar
    # comparisons still succeed.
    config = BenchmarkConfig.model_validate(
        {"duration": 10, "iterations": 1, "modes": ["tcp"]},
    )
    assert config.duration == 10
    assert config.iterations == 1
    assert not hasattr(config, "modes")


def test_node_pair_defaults():
    pair = NodePair(source="a", target="b", hardware_class="10g")
    assert pair.namespace == "default"
    assert not pair.source_zone
    assert not pair.target_zone


def test_node_pair_accepts_arbitrary_hardware_class():
    pair = NodePair(source="a", target="b", hardware_class="graviton4")
    assert pair.hardware_class == "graviton4"


def test_node_pair_rejects_empty_hardware_class():
    with pytest.raises(ValidationError):
        NodePair(source="a", target="b", hardware_class="")


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
            client_node="n1",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=3e9,
            retransmits=2,
            bytes_sent=1_000_000_000,
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
    assert trial.mean_tcp_throughput() == pytest.approx(2e9)
    # rates: iter0=5/1e9, iter1=2/1e9; mean=3.5/1e9.
    assert trial.tcp_retransmit_rate() == pytest.approx(3.5e-9)
    assert trial.total_bytes_sent() == 2_000_000_000


def test_trial_result_multi_client_per_iteration_grouping():
    """Multi-client records sum throughput within iteration, then mean."""

    def _r(bps, client, itr):
        return BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=bps,
            retransmits=1,
            bytes_sent=1_000_000_000,
            client_node=client,
            iteration=itr,
        )

    results = [
        _r(4e9, "c1", 0),
        _r(5e9, "c2", 0),
        _r(3e9, "c1", 1),
        _r(6e9, "c2", 1),
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
    assert trial.mean_tcp_throughput() == pytest.approx(9e9)
    # Rate: each iteration has 2 retx over 2GB; per-iter rate = 1e-9.
    assert trial.tcp_retransmit_rate() == pytest.approx(1e-9)
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
    assert trial.tcp_retransmit_rate() is None
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
    assert trial.tcp_retransmit_rate() == pytest.approx(1e-8)


def test_mean_throughput_filters_to_tcp_records():
    """Mixed TCP+UDP result list: UDP records must not poison TCP aggregates."""
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            client_node="n",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=4e9,
            client_node="n",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=3e9,
            client_node="n",
            iteration=1,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=5e9,
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
    # TCP-only: iter0=1e9, iter1=3e9 -> mean 2e9. UDP records dropped.
    assert trial.mean_tcp_throughput() == pytest.approx(2e9)


def test_mean_throughput_returns_zero_when_no_tcp_records():
    """All-UDP result list must produce 0.0 for TCP-only aggregates."""
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=5e9,
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
    assert trial.mean_tcp_throughput() == pytest.approx(0.0)


def test_mean_udp_throughput_filters_to_udp_records():
    """Mixed TCP+UDP results: TCP must not poison UDP throughput aggregate."""
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=9e9,
            client_node="n",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=1e9,
            client_node="n",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=3e9,
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
    # UDP-only: iter0=1e9, iter1=3e9 -> mean 2e9.
    assert trial.mean_udp_throughput() == pytest.approx(2e9)


def test_mean_udp_throughput_returns_zero_when_no_udp_records():
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
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
    assert trial.mean_udp_throughput() == pytest.approx(0.0)


def test_udp_loss_rate_typical():
    """Multi-iteration UDP trial: per-iter ratio-of-sums then mean."""
    results = [
        # iter 0: 100 lost / 10000 packets = 0.01
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=1e9,
            packets=10000,
            lost_packets=100,
            client_node="n",
            iteration=0,
        ),
        # iter 1: 30 lost / 10000 packets = 0.003
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=1e9,
            packets=10000,
            lost_packets=30,
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
    # mean(0.01, 0.003) = 0.0065
    assert trial.udp_loss_rate() == pytest.approx(0.0065)


def test_udp_loss_rate_returns_zero_when_no_udp_records():
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            retransmits=0,
            bytes_sent=1_000_000,
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
    assert trial.udp_loss_rate() == pytest.approx(0.0)


def test_udp_loss_rate_drops_iterations_missing_packets():
    """Iter where no UDP record reports packets is excluded from the mean."""
    results = [
        # iter 0: real UDP measurement, rate = 0.02
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=1e9,
            packets=5000,
            lost_packets=100,
            client_node="n",
            iteration=0,
        ),
        # iter 1: UDP record but packets missing -> excluded by filter
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=1e9,
            packets=None,
            lost_packets=None,
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
    # only iter 0 contributes -> 0.02
    assert trial.udp_loss_rate() == pytest.approx(0.02)


def test_mean_jitter_with_mixed_tcp_udp_records():
    """UDP is always observable now: jitter filter honours jitter is not None."""
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            jitter=None,
            client_node="n",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=1e9,
            jitter=0.0002,
            client_node="n",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=1e9,
            jitter=None,
            client_node="n",
            iteration=1,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=1e9,
            jitter=0.0004,
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
    # Iter 0 jitter=0.0002, iter 1 jitter=0.0004 -> mean 0.0003 s; TCP None dropped.
    assert trial.mean_udp_jitter() == pytest.approx(0.0003)


def test_trial_result_mean_jitter_single_client():
    results = [
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=1e9,
            jitter=0.0001,
            client_node="n",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=1e9,
            jitter=0.0003,
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
    assert trial.mean_udp_jitter() == pytest.approx(0.0002)


def test_trial_result_mean_jitter_multi_client():
    def _r(j, client, itr):
        return BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="udp",
            bits_per_second=1e9,
            jitter=j,
            client_node=client,
            iteration=itr,
        )

    results = [
        _r(0.0001, "c1", 0),
        _r(0.0003, "c2", 0),
        _r(0.0002, "c1", 1),
        _r(0.0004, "c2", 1),
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
    # Iter 0 mean = 0.0002, Iter 1 mean = 0.0003 -> overall mean 0.00025.
    assert trial.mean_udp_jitter() == pytest.approx(0.00025)


def test_benchmark_result_new_fields_round_trip():
    result = BenchmarkResult(
        timestamp=datetime.now(UTC),
        mode="tcp",
        bits_per_second=9e9,
        client_node="c7",
        iteration=2,
    )
    dumped = result.model_dump_json()
    restored = BenchmarkResult.model_validate_json(dumped)
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
            latency_p50=p50,
            latency_p90=p90,
            latency_p99=p99,
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
        assert trial.mean_latency_p99() == pytest.approx(15.0)

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
        assert trial.mean_latency_p50() == pytest.approx(2.0)
        assert trial.mean_latency_p90() == pytest.approx(3.0)
        assert trial.mean_latency_p99() == pytest.approx(4.0)


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
                latency_p99=5.0,
            ),
        ],
    )
    TrialLog.append(path, trial)
    loaded = TrialLog.load(path)
    assert len(loaded) == 1
    assert len(loaded[0].latency_results) == 1
    assert loaded[0].latency_results[0].workload == "fixed_qps"


class TestResumeMetadata:
    def _meta(self, *, n_sobol: int | None = 15) -> ResumeMetadata:
        return ResumeMetadata(
            objectives=ObjectivesSection(
                pareto=[
                    ParetoObjective(metric="tcp_throughput", direction="maximize"),
                    ParetoObjective(metric="udp_jitter", direction="minimize"),
                ],
                constraints=["tcp_throughput >= 1e6"],
                recommendation_weights={"udp_jitter": 0.5},
            ),
            param_space=ParamSpace(
                params=[
                    SysctlParam(
                        name="net.core.rmem_max",
                        values=[1048576, 16777216],
                        param_type="int",
                    ),
                ],
            ),
            benchmark=BenchmarkConfig(duration=10, iterations=2),
            n_sobol=n_sobol,
        )

    def test_resume_metadata_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "results.jsonl"
        meta = self._meta()
        TrialLog.write_resume_metadata(path, meta)
        loaded = TrialLog.load_resume_metadata(path)
        assert loaded is not None
        assert loaded.model_dump() == meta.model_dump()

    def test_resume_metadata_requires_param_space_and_benchmark(
        self,
        tmp_path: Path,
    ) -> None:
        # Write a partial sidecar missing benchmark; load must raise.
        path = tmp_path / "results.jsonl"
        meta_path = path.parent / f"{path.name}.meta.json"
        meta_path.write_text(
            '{"objectives": {"pareto": [{"metric": "tcp_throughput", '
            '"direction": "maximize"}], "constraints": [], '
            '"recommendationWeights": {}}, '
            '"param_space": {"params": []}}\n',
        )
        with pytest.raises(ValidationError):
            TrialLog.load_resume_metadata(path)

    def test_load_resume_metadata_rejects_old_objectives_only_file(
        self,
        tmp_path: Path,
    ) -> None:
        # Write an objectives-only sidecar — the old format.
        path = tmp_path / "results.jsonl"
        meta_path = path.parent / f"{path.name}.meta.json"
        meta_path.write_text(
            '{"pareto": [{"metric": "tcp_throughput", '
            '"direction": "maximize"}], "constraints": [], '
            '"recommendationWeights": {}}\n',
        )
        with pytest.raises(ValidationError):
            TrialLog.load_resume_metadata(path)

    def test_missing_sidecar_returns_none(self, tmp_path: Path) -> None:
        assert TrialLog.load_resume_metadata(tmp_path / "absent.jsonl") is None

    def test_resume_metadata_n_sobol_optional(self, tmp_path: Path) -> None:
        path = tmp_path / "results.jsonl"
        meta = self._meta(n_sobol=None)
        TrialLog.write_resume_metadata(path, meta)
        loaded = TrialLog.load_resume_metadata(path)
        assert loaded is not None
        assert loaded.n_sobol is None


class TestTrialLogLoadTruncatedTail:
    def _make_trial(self, iter_idx: int) -> TrialResult:
        return TrialResult(
            node_pair=NodePair(source="n1", target="n2", hardware_class="10g"),
            sysctl_values={"net.core.rmem_max": 1048576 + iter_idx},
            config=BenchmarkConfig(),
            results=[
                BenchmarkResult(
                    timestamp=datetime.now(UTC),
                    mode="tcp",
                    bits_per_second=1e9 + iter_idx,
                ),
            ],
        )

    def test_malformed_trailing_line_is_dropped(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = tmp_path / "results.jsonl"
        TrialLog.append(path, self._make_trial(0))
        TrialLog.append(path, self._make_trial(1))
        with path.open("a", encoding="utf-8") as f:
            f.write("{not valid json\n")
        caplog.set_level("WARNING")
        loaded = TrialLog.load(path)
        assert len(loaded) == 2
        assert any("truncated" in rec.message for rec in caplog.records)

    def test_partial_trailing_line_without_newline_is_dropped(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = tmp_path / "results.jsonl"
        TrialLog.append(path, self._make_trial(0))
        TrialLog.append(path, self._make_trial(1))
        # Simulate a Ctrl-C mid-write: a partial JSON line without
        # the trailing newline.
        with path.open("a", encoding="utf-8") as f:
            f.write('{"trial_id": "abc123", "node_pair"')
        caplog.set_level("WARNING")
        loaded = TrialLog.load(path)
        assert len(loaded) == 2
        assert any("truncated" in rec.message for rec in caplog.records)

    def test_mid_file_corruption_still_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "results.jsonl"
        with path.open("w", encoding="utf-8") as f:
            f.write("{not valid json\n")
            f.write(self._make_trial(0).model_dump_json() + "\n")
        with pytest.raises((ValidationError, json.JSONDecodeError)):
            TrialLog.load(path)
