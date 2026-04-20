"""Tests for ``kube_autotuner.models``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from kube_autotuner.models import (
    BenchmarkConfig,
    BenchmarkResult,
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
        memory_used_bytes=47185920,
    )
    assert result.memory_used_bytes == 47185920
    # Round-trip through JSON.
    data = result.model_dump_json()
    restored = BenchmarkResult.model_validate_json(data)
    assert restored.memory_used_bytes == 47185920


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
            cpu_utilization_percent=10.0,
            client_node="n1",
            iteration=0,
        ),
        BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=3e9,
            retransmits=2,
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
    assert trial.total_retransmits() == 7


def test_trial_result_multi_client_per_iteration_grouping():
    """Multi-client records sum throughput within iteration, then mean."""

    def _r(bps, cpu, client, itr):
        return BenchmarkResult(
            timestamp=datetime.now(UTC),
            mode="tcp",
            bits_per_second=bps,
            retransmits=1,
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
    # Retransmits: grouped sum is 4 (2 per iteration).
    assert trial.total_retransmits() == 4


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
