"""Format-level invariants for the Parquet-backed trial log.

Scope: confirm the on-disk shape written by
:meth:`kube_autotuner.trial_log.TrialLog.append` (schema conformance,
compression codec, sequence monotonicity) and the sequence-ordering
guarantee that :meth:`TrialLog.load` enforces.
"""

from __future__ import annotations

from datetime import UTC, datetime
import os
from typing import TYPE_CHECKING, cast

import pyarrow.parquet as pq

from kube_autotuner.models import (
    BenchmarkConfig,
    BenchmarkResult,
    NodePair,
    TrialResult,
)
from kube_autotuner.trial_log import TrialLog, trial_schema

if TYPE_CHECKING:
    from pathlib import Path

    import pyarrow as pa


def _trial(iter_idx: int) -> TrialResult:
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


def test_schema_conformance(tmp_path: Path) -> None:
    """Every written file matches :func:`trial_schema` column-for-column."""
    path = tmp_path / "results"
    TrialLog.append(path, _trial(0))

    files = sorted(path.glob("*.parquet"))
    assert len(files) == 1
    table = pq.read_table(files[0])
    expected = cast("pa.Schema", trial_schema())
    assert set(table.schema.names) == set(expected.names)
    for name in expected.names:
        assert table.schema.field(name).type == expected.field(name).type


def test_zstd_compression(tmp_path: Path) -> None:
    """Every column is compressed with zstd."""
    path = tmp_path / "results"
    TrialLog.append(path, _trial(0))
    files = sorted(path.glob("*.parquet"))
    meta = pq.read_metadata(files[0])
    row_group = meta.row_group(0)
    codecs = {row_group.column(i).compression for i in range(row_group.num_columns)}
    assert codecs == {"ZSTD"}


def test_sequence_numbering_is_monotonic(tmp_path: Path) -> None:
    """Sequence prefixes strictly increase across successive appends."""
    path = tmp_path / "results"
    for i in range(5):
        TrialLog.append(path, _trial(i))
    seqs = [int(p.name.split("-", 1)[0]) for p in sorted(path.glob("*.parquet"))]
    assert seqs == [1, 2, 3, 4, 5]


def test_load_returns_sequence_order(
    tmp_path: Path,
) -> None:
    """``load`` returns trials sorted by sequence, not by mtime.

    Write trials out of chronological order by appending with
    decreasing mtimes — the load path must sort by filename's seq
    prefix, not by any filesystem metadata.
    """
    path = tmp_path / "results"
    trials = [_trial(i) for i in range(3)]
    for t in trials:
        TrialLog.append(path, t)

    files = sorted(path.glob("*.parquet"))
    # Touch files so mtime order is reversed from sequence order.
    now = 1_700_000_000.0
    for i, f in enumerate(files):
        os.utime(f, (now - i, now - i))

    loaded = TrialLog.load(path)
    assert [t.trial_id for t in loaded] == [t.trial_id for t in trials]


def test_payload_fidelity(tmp_path: Path) -> None:
    """Round-trip through append+load preserves the full TrialResult."""
    path = tmp_path / "results"
    original = _trial(0)
    TrialLog.append(path, original)
    loaded = TrialLog.load(path)
    assert len(loaded) == 1
    # Pydantic equality: frozenset fields like ``BenchmarkConfig.stages``
    # are order-independent, so comparing via ``==`` sidesteps the
    # unstable JSON ordering that ``model_dump(mode="json")`` produces.
    assert loaded[0] == original
