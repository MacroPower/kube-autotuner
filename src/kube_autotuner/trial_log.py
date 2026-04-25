"""Parquet-dataset persistence for :class:`TrialResult` records.

Each ``--output`` path is a directory containing one zstd-compressed
Parquet file per trial plus a ``_meta.json`` sidecar:

.. code-block:: text

    results/                              (--output)
      _meta.json                          ResumeMetadata sidecar
      00000001-abc123def456.parquet       one TrialResult per file
      00000002-789abcdef012.parquet
      .tmp-00000003-....parquet           transient; only during a write

Writes are atomic: a tmp file is opened ``O_CREAT | O_EXCL``, written,
fsync'd, then hard-linked into place with the publish step removing the
tmp. Crash recovery is stateless — any ``.tmp-*.parquet`` observed on
``load`` or ``append`` is swept unconditionally because publication
removes the tmp in a bounded window after the final exists.

``TrialLog.append`` is single-writer. The invariant is enforced by the
per-node lease acquired before any trial runs; two concurrent writers
against the same directory would fail loudly on the ``O_EXCL`` tmp
creation after the sequence scan.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

from pydantic import ValidationError
import typer

from kube_autotuner.models import (
    ResumeMetadata,
    TrialResult,
    _ensure_resume_metadata_built,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_TMP_PREFIX = ".tmp-"
_TMP_GLOB = f"{_TMP_PREFIX}*.parquet"
_PARQUET_GLOB = "*.parquet"
_META_NAME = "_meta.json"
_FAILURES_DIR_NAME = "failures"
_MAX_TMP_COLLISION_RETRIES = 3


def _validate_output_directory(output: Path) -> None:
    """Refuse output paths that are not trial-log directories.

    A missing path passes (the first ``append`` will create it). An
    existing regular file fails. An existing non-empty directory that
    does not already look like a trial-log dataset fails — the shape
    markers are a ``_meta.json`` sidecar, a ``*.parquet`` file, or a
    ``failures/`` subdirectory.

    Args:
        output: Candidate ``--output`` path.

    Raises:
        typer.BadParameter: ``output`` exists and is not a directory,
            or is a directory whose contents do not match the
            trial-log shape.
    """
    if not output.exists():
        return
    if not output.is_dir():
        msg = f"--output exists and is not a directory: {output}"
        raise typer.BadParameter(msg)
    entries = list(output.iterdir())
    if not entries:
        return
    has_marker = (
        (output / _META_NAME).exists()
        or any(output.glob(_PARQUET_GLOB))
        or (output / _FAILURES_DIR_NAME).is_dir()
    )
    if not has_marker:
        msg = (
            "--output directory is non-empty and not a trial-log "
            f"directory: {output}; pick a fresh path or clear it."
        )
        raise typer.BadParameter(msg)


def _sweep_stale_tmp(path: Path) -> None:
    """Remove any stale ``.tmp-*.parquet`` files in ``path``.

    Stateless by design: every ``.tmp-*`` is unconditionally stale
    because publication removes the tmp in a bounded window after the
    final exists, so anything observed here is from a crashed writer.

    Args:
        path: Trial-log directory to sweep. Missing directories are
            skipped silently.
    """
    if not path.is_dir():
        return
    for tmp in path.glob(_TMP_GLOB):
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            logger.warning("could not sweep stale tmp file %s", tmp, exc_info=True)


def _sweep_and_next_sequence(path: Path) -> int:
    """Sweep stale tmp files and return the next sequence number.

    Single directory scan: each entry is routed either to the tmp
    sweep (``.tmp-*.parquet``) or the max-seq computation
    (``*.parquet`` with the ``<8-digit>-`` prefix). Non-conforming
    filenames are ignored.

    Args:
        path: Trial-log directory.

    Returns:
        The next sequence number; ``1`` when no final parquet exists.
    """
    max_seq = 0
    for entry in path.iterdir():
        name = entry.name
        if name.startswith(_TMP_PREFIX) and name.endswith(".parquet"):
            try:
                entry.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "could not sweep stale tmp file %s", entry, exc_info=True
                )
            continue
        if not name.endswith(".parquet"):
            continue
        prefix = entry.stem.split("-", 1)[0]
        if len(prefix) == 8 and prefix.isdigit():  # noqa: PLR2004 - width of zero-padded seq
            max_seq = max(max_seq, int(prefix))
    return max_seq + 1


def _publish(tmp_path: Path, final_path: Path) -> None:
    """Atomically publish ``tmp_path`` as ``final_path``.

    Uses ``os.link`` + ``os.unlink`` so a pre-existing ``final_path``
    surfaces as :class:`FileExistsError` rather than being silently
    clobbered. Falls back to :meth:`Path.rename` on filesystems that do
    not support hard links, preceded by an explicit existence check.

    Args:
        tmp_path: Source tmp file created with ``O_EXCL``.
        final_path: Target path for the published trial.

    Raises:
        FileExistsError: ``final_path`` already exists.
    """
    try:
        os.link(tmp_path, final_path)
    except OSError as link_err:
        if final_path.exists():
            raise FileExistsError(str(final_path)) from link_err
        tmp_path.rename(final_path)
        return
    tmp_path.unlink(missing_ok=True)


class TrialLog:
    """Parquet-dataset persistence for :class:`TrialResult` records.

    Each trial is written as a single-row Parquet file with a flat set
    of projection columns for cheap filtering plus a ``payload`` column
    carrying :meth:`TrialResult.model_dump_json` for lossless
    reconstruction. The dataset directory also holds a ``_meta.json``
    :class:`ResumeMetadata` sidecar.

    API surface matches the previous append-only implementation so
    callers in :mod:`kube_autotuner.runs`,
    :mod:`kube_autotuner.optimizer`, and :mod:`kube_autotuner.cli` are
    unaffected by the storage change.
    """

    @staticmethod
    def append(path: Path, trial: TrialResult) -> None:
        """Append ``trial`` as a new Parquet file under ``path``.

        The write sequence is: sweep stale tmp files, compute the next
        sequence number, open a ``.tmp-<seq>-<trial_id>.parquet`` with
        ``O_EXCL``, write the single-row table, ``fsync``, and publish
        via :func:`os.link` + :func:`os.unlink`.

        Args:
            path: Target trial-log directory. Created if missing.
            trial: The trial record to persist.

        Raises:
            FileExistsError: A tmp-file collision could not be
                resolved within the retry budget (stale or concurrent
                writer).
        """
        import pyarrow as pa  # noqa: PLC0415 - keep CLI startup fast
        import pyarrow.parquet as pq  # noqa: PLC0415

        path.mkdir(parents=True, exist_ok=True)
        seq = _sweep_and_next_sequence(path)
        for attempt in range(_MAX_TMP_COLLISION_RETRIES):
            tmp_name = f"{_TMP_PREFIX}{seq:08d}-{trial.trial_id}.parquet"
            tmp_path = path / tmp_name
            try:
                fd = os.open(
                    tmp_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                if attempt == _MAX_TMP_COLLISION_RETRIES - 1:
                    raise
                seq += 1
                continue
            os.close(fd)
            break
        else:  # pragma: no cover - loop always breaks or raises
            msg = (
                "TrialLog.append: tmp-file collision not resolved after "
                f"{_MAX_TMP_COLLISION_RETRIES} retries at {path}; "
                "stale writer or concurrent writer"
            )
            raise FileExistsError(msg)

        try:
            table = pa.Table.from_pylist(
                [_row_from_trial(trial)],
                schema=trial_schema(),
            )
            pq.write_table(
                table,
                tmp_path,
                compression="zstd",
                compression_level=7,
            )

            fsync_fd = os.open(tmp_path, os.O_RDONLY)
            try:
                os.fsync(fsync_fd)
            finally:
                os.close(fsync_fd)

            final_name = f"{seq:08d}-{trial.trial_id}.parquet"
            _publish(tmp_path, path / final_name)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def load(path: Path) -> list[TrialResult]:
        """Load every :class:`TrialResult` from ``path`` in sequence order.

        A truncated or otherwise corrupt trailing parquet file is
        tolerated at trial granularity and logged at ``WARNING``.
        Mid-sequence failures re-raise as real corruption.

        Args:
            path: Source trial-log directory. A missing or empty
                directory returns an empty list.

        Returns:
            The decoded trials, in sequence order.

        Raises:
            pyarrow.ArrowException: A non-final parquet file is
                unreadable (real corruption).
            json.JSONDecodeError: A non-final parquet's payload
                column fails JSON decoding.
            pydantic.ValidationError: A non-final payload decoded but
                failed :class:`TrialResult` validation.
        """
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        if not path.exists() or not path.is_dir():
            return []
        _sweep_stale_tmp(path)
        files = sorted(path.glob(_PARQUET_GLOB))
        if not files:
            return []

        trials: list[TrialResult] = []
        last_idx = len(files) - 1
        for idx, file in enumerate(files):
            try:
                table = pq.read_table(file, columns=["payload"])
                payload = table.column("payload")[0].as_py()
                trials.append(TrialResult.model_validate_json(payload))
            except pa.ArrowException, json.JSONDecodeError, ValidationError:
                if idx == last_idx:
                    logger.warning(
                        "dropping truncated final trial file %s (likely from "
                        "an interrupted write)",
                        file,
                    )
                    break
                raise
        return trials

    @staticmethod
    def _metadata_path(path: Path) -> Path:
        """Return the sidecar metadata path for a trial-log directory."""
        return path / _META_NAME

    @staticmethod
    def write_resume_metadata(path: Path, meta: ResumeMetadata) -> None:
        """Persist ``meta`` as a sidecar inside the trial-log directory.

        The sidecar lives at ``<path>/_meta.json``. Writes are idempotent
        for identical content; drift detection happens at resume time
        in :func:`kube_autotuner.runs._check_compatibility`.

        Args:
            path: Trial-log directory; created if missing.
            meta: :class:`ResumeMetadata` describing the run that is
                about to write into ``path``.
        """
        _ensure_resume_metadata_built()
        path.mkdir(parents=True, exist_ok=True)
        meta_path = TrialLog._metadata_path(path)
        meta_path.write_text(meta.model_dump_json() + "\n", encoding="utf-8")

    @staticmethod
    def load_resume_metadata(path: Path) -> ResumeMetadata | None:
        """Return the sidecar :class:`ResumeMetadata` for ``path``.

        Args:
            path: Trial-log directory whose ``_meta.json`` sidecar is
                consulted.

        Returns:
            The parsed :class:`ResumeMetadata`, or ``None`` when no
            sidecar exists. A malformed sidecar raises
            :class:`pydantic.ValidationError`.
        """
        _ensure_resume_metadata_built()
        meta_path = TrialLog._metadata_path(path)
        if not meta_path.exists():
            return None
        return ResumeMetadata.model_validate_json(
            meta_path.read_text(encoding="utf-8"),
        )


def _row_from_trial(trial: TrialResult) -> dict[str, object]:
    """Project a :class:`TrialResult` into the Parquet row schema.

    Args:
        trial: The trial record to serialize.

    Returns:
        A dict whose keys match the :func:`trial_schema` columns
        exactly.
    """
    return {
        "trial_id": trial.trial_id,
        "created_at": trial.created_at,
        "phase": trial.phase,
        "parent_trial_id": trial.parent_trial_id,
        "refinement_round": trial.refinement_round,
        "hardware_class": trial.node_pair.hardware_class,
        "topology": trial.topology,
        "sysctl_hash": trial.sysctl_hash,
        "kernel_version": trial.kernel_version,
        "payload": trial.model_dump_json(),
    }


def _build_schema() -> object:
    """Build the Parquet schema for :class:`TrialResult` rows.

    Factored into a function so pyarrow stays lazy-imported at module
    level. :func:`trial_schema` caches and returns the result.

    Returns:
        A ``pyarrow.Schema`` describing the flat projection columns
        plus the ``payload`` string column.
    """
    import pyarrow as pa  # noqa: PLC0415

    return pa.schema([
        pa.field("trial_id", pa.string(), nullable=False),
        pa.field(
            "created_at",
            pa.timestamp("us", tz="UTC"),
            nullable=False,
        ),
        pa.field("phase", pa.string(), nullable=True),
        pa.field("parent_trial_id", pa.string(), nullable=True),
        pa.field("refinement_round", pa.int64(), nullable=True),
        pa.field("hardware_class", pa.string(), nullable=False),
        pa.field("topology", pa.string(), nullable=False),
        pa.field("sysctl_hash", pa.string(), nullable=False),
        pa.field("kernel_version", pa.string(), nullable=False),
        pa.field("payload", pa.string(), nullable=False),
    ])


_TRIAL_SCHEMA_CACHE: object | None = None


def trial_schema() -> object:
    """Return the cached :class:`pyarrow.Schema` for trial rows.

    Keeps pyarrow lazy-imported: the schema is constructed on first
    use rather than at module import.

    Returns:
        A ``pyarrow.Schema`` describing the trial-log Parquet schema.
    """
    global _TRIAL_SCHEMA_CACHE  # noqa: PLW0603
    if _TRIAL_SCHEMA_CACHE is None:
        _TRIAL_SCHEMA_CACHE = _build_schema()
    return _TRIAL_SCHEMA_CACHE
