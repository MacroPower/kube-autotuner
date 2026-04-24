"""Tests for ``--collect-host-state`` plumbing through the four CLI commands."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.util import find_spec
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner
import yaml

from kube_autotuner.cli import app
from kube_autotuner.sysctl.fake import FakeSysctlBackend

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _fake_backend(tmp_path: Path) -> FakeSysctlBackend:
    return FakeSysctlBackend("node-a", tmp_path / "sysctl_state.json")


def _common_patches(tmp_path: Path):
    """Return (patch pair, fake backend) for the CLI command boundary.

    Returns:
        A ``((client_patch, resolve_patch), fake_backend)`` tuple: two
        uncommitted ``unittest.mock.patch`` objects that the caller
        enters in a ``with`` block, plus a pre-built
        :class:`FakeSysctlBackend` the ``_resolve_backend`` patch can
        return.
    """
    return (
        patch("kube_autotuner.cli.K8sClient"),
        patch("kube_autotuner.cli._resolve_backend"),
    ), _fake_backend(tmp_path)


def test_baseline_flag_threaded_into_context(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    (client_patch, resolve_patch), fake = _common_patches(tmp_path)
    with (
        client_patch as client_cls,
        resolve_patch as resolve,
        patch("kube_autotuner.cli.runs.run_baseline") as run_baseline,
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = fake

        result = runner.invoke(
            app,
            [
                "baseline",
                "--source",
                "a",
                "--target",
                "b",
                "--duration",
                "1",
                "--iterations",
                "1",
                "--output",
                str(out),
                "--collect-host-state",
            ],
        )
    assert result.exit_code == 0, result.output
    ctx = run_baseline.call_args.args[0]
    assert ctx.collect_host_state is True


def test_trial_flag_threaded_into_context(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    (client_patch, resolve_patch), fake = _common_patches(tmp_path)
    with (
        client_patch as client_cls,
        resolve_patch as resolve,
        patch("kube_autotuner.cli.runs.run_trial") as run_trial,
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = fake

        result = runner.invoke(
            app,
            [
                "trial",
                "--source",
                "a",
                "--target",
                "b",
                "-p",
                "net.core.rmem_max=16777216",
                "--duration",
                "1",
                "--iterations",
                "1",
                "--output",
                str(out),
                "-H",
            ],
        )
    assert result.exit_code == 0, result.output
    ctx = run_trial.call_args.args[0]
    assert ctx.collect_host_state is True


def test_optimize_flag_threaded_into_context(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    (client_patch, resolve_patch), fake = _common_patches(tmp_path)
    with (
        client_patch as client_cls,
        resolve_patch as resolve,
        patch("kube_autotuner.cli.runs.run_optimize") as run_optimize,
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = fake

        result = runner.invoke(
            app,
            [
                "optimize",
                "--source",
                "a",
                "--target",
                "b",
                "--duration",
                "1",
                "--iterations",
                "1",
                "--output",
                str(out),
                "--n-trials",
                "1",
                "--n-sobol",
                "1",
                "--collect-host-state",
            ],
        )
    assert result.exit_code == 0, result.output
    ctx = run_optimize.call_args.args[0]
    assert ctx.collect_host_state is True


def test_run_yaml_flag_threaded_into_context(tmp_path: Path) -> None:
    """The YAML-driven ``run`` command also accepts the flag."""
    out = tmp_path / "r.jsonl"
    config = tmp_path / "exp.yaml"
    config.write_text(
        yaml.safe_dump({
            "mode": "baseline",
            "nodes": {"sources": ["a"], "target": "b"},
            "benchmark": {"duration": 1, "iterations": 1},
            "output": str(out),
        })
    )
    client_mock = MagicMock()
    client_mock.preflight = MagicMock(return_value=[])
    (client_patch, resolve_patch), fake = _common_patches(tmp_path)
    with (
        client_patch as client_cls,
        resolve_patch as resolve,
        patch("kube_autotuner.cli.runs.run_baseline") as run_baseline,
    ):
        client_cls.return_value = client_mock
        resolve.return_value = fake

        result = runner.invoke(
            app,
            [
                "run",
                "-c",
                str(config),
                "--collect-host-state",
            ],
        )
    assert result.exit_code == 0, result.output
    ctx = run_baseline.call_args.args[0]
    assert ctx.collect_host_state is True


def test_baseline_default_is_false(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    (client_patch, resolve_patch), fake = _common_patches(tmp_path)
    with (
        client_patch as client_cls,
        resolve_patch as resolve,
        patch("kube_autotuner.cli.runs.run_baseline") as run_baseline,
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = fake

        result = runner.invoke(
            app,
            [
                "baseline",
                "--source",
                "a",
                "--target",
                "b",
                "--duration",
                "1",
                "--iterations",
                "1",
                "--output",
                str(out),
            ],
        )
    assert result.exit_code == 0, result.output
    ctx = run_baseline.call_args.args[0]
    assert ctx.collect_host_state is False


# --- analyze end-to-end rendering ----------------------------------------

# These tests exercise the path from a JSONL log carrying
# ``host_state_snapshots`` through the ``analyze`` CLI to an HTML report
# that surfaces the host-state section. They depend on the optional
# ``analysis`` dependency group.


def _trial_with_snapshots_dict(
    *,
    trial_id: str,
    include_snapshots: bool,
    bps: float,
) -> dict:
    """Build a JSON-ready dict mirroring ``TrialResult.model_dump``.

    Building the dict directly (rather than ``TrialResult.model_dump``)
    keeps the test fixture small and independent of model defaults;
    only the fields the analyze pipeline reads need to be populated.

    Returns:
        A dict with the ``TrialResult`` schema, suitable for writing
        to a JSONL log and loading via ``TrialLog.load``.
    """
    now = datetime.now(UTC).isoformat()
    snapshots: list[dict] = []
    if include_snapshots:
        snapshots = [
            {
                "node": "node-a",
                "iteration": None,
                "phase": "baseline",
                "timestamp": now,
                "metrics": {"conntrack_count": 10, "sockstat_tcp_inuse": 3},
                "errors": [],
            },
            {
                "node": "node-a",
                "iteration": 0,
                "phase": "post-flush",
                "timestamp": now,
                "metrics": {"conntrack_count": 12, "sockstat_tcp_inuse": 4},
                "errors": [],
            },
            {
                "node": "node-a",
                "iteration": 0,
                "phase": "post-iteration",
                "timestamp": now,
                "metrics": {"conntrack_count": 18, "sockstat_tcp_inuse": 7},
                "errors": [],
            },
        ]
    return {
        "trial_id": trial_id,
        "node_pair": {
            "source": "a",
            "target": "b",
            "hardware_class": "10g",
            "source_zone": "",
            "target_zone": "",
        },
        "sysctl_values": {"net.core.rmem_max": 212992},
        "sysctl_hash": "",
        "topology": "unknown",
        "config": {"duration": 1, "iterations": 1},
        "results": [
            {
                "timestamp": now,
                "mode": "tcp",
                "bits_per_second": bps,
                "retransmits": 5,
                "bytes_sent": 1_000_000_000,
                "iteration": 0,
                "source": "a",
                "target": "b",
            },
        ],
        "latency_results": [],
        "host_state_snapshots": snapshots,
        "created_at": now,
        "phase": None,
        "parent_trial_id": None,
    }


def _write_jsonl(path: Path, trials: list[dict]) -> None:
    import json  # noqa: PLC0415

    with path.open("w", encoding="utf-8") as f:
        for trial in trials:
            f.write(json.dumps(trial) + "\n")


def _analysis_available() -> bool:
    return find_spec("pandas") is not None and find_spec("sklearn") is not None


@pytest.mark.skipif(
    not _analysis_available(),
    reason="analyze CLI requires pandas and scikit-learn (group: analysis)",
)
def test_analyze_renders_host_state_section_when_snapshots_present(
    tmp_path: Path,
) -> None:
    jsonl = tmp_path / "trials.jsonl"
    _write_jsonl(
        jsonl,
        [
            _trial_with_snapshots_dict(
                trial_id=f"t{i}",
                include_snapshots=True,
                bps=1e9 + i * 1e8,
            )
            for i in range(3)
        ],
    )
    output_dir = tmp_path / "report"

    result = runner.invoke(
        app,
        [
            "analyze",
            "--input",
            str(jsonl),
            "--output-dir",
            str(output_dir),
            "--hardware-class",
            "10g",
        ],
    )
    assert result.exit_code == 0, result.output
    index = output_dir / "index.html"
    assert index.exists()
    html_text = index.read_text()
    assert "<h3>Host state</h3>" in html_text
    assert 'id="host-state-chart-10g"' in html_text
    # The metric union is passed through to the <select> options.
    assert 'value="conntrack_count"' in html_text
    assert 'value="sockstat_tcp_inuse"' in html_text


@pytest.mark.skipif(
    not _analysis_available(),
    reason="analyze CLI requires pandas and scikit-learn (group: analysis)",
)
def test_analyze_omits_host_state_section_without_snapshots(
    tmp_path: Path,
) -> None:
    jsonl = tmp_path / "trials.jsonl"
    _write_jsonl(
        jsonl,
        [
            _trial_with_snapshots_dict(
                trial_id=f"t{i}",
                include_snapshots=False,
                bps=1e9 + i * 1e8,
            )
            for i in range(3)
        ],
    )
    output_dir = tmp_path / "report"

    result = runner.invoke(
        app,
        [
            "analyze",
            "--input",
            str(jsonl),
            "--output-dir",
            str(output_dir),
            "--hardware-class",
            "10g",
        ],
    )
    assert result.exit_code == 0, result.output
    html_text = (output_dir / "index.html").read_text()
    assert "<h3>Host state</h3>" not in html_text
    assert 'id="host-state-chart-10g"' not in html_text
