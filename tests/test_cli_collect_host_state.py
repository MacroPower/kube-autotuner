"""Tests for ``benchmark.collectHostState`` plumbing through the CLI."""

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


def _baseline_yaml(out: Path, *, collect_host_state: bool | None) -> str:
    body: dict[str, object] = {
        "nodes": {"sources": ["a"], "target": "b"},
        "benchmark": {"duration": 1, "iterations": 1},
        "output": str(out),
    }
    if collect_host_state is not None:
        body["benchmark"] = dict(body["benchmark"])  # type: ignore[arg-type]
        body["benchmark"]["collectHostState"] = collect_host_state  # type: ignore[index]
    return yaml.safe_dump(body)


def _trial_yaml(out: Path, *, collect_host_state: bool) -> str:
    body = {
        "nodes": {"sources": ["a"], "target": "b"},
        "benchmark": {
            "duration": 1,
            "iterations": 1,
            "collectHostState": collect_host_state,
        },
        "trial": {"sysctls": {"net.core.rmem_max": "16777216"}},
        "output": str(out),
    }
    return yaml.safe_dump(body)


def _optimize_yaml(out: Path, *, collect_host_state: bool) -> str:
    body = {
        "nodes": {"sources": ["a"], "target": "b"},
        "benchmark": {
            "duration": 1,
            "iterations": 1,
            "collectHostState": collect_host_state,
        },
        "optimize": {"nTrials": 1, "nSobol": 1},
        "output": str(out),
    }
    return yaml.safe_dump(body)


@pytest.mark.parametrize(
    ("subcommand", "yaml_factory", "run_target"),
    [
        ("baseline", _baseline_yaml, "kube_autotuner.cli.runs.run_baseline"),
        ("trial", _trial_yaml, "kube_autotuner.cli.runs.run_trial"),
        ("optimize", _optimize_yaml, "kube_autotuner.cli.runs.run_optimize"),
    ],
)
def test_collect_host_state_yaml_threads_into_context(
    tmp_path: Path,
    subcommand: str,
    yaml_factory,
    run_target: str,
) -> None:
    """``benchmark.collectHostState: true`` reaches ``RunContext``."""
    out = tmp_path / "r.jsonl"
    config = tmp_path / "exp.yaml"
    if subcommand == "baseline":
        config.write_text(yaml_factory(out, collect_host_state=True))
    else:
        config.write_text(yaml_factory(out, collect_host_state=True))
    with (
        patch("kube_autotuner.cli.K8sClient") as client_cls,
        patch("kube_autotuner.cli._resolve_backend") as resolve,
        patch(run_target) as run_fn,
        patch(
            "kube_autotuner.cli.ExperimentConfig.preflight",
            return_value=[],
        ),
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = _fake_backend(tmp_path)

        result = runner.invoke(app, [subcommand, str(config)])
    assert result.exit_code == 0, result.output
    ctx = run_fn.call_args.args[0]
    assert ctx.collect_host_state is True


def test_baseline_collect_host_state_default_is_false(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    config = tmp_path / "exp.yaml"
    config.write_text(_baseline_yaml(out, collect_host_state=None))
    with (
        patch("kube_autotuner.cli.K8sClient") as client_cls,
        patch("kube_autotuner.cli._resolve_backend") as resolve,
        patch("kube_autotuner.cli.runs.run_baseline") as run_baseline,
        patch(
            "kube_autotuner.cli.ExperimentConfig.preflight",
            return_value=[],
        ),
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = _fake_backend(tmp_path)

        result = runner.invoke(app, ["baseline", str(config)])
    assert result.exit_code == 0, result.output
    ctx = run_baseline.call_args.args[0]
    assert ctx.collect_host_state is False


def test_baseline_explicit_false_threads_through(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    config = tmp_path / "exp.yaml"
    config.write_text(_baseline_yaml(out, collect_host_state=False))
    with (
        patch("kube_autotuner.cli.K8sClient") as client_cls,
        patch("kube_autotuner.cli._resolve_backend") as resolve,
        patch("kube_autotuner.cli.runs.run_baseline") as run_baseline,
        patch(
            "kube_autotuner.cli.ExperimentConfig.preflight",
            return_value=[],
        ),
    ):
        client_cls.return_value = MagicMock()
        resolve.return_value = _fake_backend(tmp_path)

        result = runner.invoke(app, ["baseline", str(config)])
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
