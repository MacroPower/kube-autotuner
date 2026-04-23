"""Tests for compat tolerance when the sidecar pre-dates verification."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from kube_autotuner import runs
from kube_autotuner.experiment import ExperimentConfig
from kube_autotuner.models import (
    BenchmarkConfig,
    BenchmarkResult,
    NodePair,
    ResumeMetadata,
    TrialLog,
    TrialResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _prior_trial(sysctl_value: int = 1048576) -> TrialResult:
    from datetime import UTC, datetime  # noqa: PLC0415

    return TrialResult(
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values={"net.core.rmem_max": sysctl_value},
        config=BenchmarkConfig(duration=1, iterations=1),
        results=[
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=9e9,
                retransmits=5,
                bytes_sent=10**9,
            ),
        ],
        phase="bayesian",
    )


def _client_stub() -> MagicMock:
    client = MagicMock()
    client.get_node_zone.return_value = ""
    return client


@patch("kube_autotuner.optimizer.OptimizationLoop")
def test_legacy_sidecar_enables_verification_with_info_log(
    mock_loop_cls: MagicMock,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    out = tmp_path / "legacy.jsonl"
    exp = ExperimentConfig.model_validate({
        "mode": "optimize",
        "nodes": {"sources": ["a"], "target": "b"},
        "benchmark": {"duration": 1, "iterations": 1},
        "optimize": {
            "n_trials": 2,
            "n_sobol": 1,
            "verification_trials": 2,
            "verification_top_k": 1,
        },
        "output": str(out),
    })
    # Legacy sidecar: both verification_* keys missing / None.
    TrialLog.append(out, _prior_trial())
    TrialLog.append(out, _prior_trial(2097152))
    assert exp.optimize is not None
    TrialLog.write_resume_metadata(
        out,
        ResumeMetadata(
            objectives=exp.objectives,
            param_space=exp.effective_param_space(),
            benchmark=exp.benchmark,
            n_sobol=exp.optimize.n_sobol,
            verification_trials=None,
            verification_top_k=None,
        ),
    )

    loop = MagicMock()
    loop.run.return_value = []
    loop.prior_count = 2
    loop.pareto_front.return_value = []
    loop._completed = [_prior_trial(), _prior_trial(2097152)]
    mock_loop_cls.return_value = loop

    ctx = runs.RunContext(
        exp=exp,
        client=_client_stub(),
        backend=MagicMock(),
        output=out,
    )
    with caplog.at_level(logging.INFO, logger="kube_autotuner.runs"):
        runs.run_optimize(ctx)

    assert any(
        "prior sidecar has no verification record" in rec.message
        for rec in caplog.records
    )
    # Sidecar refreshed with the current verification_* fields.
    refreshed = TrialLog.load_resume_metadata(out)
    assert refreshed is not None
    assert refreshed.verification_trials == 2
    assert refreshed.verification_top_k == 1
