"""Tests for resume behaviour across mixed primary + verification trials."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import typer

from kube_autotuner import runs
from kube_autotuner.experiment import ExperimentConfig
from kube_autotuner.models import (
    BenchmarkConfig,
    BenchmarkResult,
    NodePair,
    ResumeMetadata,
    TrialResult,
)
from kube_autotuner.trial_log import TrialLog

if TYPE_CHECKING:
    from pathlib import Path


def _prior_trial(
    sysctl_value: int = 1048576,
    *,
    phase: str = "bayesian",
    parent_trial_id: str | None = None,
) -> TrialResult:
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
        phase=phase,  # ty: ignore[invalid-argument-type]
        parent_trial_id=parent_trial_id,
    )


def _optimize_exp(
    out: Path,
    *,
    n_trials: int = 4,
    verification_trials: int = 0,
    verification_top_k: int = 3,
) -> ExperimentConfig:
    return ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "benchmark": {"duration": 1, "iterations": 1},
        "optimize": {
            "n_trials": n_trials,
            "n_sobol": 2,
            "verification_trials": verification_trials,
            "verification_top_k": verification_top_k,
        },
        "output": str(out),
    })


def _write_metadata(
    out: Path,
    exp: ExperimentConfig,
    *,
    verification_trials: int | None = None,
    verification_top_k: int | None = None,
) -> None:
    assert exp.optimize is not None
    TrialLog.write_resume_metadata(
        out,
        ResumeMetadata(
            objectives=exp.objectives,
            param_space=exp.effective_param_space(),
            benchmark=exp.benchmark,
            n_sobol=exp.optimize.n_sobol,
            verification_trials=verification_trials,
            verification_top_k=verification_top_k,
        ),
    )


def _client_stub() -> MagicMock:
    client = MagicMock()
    client.get_node_zone.return_value = ""
    return client


@patch("kube_autotuner.optimizer.OptimizationLoop")
def test_resume_partial_verification_runs_remaining(
    mock_loop_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Primary budget met; one verification done, remainder queued."""
    out = tmp_path / "mixed"
    exp = _optimize_exp(
        out,
        n_trials=2,
        verification_trials=2,
        verification_top_k=1,
    )
    primary = _prior_trial(1048576, phase="bayesian")
    primary2 = _prior_trial(2097152, phase="bayesian")
    one_verification = _prior_trial(
        1048576,
        phase="verification",
        parent_trial_id=primary.trial_id,
    )
    for tr in (primary, primary2, one_verification):
        TrialLog.append(out, tr)
    _write_metadata(
        out,
        exp,
        verification_trials=2,
        verification_top_k=1,
    )

    loop = MagicMock()
    loop.run.return_value = [primary, primary2]
    loop.prior_count = 2
    loop.pareto_front.return_value = []
    loop._completed = [primary, primary2, one_verification]
    mock_loop_cls.return_value = loop

    ctx = runs.RunContext(
        exp=exp,
        client=_client_stub(),
        backend=MagicMock(),
        output=out,
    )
    runs.run_optimize(ctx)

    # run_verification should have been called with the already-done map.
    loop.run_verification.assert_called_once()
    kwargs = loop.run_verification.call_args.kwargs
    assert kwargs["already_done_by_parent"] == {primary.trial_id: 1}
    assert kwargs["top_k"] == 1
    assert kwargs["repeats"] == 2


def test_resume_verification_trials_drift_rejects(tmp_path: Path) -> None:
    out = tmp_path / "drift"
    exp_prior = _optimize_exp(
        out,
        n_trials=2,
        verification_trials=2,
        verification_top_k=1,
    )
    primary = _prior_trial(phase="bayesian")
    TrialLog.append(out, primary)
    _write_metadata(
        out,
        exp_prior,
        verification_trials=2,
        verification_top_k=1,
    )
    # Current run bumps verification_trials.
    exp_current = _optimize_exp(
        out,
        n_trials=2,
        verification_trials=5,
        verification_top_k=1,
    )
    ctx = runs.RunContext(
        exp=exp_current,
        client=_client_stub(),
        backend=MagicMock(),
        output=out,
    )
    with pytest.raises(typer.BadParameter, match="verification_trials"):
        runs.run_optimize(ctx)


def test_resume_verification_top_k_drift_rejects(tmp_path: Path) -> None:
    out = tmp_path / "drift"
    exp_prior = _optimize_exp(
        out,
        n_trials=2,
        verification_trials=2,
        verification_top_k=1,
    )
    TrialLog.append(out, _prior_trial(phase="bayesian"))
    _write_metadata(
        out,
        exp_prior,
        verification_trials=2,
        verification_top_k=1,
    )
    exp_current = _optimize_exp(
        out,
        n_trials=2,
        verification_trials=2,
        verification_top_k=3,
    )
    ctx = runs.RunContext(
        exp=exp_current,
        client=_client_stub(),
        backend=MagicMock(),
        output=out,
    )
    with pytest.raises(typer.BadParameter, match="verification_top_k"):
        runs.run_optimize(ctx)
