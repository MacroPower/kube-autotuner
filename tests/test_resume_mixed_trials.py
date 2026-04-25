"""Tests for resume behaviour across mixed primary + refinement trials."""

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
    refinement_round: int | None = None,
) -> TrialResult:
    return TrialResult(
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values={"net.core.rmem_max": sysctl_value},
        config=BenchmarkConfig(iterations=1),
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
        refinement_round=refinement_round,
    )


def _optimize_exp(
    out: Path,
    *,
    n_trials: int = 4,
    refinement_rounds: int = 0,
    refinement_top_k: int = 3,
) -> ExperimentConfig:
    return ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "benchmark": {"iterations": 1},
        "iperf": {"duration": 1},
        "optimize": {
            "n_trials": n_trials,
            "n_sobol": 2,
            "refinement_rounds": refinement_rounds,
            "refinement_top_k": refinement_top_k,
        },
        "output": str(out),
    })


def _write_metadata(
    out: Path,
    exp: ExperimentConfig,
    *,
    refinement_rounds: int | None = None,
    refinement_top_k: int | None = None,
) -> None:
    assert exp.optimize is not None
    TrialLog.write_resume_metadata(
        out,
        ResumeMetadata(
            objectives=exp.objectives,
            param_space=exp.effective_param_space(),
            benchmark=exp.benchmark,
            iperf=exp.iperf,
            fortio=exp.fortio,
            n_sobol=exp.optimize.n_sobol,
            refinement_rounds=refinement_rounds,
            refinement_top_k=refinement_top_k,
        ),
    )


def _client_stub() -> MagicMock:
    client = MagicMock()
    client.get_node_zone.return_value = ""
    return client


@patch("kube_autotuner.optimizer.OptimizationLoop")
def test_resume_partial_refinement_runs_remaining(
    mock_loop_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Primary budget met; round 1 done for one parent, remainder queued."""
    out = tmp_path / "mixed"
    exp = _optimize_exp(
        out,
        n_trials=2,
        refinement_rounds=2,
        refinement_top_k=1,
    )
    primary = _prior_trial(1048576, phase="bayesian")
    primary2 = _prior_trial(2097152, phase="bayesian")
    round1_sample = _prior_trial(
        1048576,
        phase="refinement",
        parent_trial_id=primary.trial_id,
        refinement_round=1,
    )
    for tr in (primary, primary2, round1_sample):
        TrialLog.append(out, tr)
    _write_metadata(
        out,
        exp,
        refinement_rounds=2,
        refinement_top_k=1,
    )

    loop = MagicMock()
    loop.run.return_value = [primary, primary2]
    loop.prior_count = 2
    loop.pareto_front.return_value = []
    loop._completed = [primary, primary2, round1_sample]
    mock_loop_cls.return_value = loop

    ctx = runs.RunContext(
        exp=exp,
        client=_client_stub(),
        backend=MagicMock(),
        output=out,
    )
    runs.run_optimize(ctx)

    # run_refinement should have been called with the per-round done map.
    loop.run_refinement.assert_called_once()
    kwargs = loop.run_refinement.call_args.kwargs
    assert kwargs["completed_by_round"] == {1: {primary.trial_id}}
    assert kwargs["top_k"] == 1
    assert kwargs["rounds"] == 2


def test_resume_refinement_rounds_drift_rejects(tmp_path: Path) -> None:
    out = tmp_path / "drift"
    exp_prior = _optimize_exp(
        out,
        n_trials=2,
        refinement_rounds=2,
        refinement_top_k=1,
    )
    primary = _prior_trial(phase="bayesian")
    TrialLog.append(out, primary)
    _write_metadata(
        out,
        exp_prior,
        refinement_rounds=2,
        refinement_top_k=1,
    )
    # Current run bumps refinement_rounds.
    exp_current = _optimize_exp(
        out,
        n_trials=2,
        refinement_rounds=5,
        refinement_top_k=1,
    )
    ctx = runs.RunContext(
        exp=exp_current,
        client=_client_stub(),
        backend=MagicMock(),
        output=out,
    )
    with pytest.raises(typer.BadParameter, match="refinement_rounds"):
        runs.run_optimize(ctx)


def test_resume_refinement_top_k_drift_rejects(tmp_path: Path) -> None:
    out = tmp_path / "drift"
    exp_prior = _optimize_exp(
        out,
        n_trials=2,
        refinement_rounds=2,
        refinement_top_k=1,
    )
    TrialLog.append(out, _prior_trial(phase="bayesian"))
    _write_metadata(
        out,
        exp_prior,
        refinement_rounds=2,
        refinement_top_k=1,
    )
    exp_current = _optimize_exp(
        out,
        n_trials=2,
        refinement_rounds=2,
        refinement_top_k=3,
    )
    ctx = runs.RunContext(
        exp=exp_current,
        client=_client_stub(),
        backend=MagicMock(),
        output=out,
    )
    with pytest.raises(typer.BadParameter, match="refinement_top_k"):
        runs.run_optimize(ctx)
