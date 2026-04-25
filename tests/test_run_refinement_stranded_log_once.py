"""A parent dropped from top-K logs INFO once, not repeatedly."""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

pytest.importorskip("ax")

if TYPE_CHECKING:
    from unittest.mock import MagicMock

from kube_autotuner.experiment import ObjectivesSection
from kube_autotuner.models import (
    BenchmarkConfig,
    BenchmarkResult,
    NodePair,
    TrialResult,
)
from kube_autotuner.optimizer import OptimizationLoop
from kube_autotuner.sysctl.params import PARAM_SPACE


def _seed_primary(
    loop: OptimizationLoop,
    *,
    trial_id: str,
    bps: float,
) -> TrialResult:
    tr = TrialResult(
        node_pair=loop.node_pair,
        sysctl_values={"net.core.rmem_max": int(bps)},
        config=BenchmarkConfig(),
        results=[
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=bps,
                bytes_sent=int(bps),
                retransmits=1,
                iteration=0,
                client_node="a",
            ),
        ],
        phase="bayesian",
    )
    tr.trial_id = trial_id
    loop._completed.append(tr)
    return tr


def _refine_factory(
    by_parent: dict[str, list[float]],
) -> object:
    counters: dict[str, int] = {}

    def fake_evaluate(
        self: OptimizationLoop,
        parameterization: dict[str, str],  # noqa: ARG001 - shape match
        *,
        phase: str,
        parent_trial_id: str | None,
        refinement_round: int | None = None,
    ) -> tuple[TrialResult, dict[str, tuple[float, float]]]:
        assert parent_trial_id is not None
        idx = counters.get(parent_trial_id, 0)
        counters[parent_trial_id] = idx + 1
        bps = by_parent[parent_trial_id][idx]
        tr = TrialResult(
            node_pair=self.node_pair,
            sysctl_values={"net.core.rmem_max": int(bps)},
            config=BenchmarkConfig(),
            results=[
                BenchmarkResult(
                    timestamp=datetime.now(UTC),
                    mode="tcp",
                    bits_per_second=bps,
                    bytes_sent=int(bps),
                    retransmits=1,
                    iteration=0,
                    client_node="a",
                ),
            ],
            phase=phase,  # ty: ignore[invalid-argument-type]
            parent_trial_id=parent_trial_id,
            refinement_round=refinement_round,
        )
        self._completed.append(tr)
        from kube_autotuner.optimizer import _compute_metrics  # noqa: PLC0415, PLC2701

        return tr, _compute_metrics(tr)

    return fake_evaluate


@patch("kube_autotuner.optimizer.NodeLease")
@patch("kube_autotuner.optimizer.BenchmarkRunner")
@patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
def test_dropped_parent_logs_once(
    mock_setter_cls: MagicMock,  # noqa: ARG001 - patched out
    mock_runner_cls: MagicMock,  # noqa: ARG001 - patched out
    mock_lease_cls: MagicMock,  # noqa: ARG001 - patched out
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with patch.object(
        OptimizationLoop,
        "_seed_prior_trials",
        lambda self, prior: None,  # noqa: ARG005 - patch stub
    ):
        loop = OptimizationLoop(
            node_pair=NodePair(source="a", target="b", hardware_class="10g"),
            config=BenchmarkConfig(iterations=1),
            param_space=PARAM_SPACE,
            output=tmp_path / "out",
            n_trials=3,
            n_sobol=3,
            objectives=ObjectivesSection(),
        )
    a = _seed_primary(loop, trial_id="parent-a", bps=12e9)
    b = _seed_primary(loop, trial_id="parent-b", bps=11e9)
    c = _seed_primary(loop, trial_id="parent-c", bps=10e9)

    # A regresses out after round 1; rounds 2-4 do not re-log.
    by_parent: dict[str, list[float]] = {
        a.trial_id: [1e9],
        b.trial_id: [11e9, 11e9, 11e9, 11e9],
        c.trial_id: [10e9, 10e9, 10e9],
    }
    caplog.set_level(logging.INFO, logger="kube_autotuner.optimizer")
    with patch.object(OptimizationLoop, "_evaluate", _refine_factory(by_parent)):
        loop.run_refinement(top_k=2, rounds=4)

    drop_logs = [
        r
        for r in caplog.records
        if "dropped out of top-K" in r.getMessage() and a.trial_id in r.getMessage()
    ]
    assert len(drop_logs) == 1
    # b never drops, c never drops.
    assert all(b.trial_id not in r.getMessage() for r in drop_logs)
    assert all(c.trial_id not in r.getMessage() for r in drop_logs)
