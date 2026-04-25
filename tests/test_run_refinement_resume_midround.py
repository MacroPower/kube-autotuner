"""Resume mid-round: prior session's partial round R completes on resume.

Asserts the observer-index space does not collide between sessions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("ax")

if TYPE_CHECKING:
    from collections.abc import Callable

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


def _seed_refinement(
    loop: OptimizationLoop,
    *,
    parent: TrialResult,
    bps: float,
    round_index: int,
) -> TrialResult:
    tr = TrialResult(
        node_pair=loop.node_pair,
        sysctl_values=parent.sysctl_values,
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
        phase="refinement",
        parent_trial_id=parent.trial_id,
        refinement_round=round_index,
    )
    loop._completed.append(tr)
    return tr


def _refine_factory(
    by_parent_round: dict[tuple[str, int], float],
    obs_indices: list[int],
) -> tuple[
    Callable[..., tuple[TrialResult, dict[str, tuple[float, float]]]],
    Callable[..., None],
]:
    def fake_evaluate(
        self: OptimizationLoop,
        parameterization: dict[str, str],  # noqa: ARG001 - shape match
        *,
        phase: str,
        parent_trial_id: str | None,
        refinement_round: int | None = None,
    ) -> tuple[TrialResult, dict[str, tuple[float, float]]]:
        assert parent_trial_id is not None
        assert refinement_round is not None
        bps = by_parent_round[parent_trial_id, refinement_round]
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

    def fake_on_trial_start(
        index: int,
        total: int,  # noqa: ARG001 - shape match
        phase: str,  # noqa: ARG001 - shape match
        params: dict[str, str],  # noqa: ARG001 - shape match
    ) -> None:
        obs_indices.append(index)

    return fake_evaluate, fake_on_trial_start


@patch("kube_autotuner.optimizer.NodeLease")
@patch("kube_autotuner.optimizer.BenchmarkRunner")
@patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
def test_resume_completes_partial_round_and_indices_do_not_collide(
    mock_setter_cls: MagicMock,  # noqa: ARG001 - patched out
    mock_runner_cls: MagicMock,  # noqa: ARG001 - patched out
    mock_lease_cls: MagicMock,  # noqa: ARG001 - patched out
    tmp_path,
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
            n_trials=2,
            n_sobol=2,
            objectives=ObjectivesSection(),
        )

    # Two primaries seeded, plus one already-done refinement sample for
    # parent A in round 1 (the prior session's partial work).
    a = _seed_primary(loop, trial_id="parent-a", bps=12e9)
    b = _seed_primary(loop, trial_id="parent-b", bps=11e9)
    _seed_refinement(loop, parent=a, bps=11e9, round_index=1)

    # All subsequent samples hold A and B's primary numbers so top-K
    # stays {A, B} every round.
    by_parent_round: dict[tuple[str, int], float] = {
        (b.trial_id, 1): 11e9,  # round-1 finishes B
        (a.trial_id, 2): 12e9,  # round-2 A
        (b.trial_id, 2): 11e9,  # round-2 B
    }
    obs_indices: list[int] = []
    fake_evaluate, fake_on_trial_start = _refine_factory(
        by_parent_round,
        obs_indices,
    )
    loop.observer = MagicMock()
    loop.observer.on_trial_start = fake_on_trial_start

    with patch.object(OptimizationLoop, "_evaluate", fake_evaluate):
        created = loop.run_refinement(
            top_k=2,
            rounds=2,
            completed_by_round={1: {a.trial_id}},
        )

    # Round 1: only B sampled (A already done); round 2: A + B sampled.
    assert len(created) == 3
    by_round: dict[int, set[str]] = {}
    for tr in created:
        assert tr.refinement_round is not None
        by_round.setdefault(tr.refinement_round, set()).add(tr.parent_trial_id or "")
    assert by_round[1] == {b.trial_id}
    assert by_round[2] == {a.trial_id, b.trial_id}

    # Observer indices: base = prior_count(0) + live_primary(2) +
    # total_prior_refinement(1) = 3, so the three new samples land at
    # 3, 4, 5 -- never colliding with the prior round-1 sample's slot.
    assert obs_indices == [3, 4, 5]
