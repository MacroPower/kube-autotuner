"""Top-K shifts between rounds when a regressed parent drops out.

Round 1 picks the top two parents A and B. A's refinement sample
regresses sharply, so its combined mean falls behind C's primary.
Round 2 re-picks top-K and now selects B and C; the round-2 trial
assignments reflect that.
"""

from __future__ import annotations

from datetime import UTC, datetime
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
    is_primary,
)
from kube_autotuner.optimizer import OptimizationLoop
from kube_autotuner.sysctl.params import PARAM_SPACE


def _seed_primary(
    loop: OptimizationLoop,
    *,
    trial_id: str,
    bps: float,
) -> TrialResult:
    """Inject a synthetic primary trial directly into ``loop._completed``.

    Returns:
        The :class:`TrialResult` that was appended.
    """
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
    """Return an ``_evaluate`` replacement that emits parent-specific metrics."""
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


def _build_loop(tmp_path) -> OptimizationLoop:
    """Construct a loop that does not perform Ax setup or seed_prior_trials.

    Returns:
        The constructed :class:`OptimizationLoop`.
    """
    with patch.object(
        OptimizationLoop,
        "_seed_prior_trials",
        lambda self, prior: None,  # noqa: ARG005 - patch stub
    ):
        return OptimizationLoop(
            node_pair=NodePair(source="a", target="b", hardware_class="10g"),
            config=BenchmarkConfig(iterations=1),
            param_space=PARAM_SPACE,
            output=tmp_path / "out",
            n_trials=3,
            n_sobol=3,
            objectives=ObjectivesSection(),
        )


@patch("kube_autotuner.optimizer.NodeLease")
@patch("kube_autotuner.optimizer.BenchmarkRunner")
@patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
def test_top_k_shifts_after_round_1_regression(
    mock_setter_cls: MagicMock,  # noqa: ARG001 - patched out
    mock_runner_cls: MagicMock,  # noqa: ARG001 - patched out
    mock_lease_cls: MagicMock,  # noqa: ARG001 - patched out
    tmp_path,
) -> None:
    loop = _build_loop(tmp_path)
    a = _seed_primary(loop, trial_id="parent-a", bps=12e9)
    b = _seed_primary(loop, trial_id="parent-b", bps=11e9)
    c = _seed_primary(loop, trial_id="parent-c", bps=10e9)

    # Round-1 sampling: A regresses to 1e9, B holds at 11e9. After round
    # 1, A's mean is (12 + 1) / 2 = 6.5 < C's 10; A drops out, C enters.
    # Round-2 sampling: B and C each get one sample.
    by_parent: dict[str, list[float]] = {
        a.trial_id: [1e9],  # round-1 only; A drops out before round 2
        b.trial_id: [11e9, 11e9],  # round-1 + round-2
        c.trial_id: [10e9],  # round-2 only
    }

    with patch.object(OptimizationLoop, "_evaluate", _refine_factory(by_parent)):
        created = loop.run_refinement(top_k=2, rounds=2)

    # Total 4 samples (top_k=2 * rounds=2).
    assert len(created) == 4
    by_round: dict[int, set[str]] = {}
    for tr in created:
        assert tr.refinement_round is not None
        by_round.setdefault(tr.refinement_round, set()).add(tr.parent_trial_id or "")
    # Round 1 picked {a, b}; round 2 picked {b, c} (A dropped, C entered).
    assert by_round[1] == {a.trial_id, b.trial_id}
    assert by_round[2] == {b.trial_id, c.trial_id}
    # Sanity: every parent's `is_primary` rows are still in _completed.
    primary_ids = {tr.trial_id for tr in loop._completed if is_primary(tr)}
    assert primary_ids == {a.trial_id, b.trial_id, c.trial_id}
