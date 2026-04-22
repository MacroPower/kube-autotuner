"""Tests for :meth:`OptimizationLoop._seed_prior_trials` with mixed priors.

Verification rows must land in ``self._completed`` but skip Ax's
``attach_trial`` / ``complete_trial`` (Ax cannot accept the same arm
twice and has no use for a repeat observation).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("ax")

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


def _default_sysctls(offset: int = 0) -> dict[str, str | int]:
    """Return a full-coverage sysctl dict satisfying the PARAM_SPACE search space.

    The Ax client rejects attach_trial calls whose parameterization does
    not include every search-space key. Priors produced by a real run
    always carry every key (via ``_evaluate``'s decode of the full Ax
    parameterization); the seed-path tests reproduce that shape here.

    ``offset`` picks a different value index per param so two calls
    produce distinct Ax arms (Ax rejects duplicate arms on
    ``attach_trial``).
    """
    return {p.name: p.values[offset % len(p.values)] for p in PARAM_SPACE.params}


def _prior(
    *,
    phase: str,
    parent_trial_id: str | None = None,
    offset: int = 0,
) -> TrialResult:
    return TrialResult(
        node_pair=NodePair(source="a", target="b", hardware_class="10g"),
        sysctl_values=_default_sysctls(offset),
        config=BenchmarkConfig(),
        results=[
            BenchmarkResult(
                timestamp=datetime.now(UTC),
                mode="tcp",
                bits_per_second=9e9,
                retransmits=5,
                bytes_sent=10**9,
                cpu_utilization_percent=20.0,
            ),
        ],
        phase=phase,  # ty: ignore[invalid-argument-type]
        parent_trial_id=parent_trial_id,
    )


@patch("kube_autotuner.optimizer.NodeLease")
@patch("kube_autotuner.optimizer.BenchmarkRunner")
@patch("kube_autotuner.optimizer.make_sysctl_setter_from_env")
def test_mixed_priors_seed_only_primaries_into_ax(
    mock_setter_cls: MagicMock,  # noqa: ARG001 - patched out
    mock_runner_cls: MagicMock,  # noqa: ARG001 - patched out
    mock_lease_cls: MagicMock,  # noqa: ARG001 - patched out
    tmp_path,
) -> None:
    primary_a = _prior(phase="sobol", offset=0)
    primary_b = _prior(phase="bayesian", offset=1)
    verification_child = _prior(
        phase="verification",
        offset=0,
        parent_trial_id=primary_a.trial_id,
    )

    with (
        patch.object(
            OptimizationLoop,
            "_seed_prior_trials",
            lambda self, prior: None,  # noqa: ARG005 - patch stub
        ),
    ):
        loop = OptimizationLoop(
            node_pair=NodePair(
                source="a",
                target="b",
                hardware_class="10g",
            ),
            config=BenchmarkConfig(),
            param_space=PARAM_SPACE,
            output=tmp_path / "out.jsonl",
            n_trials=3,
            n_sobol=1,
            objectives=ObjectivesSection(),
        )

    # Spy the real Ax client.
    attach_spy = MagicMock(side_effect=loop.client.attach_trial)
    complete_spy = MagicMock(side_effect=loop.client.complete_trial)
    loop.client.attach_trial = attach_spy  # ty: ignore[invalid-assignment]
    loop.client.complete_trial = complete_spy  # ty: ignore[invalid-assignment]

    prior = [primary_a, verification_child, primary_b]
    loop._seed_prior_trials(prior)

    # Both primaries attached + completed; the verification row did not.
    assert attach_spy.call_count == 2
    assert complete_spy.call_count == 2

    # _completed holds all three rows in file order.
    assert loop._completed == [primary_a, verification_child, primary_b]
    assert sum(1 for t in loop._completed if is_primary(t)) == 2
