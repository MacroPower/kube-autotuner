"""End-to-end integration test for the optimize CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from kube_autotuner.cli import app
from kube_autotuner.k8s.lease import NodeLease
from kube_autotuner.models import TrialResult

if TYPE_CHECKING:
    from pathlib import Path

    from kube_autotuner.benchmark.runner import BenchmarkRunner
    from kube_autotuner.k8s.client import K8sClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(900),
]


def test_optimize_runs_trials_and_writes_jsonl(
    kubeconfig_env: str,  # noqa: ARG001 - activates KUBECONFIG env var
    k8s_client: K8sClient,
    node_names: dict[str, str],
    test_namespace: str,
    fake_sysctl_env: Path,  # noqa: ARG001 - activates fake backend env vars
    tmp_path: Path,
) -> None:
    pytest.importorskip("ax")

    output_file = tmp_path / "optimize.jsonl"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "optimize",
            "--source",
            node_names["source"],
            "--target",
            node_names["target"],
            "--hardware-class",
            "1g",
            "--ip-family-policy",
            "SingleStack",
            "--namespace",
            test_namespace,
            "--duration",
            "5",
            "--iterations",
            "1",
            "--n-trials",
            "2",
            "--n-sobol",
            "2",
            "--output",
            str(output_file),
        ],
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    assert "Completed 2 trials" in result.output

    lines = output_file.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        trial = TrialResult.model_validate_json(line)
        assert trial.sysctl_values
        assert trial.results[0].bits_per_second > 0

    lease_name = f"{NodeLease.LEASE_PREFIX}-{node_names['target']}"
    assert k8s_client.get_json("lease", lease_name, test_namespace) is None


def _invoke_optimize(
    output_file: Path,
    *,
    node_names: dict[str, str],
    test_namespace: str,
    n_trials: int,
    n_sobol: int = 2,
    extra_args: list[str] | None = None,
) -> Any:  # typer.testing.Result
    """Invoke the optimize CLI with the canonical integration-test flags."""
    runner = CliRunner()
    return runner.invoke(
        app,
        [
            "optimize",
            "--source",
            node_names["source"],
            "--target",
            node_names["target"],
            "--hardware-class",
            "1g",
            "--ip-family-policy",
            "SingleStack",
            "--namespace",
            test_namespace,
            "--duration",
            "5",
            "--iterations",
            "1",
            "--n-trials",
            str(n_trials),
            "--n-sobol",
            str(n_sobol),
            "--output",
            str(output_file),
            *(extra_args or []),
        ],
    )


class _BenchmarkRunCounter:
    """Wrap ``BenchmarkRunner.run`` to count invocations without changing its behaviour.

    Integration-test monkeypatch: the wrapper forwards to the original
    method so real benchmarks still execute against the Talos cluster,
    but the counter lets the test assert how many benchmark runs each
    CLI invocation triggered. That is the direct signal the plan
    asked for (`only two new benchmark runs occurred`).
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kube_autotuner.benchmark.runner import BenchmarkRunner  # noqa: PLC0415

        self.calls = 0
        self._original = BenchmarkRunner.run

        def _wrapped(inner_self: BenchmarkRunner) -> Any:
            self.calls += 1
            return self._original(inner_self)

        monkeypatch.setattr(BenchmarkRunner, "run", _wrapped)


def test_optimize_resumes_from_prior_jsonl(
    kubeconfig_env: str,  # noqa: ARG001 - activates KUBECONFIG env var
    node_names: dict[str, str],
    test_namespace: str,
    fake_sysctl_env: Path,  # noqa: ARG001 - activates fake backend env vars
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-invoking optimize with the same output resumes from prior trials."""
    pytest.importorskip("ax")

    output_file = tmp_path / "optimize.jsonl"
    counter = _BenchmarkRunCounter(monkeypatch)

    first = _invoke_optimize(
        output_file,
        node_names=node_names,
        test_namespace=test_namespace,
        n_trials=2,
    )
    assert first.exit_code == 0, f"CLI failed:\n{first.output}"
    assert counter.calls == 2, (
        f"first invocation should run 2 benchmarks, got {counter.calls}"
    )

    first_two_lines = output_file.read_text().strip().splitlines()
    assert len(first_two_lines) == 2

    second = _invoke_optimize(
        output_file,
        node_names=node_names,
        test_namespace=test_namespace,
        n_trials=4,
    )
    assert second.exit_code == 0, f"CLI failed:\n{second.output}"

    # Only 2 new benchmarks (4 total across the two invocations).
    assert counter.calls == 4, (
        f"second invocation should only run 2 new benchmarks, got {counter.calls - 2}"
    )

    # Resume arithmetic surfaced in the logs. ``Ax``'s own ``Trial N
    # marked COMPLETED`` lines come from the ``ax.api.client`` logger
    # which doesn't propagate through our ``basicConfig`` + CliRunner
    # stderr swap; the unit test
    # ``test_seed_prior_trials_attaches_and_completes`` pins the
    # attach_trial/complete_trial calls directly instead.
    assert "Resuming: 2 prior trials; running 2 more (budget=4)" in second.output
    assert "2 prior + 2 new" in second.output
    # The live log numerator continues past the priors rather than
    # restarting at 1 — pins the _prior_count + i + 1 rule against a
    # real Ax loop, not just a mock.
    assert "Trial 3/4" in second.output
    assert "Trial 4/4" in second.output

    # Prior rows survive byte-identically; live loop only appended two.
    all_lines = output_file.read_text().strip().splitlines()
    assert len(all_lines) == 4
    assert all_lines[:2] == first_two_lines


def test_optimize_fresh_archives_prior_artefacts(
    kubeconfig_env: str,  # noqa: ARG001 - activates KUBECONFIG env var
    node_names: dict[str, str],
    test_namespace: str,
    fake_sysctl_env: Path,  # noqa: ARG001 - activates fake backend env vars
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--fresh`` renames the prior JSONL and sidecar aside, runs from zero."""
    pytest.importorskip("ax")

    output_file = tmp_path / "optimize.jsonl"
    counter = _BenchmarkRunCounter(monkeypatch)

    first = _invoke_optimize(
        output_file,
        node_names=node_names,
        test_namespace=test_namespace,
        n_trials=2,
    )
    assert first.exit_code == 0, f"CLI failed:\n{first.output}"
    assert counter.calls == 2

    # Re-invoke with --fresh. The prior JSONL + sidecar should be
    # archived aside and the run should start from zero.
    # ``n_sobol=1`` keeps the config valid against the
    # ``n_sobol <= n_trials`` invariant.
    second = _invoke_optimize(
        output_file,
        node_names=node_names,
        test_namespace=test_namespace,
        n_trials=1,
        n_sobol=1,
        extra_args=["--fresh"],
    )
    assert second.exit_code == 0, f"CLI failed:\n{second.output}"
    assert counter.calls == 3, (
        f"--fresh invocation should run 1 benchmark from zero, total={counter.calls}"
    )

    jsonl_backups = [
        p for p in tmp_path.glob("optimize.jsonl.*.bak") if ".meta.json" not in p.name
    ]
    meta_backups = list(tmp_path.glob("optimize.jsonl.meta.json.*.bak"))
    assert len(jsonl_backups) == 1, f"expected 1 JSONL backup; got {jsonl_backups}"
    assert len(meta_backups) == 1, f"expected 1 sidecar backup; got {meta_backups}"

    # Current JSONL has one fresh trial (not appended to the prior).
    current_lines = output_file.read_text().strip().splitlines()
    assert len(current_lines) == 1

    # Backups preserve the prior two trials verbatim.
    archived = jsonl_backups[0].read_text().strip().splitlines()
    assert len(archived) == 2


def test_optimize_rejects_incompatible_n_sobol(
    kubeconfig_env: str,  # noqa: ARG001 - activates KUBECONFIG env var
    node_names: dict[str, str],
    test_namespace: str,
    fake_sysctl_env: Path,  # noqa: ARG001 - activates fake backend env vars
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing ``--n-sobol`` across sessions is a hard error without ``--fresh``."""
    pytest.importorskip("ax")

    output_file = tmp_path / "optimize.jsonl"
    counter = _BenchmarkRunCounter(monkeypatch)

    first = _invoke_optimize(
        output_file,
        node_names=node_names,
        test_namespace=test_namespace,
        n_trials=1,
        n_sobol=1,
    )
    assert first.exit_code == 0, f"CLI failed:\n{first.output}"
    assert counter.calls == 1

    # Same output, different n_sobol -> compat check fails, no benchmark runs.
    second = _invoke_optimize(
        output_file,
        node_names=node_names,
        test_namespace=test_namespace,
        n_trials=2,
        n_sobol=2,
    )
    assert second.exit_code != 0
    assert counter.calls == 1, (
        f"rejected invocation should not run benchmarks; total={counter.calls}"
    )
    assert "n_sobol" in second.output


def test_optimize_short_circuits_when_budget_already_met(
    kubeconfig_env: str,  # noqa: ARG001 - activates KUBECONFIG env var
    node_names: dict[str, str],
    test_namespace: str,
    fake_sysctl_env: Path,  # noqa: ARG001 - activates fake backend env vars
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-invoking with ``n_trials <= prior_count`` skips the loop entirely."""
    pytest.importorskip("ax")

    output_file = tmp_path / "optimize.jsonl"
    counter = _BenchmarkRunCounter(monkeypatch)

    first = _invoke_optimize(
        output_file,
        node_names=node_names,
        test_namespace=test_namespace,
        n_trials=2,
    )
    assert first.exit_code == 0, f"CLI failed:\n{first.output}"
    assert counter.calls == 2

    # Budget already met (prior=2, n_trials=2) -> loop is skipped.
    second = _invoke_optimize(
        output_file,
        node_names=node_names,
        test_namespace=test_namespace,
        n_trials=2,
    )
    assert second.exit_code == 0, f"CLI failed:\n{second.output}"
    assert counter.calls == 2, (
        f"short-circuit path should run no new benchmarks; total={counter.calls}"
    )
    assert "Budget already met" in second.output
    assert len(output_file.read_text().strip().splitlines()) == 2
