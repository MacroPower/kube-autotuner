"""Unit tests for :mod:`kube_autotuner.experiment`."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from kube_autotuner import experiment as experiment_mod
from kube_autotuner.experiment import (
    CLIENT_FLAG_DENYLIST,
    FORTIO_CLIENT_FLAG_DENYLIST,
    FORTIO_SERVER_FLAG_DENYLIST,
    SERVER_FLAG_DENYLIST,
    ExperimentConfig,
    ExperimentConfigError,
    ObjectivesSection,
    ParetoObjective,
)
from kube_autotuner.sysctl.params import PARAM_SPACE

if TYPE_CHECKING:
    from kube_autotuner.experiment import PreflightResult

FIXTURE_YAML = Path("tests/fixtures/experiment_example.yaml")


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "exp.yaml"
    p.write_text(body)
    return p


def _find(results: list[PreflightResult], name: str) -> PreflightResult:
    for r in results:
        if r.name == name:
            return r
    pytest.fail(f"no PreflightResult named {name!r} in {results!r}")


# --- YAML loader ---------------------------------------------------------


def test_minimal_baseline_roundtrips(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [kmain07]
  target: kmain08
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    assert exp.nodes.sources == ["kmain07"]
    assert exp.nodes.target == "kmain08"
    assert exp.iperf.client.extra_args == []
    assert exp.patches == []
    assert exp.optimize is None
    assert exp.trial is None


def test_full_optimize_roundtrips(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [kmain07, kmain09]
  target: kmain08
  hardwareClass: 10g
benchmark:
  iterations: 3
optimize:
  nTrials: 50
  nSobol: 15
  applySource: true
  paramSpace:
    - {name: net.core.rmem_max, paramType: int, values: [4194304, 67108864]}
iperf:
  duration: 30
  client:
    extraArgs: ["--bidir", "-Z"]
  server:
    extraArgs: ["--forceflush"]
patches:
  - target: {kind: Job}
    patch:
      spec:
        template:
          spec:
            containers:
              - name: iperf3-client
                resources:
                  limits: {memory: "2Gi"}
  - target: {kind: Deployment}
    patch:
      - op: add
        path: /spec/template/spec/hostNetwork
        value: true
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    assert exp.optimize is not None
    assert exp.optimize.n_trials == 50
    assert exp.iperf.client.extra_args == ["--bidir", "-Z"]
    assert len(exp.patches) == 2
    assert exp.patches[0].target.kind == "Job"
    assert isinstance(exp.patches[0].patch, dict)
    assert isinstance(exp.patches[1].patch, list)
    assert exp.optimize.param_space is not None
    assert exp.optimize.param_space[0].name == "net.core.rmem_max"


def test_custom_hardware_class_roundtrips(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [kmain07]
  target: kmain08
  hardwareClass: epyc-9454p
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    assert exp.nodes.hardware_class == "epyc-9454p"


def test_empty_hardware_class_raises(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [kmain07]
  target: kmain08
  hardwareClass: ""
""",
    )
    with pytest.raises(ExperimentConfigError):
        ExperimentConfig.from_yaml(path)


def test_fixture_yaml_roundtrips():
    """The shipped fixture must load cleanly and survive model validation."""
    exp = ExperimentConfig.from_yaml(FIXTURE_YAML)
    assert exp.nodes.sources == ["kmain07", "kmain09"]
    assert exp.nodes.target == "kmain08"
    assert exp.nodes.hardware_class == "10g"
    assert exp.optimize is not None
    assert exp.optimize.n_trials == 50
    assert exp.optimize.n_sobol == 15
    assert exp.optimize.param_space is not None
    assert [p.name for p in exp.optimize.param_space] == [
        "net.core.rmem_max",
        "net.ipv4.tcp_congestion_control",
    ]
    assert exp.iperf.client.extra_args == ["--bidir", "-Z"]
    assert exp.iperf.server.extra_args == ["--forceflush"]
    assert len(exp.patches) == 2
    assert exp.patches[0].target.kind == "Job"
    assert exp.patches[1].target.kind == "Deployment"
    assert exp.output == "out/results"


def test_multi_doc_yaml_raises(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
---
nodes:
  sources: [c]
  target: d
""",
    )
    with pytest.raises(ExperimentConfigError, match="multi-document YAML"):
        ExperimentConfig.from_yaml(path)


def test_typo_field_raises(tmp_path: Path):
    """extra='forbid' catches typos at parse time."""
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
iperf:
  clint:
    extraArgs: []
""",
    )
    with pytest.raises(ExperimentConfigError):
        ExperimentConfig.from_yaml(path)


def test_iperf_section_default_clients_per_node_is_one():
    """``clients_per_node`` defaults to 1: one client Job per source."""
    from kube_autotuner.experiment import IperfSection  # noqa: PLC0415

    section = IperfSection()
    assert section.clients_per_node == 1


def test_iperf_section_clients_per_node_zero_raises():
    """``ge=1`` rejects ``clients_per_node=0``."""
    from pydantic import ValidationError  # noqa: PLC0415

    from kube_autotuner.experiment import IperfSection  # noqa: PLC0415

    with pytest.raises(ValidationError):
        IperfSection.model_validate({"clients_per_node": 0})


def test_iperf_section_clients_per_node_negative_raises():
    """``ge=1`` rejects negative ``clients_per_node`` values."""
    from pydantic import ValidationError  # noqa: PLC0415

    from kube_autotuner.experiment import IperfSection  # noqa: PLC0415

    with pytest.raises(ValidationError):
        IperfSection.model_validate({"clients_per_node": -1})


def test_iperf_section_camelcase_clients_per_node_rejected():
    """``IperfSection`` has no alias generator: ``clientsPerNode`` is unknown."""
    from pydantic import ValidationError  # noqa: PLC0415

    from kube_autotuner.experiment import IperfSection  # noqa: PLC0415

    with pytest.raises(ValidationError):
        IperfSection.model_validate({"clientsPerNode": 2})


def test_invalid_yaml_raises(tmp_path: Path):
    path = _write(tmp_path, "nodes:\n  sources: oops\n: : :\n")
    with pytest.raises(ExperimentConfigError, match="invalid YAML"):
        ExperimentConfig.from_yaml(path)


# --- section-level validators -------------------------------------------


def test_legacy_mode_field_in_yaml_rejected(tmp_path: Path):
    """A legacy YAML carrying ``mode:`` fails fast with ``mode`` in the message."""
    path = _write(
        tmp_path,
        """\
mode: optimize
nodes:
  sources: [a]
  target: b
""",
    )
    with pytest.raises(ExperimentConfigError, match="mode"):
        ExperimentConfig.from_yaml(path)


def test_trial_section_rejects_empty_sysctls():
    """``TrialSection`` itself rejects an empty sysctls map at construction."""
    from kube_autotuner.experiment import TrialSection  # noqa: PLC0415

    with pytest.raises(ValueError, match="at least one entry"):
        TrialSection.model_validate({"sysctls": {}})


def test_trial_with_empty_sysctls_in_yaml_raises(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
trial:
  sysctls: {}
""",
    )
    with pytest.raises(ExperimentConfigError, match="at least one entry"):
        ExperimentConfig.from_yaml(path)


def test_trial_with_sysctls_accepted(tmp_path: Path):
    """Regression guard for the trial happy path."""
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
trial:
  sysctls:
    net.core.rmem_max: 67108864
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    assert exp.trial is not None
    assert exp.trial.sysctls == {"net.core.rmem_max": 67108864}


def test_optimize_section_rejects_n_sobol_above_n_trials():
    """``OptimizeSection`` itself rejects ``n_sobol > n_trials``."""
    from kube_autotuner.experiment import OptimizeSection  # noqa: PLC0415

    with pytest.raises(ValueError, match="n_sobol must be <="):
        OptimizeSection.model_validate({"nTrials": 5, "nSobol": 10})


def test_n_sobol_greater_than_n_trials_raises(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
optimize:
  nTrials: 5
  nSobol: 10
""",
    )
    with pytest.raises(ExperimentConfigError, match="n_sobol must be <="):
        ExperimentConfig.from_yaml(path)


def test_n_sobol_equal_to_n_trials_accepted(tmp_path: Path):
    """Boundary guard: equal values are allowed."""
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
optimize:
  nTrials: 5
  nSobol: 5
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    assert exp.optimize is not None
    assert exp.optimize.n_trials == 5
    assert exp.optimize.n_sobol == 5


# --- benchmark.stages / objectives pruning ------------------------------


def test_stages_default_keeps_every_objective(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    metrics = {obj.metric for obj in exp.objectives.pareto}
    assert metrics == {
        "tcp_throughput",
        "udp_throughput",
        "tcp_retransmit_rate",
        "udp_loss_rate",
        "udp_jitter",
        "rps",
        "latency_p50",
        "latency_p90",
        "latency_p99",
    }


def test_stages_subset_prunes_default_objectives(tmp_path: Path, caplog):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
benchmark:
  stages: ["bw-tcp"]
""",
    )
    with caplog.at_level("INFO", logger="kube_autotuner.experiment"):
        exp = ExperimentConfig.from_yaml(path)
    metrics = {obj.metric for obj in exp.objectives.pareto}
    assert metrics == {"tcp_throughput", "tcp_retransmit_rate"}
    for constraint in exp.objectives.constraints:
        metric = constraint.split()[0]
        assert metric in {"tcp_throughput", "tcp_retransmit_rate"}
    assert "udp_throughput" not in exp.objectives.recommendation_weights
    assert any(
        "produced only by disabled stages" in rec.message for rec in caplog.records
    )


def test_stages_prune_error_when_pareto_empties(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
benchmark:
  stages: ["bw-tcp"]
objectives:
  pareto:
    - metric: udp_throughput
      direction: maximize
  constraints: []
  recommendationWeights: {}
""",
    )
    with pytest.raises(
        ExperimentConfigError,
        match=r"objectives\.pareto is empty after pruning",
    ):
        ExperimentConfig.from_yaml(path)


# --- projections ---------------------------------------------------------


def test_effective_param_space_default_returns_full_param_space():
    """With no optimize override, the canonical full PARAM_SPACE is returned."""
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
    })
    assert exp.effective_param_space() is PARAM_SPACE


def test_effective_param_space_user_override_replaces_default():
    """A user-supplied param_space is returned verbatim."""
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "optimize": {
            "nTrials": 2,
            "nSobol": 1,
            "paramSpace": [
                {
                    "name": "net.ipv4.udp_mem",
                    "paramType": "choice",
                    "values": ["a b c"],
                },
            ],
        },
    })
    ps = exp.effective_param_space()
    assert ps.param_names() == ["net.ipv4.udp_mem"]


def test_benchmark_modes_key_rejected(tmp_path: Path):
    """An unknown ``benchmark.modes`` key must be rejected at load time."""
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
benchmark:
  iterations: 1
  modes: [tcp]
""",
    )
    with pytest.raises(ExperimentConfigError):
        ExperimentConfig.from_yaml(path)


def test_effective_param_space_override():
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "optimize": {
            "nTrials": 2,
            "nSobol": 1,
            "paramSpace": [
                {"name": "net.core.rmem_max", "paramType": "int", "values": [1, 2]},
            ],
        },
    })
    ps = exp.effective_param_space()
    assert ps.param_names() == ["net.core.rmem_max"]


def test_to_node_pair_multi_source():
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["primary", "extra1", "extra2"], "target": "t"},
    })
    np = exp.to_node_pair()
    assert np.source == "primary"
    assert np.extra_sources == ["extra1", "extra2"]
    assert np.target == "t"


# --- SMP structural validators ------------------------------------------


def test_smp_dict_without_kind_rejected(tmp_path: Path):
    """Dict body + labelSelector-only target must declare a kind."""
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
patches:
  - target:
      labelSelector: "app=foo"
    patch:
      spec:
        replicas: 3
""",
    )
    with pytest.raises(ExperimentConfigError, match=r"require `target\.kind`"):
        ExperimentConfig.from_yaml(path)


def test_smp_dict_with_body_kind_accepted(tmp_path: Path):
    """Dict body with explicit `kind:` survives the validator."""
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
patches:
  - target:
      labelSelector: "app=foo"
    patch:
      kind: Deployment
      apiVersion: apps/v1
      metadata:
        name: x
      spec:
        replicas: 3
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    assert len(exp.patches) == 1


# --- individual preflight checks ----------------------------------------


@pytest.mark.parametrize("flag", ["-c", "--client", "-J", "--json", "-B", "--logfile"])
def test_client_denylist(tmp_path: Path, flag: str):
    path = _write(
        tmp_path,
        f"""\
nodes:
  sources: [a]
  target: b
iperf:
  client:
    extraArgs: ["{flag}"]
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    result = exp._check_denylists()
    assert not result.passed
    assert "reserved flag" in result.detail
    assert flag in result.detail


def test_client_denylist_whole_token(tmp_path: Path):
    """--timestamps is not --time; must pass."""
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
iperf:
  client:
    extraArgs: ["--timestamps"]
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    assert exp._check_denylists().passed


def test_client_extra_args_window_round_trip(tmp_path: Path):
    """``-w`` is now a user-supplied iperf3 flag; extraArgs must accept it."""
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
iperf:
  client:
    extraArgs: ["-w", "256K"]
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    assert exp._check_denylists().passed


def test_server_denylist(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
iperf:
  server:
    extraArgs: ["-s"]
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    result = exp._check_denylists()
    assert not result.passed
    assert "reserved flag" in result.detail


def test_sysctl_name_regex():
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "trial": {"sysctls": {"NOT-VALID": "1"}},
    })
    result = exp._check_sysctl_names()
    assert not result.passed
    assert "invalid sysctl name" in result.detail


def test_sysctl_name_regex_happy_path():
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "trial": {"sysctls": {"net.core.rmem_max": 67108864}},
    })
    assert exp._check_sysctl_names().passed


def test_patch_metadata_namespace_rejected(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
patches:
  - target: {kind: Job}
    patch:
      metadata:
        namespace: other
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    result = exp._check_patches_shape()
    assert not result.passed
    assert "metadata.namespace" in result.detail


def test_patch_jsonpatch_namespace_op_rejected(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
patches:
  - target: {kind: Job}
    patch:
      - op: replace
        path: /metadata/namespace
        value: other
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    result = exp._check_patches_shape()
    assert not result.passed
    assert "/metadata/namespace" in result.detail


def test_target_namespace_rejected(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
patches:
  - target:
      kind: Deployment
      namespace: other
    patch:
      spec: {replicas: 1}
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    result = exp._check_patches_shape()
    assert not result.passed
    assert "target.namespace" in result.detail


def test_patches_shape_clean_passes(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
patches:
  - target: {kind: Deployment}
    patch:
      spec: {replicas: 1}
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    assert exp._check_patches_shape().passed


def test_check_kustomize_available_missing(monkeypatch):
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "patches": [
            {"target": {"kind": "Deployment"}, "patch": {"spec": {}}},
        ],
    })
    monkeypatch.setattr(shutil, "which", lambda _: None)
    result = exp._check_kustomize_available()
    assert not result.passed
    assert "kustomize" in result.detail
    assert "binary not found" in result.detail


def test_check_kustomize_available_skips_when_no_patches():
    """No patches -> kustomize not required."""
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
    })
    result = exp._check_kustomize_available()
    assert result.passed
    assert "skipped" in result.detail


def test_check_kustomize_available_routes_through_run_tool(monkeypatch):
    """The probe routes through kube_autotuner.subproc.run_tool, not bare subprocess."""
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "patches": [
            {"target": {"kind": "Deployment"}, "patch": {"spec": {}}},
        ],
    })
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/kustomize")

    calls: list[tuple[str, list[str]]] = []

    def _fake_run_tool(
        binary: str,
        args,
        *,
        check: bool = False,
        input_: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del check, input_
        calls.append((binary, list(args)))
        return subprocess.CompletedProcess(
            args=[binary, *args], returncode=0, stdout="v5", stderr=""
        )

    monkeypatch.setattr(experiment_mod, "run_tool", _fake_run_tool)
    result = exp._check_kustomize_available()
    assert result.passed
    assert calls == [("kustomize", ["version"])]


def test_check_kustomize_available_nonzero_rc(monkeypatch):
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "patches": [
            {"target": {"kind": "Deployment"}, "patch": {"spec": {}}},
        ],
    })
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/kustomize")
    monkeypatch.setattr(
        experiment_mod,
        "run_tool",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=["kustomize", "version"],
            returncode=2,
            stdout="",
            stderr="boom",
        ),
    )
    result = exp._check_kustomize_available()
    assert not result.passed
    assert "rc=2" in result.detail
    assert "boom" in result.detail


def test_check_nodes_exist_surfaces_api_failure():
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
    })
    client = MagicMock()
    client.get_node_zone.side_effect = RuntimeError("NotFound")
    result = exp._check_nodes_exist(client)
    assert not result.passed
    assert "not found or unreachable" in result.detail


def test_check_nodes_exist_happy_path():
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a", "extra"], "target": "b"},
    })
    client = MagicMock()
    client.get_node_zone.return_value = "zone-a"
    assert exp._check_nodes_exist(client).passed
    assert client.get_node_zone.call_count == 3


def test_output_path_creates_directory(tmp_path: Path):
    out = tmp_path / "subdir" / "results"
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "output": str(out),
    })
    result = exp._check_output_path()
    assert result.passed
    # Preflight creates the trial-log directory up-front so the first
    # append does not have to.
    assert out.is_dir()


def test_output_path_not_writable(tmp_path: Path, monkeypatch):
    out = tmp_path / "results"
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "output": str(out),
    })
    monkeypatch.setattr(os, "access", lambda _p, _mode: False)
    result = exp._check_output_path()
    assert not result.passed
    assert "not writable" in result.detail


def test_output_path_rejects_existing_file(tmp_path: Path):
    out = tmp_path / "results"
    out.write_text("regular file")
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "output": str(out),
    })
    result = exp._check_output_path()
    assert not result.passed
    assert "not a directory" in result.detail


def test_output_path_rejects_unrelated_directory(tmp_path: Path):
    out = tmp_path / "results"
    out.mkdir()
    (out / "random.txt").write_text("not a trial log")
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "output": str(out),
    })
    result = exp._check_output_path()
    assert not result.passed
    assert "not a trial-log" in result.detail


# --- preflight orchestration --------------------------------------------


def test_preflight_orchestration_smoke():
    """preflight() runs every check in order and returns all results."""
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
    })
    client = MagicMock()
    client.get_node_zone.return_value = ""
    results = exp.preflight(client)
    names = [r.name for r in results]
    assert names == [
        "denylists",
        "sysctl-names",
        "patches-shape",
        "kustomize-available",
        "dry-render-patches",
        "output-path",
        "nodes-exist",
    ]
    assert all(r.passed for r in results)


def test_preflight_collects_multiple_failures(tmp_path: Path):
    """preflight() does not fail fast -- every failing check is returned."""
    out = tmp_path / "results"
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "iperf": {"client": {"extraArgs": ["-c"]}},
        "output": str(out),
    })
    client = MagicMock()
    client.get_node_zone.side_effect = RuntimeError("NotFound")
    results = exp.preflight(client)

    assert not _find(results, "denylists").passed
    assert not _find(results, "nodes-exist").passed
    # Unrelated checks still report as passing.
    assert _find(results, "sysctl-names").passed
    assert _find(results, "patches-shape").passed


# --- dry-render (requires real kustomize binary) ------------------------


def test_strict_patch_with_zero_matches_fails():
    """A strict patch that matches no resources must fail during dry-render."""
    if shutil.which("kustomize") is None:
        pytest.skip("kustomize binary required on PATH")

    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "patches": [
            {
                "target": {"kind": "ConfigMap"},  # absent from rendered output
                "patch": [
                    {"op": "add", "path": "/data/x", "value": "y"},
                ],
            },
        ],
    })
    result = exp._dry_render_patches()
    assert not result.passed
    assert "matched zero resources" in result.detail


def test_non_strict_patch_with_zero_matches_allowed():
    if shutil.which("kustomize") is None:
        pytest.skip("kustomize binary required on PATH")

    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
        "patches": [
            {
                "target": {"kind": "ConfigMap"},
                "strict": False,
                "patch": [
                    {"op": "add", "path": "/data/x", "value": "y"},
                ],
            },
        ],
    })
    assert exp._dry_render_patches().passed


def test_dry_render_skipped_when_no_patches():
    """No patches -> dry-render is a no-op."""
    exp = ExperimentConfig.model_validate({
        "nodes": {"sources": ["a"], "target": "b"},
    })
    result = exp._dry_render_patches()
    assert result.passed
    assert "skipped" in result.detail


# --- constants -----------------------------------------------------------


def test_denylist_constants_split_correctly():
    """Client denylist includes iperf-client flags; server denylist is tighter."""
    assert "-c" in CLIENT_FLAG_DENYLIST
    assert "--json" in CLIENT_FLAG_DENYLIST
    assert "-s" in SERVER_FLAG_DENYLIST
    assert "-c" not in SERVER_FLAG_DENYLIST
    assert "--json" not in SERVER_FLAG_DENYLIST


@pytest.mark.parametrize("flag", sorted(FORTIO_CLIENT_FLAG_DENYLIST))
def test_fortio_client_denylist_flags_every_entry(tmp_path: Path, flag: str):
    path = _write(
        tmp_path,
        f"""\
nodes:
  sources: [a]
  target: b
fortio:
  client:
    extraArgs: ["{flag}"]
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    result = exp._check_denylists()
    assert not result.passed
    assert "reserved flag" in result.detail
    assert flag in result.detail


def test_fortio_server_denylist(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
nodes:
  sources: [a]
  target: b
fortio:
  server:
    extraArgs: ["-http-port"]
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    result = exp._check_denylists()
    assert not result.passed
    assert "reserved flag" in result.detail
    assert "-http-port" in result.detail


def test_fortio_denylist_constants():
    assert "-qps" in FORTIO_CLIENT_FLAG_DENYLIST
    assert "-json" in FORTIO_CLIENT_FLAG_DENYLIST
    assert "-http-port" in FORTIO_SERVER_FLAG_DENYLIST


class TestObjectivesSection:
    def test_defaults_round_trip(self) -> None:
        section = ObjectivesSection()
        dumped = section.model_dump_json()
        reloaded = ObjectivesSection.model_validate_json(dumped)
        assert reloaded.model_dump() == section.model_dump()

    def test_default_pareto_shape(self) -> None:
        section = ObjectivesSection()
        assert [(p.metric, p.direction) for p in section.pareto] == [
            ("tcp_throughput", "maximize"),
            ("udp_throughput", "maximize"),
            ("tcp_retransmit_rate", "minimize"),
            ("udp_loss_rate", "minimize"),
            ("udp_jitter", "minimize"),
            ("rps", "maximize"),
            ("latency_p50", "minimize"),
            ("latency_p90", "minimize"),
            ("latency_p99", "minimize"),
        ]

    def test_obsolete_memory_metric_rejected(self) -> None:
        with pytest.raises(Exception, match="memory"):
            ObjectivesSection.model_validate(
                {
                    "pareto": [{"metric": "memory", "direction": "minimize"}],
                    "constraints": [],
                    "recommendationWeights": {},
                },
            )

    def test_default_constraints_include_retransmit_rate(self) -> None:
        section = ObjectivesSection()
        assert "tcp_retransmit_rate <= 1000" in section.constraints

    def test_weight_on_maximize_metric_accepted(self) -> None:
        section = ObjectivesSection(
            recommendation_weights={"tcp_throughput": 2.0},
        )
        assert section.recommendation_weights["tcp_throughput"] == pytest.approx(2.0)

    def test_zero_weight_on_maximize_metric_accepted(self) -> None:
        section = ObjectivesSection(
            recommendation_weights={"tcp_throughput": 0.0},
        )
        assert section.recommendation_weights["tcp_throughput"] == pytest.approx(0.0)

    def test_weight_on_maximize_metric_not_in_pareto_rejected(self) -> None:
        with pytest.raises(ValueError, match="not in pareto objectives"):
            ObjectivesSection(
                pareto=[
                    ParetoObjective(metric="udp_throughput", direction="maximize"),
                ],
                recommendation_weights={"tcp_throughput": 1.0},
            )

    def test_weight_on_unknown_metric_rejected(self) -> None:
        with pytest.raises(ValueError, match="not in pareto objectives"):
            ObjectivesSection(
                recommendation_weights={
                    "nosuch": 0.1,
                    "tcp_retransmit_rate": 0.3,
                },
            )

    @pytest.mark.parametrize("metric", ["tcp_retransmit_rate", "tcp_throughput"])
    def test_negative_weight_rejected(self, metric: str) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ObjectivesSection(
                recommendation_weights={metric: -0.1},
            )

    def test_malformed_constraint_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            ObjectivesSection(constraints=["tcp_retransmit_rate !! 0.1"])

    def test_unknown_constraint_metric_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            ObjectivesSection(constraints=["latency_p42 <= 50"])

    def test_empty_pareto_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            ObjectivesSection(pareto=[])

    def test_yaml_alias_recommendation_weights(self) -> None:
        section = ObjectivesSection.model_validate(
            {
                "recommendationWeights": {
                    "tcp_retransmit_rate": 0.4,
                    "udp_jitter": 0.2,
                },
            },
        )
        assert section.recommendation_weights == {
            "tcp_retransmit_rate": 0.4,
            "udp_jitter": 0.2,
        }

    def test_pareto_with_custom_metrics(self) -> None:
        section = ObjectivesSection(
            pareto=[
                ParetoObjective(metric="tcp_throughput", direction="maximize"),
                ParetoObjective(metric="udp_jitter", direction="minimize"),
            ],
            constraints=[],
            recommendation_weights={"udp_jitter": 0.5},
        )
        assert len(section.pareto) == 2
        assert section.recommendation_weights == {"udp_jitter": 0.5}


class TestObjectivesSectionConstraintUnits:
    """k8s-style suffix normalization in ``ObjectivesSection.constraints``."""

    def test_default_constraints_self_stable(self) -> None:
        """Defaults are already in post-normalization form.

        Pins that validation is a no-op on the shipped defaults so the
        ``_DEFAULT_CONSTRAINTS`` literal stays in sync with the output
        shape produced by :func:`_normalize_constraint`.
        """
        expected = [
            "tcp_throughput >= 1000000",
            "udp_throughput >= 1000000",
            "tcp_retransmit_rate <= 1000",
            "udp_loss_rate <= 0.05",
            "rps >= 100",
            "latency_p99 <= 1",
            "udp_jitter <= 0.01",
            "latency_p50 <= 0.1",
            "latency_p90 <= 0.5",
        ]
        section = ObjectivesSection()
        assert section.constraints == expected

    def test_iec_suffix_normalized(self) -> None:
        section = ObjectivesSection(constraints=["tcp_throughput >= 1Gi"])
        assert section.constraints == ["tcp_throughput >= 1073741824"]

    def test_milli_suffix_normalized(self) -> None:
        section = ObjectivesSection(constraints=["udp_jitter <= 500m"])
        assert section.constraints == ["udp_jitter <= 0.5"]

    def test_nano_suffix_normalized(self) -> None:
        section = ObjectivesSection(constraints=["tcp_retransmit_rate <= 1n"])
        assert section.constraints == ["tcp_retransmit_rate <= 1e-09"]

    def test_milli_rate_normalized(self) -> None:
        section = ObjectivesSection(constraints=["tcp_retransmit_rate <= 1m"])
        assert section.constraints == ["tcp_retransmit_rate <= 0.001"]

    def test_fractional_iec_normalized(self) -> None:
        section = ObjectivesSection(constraints=["tcp_throughput >= 1.5Gi"])
        assert section.constraints == ["tcp_throughput >= 1610612736"]

    def test_decimal_exponent_suffix_normalized(self) -> None:
        section = ObjectivesSection(constraints=["tcp_throughput >= 1e9"])
        assert section.constraints == ["tcp_throughput >= 1000000000"]

    def test_mixed_list_preserves_bare(self) -> None:
        section = ObjectivesSection(
            constraints=["tcp_throughput >= 1Gi", "rps >= 200"],
        )
        assert section.constraints == [
            "tcp_throughput >= 1073741824",
            "rps >= 200",
        ]

    def test_unknown_suffix_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            ObjectivesSection(constraints=["cpu <= 5Xi"])

    def test_capital_k_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            ObjectivesSection(constraints=["cpu <= 1K"])

    def test_mixed_form_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            ObjectivesSection(constraints=["tcp_throughput >= 1e3Ki"])

    def test_normalization_idempotent(self) -> None:
        first = ObjectivesSection(
            constraints=["tcp_throughput >= 1Gi", "udp_jitter <= 500m"],
        )
        second = ObjectivesSection(constraints=list(first.constraints))
        assert second.constraints == first.constraints

    def test_latency_milli_suffix_ergonomics(self) -> None:
        """``500m`` on a time-valued metric parses as 500 ms = 0.5 s."""
        section = ObjectivesSection(constraints=["latency_p99 <= 500m"])
        assert section.constraints == ["latency_p99 <= 0.5"]


def test_experiment_config_default_objectives() -> None:
    config = ExperimentConfig.model_validate({
        "nodes": {"sources": ["kmain07"], "target": "kmain08"},
    })
    assert config.objectives == ObjectivesSection()


def test_objectives_memory_cost_weight_default() -> None:
    """``memory_cost_weight`` ships at 0.1 (on, gentle)."""
    section = ObjectivesSection()
    assert section.memory_cost_weight == pytest.approx(0.1)


def test_objectives_memory_cost_weight_camel_case_alias() -> None:
    """YAML users spell the field ``memoryCostWeight``."""
    section = ObjectivesSection.model_validate({"memoryCostWeight": 0.25})
    assert section.memory_cost_weight == pytest.approx(0.25)
    dumped = section.model_dump(by_alias=True)
    assert dumped["memoryCostWeight"] == pytest.approx(0.25)


def test_objectives_memory_cost_weight_rejects_negative() -> None:
    """Negative weights are rejected by the ``ge=0.0`` Field constraint."""
    with pytest.raises(ValueError, match="greater than or equal"):
        ObjectivesSection(memory_cost_weight=-0.1)


def test_objectives_default_memory_cost_weight_when_omitted() -> None:
    """``memory_cost_weight`` defaults when the YAML omits the key."""
    data = {
        "pareto": [{"metric": "tcp_throughput", "direction": "maximize"}],
        "constraints": [],
        "recommendationWeights": {},
    }
    section = ObjectivesSection.model_validate(data)
    assert section.memory_cost_weight == pytest.approx(0.1)
