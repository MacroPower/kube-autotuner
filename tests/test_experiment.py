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
mode: baseline
nodes:
  sources: [kmain07]
  target: kmain08
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    assert exp.mode == "baseline"
    assert exp.nodes.sources == ["kmain07"]
    assert exp.nodes.target == "kmain08"
    assert exp.iperf.client.extra_args == []
    assert exp.patches == []


def test_full_optimize_roundtrips(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
mode: optimize
nodes:
  sources: [kmain07, kmain09]
  target: kmain08
  hardwareClass: 10g
benchmark:
  duration: 30
  iterations: 3
  modes: [tcp]
optimize:
  nTrials: 50
  nSobol: 15
  applySource: true
  paramSpace:
    - {name: net.core.rmem_max, paramType: int, values: [4194304, 67108864]}
iperf:
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


def test_fixture_yaml_roundtrips():
    """The shipped fixture must load cleanly and survive model validation."""
    exp = ExperimentConfig.from_yaml(FIXTURE_YAML)
    assert exp.mode == "optimize"
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
    assert exp.output == "out/results.jsonl"


def test_multi_doc_yaml_raises(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
mode: baseline
nodes:
  sources: [a]
  target: b
---
mode: baseline
""",
    )
    with pytest.raises(ExperimentConfigError, match="multi-document YAML"):
        ExperimentConfig.from_yaml(path)


def test_typo_field_raises(tmp_path: Path):
    """extra='forbid' catches typos at parse time."""
    path = _write(
        tmp_path,
        """\
mode: baseline
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


def test_invalid_yaml_raises(tmp_path: Path):
    path = _write(tmp_path, "mode: baseline\n  sources: oops\n: : :\n")
    with pytest.raises(ExperimentConfigError, match="invalid YAML"):
        ExperimentConfig.from_yaml(path)


# --- mode-specific validators -------------------------------------------


def test_optimize_without_section_raises(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
mode: optimize
nodes:
  sources: [kmain07]
  target: kmain08
""",
    )
    with pytest.raises(ExperimentConfigError, match="requires `optimize:`"):
        ExperimentConfig.from_yaml(path)


def test_trial_without_sysctls_raises(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
mode: trial
nodes:
  sources: [kmain07]
  target: kmain08
""",
    )
    with pytest.raises(ExperimentConfigError, match="requires a `trial:` section"):
        ExperimentConfig.from_yaml(path)


def test_trial_with_empty_sysctls_raises(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
mode: trial
nodes:
  sources: [a]
  target: b
trial:
  sysctls: {}
""",
    )
    with pytest.raises(ExperimentConfigError, match="non-empty"):
        ExperimentConfig.from_yaml(path)


def test_trial_with_sysctls_accepted(tmp_path: Path):
    """Regression guard for mode=trial happy path."""
    path = _write(
        tmp_path,
        """\
mode: trial
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


def test_n_sobol_greater_than_n_trials_raises(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
mode: optimize
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
mode: optimize
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


# --- projections ---------------------------------------------------------


def test_effective_param_space_default():
    """When unset, falls back to the canonical PARAM_SPACE."""
    exp = ExperimentConfig.model_validate({
        "mode": "baseline",
        "nodes": {"sources": ["a"], "target": "b"},
    })
    assert exp.effective_param_space() is PARAM_SPACE


def test_effective_param_space_override():
    exp = ExperimentConfig.model_validate({
        "mode": "optimize",
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
        "mode": "baseline",
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
mode: baseline
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
mode: baseline
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
mode: baseline
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
    """--windowsize-hint is not --window; must pass."""
    path = _write(
        tmp_path,
        """\
mode: baseline
nodes:
  sources: [a]
  target: b
iperf:
  client:
    extraArgs: ["--windowsize-hint"]
""",
    )
    exp = ExperimentConfig.from_yaml(path)
    assert exp._check_denylists().passed


def test_server_denylist(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
mode: baseline
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
        "mode": "trial",
        "nodes": {"sources": ["a"], "target": "b"},
        "trial": {"sysctls": {"NOT-VALID": "1"}},
    })
    result = exp._check_sysctl_names()
    assert not result.passed
    assert "invalid sysctl name" in result.detail


def test_sysctl_name_regex_happy_path():
    exp = ExperimentConfig.model_validate({
        "mode": "trial",
        "nodes": {"sources": ["a"], "target": "b"},
        "trial": {"sysctls": {"net.core.rmem_max": 67108864}},
    })
    assert exp._check_sysctl_names().passed


def test_patch_metadata_namespace_rejected(tmp_path: Path):
    path = _write(
        tmp_path,
        """\
mode: baseline
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
mode: baseline
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
mode: baseline
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
mode: baseline
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
        "mode": "baseline",
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
        "mode": "baseline",
        "nodes": {"sources": ["a"], "target": "b"},
    })
    result = exp._check_kustomize_available()
    assert result.passed
    assert "skipped" in result.detail


def test_check_kustomize_available_routes_through_run_tool(monkeypatch):
    """The probe routes through kube_autotuner.subproc.run_tool, not bare subprocess."""
    exp = ExperimentConfig.model_validate({
        "mode": "baseline",
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
        "mode": "baseline",
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
        "mode": "baseline",
        "nodes": {"sources": ["a"], "target": "b"},
    })
    client = MagicMock()
    client.get_node_zone.side_effect = RuntimeError("NotFound")
    result = exp._check_nodes_exist(client)
    assert not result.passed
    assert "not found or unreachable" in result.detail


def test_check_nodes_exist_happy_path():
    exp = ExperimentConfig.model_validate({
        "mode": "baseline",
        "nodes": {"sources": ["a", "extra"], "target": "b"},
    })
    client = MagicMock()
    client.get_node_zone.return_value = "zone-a"
    assert exp._check_nodes_exist(client).passed
    assert client.get_node_zone.call_count == 3


def test_output_path_creates_parent_dir(tmp_path: Path):
    out = tmp_path / "subdir" / "results.jsonl"
    exp = ExperimentConfig.model_validate({
        "mode": "baseline",
        "nodes": {"sources": ["a"], "target": "b"},
        "output": str(out),
    })
    result = exp._check_output_path()
    assert result.passed
    assert out.parent.exists()
    # Must not touch the output file itself -- avoid leaking empty files
    # when later preflight steps fail.
    assert not out.exists()


def test_output_path_parent_not_writable(tmp_path: Path, monkeypatch):
    out = tmp_path / "results.jsonl"
    exp = ExperimentConfig.model_validate({
        "mode": "baseline",
        "nodes": {"sources": ["a"], "target": "b"},
        "output": str(out),
    })
    monkeypatch.setattr(os, "access", lambda _p, _mode: False)
    result = exp._check_output_path()
    assert not result.passed
    assert "not writable" in result.detail


# --- preflight orchestration --------------------------------------------


def test_preflight_orchestration_smoke():
    """preflight() runs every check in order and returns all results."""
    exp = ExperimentConfig.model_validate({
        "mode": "baseline",
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
    out = tmp_path / "results.jsonl"
    exp = ExperimentConfig.model_validate({
        "mode": "baseline",
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
        "mode": "baseline",
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
        "mode": "baseline",
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
        "mode": "baseline",
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


class TestObjectivesSection:
    def test_defaults_round_trip(self) -> None:
        section = ObjectivesSection()
        dumped = section.model_dump_json()
        reloaded = ObjectivesSection.model_validate_json(dumped)
        assert reloaded.model_dump() == section.model_dump()

    def test_default_pareto_shape(self) -> None:
        section = ObjectivesSection()
        assert [(p.metric, p.direction) for p in section.pareto] == [
            ("throughput", "maximize"),
            ("cpu", "minimize"),
            ("retransmits", "minimize"),
            ("memory", "minimize"),
        ]

    def test_weight_on_maximize_metric_rejected(self) -> None:
        with pytest.raises(ValueError, match="maximize-direction"):
            ObjectivesSection(
                recommendation_weights={"throughput": 1.0},
            )

    def test_weight_on_unknown_metric_rejected(self) -> None:
        with pytest.raises(ValueError, match="not in pareto objectives"):
            ObjectivesSection(
                recommendation_weights={"nosuch": 0.1, "cpu": 0.15, "memory": 0.15},
            )

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ObjectivesSection(
                recommendation_weights={"cpu": -0.1, "memory": 0.15},
            )

    def test_malformed_constraint_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            ObjectivesSection(constraints=["cpu !! 200"])

    def test_unknown_constraint_metric_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            ObjectivesSection(constraints=["jitter <= 50"])

    def test_empty_pareto_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            ObjectivesSection(pareto=[])

    def test_yaml_alias_recommendation_weights(self) -> None:
        section = ObjectivesSection.model_validate(
            {"recommendationWeights": {"cpu": 0.2, "memory": 0.2, "retransmits": 0.4}},
        )
        assert section.recommendation_weights == {
            "cpu": 0.2,
            "memory": 0.2,
            "retransmits": 0.4,
        }

    def test_pareto_with_custom_metrics(self) -> None:
        section = ObjectivesSection(
            pareto=[
                ParetoObjective(metric="throughput", direction="maximize"),
                ParetoObjective(metric="memory", direction="minimize"),
            ],
            constraints=[],
            recommendation_weights={"memory": 0.5},
        )
        assert len(section.pareto) == 2
        assert section.recommendation_weights == {"memory": 0.5}


def test_experiment_config_default_objectives() -> None:
    config = ExperimentConfig.model_validate({
        "mode": "baseline",
        "nodes": {"sources": ["kmain07"], "target": "kmain08"},
    })
    assert config.objectives == ObjectivesSection()
