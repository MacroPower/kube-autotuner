"""Unit tests for :mod:`kube_autotuner.benchmark.patch`.

These exercises require a real ``kustomize`` binary on ``PATH`` because
the module shells out for patch evaluation. The devshell provisions it
via ``pkgs.kustomize``; the ``skipif`` guard keeps the tests green on
barebones machines.
"""

from __future__ import annotations

import shutil

import pytest
import yaml

from kube_autotuner.benchmark.patch import apply_patches
from kube_autotuner.benchmark.server_spec import build_server_yaml
from kube_autotuner.experiment import ExperimentConfigError, Patch, PatchTarget

pytestmark = pytest.mark.skipif(
    shutil.which("kustomize") is None,
    reason="kustomize binary required on PATH",
)


def test_no_patches_returns_input_unchanged():
    yaml_text = "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: foo\n"
    assert apply_patches(yaml_text, []) == yaml_text


def test_json_patch_adds_field():
    base = build_server_yaml("kmain08", [5201], "RequireDualStack")
    patch = Patch(
        target=PatchTarget(kind="Deployment"),
        patch=[
            {"op": "add", "path": "/spec/template/spec/hostNetwork", "value": True},
        ],
    )
    rendered = apply_patches(base, [patch])
    docs = list(yaml.safe_load_all(rendered))
    dep = next(d for d in docs if d["kind"] == "Deployment")
    assert dep["spec"]["template"]["spec"]["hostNetwork"] is True


def test_strategic_merge_by_container_name():
    """Kustomize SMP merges the containers list by ``name``."""
    base = build_server_yaml("kmain08", [5201, 5202], "RequireDualStack")
    patch = Patch(
        target=PatchTarget(kind="Deployment"),
        patch={
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "iperf3-server-5201",
                                "resources": {"limits": {"memory": "2Gi"}},
                            },
                        ],
                    },
                },
            },
        },
    )
    rendered = apply_patches(base, [patch])
    docs = list(yaml.safe_load_all(rendered))
    dep = next(d for d in docs if d["kind"] == "Deployment")
    containers = dep["spec"]["template"]["spec"]["containers"]
    names = [c["name"] for c in containers]
    assert "iperf3-server-5201" in names
    assert "iperf3-server-5202" in names  # untouched
    patched = next(c for c in containers if c["name"] == "iperf3-server-5201")
    assert patched["resources"]["limits"]["memory"] == "2Gi"


def test_tolerations_not_clobbered_by_unrelated_smp():
    """Patching only ``nodeSelector`` must not drop the existing tolerations."""
    base = build_server_yaml("kmain08", [5201], "RequireDualStack")
    patch = Patch(
        target=PatchTarget(kind="Deployment"),
        patch={
            "spec": {
                "template": {
                    "spec": {
                        "nodeSelector": {"kubernetes.io/os": "linux"},
                    },
                },
            },
        },
    )
    rendered = apply_patches(base, [patch])
    docs = list(yaml.safe_load_all(rendered))
    dep = next(d for d in docs if d["kind"] == "Deployment")
    spec = dep["spec"]["template"]["spec"]
    assert spec["tolerations"], "tolerations dropped by unrelated SMP patch"
    assert spec["nodeSelector"]["kubernetes.io/os"] == "linux"


def test_target_kind_scopes_patch_to_service_only():
    base = build_server_yaml("kmain08", [5201], "RequireDualStack")
    patch = Patch(
        target=PatchTarget(kind="Service"),
        patch={"spec": {"type": "NodePort"}},
    )
    rendered = apply_patches(base, [patch])
    docs = list(yaml.safe_load_all(rendered))
    svc = next(d for d in docs if d["kind"] == "Service")
    dep = next(d for d in docs if d["kind"] == "Deployment")
    assert svc["spec"]["type"] == "NodePort"
    # Deployment is untouched; spec.type is not a Deployment field.
    assert "type" not in dep["spec"]


def test_label_selector_target():
    base = build_server_yaml("kmain08", [5201], "RequireDualStack")
    patch = Patch(
        target=PatchTarget(labelSelector="app.kubernetes.io/name=iperf3-server"),
        patch=[
            {"op": "add", "path": "/metadata/labels/patched", "value": "yes"},
        ],
    )
    rendered = apply_patches(base, [patch])
    docs = list(yaml.safe_load_all(rendered))
    for d in docs:
        assert d["metadata"]["labels"].get("patched") == "yes"


def test_bad_json_patch_path_surfaces_kustomize_error():
    base = build_server_yaml("kmain08", [5201], "RequireDualStack")
    patch = Patch(
        target=PatchTarget(kind="Deployment"),
        patch=[
            {"op": "replace", "path": "/does/not/exist", "value": "x"},
        ],
    )
    with pytest.raises(ExperimentConfigError, match="kustomize build failed"):
        apply_patches(base, [patch])
