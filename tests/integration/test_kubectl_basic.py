"""Integration tests for Kubectl wrapper against a real Talos cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kube_autotuner.k8s.client import KubectlError

if TYPE_CHECKING:
    from kube_autotuner.k8s.client import Kubectl

pytestmark = pytest.mark.integration

CONFIGMAP_YAML = """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: test-cm
data:
  key: value
"""

BUSYBOX_POD_YAML = """\
apiVersion: v1
kind: Pod
metadata:
  name: test-busybox
spec:
  containers:
    - name: busybox
      image: busybox:latest
      command: ["sleep", "300"]
  restartPolicy: Never
"""


def test_apply_get_delete(kubectl: Kubectl, test_namespace: str) -> None:
    kubectl.apply(CONFIGMAP_YAML, test_namespace)

    obj = kubectl.get_json("configmap", "test-cm", test_namespace)
    assert obj is not None
    assert obj["data"]["key"] == "value"

    kubectl.delete("configmap", "test-cm", test_namespace)
    assert kubectl.get_json("configmap", "test-cm", test_namespace) is None


def test_get_json_returns_none_for_missing(
    kubectl: Kubectl, test_namespace: str
) -> None:
    assert kubectl.get_json("configmap", "nonexistent", test_namespace) is None


def test_wait_for_pod_ready(kubectl: Kubectl, test_namespace: str) -> None:
    kubectl.apply(BUSYBOX_POD_YAML, test_namespace)
    kubectl.wait("pod", "test-busybox", "condition=Ready", test_namespace, timeout=60)

    obj = kubectl.get_json("pod", "test-busybox", test_namespace)
    assert obj is not None
    phase = obj["status"]["phase"]
    assert phase == "Running"


def test_kubectl_error_on_bad_resource(kubectl: Kubectl, test_namespace: str) -> None:
    with pytest.raises(KubectlError):
        kubectl._run(["get", "notarealresource", "-n", test_namespace])
