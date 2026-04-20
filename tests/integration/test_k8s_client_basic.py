"""Integration tests for K8sClient against a real Talos cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kube_autotuner.k8s.client import K8sApiError

if TYPE_CHECKING:
    from kube_autotuner.k8s.client import K8sClient

pytestmark = pytest.mark.integration

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


def test_apply_wait_delete_pod(k8s_client: K8sClient, test_namespace: str) -> None:
    k8s_client.apply(BUSYBOX_POD_YAML, test_namespace)
    k8s_client.wait(
        "pod", "test-busybox", "condition=Ready", test_namespace, timeout=60
    )
    k8s_client.delete("pod", "test-busybox", test_namespace)


def test_get_json_returns_none_for_missing(
    k8s_client: K8sClient, test_namespace: str
) -> None:
    assert k8s_client.get_json("lease", "nonexistent", test_namespace) is None


def test_delete_missing_raises_when_not_ignoring(
    k8s_client: K8sClient, test_namespace: str
) -> None:
    # Default ignore_not_found=True swallows the 404.
    k8s_client.delete("pod", "does-not-exist", test_namespace)
    with pytest.raises(K8sApiError) as excinfo:
        k8s_client.delete(
            "pod", "does-not-exist", test_namespace, ignore_not_found=False
        )
    assert excinfo.value.reason == "NotFound"
