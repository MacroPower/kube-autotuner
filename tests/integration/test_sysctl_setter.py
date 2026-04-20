"""Backend-agnostic integration tests.

Dispatches via ``KUBE_AUTOTUNER_SYSCTL_BACKEND`` (``real``, ``talos``, or
``fake``). Read tests run everywhere. Write tests require a backend that can
actually write host sysctls. On Talos Docker this means the ``talos`` backend
(``talosctl patch mc``); the ``real`` privileged-pod backend works only on
bare-metal Talos where the pod runs in the init user namespace. Tests that
depend on working writes are marked ``requires_real_sysctl_write`` so they
can be opted out via ``KUBE_AUTOTUNER_ALLOW_SYSCTL_SKIP=1``.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest

from kube_autotuner.sysctl.setter import make_sysctl_setter_from_env

if TYPE_CHECKING:
    from kube_autotuner.k8s.client import Kubectl
    from kube_autotuner.sysctl.backend import SysctlBackend

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(120),
]

SAFE_PARAMS = ["net.core.rmem_max", "net.core.wmem_max"]


def _make_setter(
    kubectl: Kubectl, node_names: dict[str, str], test_namespace: str
) -> SysctlBackend:
    return make_sysctl_setter_from_env(
        node=node_names["target"],
        namespace=test_namespace,
        kubectl=kubectl,
    )


def test_get_reads_sysctl_values(
    kubectl: Kubectl,
    node_names: dict[str, str],
    test_namespace: str,
    sysctls_available: None,  # noqa: ARG001 - activation fixture
) -> None:
    setter = _make_setter(kubectl, node_names, test_namespace)
    values = setter.get(SAFE_PARAMS)

    for param in SAFE_PARAMS:
        assert param in values
        assert values[param].isdigit(), (
            f"Expected numeric value for {param}, got {values[param]!r}"
        )


@pytest.mark.requires_real_sysctl_write
def test_apply_and_verify(
    kubectl: Kubectl, node_names: dict[str, str], test_namespace: str
) -> None:
    setter = _make_setter(kubectl, node_names, test_namespace)
    original = setter.get(["net.core.rmem_max"])

    test_value = "16777216"
    try:
        setter.apply({"net.core.rmem_max": test_value})
        current = setter.get(["net.core.rmem_max"])
        assert current["net.core.rmem_max"] == test_value
    finally:
        # Best-effort restore.
        with contextlib.suppress(Exception):
            setter.apply({"net.core.rmem_max": original["net.core.rmem_max"]})


@pytest.mark.requires_real_sysctl_write
def test_snapshot_and_restore(
    kubectl: Kubectl, node_names: dict[str, str], test_namespace: str
) -> None:
    setter = _make_setter(kubectl, node_names, test_namespace)
    original = setter.snapshot(["net.core.rmem_max"])

    setter.apply({"net.core.rmem_max": "8388608"})
    setter.restore(original)
    current = setter.get(["net.core.rmem_max"])
    assert current["net.core.rmem_max"] == original["net.core.rmem_max"]
