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
    from kube_autotuner.k8s.client import K8sClient
    from kube_autotuner.sysctl.backend import SysctlBackend

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(120),
]

SAFE_PARAMS = ["net.core.rmem_max", "net.core.wmem_max"]


def _make_setter(
    k8s_client: K8sClient, node_names: dict[str, str], test_namespace: str
) -> SysctlBackend:
    return make_sysctl_setter_from_env(
        node=node_names["target"],
        namespace=test_namespace,
        client=k8s_client,
    )


def test_get_reads_sysctl_values(
    k8s_client: K8sClient,
    node_names: dict[str, str],
    test_namespace: str,
    sysctls_available: None,  # noqa: ARG001 - activation fixture
) -> None:
    setter = _make_setter(k8s_client, node_names, test_namespace)
    values = setter.get(SAFE_PARAMS)

    for param in SAFE_PARAMS:
        assert param in values
        assert values[param].isdigit(), (
            f"Expected numeric value for {param}, got {values[param]!r}"
        )


def test_collect_host_state_records_tcp_metrics_rows(
    k8s_client: K8sClient,
    node_names: dict[str, str],
    test_namespace: str,
    sysctls_available: None,  # noqa: ARG001 - activation fixture
) -> None:
    """``tcp_metrics_rows`` must be read via netlink, not ``/proc/net/tcp_metrics``.

    Mainline Linux exposes the TCP metrics cache only through the
    ``tcp_metrics`` generic netlink family; ``/proc/net/tcp_metrics``
    does not exist, so the older ``wc -l`` form silently produced an
    ``NA`` fallback for every snapshot on every node. Regression guard
    for the fix that swapped the script line to
    ``ip tcp_metrics show | wc -l``.
    """
    setter = _make_setter(k8s_client, node_names, test_namespace)
    snapshot = setter.collect_host_state(iteration=None, phase="baseline")
    if snapshot is None:
        pytest.skip("backend does not support host-state collection")

    tcp_errors = [e for e in snapshot.errors if e.startswith("tcp_metrics")]
    assert not tcp_errors, (
        f"tcp_metrics section produced collection errors: {tcp_errors!r}"
    )
    assert "tcp_metrics_rows" in snapshot.metrics, (
        f"tcp_metrics_rows missing from metrics: {snapshot.metrics!r}"
    )
    rows = snapshot.metrics["tcp_metrics_rows"]
    assert isinstance(rows, int)
    assert rows >= 0


@pytest.mark.requires_real_sysctl_write
def test_apply_and_verify(
    k8s_client: K8sClient, node_names: dict[str, str], test_namespace: str
) -> None:
    setter = _make_setter(k8s_client, node_names, test_namespace)
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
    k8s_client: K8sClient, node_names: dict[str, str], test_namespace: str
) -> None:
    setter = _make_setter(k8s_client, node_names, test_namespace)
    original = setter.snapshot(["net.core.rmem_max"])

    setter.apply({"net.core.rmem_max": "8388608"})
    setter.restore(original)
    current = setter.get(["net.core.rmem_max"])
    assert current["net.core.rmem_max"] == original["net.core.rmem_max"]
