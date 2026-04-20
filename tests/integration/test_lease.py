"""Integration tests for NodeLease against a real Talos cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kube_autotuner.k8s.lease import LeaseHeldError, NodeLease

if TYPE_CHECKING:
    from kube_autotuner.k8s.client import K8sClient

pytestmark = pytest.mark.integration


def test_acquire_and_release(k8s_client: K8sClient, test_namespace: str) -> None:
    lease = NodeLease("test-node", namespace=test_namespace, client=k8s_client)
    lease.acquire()

    obj = k8s_client.get_json("lease", lease.lease_name, test_namespace)
    assert obj is not None
    assert obj["spec"]["holderIdentity"] == lease.holder

    lease.release()
    assert k8s_client.get_json("lease", lease.lease_name, test_namespace) is None


def test_context_manager(k8s_client: K8sClient, test_namespace: str) -> None:
    lease = NodeLease("test-node-cm", namespace=test_namespace, client=k8s_client)
    with lease:
        obj = k8s_client.get_json("lease", lease.lease_name, test_namespace)
        assert obj is not None

    assert k8s_client.get_json("lease", lease.lease_name, test_namespace) is None


def test_same_holder_reacquire(k8s_client: K8sClient, test_namespace: str) -> None:
    lease = NodeLease(
        "test-node-reacq",
        namespace=test_namespace,
        holder="holder-a",
        client=k8s_client,
    )
    lease.acquire()

    # Same holder acquires again -- should succeed.
    lease2 = NodeLease(
        "test-node-reacq",
        namespace=test_namespace,
        holder="holder-a",
        client=k8s_client,
    )
    lease2.acquire()

    # Clean up.
    lease2.release()


def test_held_by_other_raises(k8s_client: K8sClient, test_namespace: str) -> None:
    lease_a = NodeLease(
        "test-node-held",
        namespace=test_namespace,
        holder="holder-a",
        client=k8s_client,
    )
    lease_a.acquire()

    lease_b = NodeLease(
        "test-node-held",
        namespace=test_namespace,
        holder="holder-b",
        client=k8s_client,
    )
    with pytest.raises(LeaseHeldError):
        lease_b.acquire()

    lease_a.release()
