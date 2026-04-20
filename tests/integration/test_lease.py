"""Integration tests for NodeLease against a real Talos cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kube_autotuner.k8s.lease import LeaseHeldError, NodeLease

if TYPE_CHECKING:
    from kube_autotuner.k8s.client import Kubectl

pytestmark = pytest.mark.integration


def test_acquire_and_release(kubectl: Kubectl, test_namespace: str) -> None:
    lease = NodeLease("test-node", namespace=test_namespace, kubectl=kubectl)
    lease.acquire()

    obj = kubectl.get_json("lease", lease.lease_name, test_namespace)
    assert obj is not None
    assert obj["spec"]["holderIdentity"] == lease.holder

    lease.release()
    assert kubectl.get_json("lease", lease.lease_name, test_namespace) is None


def test_context_manager(kubectl: Kubectl, test_namespace: str) -> None:
    lease = NodeLease("test-node-cm", namespace=test_namespace, kubectl=kubectl)
    with lease:
        obj = kubectl.get_json("lease", lease.lease_name, test_namespace)
        assert obj is not None

    assert kubectl.get_json("lease", lease.lease_name, test_namespace) is None


def test_same_holder_reacquire(kubectl: Kubectl, test_namespace: str) -> None:
    lease = NodeLease(
        "test-node-reacq",
        namespace=test_namespace,
        holder="holder-a",
        kubectl=kubectl,
    )
    lease.acquire()

    # Same holder acquires again -- should succeed.
    lease2 = NodeLease(
        "test-node-reacq",
        namespace=test_namespace,
        holder="holder-a",
        kubectl=kubectl,
    )
    lease2.acquire()

    # Clean up.
    lease2.release()


def test_held_by_other_raises(kubectl: Kubectl, test_namespace: str) -> None:
    lease_a = NodeLease(
        "test-node-held",
        namespace=test_namespace,
        holder="holder-a",
        kubectl=kubectl,
    )
    lease_a.acquire()

    lease_b = NodeLease(
        "test-node-held",
        namespace=test_namespace,
        holder="holder-b",
        kubectl=kubectl,
    )
    with pytest.raises(LeaseHeldError):
        lease_b.acquire()

    lease_a.release()
