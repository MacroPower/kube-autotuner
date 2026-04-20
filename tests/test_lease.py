from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from kube_autotuner.k8s.client import KubectlError
from kube_autotuner.k8s.lease import LEASE_TTL_SECONDS, LeaseHeldError, NodeLease


@pytest.fixture
def kubectl():
    return MagicMock()


def _existing_lease(
    holder: str = "kube-autotuner-other",
    renew_time: datetime | None = None,
    ttl: int = LEASE_TTL_SECONDS,
    resource_version: str = "12345",
) -> dict:
    if renew_time is None:
        renew_time = datetime.now(UTC)
    return {
        "metadata": {
            "name": "kube-autotuner-lock-kmain07",
            "resourceVersion": resource_version,
        },
        "spec": {
            "holderIdentity": holder,
            "leaseDurationSeconds": ttl,
            "renewTime": renew_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        },
    }


class TestAcquire:
    def test_create_succeeds(self, kubectl):
        """When no lease exists, create succeeds atomically."""
        lease = NodeLease("kmain07", namespace="default", holder="me", kubectl=kubectl)
        lease.acquire()

        kubectl.create.assert_called_once()
        assert lease._acquired is True

    def test_takeover_expired(self, kubectl):
        """When an expired lease exists, replace with resourceVersion."""
        kubectl.create.side_effect = KubectlError(["kubectl"], 1, "AlreadyExists")
        expired_time = datetime.now(UTC) - timedelta(seconds=LEASE_TTL_SECONDS + 60)
        kubectl.get_json.return_value = _existing_lease(renew_time=expired_time)

        lease = NodeLease("kmain07", namespace="default", holder="me", kubectl=kubectl)
        lease.acquire()

        kubectl.replace.assert_called_once()
        assert lease._acquired is True

    def test_reentrant_same_holder(self, kubectl):
        """Same holder can re-acquire an active lease."""
        kubectl.create.side_effect = KubectlError(["kubectl"], 1, "AlreadyExists")
        kubectl.get_json.return_value = _existing_lease(holder="me")

        lease = NodeLease("kmain07", namespace="default", holder="me", kubectl=kubectl)
        lease.acquire()

        kubectl.replace.assert_called_once()
        assert lease._acquired is True

    def test_raises_when_held_by_other(self, kubectl):
        """Active lease held by another process raises LeaseHeldError."""
        kubectl.create.side_effect = KubectlError(["kubectl"], 1, "AlreadyExists")
        kubectl.get_json.return_value = _existing_lease(holder="kube-autotuner-other")

        lease = NodeLease("kmain07", namespace="default", holder="me", kubectl=kubectl)
        with pytest.raises(LeaseHeldError, match="kube-autotuner-other"):
            lease.acquire()

        assert lease._acquired is False

    def test_retry_on_conflict(self, kubectl):
        """Conflict on replace retries once."""
        kubectl.create.side_effect = KubectlError(["kubectl"], 1, "AlreadyExists")
        expired_time = datetime.now(UTC) - timedelta(seconds=LEASE_TTL_SECONDS + 60)
        kubectl.get_json.return_value = _existing_lease(renew_time=expired_time)
        kubectl.replace.side_effect = [
            KubectlError(["kubectl"], 1, "Conflict"),
            None,
        ]

        lease = NodeLease("kmain07", namespace="default", holder="me", kubectl=kubectl)
        lease.acquire()

        assert kubectl.replace.call_count == 2
        assert lease._acquired is True

    def test_deleted_between_create_and_get(self, kubectl):
        """Lease deleted between create attempt and get -> retry create."""
        kubectl.create.side_effect = [
            KubectlError(["kubectl"], 1, "AlreadyExists"),
            None,
        ]
        kubectl.get_json.return_value = None

        lease = NodeLease("kmain07", namespace="default", holder="me", kubectl=kubectl)
        lease.acquire()

        assert kubectl.create.call_count == 2
        assert lease._acquired is True

    def test_non_already_exists_error_propagates(self, kubectl):
        """Non-AlreadyExists errors from create propagate."""
        kubectl.create.side_effect = KubectlError(["kubectl"], 1, "connection refused")

        lease = NodeLease("kmain07", namespace="default", holder="me", kubectl=kubectl)
        with pytest.raises(KubectlError, match="connection refused"):
            lease.acquire()


class TestRelease:
    def test_release_deletes(self, kubectl):
        lease = NodeLease("kmain07", namespace="default", holder="me", kubectl=kubectl)
        lease._acquired = True
        lease.release()

        kubectl.delete.assert_called_once_with(
            "lease", "kube-autotuner-lock-kmain07", "default"
        )
        assert lease._acquired is False

    def test_release_noop_when_not_acquired(self, kubectl):
        lease = NodeLease("kmain07", namespace="default", holder="me", kubectl=kubectl)
        lease.release()
        kubectl.delete.assert_not_called()


class TestContextManager:
    def test_acquire_and_release(self, kubectl):
        with NodeLease(
            "kmain07", namespace="default", holder="me", kubectl=kubectl
        ) as lease:
            assert lease._acquired is True
        kubectl.delete.assert_called_once()

    def test_release_on_exception(self, kubectl):
        with (
            pytest.raises(RuntimeError),
            NodeLease("kmain07", namespace="default", holder="me", kubectl=kubectl),
        ):
            raise RuntimeError("boom")
        kubectl.delete.assert_called_once()

    def test_exit_suppresses_release_error(self, kubectl):
        """Release failure in __exit__ should not mask the original exception."""
        kubectl.delete.side_effect = KubectlError(["kubectl"], 1, "timeout")
        with (
            pytest.raises(RuntimeError, match="boom"),
            NodeLease("kmain07", namespace="default", holder="me", kubectl=kubectl),
        ):
            raise RuntimeError("boom")
        kubectl.delete.assert_called_once()

    def test_exit_suppresses_release_error_no_original(self, kubectl):
        """Release failure without an original exception should not propagate."""
        kubectl.delete.side_effect = KubectlError(["kubectl"], 1, "timeout")
        with NodeLease("kmain07", namespace="default", holder="me", kubectl=kubectl):
            pass


class TestHolderDefault:
    def test_default_holder_uses_kube_autotuner_prefix(self, kubectl):
        """Auto-generated holder IDs use the ``kube-autotuner-`` prefix."""
        lease = NodeLease("kmain07", kubectl=kubectl)
        assert lease.holder.startswith("kube-autotuner-")
