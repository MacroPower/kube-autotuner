from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from kube_autotuner.k8s.client import K8sApiError
from kube_autotuner.k8s.lease import LEASE_TTL_SECONDS, LeaseHeldError, NodeLease


def _api_error(reason: str, status: int = 409, message: str = "") -> K8sApiError:
    return K8sApiError(
        op="create lease/kube-autotuner-lock-kmain07",
        status=status,
        reason=reason,
        message=message or reason,
    )


@pytest.fixture
def client():
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
    def test_create_succeeds(self, client):
        """When no lease exists, create succeeds atomically."""
        lease = NodeLease("kmain07", namespace="default", holder="me", client=client)
        lease.acquire()

        client.create.assert_called_once()
        assert lease._acquired is True

    def test_takeover_expired(self, client):
        """When an expired lease exists, replace with resourceVersion."""
        client.create.side_effect = _api_error("AlreadyExists")
        expired_time = datetime.now(UTC) - timedelta(seconds=LEASE_TTL_SECONDS + 60)
        client.get_json.return_value = _existing_lease(renew_time=expired_time)

        lease = NodeLease("kmain07", namespace="default", holder="me", client=client)
        lease.acquire()

        client.replace.assert_called_once()
        assert lease._acquired is True

    def test_reentrant_same_holder(self, client):
        """Same holder can re-acquire an active lease."""
        client.create.side_effect = _api_error("AlreadyExists")
        client.get_json.return_value = _existing_lease(holder="me")

        lease = NodeLease("kmain07", namespace="default", holder="me", client=client)
        lease.acquire()

        client.replace.assert_called_once()
        assert lease._acquired is True

    def test_raises_when_held_by_other(self, client):
        """Active lease held by another process raises LeaseHeldError."""
        client.create.side_effect = _api_error("AlreadyExists")
        client.get_json.return_value = _existing_lease(holder="kube-autotuner-other")

        lease = NodeLease("kmain07", namespace="default", holder="me", client=client)
        with pytest.raises(LeaseHeldError, match="kube-autotuner-other"):
            lease.acquire()

        assert lease._acquired is False

    def test_retry_on_conflict(self, client):
        """Conflict on replace retries once."""
        client.create.side_effect = _api_error("AlreadyExists")
        expired_time = datetime.now(UTC) - timedelta(seconds=LEASE_TTL_SECONDS + 60)
        client.get_json.return_value = _existing_lease(renew_time=expired_time)
        client.replace.side_effect = [
            _api_error("Conflict"),
            None,
        ]

        lease = NodeLease("kmain07", namespace="default", holder="me", client=client)
        lease.acquire()

        assert client.replace.call_count == 2
        assert lease._acquired is True

    def test_deleted_between_create_and_get(self, client):
        """Lease deleted between create attempt and get -> retry create."""
        client.create.side_effect = [
            _api_error("AlreadyExists"),
            None,
        ]
        client.get_json.return_value = None

        lease = NodeLease("kmain07", namespace="default", holder="me", client=client)
        lease.acquire()

        assert client.create.call_count == 2
        assert lease._acquired is True

    def test_non_already_exists_error_propagates(self, client):
        """Non-AlreadyExists errors from create propagate."""
        client.create.side_effect = _api_error(
            "ServerTimeout", status=504, message="connection refused"
        )

        lease = NodeLease("kmain07", namespace="default", holder="me", client=client)
        with pytest.raises(K8sApiError, match="connection refused"):
            lease.acquire()


class TestRelease:
    def test_release_deletes(self, client):
        lease = NodeLease("kmain07", namespace="default", holder="me", client=client)
        lease._acquired = True
        lease.release()

        client.delete.assert_called_once_with(
            "lease", "kube-autotuner-lock-kmain07", "default"
        )
        assert lease._acquired is False

    def test_release_noop_when_not_acquired(self, client):
        lease = NodeLease("kmain07", namespace="default", holder="me", client=client)
        lease.release()
        client.delete.assert_not_called()


class TestContextManager:
    def test_acquire_and_release(self, client):
        with NodeLease(
            "kmain07", namespace="default", holder="me", client=client
        ) as lease:
            assert lease._acquired is True
        client.delete.assert_called_once()

    def test_release_on_exception(self, client):
        with (
            pytest.raises(RuntimeError),
            NodeLease("kmain07", namespace="default", holder="me", client=client),
        ):
            raise RuntimeError("boom")
        client.delete.assert_called_once()

    def test_exit_suppresses_release_error(self, client):
        """Release failure in __exit__ should not mask the original exception."""
        client.delete.side_effect = _api_error("Timeout", status=504, message="timeout")
        with (
            pytest.raises(RuntimeError, match="boom"),
            NodeLease("kmain07", namespace="default", holder="me", client=client),
        ):
            raise RuntimeError("boom")
        client.delete.assert_called_once()

    def test_exit_suppresses_release_error_no_original(self, client):
        """Release failure without an original exception should not propagate."""
        client.delete.side_effect = _api_error("Timeout", status=504, message="timeout")
        with NodeLease("kmain07", namespace="default", holder="me", client=client):
            pass


class TestHolderDefault:
    def test_default_holder_uses_kube_autotuner_prefix(self, client):
        """Auto-generated holder IDs use the ``kube-autotuner-`` prefix."""
        lease = NodeLease("kmain07", client=client)
        assert lease.holder.startswith("kube-autotuner-")
