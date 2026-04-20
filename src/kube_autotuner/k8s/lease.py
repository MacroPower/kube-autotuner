"""Kubernetes Lease-based node lock for exclusive sysctl tuning.

The lock is stored as a ``coordination.k8s.io/v1`` Lease resource;
mutual exclusion relies on the API server's compare-and-swap semantics
rather than any extra coordination service.
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from typing import TYPE_CHECKING, Self
from uuid import uuid4

from kube_autotuner.k8s.client import Kubectl, KubectlError
from kube_autotuner.k8s.templates import render_template

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)

LEASE_TTL_SECONDS = 900


class LeaseHeldError(Exception):
    """Raised when a lease is actively held by a different process."""

    def __init__(self, lease_name: str, holder: str, expires: str) -> None:
        """Store context and format a helpful message.

        Args:
            lease_name: Name of the contested lease.
            holder: ``holderIdentity`` observed on the live object.
            expires: Approximate expiry timestamp (``renewTime``).
        """
        self.lease_name = lease_name
        self.holder = holder
        self.expires = expires
        msg = f"Lease {lease_name!r} held by {holder!r} (expires ~{expires})"
        super().__init__(msg)


def _utc_now_rfc3339() -> str:
    """Return the current UTC time formatted as RFC3339Micro."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_k8s_time(ts: str) -> datetime:
    """Parse a Kubernetes RFC3339Micro (or second-resolution) timestamp.

    Args:
        ts: Timestamp string produced by the Kubernetes API
            (``"2026-04-17T00:00:00.000000Z"`` or
            ``"2026-04-17T00:00:00Z"``).

    Returns:
        A timezone-aware UTC :class:`datetime`.

    Raises:
        ValueError: If ``ts`` matches neither recognised format.
    """
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    msg = f"Cannot parse timestamp: {ts!r}"
    raise ValueError(msg)


class NodeLease:
    """Kubernetes Lease-based mutex guarding exclusive node access.

    Acquisition strategy:

    1. **Fast path.** ``kubectl create`` of a fresh ``Lease`` is an
       atomic CAS on the object's name — the API server rejects a second
       creator with ``AlreadyExists``. A successful create takes the
       lock.
    2. **Takeover path.** ``AlreadyExists`` means a lease with the same
       name already lives in the namespace. :meth:`_try_takeover` fetches
       it and decides whether the current caller is entitled to replace
       it: either it is expired (``renewTime`` older than
       ``leaseDurationSeconds``) or the caller is already the recorded
       ``holderIdentity`` (re-entrant). In either case we issue a
       ``kubectl replace`` that echoes back the observed
       ``resourceVersion``; the API server rejects the replace with
       ``Conflict`` if anyone else mutated the object in between,
       giving us optimistic concurrency control without explicit locks.
    3. **Retry rationale.** Two races are expected during normal
       operation and are handled with a single retry:

       - The live lease is deleted between our failed create and our
         ``get_json`` call — then :meth:`_try_takeover` loops back to
         create, and the second create may again race a new creator.
       - Two processes observe the same expired lease and both issue a
         ``replace`` with the same ``resourceVersion``; the loser gets
         ``Conflict`` and retries the takeover, re-reading the (now
         updated) ``resourceVersion``. A single retry is enough — the
         winner will have bumped the version so the loser either takes
         over with a fresh read or fails cleanly with
         :class:`LeaseHeldError` if the winner still holds it.

    The lease is released with ``kubectl delete`` on context-manager
    exit; delete failures are logged but not re-raised so they never
    mask a caller-level exception.
    """

    LEASE_PREFIX = "kube-autotuner-lock"

    def __init__(
        self,
        node: str,
        namespace: str = "default",
        holder: str | None = None,
        kubectl: Kubectl | None = None,
    ) -> None:
        """Configure the lease wrapper.

        Args:
            node: Node name the lease guards; also suffixes
                ``lease_name``.
            namespace: Namespace hosting the Lease object.
            holder: ``holderIdentity`` to claim. Defaults to a random
                ``kube-autotuner-<hex8>`` identifier unique to this
                process.
            kubectl: Injected :class:`Kubectl` client (tests pass a
                mock). Defaults to a freshly constructed real client.
        """
        self.node = node
        self.namespace = namespace
        self.holder = holder or f"kube-autotuner-{uuid4().hex[:8]}"
        self.kubectl = kubectl or Kubectl()
        self.lease_name = f"{self.LEASE_PREFIX}-{node}"
        self._acquired = False

    def _render_lease(self, resource_version: str = "") -> str:
        """Render the lease manifest with the current wall-clock timestamp.

        Args:
            resource_version: When non-empty, embed this
                ``resourceVersion`` so a subsequent ``kubectl replace``
                participates in optimistic-concurrency control.

        Returns:
            The rendered YAML manifest text.
        """
        rv_line = f'resourceVersion: "{resource_version}"' if resource_version else ""
        now = _utc_now_rfc3339()
        return render_template(
            "lease.yaml",
            {
                "LEASE_NAME": self.lease_name,
                "LEASE_NAMESPACE": self.namespace,
                "HOLDER_ID": self.holder,
                "LEASE_TTL": str(LEASE_TTL_SECONDS),
                "ACQUIRE_TIME": now,
                "RENEW_TIME": now,
                "RESOURCE_VERSION_LINE": rv_line,
            },
        )

    def acquire(self) -> None:
        """Acquire the node lease.

        Delegates to :meth:`_try_takeover` when the lease already exists;
        see that method for the ``LeaseHeldError`` / ``KubectlError``
        conditions.

        Raises:
            KubectlError: On any ``kubectl`` failure other than
                ``AlreadyExists``.
        """
        try:
            self.kubectl.create(self._render_lease(), self.namespace)
        except KubectlError as e:
            if "AlreadyExists" not in e.stderr:
                raise
            self._try_takeover()
            return
        self._acquired = True
        logger.info("Acquired lease %s (holder=%s)", self.lease_name, self.holder)

    def _try_takeover(self, retries: int = 1) -> None:
        """Attempt to take over an existing lease via optimistic concurrency.

        See the class docstring for the retry rationale.

        Args:
            retries: Remaining retry budget for recoverable
                ``AlreadyExists`` / ``Conflict`` races.

        Raises:
            LeaseHeldError: A non-expired lease is held by another
                process.
            KubectlError: A ``kubectl`` call failed with an error other
                than the ones we retry on.
        """
        existing = self.kubectl.get_json("lease", self.lease_name, self.namespace)
        if existing is None:
            try:
                self.kubectl.create(self._render_lease(), self.namespace)
            except KubectlError as e:
                if "AlreadyExists" not in e.stderr or retries <= 0:
                    raise
                self._try_takeover(retries - 1)
                return
            self._acquired = True
            logger.info(
                "Acquired lease %s (holder=%s)",
                self.lease_name,
                self.holder,
            )
            return

        spec = existing.get("spec") or {}
        holder = spec.get("holderIdentity", "")
        ttl = spec.get("leaseDurationSeconds", LEASE_TTL_SECONDS)
        renew_str = spec.get("renewTime", "")
        metadata = existing.get("metadata") or {}
        resource_version = metadata.get("resourceVersion", "")

        if renew_str:
            renew_time = _parse_k8s_time(str(renew_str))
            now = datetime.now(UTC)
            expired = (now - renew_time).total_seconds() > float(ttl)
        else:
            expired = True

        if not expired and holder != self.holder:
            raise LeaseHeldError(self.lease_name, str(holder), str(renew_str))

        yaml = self._render_lease(str(resource_version))
        try:
            self.kubectl.replace(yaml, self.namespace)
        except KubectlError as e:
            if retries > 0 and "Conflict" in e.stderr:
                logger.debug("Conflict on replace, retrying takeover")
                self._try_takeover(retries - 1)
                return
            raise
        self._acquired = True
        logger.info(
            "Took over %s lease %s (holder=%s)",
            "expired" if expired else "own",
            self.lease_name,
            self.holder,
        )

    def release(self) -> None:
        """Release the lease if it is currently held by this wrapper."""
        if not self._acquired:
            return
        self.kubectl.delete("lease", self.lease_name, self.namespace)
        self._acquired = False
        logger.info("Released lease %s", self.lease_name)

    def __enter__(self) -> Self:
        """Acquire the lease on context entry.

        Returns:
            ``self`` so callers can use ``with NodeLease(...) as lease:``.
        """
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release the lease on exit; never mask the caller's exception."""
        try:
            self.release()
        except KubectlError:
            logger.warning(
                "Failed to release lease %s",
                self.lease_name,
                exc_info=True,
            )
