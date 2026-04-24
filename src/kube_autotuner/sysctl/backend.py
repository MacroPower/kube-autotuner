"""Pluggable sysctl backend protocol and shared input validators.

Defines :class:`SysctlBackend`, the protocol every concrete sysctl
backend implements, along with the regex-based validators used to reject
malformed keys and shell-unsafe values before they reach the node.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from contextlib import AbstractContextManager

_SYSCTL_KEY_RE = re.compile(r"^[a-z0-9_.]+$")
_SYSCTL_VALUE_RE = re.compile(r"^[a-zA-Z0-9_ .-]+$")


def _validate_sysctl_key(key: str) -> None:
    """Reject ``key`` if it is not a well-formed sysctl key.

    Args:
        key: Candidate sysctl key (e.g. ``"net.core.rmem_max"``).

    Raises:
        ValueError: If ``key`` contains characters outside
            ``[a-z0-9_.]``.
    """
    if not _SYSCTL_KEY_RE.match(key):
        msg = f"Invalid sysctl key: {key!r}"
        raise ValueError(msg)


def _validate_sysctl_value(value: str | int) -> None:
    """Reject ``value`` if it contains shell-unsafe characters.

    Args:
        value: Candidate sysctl value. Ints are coerced via ``str``.

    Raises:
        ValueError: If ``value`` contains characters outside
            ``[a-zA-Z0-9_ .-]``.
    """
    if not _SYSCTL_VALUE_RE.match(str(value)):
        msg = f"Invalid sysctl value: {value!r}"
        raise ValueError(msg)


@runtime_checkable
class SysctlBackend(Protocol):
    """Protocol implemented by every sysctl backend.

    Concrete implementations cover the privileged-pod writer, the Talos
    machine-config writer, and the in-memory fake used by tests. The
    ``@runtime_checkable`` decorator lets
    ``isinstance(obj, SysctlBackend)`` gate ad-hoc duck-typed code
    paths.
    """

    def apply(self, params: Mapping[str, str | int]) -> None:
        """Apply ``params`` on the target node."""
        ...

    def get(self, param_names: list[str]) -> dict[str, str]:
        """Return current values for ``param_names``."""
        ...

    def snapshot(self, param_names: list[str]) -> dict[str, str]:
        """Capture current values of ``param_names`` for later rollback."""
        ...

    def restore(self, original: dict[str, str]) -> None:
        """Re-apply a previously captured snapshot."""
        ...

    def flush_network_state(self) -> None:
        """Clear per-iteration kernel network state on the target node.

        Evicts two caches that otherwise bleed measurement state from
        the previous iteration into the next:

        * ``/proc/net/tcp_metrics`` -- ``net.ipv4.tcp_no_metrics_save=1``
          stops *new* rows from being cached but does not evict rows
          already present, so cached ssthresh / RTT survives the pin.
        * The netfilter conntrack table -- residual entries from the
          previous saturation sub-stage skew the next iteration's rps
          ceiling and p99 latency once ``nf_conntrack_*`` timeouts are
          in the tuning search space.

        Implementations should log-and-continue on failure (SELinux
        denial, missing iproute2, conntrack module not loaded) rather
        than fail the iteration.

        Warning:
            The real backend flushes the **entire** host conntrack
            table, not just benchmark entries. On nodes that carry
            unrelated traffic (SSH, CNI control plane, sidecars) this
            briefly disrupts those in-flight connections. Dedicated
            benchmark nodes are the documented deployment model.
        """
        ...

    def lock(self) -> AbstractContextManager[object]:
        """Return a context manager serialising access to the node."""
        ...
