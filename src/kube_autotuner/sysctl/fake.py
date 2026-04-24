"""In-memory sysctl backend for integration tests.

Backed by a JSON file so CLI invocations launched by a single test can
share state. The constructor requires an explicit ``state_path``; the
``make_sysctl_setter_from_env`` factory is responsible for resolving
``KUBE_AUTOTUNER_SYSCTL_FAKE_STATE`` and forwarding the path to this
class.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import TYPE_CHECKING

from kube_autotuner.sysctl.backend import (
    _validate_sysctl_key,
    _validate_sysctl_value,
)
from kube_autotuner.sysctl.params import PARAM_SPACE

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from kube_autotuner.models import HostStatePhase, HostStateSnapshot

logger = logging.getLogger(__name__)


def _seed_defaults() -> dict[str, str]:
    """Return deterministic defaults for every tunable plus kernel.osrelease."""
    defaults: dict[str, str] = {p.name: str(p.values[0]) for p in PARAM_SPACE.params}
    defaults["kernel.osrelease"] = "6.1.0-talos"
    return defaults


_DEFAULTS = _seed_defaults()


class FakeSysctlBackend:
    """JSON-file-backed sysctl backend for tests that exercise CLI pipelines."""

    client: object | None = None

    def __init__(self, node: str, state_path: Path) -> None:
        """Initialise the fake backend.

        Args:
            node: Name of the node whose sysctls are being simulated.
                Stored for logging; state is global to ``state_path``.
            state_path: JSON file that persists sysctl values between
                calls. The parent directory is created on first write.
        """
        self.node = node
        self.state_path = state_path

    def _load(self) -> dict[str, str]:
        """Return the current JSON state, or an empty dict if missing."""
        if not self.state_path.exists():
            return {}
        loaded: dict[str, str] = json.loads(self.state_path.read_text())
        return loaded

    def _save(self, state: dict[str, str]) -> None:
        """Persist ``state`` to disk, creating parents as needed."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state))

    def apply(self, params: Mapping[str, str | int]) -> None:
        """Write ``params`` into the JSON state file after validation."""
        for k, v in params.items():
            _validate_sysctl_key(k)
            _validate_sysctl_value(v)
        state = self._load()
        for k, v in params.items():
            state[k] = str(v)
        self._save(state)
        logger.info(
            "FakeSysctlBackend applied %d sysctl(s) on %s", len(params), self.node
        )

    def get(self, param_names: list[str]) -> dict[str, str]:
        """Return current values for ``param_names``, seeded defaults on miss."""
        for name in param_names:
            _validate_sysctl_key(name)
        state = self._load()
        return {name: state.get(name, _DEFAULTS.get(name, "0")) for name in param_names}

    def snapshot(self, param_names: list[str]) -> dict[str, str]:
        """Return the current values for ``param_names`` (equivalent to :meth:`get`)."""
        return self.get(param_names)

    def restore(self, original: dict[str, str]) -> None:
        """Re-apply a previously captured snapshot."""
        self.apply(original)

    def flush_network_state(self) -> None:
        """No-op: fake backend has no kernel tcp_metrics or conntrack table to flush."""
        logger.debug(
            "FakeSysctlBackend network-state flush is a no-op on %s", self.node
        )

    def collect_host_state(  # noqa: PLR6301 - protocol conformance requires instance method
        self,
        iteration: int | None,  # noqa: ARG002
        phase: HostStatePhase,  # noqa: ARG002
    ) -> HostStateSnapshot | None:
        """No-op: fake backend has no host kernel state to snapshot.

        Returns:
            Always ``None``.
        """
        return None

    def lock(self) -> contextlib.AbstractContextManager[None]:  # noqa: PLR6301
        """Return a no-op context manager; serialisation is unnecessary in-memory."""
        return contextlib.nullcontext()
