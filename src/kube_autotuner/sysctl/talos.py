"""Sysctl backend that routes writes through ``talosctl patch mc``.

:meth:`TalosSysctlBackend.apply` persists the desired sysctls into the
machine config with ``talosctl patch mc --mode=no-reboot`` and then
polls ``/proc/sys`` until Talos's ``KernelParamSpecController``
reconciles the new values. No reboot is needed on provisioners where
the controller can actually write ``/proc/sys`` -- QEMU and bare-metal
Talos.

The ``docker`` provisioner is *not* supported: the controller gets
``EACCES`` on ``/proc/sys`` because Talos's container does not own
``init_net``, and ``talosctl reboot`` is rejected in container mode, so
there is no code path that can land a sysctl change. Integration tests
running against Talos Docker opt out via the
``requires_real_sysctl_write`` pytest marker.

Every ``talosctl`` shell-out routes through
:func:`kube_autotuner.subproc.run_tool`, the sanctioned subprocess
entrypoint in the repo. The module reads no environment variables; the
endpoint is either supplied explicitly or discovered via
:meth:`kube_autotuner.k8s.client.Kubectl.get_node_internal_ip`. The
``make_sysctl_setter_from_env`` helper in
:mod:`kube_autotuner.sysctl.setter` is the single place env vars are
consulted.
"""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile
import time
from typing import TYPE_CHECKING

import yaml

from kube_autotuner.k8s.client import Kubectl, KubectlError
from kube_autotuner.k8s.lease import NodeLease
from kube_autotuner.subproc import run_tool
from kube_autotuner.sysctl.backend import (
    _validate_sysctl_key,
    _validate_sysctl_value,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

_APPLY_PROPAGATION_TIMEOUT_SECONDS = 120
_APPLY_PROPAGATION_POLL_INTERVAL_SECONDS = 1.0


class TalosSysctlBackend:
    """Applies and reads sysctls on a Talos node via ``talosctl``."""

    kubectl: Kubectl

    def __init__(
        self,
        node: str,
        namespace: str = "default",
        kubectl: Kubectl | None = None,
        endpoint: str | None = None,
    ) -> None:
        """Configure the Talos backend.

        Args:
            node: Kubernetes node name. Also used as the default
                ``talosctl -n`` target when ``endpoint`` is not supplied
                and the InternalIP lookup succeeds.
            namespace: Namespace hosting the coordination Lease.
            kubectl: Injected :class:`Kubectl` client. Defaults to a
                freshly constructed real client.
            endpoint: Explicit ``talosctl`` endpoint address. When
                ``None``, the endpoint is resolved lazily from
                ``kubectl get node <node>`` on first access.
        """
        self.node = node
        self.namespace = namespace
        self.kubectl = kubectl or Kubectl()
        self._endpoint_override = endpoint
        self._resolved_endpoint: str | None = None

    @property
    def endpoint(self) -> str:
        """Return the resolved ``talosctl -n`` target, resolving lazily.

        Resolution order:

        1. Cached ``_resolved_endpoint`` (memoised after first resolution).
        2. ``endpoint=`` kwarg passed to :meth:`__init__`.
        3. The node's ``InternalIP`` from ``kubectl get node``.

        Raises:
            RuntimeError: If ``kubectl`` fails, or the node has no
                ``InternalIP``.
        """
        if self._resolved_endpoint:
            return self._resolved_endpoint
        if self._endpoint_override:
            self._resolved_endpoint = self._endpoint_override
            return self._resolved_endpoint
        try:
            ip = self.kubectl.get_node_internal_ip(self.node)
        except KubectlError as e:
            msg = (
                f"Could not resolve Talos endpoint for node {self.node!r}: "
                f"{e.stderr.strip()}"
            )
            raise RuntimeError(msg) from e
        if not ip:
            msg = (
                f"Node {self.node!r} has no InternalIP; pass endpoint= "
                "explicitly to TalosSysctlBackend"
            )
            raise RuntimeError(msg)
        self._resolved_endpoint = ip
        return self._resolved_endpoint

    def _talosctl(self, *args: str) -> str:
        """Invoke ``talosctl -n <endpoint> <args>`` via :func:`run_tool`.

        Args:
            *args: Arguments appended after ``-n <endpoint>``.

        Returns:
            The child's ``stdout`` decoded as text.

        Raises:
            RuntimeError: If ``talosctl`` is missing from ``PATH`` or
                exits non-zero. The error message carries the failing
                argv and captured ``stderr`` for diagnosis.
        """
        cmd_args = ["-n", self.endpoint, *args]
        try:
            result = run_tool("talosctl", cmd_args)
        except FileNotFoundError as e:
            msg = (
                "talosctl not found in PATH; install talosctl to use "
                "the Talos sysctl backend"
            )
            raise RuntimeError(msg) from e
        if result.returncode != 0:
            msg = (
                f"talosctl failed (rc={result.returncode}): "
                f"talosctl {' '.join(cmd_args)}\n{result.stderr}"
            )
            raise RuntimeError(msg)
        return result.stdout

    def apply(self, params: Mapping[str, str | int]) -> None:
        """Persist ``params`` into the Talos machine config and await propagation.

        Strategy: render a strategic-merge patch under ``machine.sysctls``,
        write it to a temporary file, apply it with
        ``talosctl patch mc --mode=no-reboot``, then poll ``/proc/sys``
        until every requested value appears at runtime.

        Args:
            params: Mapping of sysctl key to desired value. Values are
                coerced to strings so YAML does not re-type numeric-
                looking sysctl values. Invalid keys or values propagate
                ``ValueError``; ``talosctl`` failures and propagation
                timeouts propagate ``RuntimeError``.
        """
        for k, v in params.items():
            _validate_sysctl_key(k)
            _validate_sysctl_value(v)
        expected = {k: str(v) for k, v in params.items()}
        # Strategic-merge patch against the base MachineConfig. JSON6902
        # patches are rejected on multi-document configs, and the
        # Kubernetes-style apiVersion/kind wrapper is rejected because
        # the base MachineConfig doc uses ``version: v1alpha1`` at the
        # top level. A bare YAML fragment under ``machine.sysctls`` is
        # the form every Talos doc example uses.
        patch_doc = {"machine": {"sysctls": expected}}
        patch_yaml = yaml.safe_dump(patch_doc, default_flow_style=False)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(patch_yaml)
            patch_path = Path(f.name)
        try:
            self._talosctl(
                "patch",
                "mc",
                "--mode=no-reboot",
                "-p",
                f"@{patch_path}",
            )
            self._await_runtime_values(expected)
        finally:
            patch_path.unlink(missing_ok=True)
        logger.info(
            "Applied %d sysctl(s) on %s via talosctl",
            len(params),
            self.node,
        )

    def _await_runtime_values(self, expected: dict[str, str]) -> None:
        """Poll ``/proc/sys`` until ``expected`` is observed, or time out.

        Args:
            expected: Mapping of sysctl key to stringified expected value.

        Raises:
            RuntimeError: Runtime values did not converge within
                :data:`_APPLY_PROPAGATION_TIMEOUT_SECONDS`. The message
                carries the latest observed state and a best-effort
                dump of ``machine.sysctls`` for diagnosis.
        """
        deadline = time.monotonic() + _APPLY_PROPAGATION_TIMEOUT_SECONDS
        current: dict[str, str] = {}
        while True:
            current = self.get(list(expected))
            if all(current.get(k) == v for k, v in expected.items()):
                return
            if time.monotonic() >= deadline:
                msg = (
                    "sysctl values did not propagate within "
                    f"{_APPLY_PROPAGATION_TIMEOUT_SECONDS}s on "
                    f"{self.node}: expected {expected}, got {current}. "
                    f"Machine config machine.sysctls: "
                    f"{self._read_machine_sysctls()!r}"
                )
                raise RuntimeError(msg)
            time.sleep(_APPLY_PROPAGATION_POLL_INTERVAL_SECONDS)

    def _read_machine_sysctls(self) -> str:
        """Return the live ``machineconfig`` YAML for diagnostic logging.

        Never raises; errors are stringified into the return value so the
        caller (``_await_runtime_values``) can embed the result in the
        propagation-timeout message without shadowing the original
        failure.
        """
        try:
            return self._talosctl("get", "machineconfig", "-o", "yaml")
        except RuntimeError as e:
            return f"<unavailable: {e}>"

    def get(self, param_names: list[str]) -> dict[str, str]:
        """Return current runtime values for ``param_names``.

        Each key translates to a ``/proc/sys/<path>`` read through
        ``talosctl read``.
        """
        for name in param_names:
            _validate_sysctl_key(name)
        values: dict[str, str] = {}
        for name in param_names:
            path = "/proc/sys/" + name.replace(".", "/")
            values[name] = self._talosctl("read", path).strip()
        return values

    def snapshot(self, param_names: list[str]) -> dict[str, str]:
        """Capture current values of ``param_names`` for later rollback.

        Args:
            param_names: Keys to capture.

        Returns:
            Mapping of key to current runtime value, suitable for
            passing to :meth:`restore`.
        """
        return self.get(param_names)

    def restore(self, original: dict[str, str]) -> None:
        """Re-apply a previously captured snapshot."""
        self.apply(original)

    def lock(self) -> NodeLease:
        """Return a :class:`NodeLease` guarding exclusive node access."""
        return NodeLease(self.node, namespace=self.namespace, kubectl=self.kubectl)
