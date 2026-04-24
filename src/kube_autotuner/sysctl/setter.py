"""Privileged-pod sysctl backend and the ``make_sysctl_setter`` factory.

:class:`SysctlSetter` is the default production backend: it schedules a
privileged ``hostNetwork`` pod onto the target node, runs
``sysctl -w`` / ``sysctl -n`` inside the host network namespace, and
tears the pod down. All Kubernetes calls route through the
:class:`kube_autotuner.k8s.client.K8sClient` wrapper.

The factory layer (:func:`make_sysctl_setter` and
:func:`make_sysctl_setter_from_env`) also lives here. The env-reading
helper is the single sanctioned place in the package that consults the
``KUBE_AUTOTUNER_SYSCTL_BACKEND`` / ``KUBE_AUTOTUNER_SYSCTL_FAKE_STATE``
environment variables; the CLI calls it at the boundary. Neither
factory is invoked at module import time.
"""

from __future__ import annotations

from contextlib import suppress
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from kube_autotuner.k8s.client import K8sApiError, K8sClient
from kube_autotuner.k8s.lease import NodeLease
from kube_autotuner.k8s.templates import render_template
from kube_autotuner.sysctl.backend import (
    _validate_sysctl_key,
    _validate_sysctl_value,
)
from kube_autotuner.sysctl.fake import FakeSysctlBackend
from kube_autotuner.sysctl.talos import TalosSysctlBackend

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kube_autotuner.sysctl.backend import SysctlBackend

logger = logging.getLogger(__name__)

_POD_READY_TIMEOUT_SECONDS = 60
_LOG_TAIL_LINES = 20

BackendName = Literal["fake", "real", "talos"]
"""Names accepted by :func:`make_sysctl_setter`'s ``backend`` argument."""

_VALID_BACKENDS: frozenset[str] = frozenset({"fake", "real", "talos"})


class PodExecutionError(RuntimeError):
    """Raised when a sysctl-setter pod fails to reach ``Succeeded``.

    The exception message embeds the pod name, target node, and the last
    :data:`_LOG_TAIL_LINES` lines of the pod's logs so operators do not
    need to fetch logs manually to triage a failure.

    Attributes:
        pod_name: Name of the failed pod.
        node: Target node the pod was pinned to.
        logs_tail: Tail of the pod's log output (best effort; falls back
            to a diagnostic string if log collection itself fails).
    """

    def __init__(
        self,
        pod_name: str,
        node: str,
        logs_tail: str,
        cause: str,
    ) -> None:
        """Format the context-rich failure message.

        Args:
            pod_name: Name of the failed pod.
            node: Target node.
            logs_tail: Tail of the pod's log output.
            cause: Short description of the triggering API error.
        """
        self.pod_name = pod_name
        self.node = node
        self.logs_tail = logs_tail
        msg = (
            f"sysctl-setter pod {pod_name!r} on node {node!r} failed: "
            f"{cause}\nLast {_LOG_TAIL_LINES} log line(s):\n{logs_tail}"
        )
        super().__init__(msg)


class SysctlSetter:
    """Applies and reads sysctls on a Kubernetes node via a privileged pod.

    Each call to :meth:`apply` or :meth:`get` schedules a short-lived
    ``hostNetwork`` pod on ``self.node``, runs the sysctl commands, and
    deletes the pod in a ``finally`` block. Pod failures surface as
    :class:`PodExecutionError` with the pod's log tail attached.
    """

    def __init__(
        self,
        node: str,
        namespace: str = "default",
        client: K8sClient | None = None,
    ) -> None:
        """Configure the real sysctl backend.

        Args:
            node: Kubernetes node the sysctls are applied to.
            namespace: Namespace for the ephemeral setter pods.
            client: Injected :class:`K8sClient` (tests pass a mock).
                Defaults to a freshly constructed real client.
        """
        self.node = node
        self.namespace = namespace
        self.client = client or K8sClient()

    def _render_pod(self, pod_name: str, commands: str) -> str:
        """Return the rendered setter-pod manifest.

        Args:
            pod_name: Pod name; must be unique within ``namespace`` for
                the duration of the call.
            commands: Shell snippet run inside the privileged container.

        Returns:
            The rendered YAML manifest ready for :meth:`K8sClient.apply`.
        """
        return render_template(
            "sysctl_setter.yaml",
            {
                "NODE": self.node,
                "POD_NAME": pod_name,
                "SYSCTL_COMMANDS": commands,
            },
        )

    def _await_ready(self, pod_name: str) -> None:
        """Block until the setter pod reaches ``Succeeded``.

        Propagates :class:`K8sApiError` from the underlying wait call
        unchanged; :meth:`_run_pod` wraps that error with pod/log
        context.
        """
        self.client.wait(
            "pod",
            pod_name,
            "jsonpath={.status.phase}=Succeeded",
            self.namespace,
            timeout=_POD_READY_TIMEOUT_SECONDS,
        )

    def _collect_logs(self, pod_name: str) -> str:
        """Return the pod's full log output."""
        return self.client.logs("pod", pod_name, self.namespace)

    def _tail_logs(self, pod_name: str) -> str:
        """Return the last :data:`_LOG_TAIL_LINES` lines, never raising."""
        try:
            logs = self._collect_logs(pod_name)
        except K8sApiError as e:
            return f"<logs unavailable: {e.message.strip() or e}>"
        lines = logs.splitlines()[-_LOG_TAIL_LINES:]
        return "\n".join(lines)

    def _run_pod(self, pod_name: str, commands: str) -> str:
        """Apply the pod manifest, await readiness, collect logs, delete.

        Args:
            pod_name: Pod name.
            commands: Shell snippet run inside the privileged container.

        Returns:
            The pod's ``stdout`` (captured from the pod's log stream)
            on successful completion.

        Raises:
            PodExecutionError: The pod failed to reach ``Succeeded``;
                the exception carries pod name, node, and a log tail.
        """
        yaml_manifest = self._render_pod(pod_name, commands)
        try:
            self.client.apply(yaml_manifest, self.namespace)
            try:
                self._await_ready(pod_name)
            except K8sApiError as e:
                tail = self._tail_logs(pod_name)
                raise PodExecutionError(
                    pod_name,
                    self.node,
                    tail,
                    f"{e}",
                ) from e
            return self._collect_logs(pod_name)
        finally:
            with suppress(K8sApiError):
                self.client.delete("pod", pod_name, self.namespace)

    def apply(self, params: Mapping[str, str | int]) -> None:
        """Apply ``params`` on the target node.

        Args:
            params: Mapping of sysctl key to desired value. Invalid keys
                or values propagate ``ValueError`` from the shared
                validators; pod failures propagate
                :class:`PodExecutionError` with a log tail attached.
        """
        for k, v in params.items():
            _validate_sysctl_key(k)
            _validate_sysctl_value(v)
        cmds = "; ".join(f"sysctl -w {k}='{v}'" for k, v in params.items())
        pod_name = f"sysctl-set-{self.node}"
        output = self._run_pod(pod_name, cmds)
        logger.info("Applied sysctls on %s:\n%s", self.node, output)

    def get(self, param_names: list[str]) -> dict[str, str]:
        """Read current sysctl values from the target node.

        Args:
            param_names: Keys to read. Invalid keys propagate
                ``ValueError``; pod failures propagate
                :class:`PodExecutionError` with a log tail attached.

        Returns:
            Mapping of key to current runtime value.
        """
        for name in param_names:
            _validate_sysctl_key(name)
        cmds = "; ".join(f"echo '{name}='$(sysctl -n {name})" for name in param_names)
        pod_name = f"sysctl-get-{self.node}"
        output = self._run_pod(pod_name, cmds)
        values: dict[str, str] = {}
        for line in output.strip().splitlines():
            if "=" in line:
                key, val = line.split("=", 1)
                values[key.strip()] = val.strip()
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
        logger.info("Restoring sysctls on %s", self.node)
        self.apply(original)

    def flush_network_state(self) -> None:
        """Clear cached per-peer TCP metrics and the conntrack table.

        Runs ``ip tcp_metrics flush all; conntrack -F`` inside one
        privileged hostNetwork pod so both flushes amortise the same
        pod schedule / image pull / teardown. ``;`` (not ``&&``) so a
        missing tcp_metrics netlink module does not prevent the
        conntrack flush. Failures (SELinux denial on RHEL/OpenShift,
        image missing iproute2 or conntrack-tools, nf_conntrack module
        not loaded) are logged and swallowed so the per-iteration
        methodology hook cannot stall the whole optimizer.
        """
        pod_name = f"net-flush-{self.node}"
        try:
            self._run_pod(pod_name, "ip tcp_metrics flush all; conntrack -F")
        except (PodExecutionError, K8sApiError) as e:
            logger.warning(
                "network-state flush failed on %s (continuing): %s",
                self.node,
                e,
            )

    def lock(self) -> NodeLease:
        """Return a :class:`NodeLease` guarding exclusive node access."""
        return NodeLease(self.node, namespace=self.namespace, client=self.client)


def make_sysctl_setter(
    *,
    backend: BackendName,
    node: str,
    namespace: str = "default",
    client: K8sClient | None = None,
    fake_state_path: Path | None = None,
    talos_endpoint: str | None = None,
) -> SysctlBackend:
    """Construct a sysctl backend by explicit name.

    This factory reads no environment variables; the CLI layer is
    responsible for supplying concrete arguments. See
    :func:`make_sysctl_setter_from_env` for the env-bridging helper.

    Args:
        backend: Backend to instantiate. ``"real"`` returns a
            :class:`SysctlSetter`, ``"talos"`` returns a
            :class:`~kube_autotuner.sysctl.talos.TalosSysctlBackend`,
            and ``"fake"`` returns a
            :class:`~kube_autotuner.sysctl.fake.FakeSysctlBackend`.
        node: Kubernetes node name.
        namespace: Namespace for coordination resources.
        client: Injected :class:`K8sClient`. Ignored by the fake
            backend.
        fake_state_path: JSON state file required when
            ``backend == "fake"``. Ignored by the real/Talos backends.
        talos_endpoint: Explicit ``talosctl -n`` target. Ignored by the
            real/fake backends.

    Returns:
        A concrete :class:`SysctlBackend`.

    Raises:
        ValueError: ``backend`` is not one of ``{"fake", "real",
            "talos"}``.
        RuntimeError: ``backend == "fake"`` and ``fake_state_path`` is
            missing.
    """
    if backend == "real":
        return SysctlSetter(node, namespace=namespace, client=client)
    if backend == "talos":
        return TalosSysctlBackend(
            node,
            namespace=namespace,
            client=client,
            endpoint=talos_endpoint,
        )
    if backend == "fake":
        if fake_state_path is None:
            msg = (
                "make_sysctl_setter(backend='fake', ...) requires "
                "fake_state_path to point to a JSON state file"
            )
            raise RuntimeError(msg)
        return FakeSysctlBackend(node, fake_state_path)
    msg = f"Unknown sysctl backend {backend!r} (expected 'real', 'talos', or 'fake')"
    raise ValueError(msg)


def make_sysctl_setter_from_env(
    *,
    node: str,
    namespace: str = "default",
    client: K8sClient | None = None,
    talos_endpoint: str | None = None,
    env: Mapping[str, str] | None = None,
) -> SysctlBackend:
    """Construct a sysctl backend using env-var-driven configuration.

    Honours two environment variables:

    * ``KUBE_AUTOTUNER_SYSCTL_BACKEND`` (default ``"real"``) selects the
      backend; its value is forwarded to :func:`make_sysctl_setter`.
    * ``KUBE_AUTOTUNER_SYSCTL_FAKE_STATE`` supplies the state-file path
      when the fake backend is requested.

    Args:
        node: Kubernetes node name.
        namespace: Namespace for coordination resources.
        client: Injected :class:`K8sClient`.
        talos_endpoint: Explicit ``talosctl -n`` target. Forwarded to
            :func:`make_sysctl_setter` for the Talos backend.
        env: Environment mapping to consult. Defaults to
            :data:`os.environ`.

    Returns:
        The :class:`SysctlBackend` produced by
        :func:`make_sysctl_setter`.

    Raises:
        ValueError: ``KUBE_AUTOTUNER_SYSCTL_BACKEND`` is not one of
            ``{"fake", "real", "talos"}``.
        RuntimeError: The fake backend is selected without
            ``KUBE_AUTOTUNER_SYSCTL_FAKE_STATE`` being set.
    """
    env_map: Mapping[str, str] = env if env is not None else os.environ
    backend_name = env_map.get("KUBE_AUTOTUNER_SYSCTL_BACKEND", "real")
    if backend_name not in _VALID_BACKENDS:
        msg = (
            f"Unknown KUBE_AUTOTUNER_SYSCTL_BACKEND={backend_name!r} "
            "(expected 'real', 'talos', or 'fake')"
        )
        raise ValueError(msg)
    state_raw = env_map.get("KUBE_AUTOTUNER_SYSCTL_FAKE_STATE")
    fake_state_path: Path | None = None
    if backend_name == "fake":
        if not state_raw:
            msg = (
                "KUBE_AUTOTUNER_SYSCTL_BACKEND=fake requires "
                "KUBE_AUTOTUNER_SYSCTL_FAKE_STATE to point to a JSON "
                "state file"
            )
            raise RuntimeError(msg)
        fake_state_path = Path(state_raw)
    return make_sysctl_setter(
        backend=cast("BackendName", backend_name),
        node=node,
        namespace=namespace,
        client=client,
        fake_state_path=fake_state_path,
        talos_endpoint=talos_endpoint,
    )
