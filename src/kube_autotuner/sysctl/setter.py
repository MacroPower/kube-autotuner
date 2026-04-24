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
from kube_autotuner.models import HostStateSnapshot
from kube_autotuner.sysctl.backend import (
    _validate_sysctl_key,
    _validate_sysctl_value,
)
from kube_autotuner.sysctl.fake import FakeSysctlBackend
from kube_autotuner.sysctl.talos import TalosSysctlBackend

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kube_autotuner.models import HostStatePhase
    from kube_autotuner.sysctl.backend import SysctlBackend

logger = logging.getLogger(__name__)

_POD_READY_TIMEOUT_SECONDS = 60
_LOG_TAIL_LINES = 20

_HOST_STATE_COMMANDS: tuple[str, ...] = (
    "echo '===conntrack_count==='",
    "conntrack -C 2>/dev/null || echo NA",
    "echo '===conntrack_stats==='",
    "conntrack -S 2>/dev/null || echo NA",
    "echo '===sockstat==='",
    "cat /proc/net/sockstat 2>/dev/null || echo NA",
    "echo '===sockstat6==='",
    "cat /proc/net/sockstat6 2>/dev/null || echo NA",
    "echo '===netstat==='",
    "cat /proc/net/netstat 2>/dev/null || echo NA",
    "echo '===snmp==='",
    "cat /proc/net/snmp 2>/dev/null || echo NA",
    "echo '===tcp_metrics==='",
    "wc -l < /proc/net/tcp_metrics 2>/dev/null || echo NA",
    "echo '===route==='",
    "wc -l < /proc/net/route 2>/dev/null || echo NA",
    "echo '===arp==='",
    "wc -l < /proc/net/arp 2>/dev/null || echo NA",
    "echo '===meminfo==='",
    "cat /proc/meminfo 2>/dev/null || echo NA",
    "echo '===slabinfo==='",
    "grep nf_conntrack /proc/slabinfo 2>/dev/null || echo NA",
    "echo '===file_nr==='",
    "cat /proc/sys/fs/file-nr 2>/dev/null || echo NA",
    "echo '===end==='",
)

_HOST_STATE_SCRIPT = "; ".join(_HOST_STATE_COMMANDS)
"""Shell script that dumps labelled host-state sections to stdout.

Read by :func:`_parse_host_state_output`. Must stay on a single line
so the rendered setter-pod YAML scalar round-trips cleanly, and must
avoid ``$NODE`` / ``$POD_NAME`` / ``$SYSCTL_COMMANDS`` / ``$$`` /
``"`` / ``\\`` tokens -- the template substitutes ``${SYSCTL_COMMANDS}``
via :func:`string.Template.safe_substitute`, and the target YAML
scalar is double-quoted which would interpret ``\\`` as an escape.
"""

_TCPEXT_KEYS = frozenset({
    "TCPTimeWaitOverflow",
    "TW",
    "TWRecycled",
    "TCPOrphanQueued",
    "ListenDrops",
    "ListenOverflows",
    "DelayedACKs",
    "TCPAbortOnData",
    "TCPAbortOnClose",
    "TCPKeepAlive",
})

_SNMP_KEYS = frozenset({
    "InSegs",
    "OutSegs",
    "RetransSegs",
    "OutRsts",
    "CurrEstab",
    "InDatagrams",
    "OutDatagrams",
    "RcvbufErrors",
    "SndbufErrors",
})

_CONNTRACK_STATS_KEYS = frozenset({
    "found",
    "invalid",
    "insert",
    "insert_failed",
    "drop",
    "early_drop",
    "error",
    "search_restart",
})

_MEMINFO_KEYS = {
    "Slab": "slab_kb",
    "SReclaimable": "sreclaimable_kb",
    "SUnreclaim": "sunreclaim_kb",
}

_SECTION_MARKER_MIN_LEN = 7
"""Minimum length of a ``===name===`` marker line (two non-empty names)."""

_SLABINFO_MIN_COLUMNS = 3
"""``name active_objs num_objs`` -- the minimum we need to record."""

_FILE_NR_COLUMNS = 3
"""``/proc/sys/fs/file-nr`` always emits ``allocated unused max``."""


def _split_sections(text: str) -> dict[str, str]:
    """Split ``_HOST_STATE_SCRIPT`` stdout into ``name -> body`` sections.

    Sections are delimited by ``===name===`` marker lines; the trailing
    ``===end===`` marker is discarded.

    Args:
        text: Raw stdout captured from a successful script run.

    Returns:
        Mapping of section name to the joined body text with
        surrounding whitespace stripped.
    """
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if (
            stripped.startswith("===")
            and stripped.endswith("===")
            and len(stripped) >= _SECTION_MARKER_MIN_LEN
        ):
            if current is not None and current != "end":
                sections[current] = "\n".join(buffer).strip()
            current = stripped[3:-3]
            buffer = []
            continue
        buffer.append(line)
    if current is not None and current != "end":
        sections[current] = "\n".join(buffer).strip()
    return sections


def _parse_int(
    raw: str,
    *,
    section: str,
    key: str,
    metrics: dict[str, int],
    errors: list[str],
) -> None:
    """Parse ``raw`` as an int and write to ``metrics`` or ``errors``."""
    try:
        metrics[key] = int(raw)
    except ValueError:
        errors.append(f"{section}: {key} unparseable value {raw!r}")


def _parse_conntrack_stats(
    block: str,
    metrics: dict[str, int],
    errors: list[str],
) -> None:
    """Sum ``conntrack -S`` per-cpu counters into total metrics."""
    totals: dict[str, int] = dict.fromkeys(_CONNTRACK_STATS_KEYS, 0)
    seen: set[str] = set()
    for line in block.splitlines():
        for part in line.split():
            k, sep, v = part.partition("=")
            if not sep or k not in _CONNTRACK_STATS_KEYS:
                continue
            try:
                totals[k] += int(v)
            except ValueError:
                # Aggregation: one bad per-cpu token does not tank the total.
                # A key only ends up "missing" from errors if no cpu row ever
                # parsed cleanly for it.
                continue
            seen.add(k)
    for k in seen:
        metrics[f"conntrack_{k}"] = totals[k]
    errors.extend(f"conntrack_stats: missing {k}" for k in _CONNTRACK_STATS_KEYS - seen)


def _parse_sockstat(
    block: str,
    *,
    prefix: str,
    section: str,
    metrics: dict[str, int],
    errors: list[str],
) -> None:
    """Parse ``/proc/net/sockstat``-style ``Proto: key val key val`` lines.

    Captures *every* key the kernel emits (TCP, UDP, UDPLITE, RAW,
    FRAG, sockets, etc.) rather than a curated subset. ``/proc/net/
    sockstat`` is small and cheap to ingest, and the extra protocols
    are occasionally useful leak signals (e.g. FRAG memory growth).
    The plan's named keys (tcp_inuse, tcp_tw, tcp_mem, udp_inuse,
    udp_mem, ...) are the documented *minimum*, not a hard filter.
    """
    for line in block.splitlines():
        head, sep, tail = line.partition(":")
        if not sep:
            continue
        proto = head.strip().lower().replace(" ", "_")
        if not proto:
            continue
        tokens = tail.split()
        for i in range(0, len(tokens) - 1, 2):
            key = f"{prefix}{proto}_{tokens[i].lower()}"
            _parse_int(
                tokens[i + 1],
                section=section,
                key=key,
                metrics=metrics,
                errors=errors,
            )


def _parse_netstat_like(
    block: str,
    *,
    prefix: str,
    wanted: frozenset[str],
    section: str,
    metrics: dict[str, int],
    errors: list[str],
) -> None:
    """Parse paired header/values lines from ``/proc/net/{netstat,snmp}``.

    Any key in ``wanted`` that doesn't appear in any header row is
    recorded in ``errors`` so a reader diffing two snapshots can tell
    "kernel didn't emit it" apart from "we didn't look".
    """
    lines = block.splitlines()
    seen: set[str] = set()
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        header = lines[i]
        values = lines[i + 1]
        h_name, h_sep, h_rest = header.partition(":")
        v_name, v_sep, v_rest = values.partition(":")
        if not h_sep or not v_sep or h_name != v_name:
            continue
        keys = h_rest.split()
        vals = v_rest.split()
        for k, v in zip(keys, vals, strict=False):
            if k not in wanted:
                continue
            metric_key = f"{prefix}{h_name.lower()}_{k}"
            _parse_int(
                v,
                section=section,
                key=metric_key,
                metrics=metrics,
                errors=errors,
            )
            seen.add(k)
    errors.extend(f"{section}: missing {k}" for k in wanted - seen)


def _parse_meminfo(block: str, metrics: dict[str, int], errors: list[str]) -> None:
    """Parse the curated subset of ``/proc/meminfo``, stripping ``kB`` suffixes."""
    seen: set[str] = set()
    for line in block.splitlines():
        name, sep, rest = line.partition(":")
        name = name.strip()
        if not sep or name not in _MEMINFO_KEYS:
            continue
        parts = rest.split()
        if not parts:
            errors.append(f"meminfo: empty value for {name}")
            continue
        _parse_int(
            parts[0],
            section="meminfo",
            key=_MEMINFO_KEYS[name],
            metrics=metrics,
            errors=errors,
        )
        seen.add(name)
    errors.extend(f"meminfo: missing {name}" for name in set(_MEMINFO_KEYS) - seen)


def _parse_slabinfo(block: str, metrics: dict[str, int], errors: list[str]) -> None:
    """Extract ``active_objs`` / ``num_objs`` for the ``nf_conntrack`` slab."""
    for line in block.splitlines():
        parts = line.split()
        if not parts or parts[0] != "nf_conntrack":
            continue
        if len(parts) < _SLABINFO_MIN_COLUMNS:
            errors.append("slabinfo: nf_conntrack row malformed")
            return
        _parse_int(
            parts[1],
            section="slabinfo",
            key="slab_nf_conntrack_active_objs",
            metrics=metrics,
            errors=errors,
        )
        _parse_int(
            parts[2],
            section="slabinfo",
            key="slab_nf_conntrack_num_objs",
            metrics=metrics,
            errors=errors,
        )
        return
    errors.append("slabinfo: nf_conntrack row not found")


def _parse_file_nr(block: str, metrics: dict[str, int], errors: list[str]) -> None:
    """Parse ``/proc/sys/fs/file-nr``: ``allocated  unused  max``."""
    parts = block.split()
    if len(parts) < _FILE_NR_COLUMNS:
        errors.append(f"file_nr: malformed {block!r}")
        return
    for raw, key in zip(
        parts[:_FILE_NR_COLUMNS],
        ("file_nr_allocated", "file_nr_unused", "file_nr_max"),
        strict=True,
    ):
        _parse_int(raw, section="file_nr", key=key, metrics=metrics, errors=errors)


def _parse_host_state_output(  # noqa: C901 - dispatcher across many /proc sources
    text: str,
) -> tuple[dict[str, int], list[str]]:
    """Turn ``_HOST_STATE_SCRIPT`` stdout into ``(metrics, errors)``.

    A source whose command emitted the literal ``NA`` fallback routes
    to ``errors`` and contributes no entry to ``metrics`` -- sentinel
    zeros would defeat the leak-detection signal by being
    indistinguishable from a true zero reading.

    Args:
        text: Raw stdout from a successful ``_HOST_STATE_SCRIPT`` run.

    Returns:
        ``(metrics, errors)``: a flat ``dict[str, int]`` of scalar
        counters, plus a list of human-readable parse notes (empty on
        a clean run).
    """
    sections = _split_sections(text)
    metrics: dict[str, int] = {}
    errors: list[str] = []

    def section_or_na(name: str) -> str | None:
        block = sections.get(name)
        if block is None:
            errors.append(f"{name}: section missing")
            return None
        if block == "NA":
            errors.append(f"{name}: NA")
            return None
        return block

    block = section_or_na("conntrack_count")
    if block is not None:
        _parse_int(
            block.strip(),
            section="conntrack_count",
            key="conntrack_count",
            metrics=metrics,
            errors=errors,
        )

    block = section_or_na("conntrack_stats")
    if block is not None:
        _parse_conntrack_stats(block, metrics, errors)

    for name, prefix in (("sockstat", "sockstat_"), ("sockstat6", "sockstat6_")):
        block = section_or_na(name)
        if block is not None:
            _parse_sockstat(
                block,
                prefix=prefix,
                section=name,
                metrics=metrics,
                errors=errors,
            )

    for name, prefix, wanted in (
        ("netstat", "netstat_", _TCPEXT_KEYS),
        ("snmp", "snmp_", _SNMP_KEYS),
    ):
        block = section_or_na(name)
        if block is not None:
            _parse_netstat_like(
                block,
                prefix=prefix,
                wanted=wanted,
                section=name,
                metrics=metrics,
                errors=errors,
            )

    for name, key in (
        ("tcp_metrics", "tcp_metrics_rows"),
        ("route", "route_rows"),
        ("arp", "arp_rows"),
    ):
        block = section_or_na(name)
        if block is not None:
            _parse_int(
                block.strip(),
                section=name,
                key=key,
                metrics=metrics,
                errors=errors,
            )

    block = section_or_na("meminfo")
    if block is not None:
        _parse_meminfo(block, metrics, errors)

    block = section_or_na("slabinfo")
    if block is not None:
        _parse_slabinfo(block, metrics, errors)

    block = section_or_na("file_nr")
    if block is not None:
        _parse_file_nr(block, metrics, errors)

    return metrics, errors


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

    def collect_host_state(
        self,
        iteration: int | None,
        phase: HostStatePhase,
    ) -> HostStateSnapshot | None:
        """Snapshot cheap-to-read host kernel counters via a privileged pod.

        Runs :data:`_HOST_STATE_SCRIPT` in one ``sh -c`` invocation on
        the target node, parses the labelled sections into scalar
        counters, and returns a :class:`HostStateSnapshot`. Pod
        failures are logged and surfaced as a snapshot with empty
        ``metrics`` and a single entry in ``errors`` so the caller
        still records that a collection attempt was made.

        Args:
            iteration: Iteration index; ``None`` for the per-run
                baseline.
            phase: Collection point label; used in the pod name so
                parallel ``apply``/``flush``/``snapshot`` pods on the
                same node cannot collide.

        Returns:
            A populated :class:`HostStateSnapshot`. Never ``None``
            for this backend.
        """
        pod_name = f"host-snap-{self.node}-{phase}"
        try:
            output = self._run_pod(pod_name, _HOST_STATE_SCRIPT)
        except Exception as e:  # noqa: BLE001 - instrumentation must never fail the trial
            logger.warning(
                "host-state snapshot pod failed on %s phase=%s (continuing): %s",
                self.node,
                phase,
                e,
            )
            return HostStateSnapshot(
                node=self.node,
                iteration=iteration,
                phase=phase,
                metrics={},
                errors=[f"pod failed: {e}"],
            )
        metrics, errors = _parse_host_state_output(output)
        return HostStateSnapshot(
            node=self.node,
            iteration=iteration,
            phase=phase,
            metrics=metrics,
            errors=errors,
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
