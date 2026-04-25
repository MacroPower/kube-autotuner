"""Stateless diagnostic-capture helpers shared by the benchmark runner.

Each helper takes its :class:`K8sClient` dependency explicitly so the
runner can keep its orchestration code free of diagnostic-formatting
noise. Every function in this module is best-effort: a failure inside
the diagnostic path must never mask the primary failure being
described, so all helpers swallow exceptions and degrade to empty rows
or ``None``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from kube_autotuner.benchmark.errors import ClientJobFailed

if TYPE_CHECKING:
    from kube_autotuner.k8s.client import JobFailureDiagnostics, K8sClient

logger = logging.getLogger(__name__)


def diagnostics_from(exc: BaseException) -> list[JobFailureDiagnostics]:
    """Extract per-attempt diagnostics from a :class:`ClientJobFailed`.

    Returns ``[]`` for any other exception type so the stage method
    can unconditionally call this on ``first_exc``.

    Args:
        exc: The first exception raised by a stage's per-client
            future.

    Returns:
        The per-attempt diagnostics list, or ``[]`` when ``exc`` is
        not a :class:`ClientJobFailed` (e.g. a cleanup
        :class:`RuntimeError` surfaced from the retry loop's
        ``finally`` block).
    """
    if isinstance(exc, ClientJobFailed):
        return list(exc.diagnostics)
    return []


def _server_container_status_rows(pod: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Flatten a server pod's ``containerStatuses`` to plain dicts.

    Captures the fields the plan calls out for the stage snapshot:
    container name, ``ready``, ``restartCount``, and the
    ``lastState.terminated`` reason/exit code so a recently crashed
    iperf3 server is visible as a non-zero ``restartCount`` plus a
    ``last_terminated`` payload.

    Args:
        pod: Typed pod object from the CoreV1 API.

    Returns:
        A list of dicts, one per container. Empty list when the pod
        has no container statuses yet.
    """
    rows: list[dict[str, Any]] = []
    statuses = getattr(getattr(pod, "status", None), "container_statuses", None) or []
    for cs in statuses:
        last_state = getattr(cs, "last_state", None)
        term = getattr(last_state, "terminated", None) if last_state else None
        last_terminated: dict[str, Any] = {}
        if term is not None:
            last_terminated = {
                "reason": str(getattr(term, "reason", "") or ""),
                "exit_code": (
                    int(getattr(term, "exit_code", 0) or 0)
                    if getattr(term, "exit_code", None) is not None
                    else None
                ),
                "message": str(getattr(term, "message", "") or ""),
            }
        rows.append({
            "name": str(getattr(cs, "name", "") or ""),
            "ready": bool(getattr(cs, "ready", False)),
            "restart_count": int(getattr(cs, "restart_count", 0) or 0),
            "last_terminated": last_terminated,
        })
    return rows


def log_job_diagnostics(
    client: K8sClient,
    job_name: str,
    namespace: str,
    kind: str,
    attempt: int,
    *,
    stage_label: str,
    iteration: int,
) -> JobFailureDiagnostics | None:
    """Emit a single warning line describing a failed Job attempt.

    Pulls :meth:`K8sClient.describe_job_failure` and renders the
    returned :class:`JobFailureDiagnostics` into one structured log
    record. Diagnostics must never mask the primary failure: every
    downstream call in the diagnostic path is best-effort and we
    log at ``warning`` rather than raising.

    Invariant: this function MUST NOT propagate any exception. The
    retry loop's per-attempt ``finally`` block runs this before the
    Job ``delete``; if this raised, ``delete`` would be skipped and
    the Job would leak until the stage-level label sweep runs.

    Args:
        client: Injected :class:`K8sClient`.
        job_name: Job metadata.name.
        namespace: Target namespace.
        kind: Short label for log grouping (``"iperf3"``, ``"fortio"``).
        attempt: 1-based attempt index.
        stage_label: Sub-stage label (``"bw-tcp"`` / ``"bw-udp"`` /
            ``"fortio-saturation"`` / ``"fortio-fixed_qps"``).
        iteration: Zero-based iteration index.

    Returns:
        The :class:`JobFailureDiagnostics` payload on success, or
        ``None`` if the describe call itself failed. The retry loop
        accumulates non-``None`` returns into the envelope carried
        by :class:`ClientJobFailed`.
    """
    try:
        diag = client.describe_job_failure(job_name, namespace)
    except Exception:
        logger.warning(
            "Could not describe %s client job %s [stage=%s iter=%d] (attempt %d)",
            kind,
            job_name,
            stage_label,
            iteration,
            attempt,
            exc_info=True,
        )
        return None
    logger.warning(
        "%s client Job %s [stage=%s iter=%d] attempt %d diagnostics: %s",
        kind,
        job_name,
        stage_label,
        iteration,
        attempt,
        diag,
    )
    return diag


def collect_server_snapshot(
    client: K8sClient,
    *,
    namespace: str,
    label: str,
) -> list[dict[str, Any]]:
    """Snapshot server pods matching ``label``; log a warning, return rows.

    Fires once per failed stage from the runner's stage methods.
    Best-effort: every downstream call is wrapped so a missing
    server, vanished pod, events-listing failure, or log-read
    failure degrades to an empty row (or empty return) rather than
    raising. Diagnostics must never mask the primary failure they
    describe.

    Args:
        client: Injected :class:`K8sClient`.
        namespace: Target namespace.
        label: Label selector for the server pods, e.g.
            ``SERVER_LABEL`` or ``FORTIO_SERVER_LABEL``.

    Returns:
        One dict per server pod with ``name``, ``phase``,
        ``container_statuses`` (name, ready, restart_count,
        last_terminated.reason/exit_code), ``events``, and
        ``log_tail`` fields. Empty list when listing failed or no
        pods matched.
    """
    try:
        pods = client.list_pods_by_label(label, namespace)
    except Exception:
        logger.warning(
            "server snapshot [label=%s] list_pods_by_label failed",
            label,
            exc_info=True,
        )
        return []
    rows: list[dict[str, Any]] = []
    for pod in pods:
        name = str(getattr(getattr(pod, "metadata", None), "name", "") or "")
        phase = str(getattr(getattr(pod, "status", None), "phase", "") or "")
        container_statuses = _server_container_status_rows(pod)
        events: list[Any] = []
        log_tail = ""
        if name:
            try:
                events = list(
                    client._recent_pod_events(name, namespace, 10),  # noqa: SLF001
                )
            except Exception:  # noqa: BLE001 - best-effort diagnostic, swallow silently
                events = []
            try:
                log_tail = client._read_pod_log_tail(name, namespace, 100)  # noqa: SLF001
            except Exception:  # noqa: BLE001 - best-effort diagnostic, swallow silently
                log_tail = ""
        rows.append({
            "name": name,
            "phase": phase,
            "container_statuses": container_statuses,
            "events": events,
            "log_tail": log_tail,
        })
    logger.warning(
        "server snapshot [label=%s] pods=%d rows=%s",
        label,
        len(rows),
        rows,
    )
    return rows
