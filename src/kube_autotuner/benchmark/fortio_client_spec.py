"""Programmatic fortio client Job YAML builder."""

from __future__ import annotations

import json
import shlex
from typing import Literal

_FORTIO_HTTP_PORT = 8080
_FORTIO_RESULT_FILE = "/tmp/fortio-result.json"  # noqa: S108

Workload = Literal["saturation", "fixed_qps"]


def _barrier_prologue(start_at_epoch: int | None) -> str:
    """Return the sleep-until-epoch shell prologue, or ``""`` when disabled.

    Emits no log output. ``kubectl logs`` merges stdout and stderr
    line-by-line; a stray ``echo`` would leak into the fortio logger's
    own stderr stream and interleave inside the merged log that
    :func:`extract_fortio_result_json` scans. Operators can reconstruct
    drift between clients from each result's ``StartTime``.

    Args:
        start_at_epoch: Absolute Unix timestamp the client should begin
            executing fortio at. ``None`` disables the barrier and the
            function returns ``""``.

    Returns:
        A ``;``-terminated shell fragment to splice after ``set -e ;``
        and before the fortio argv, or ``""`` when no barrier is needed.
    """
    if start_at_epoch is None:
        return ""
    epoch = int(start_at_epoch)
    return (
        f"NOW=$(date +%s) ; "
        f"DELTA=$(( {epoch} - NOW )) ; "
        f'if [ "$DELTA" -gt 0 ]; then sleep "$DELTA"; fi ; '
    )


def fortio_client_job_name(node: str, workload: Workload, iteration: int) -> str:
    """Return the RFC 1123 subdomain Job name for a fortio client run.

    The ``fixed_qps`` workload literal contains an underscore, which the
    Kubernetes API rejects in ``metadata.name``. The slug here normalises
    underscores to hyphens so both this builder and the runner agree on
    the same (valid) Job name.

    Args:
        node: Client node name; embedded verbatim.
        workload: Workload literal; underscores are replaced with
            hyphens before embedding.
        iteration: Zero-based iteration index.

    Returns:
        A Job name that satisfies the RFC 1123 subdomain regex.
    """
    slug = workload.replace("_", "-")
    return f"fortio-client-{node}-{slug}-i{iteration}"


def build_fortio_client_yaml(  # noqa: PLR0913, PLR0917 - fortio load flag surface
    node: str,
    target: str,
    iteration: int,
    workload: Workload,
    qps: int,
    connections: int,
    duration: int,
    extra_args: list[str] | None = None,
    *,
    start_at_epoch: int | None = None,
) -> str:
    """Build a Job YAML that runs ``fortio load`` against the fortio server.

    Args:
        node: Kubernetes node name to pin the client pod to via
            ``nodeSelector``. Also embedded in the Job's name.
        target: Server node name; the client connects to
            ``fortio-server-<target>`` Service.
        iteration: Zero-based iteration index; embedded in the Job
            name to prevent cross-iteration name collisions during
            cleanup.
        workload: ``"saturation"`` (``-qps 0``, drives max rate) or
            ``"fixed_qps"`` (drives the configured offered load).
        qps: Target QPS forwarded to fortio's ``-qps`` flag. For
            saturation runs, pass ``0``.
        connections: Concurrent connections forwarded to ``-c``.
        duration: Test duration in seconds; rendered as ``-t <n>s``.
        extra_args: Additional fortio flags appended after the
            controlled arguments. Reserved-flag enforcement happens in
            the experiment config layer, not here.
        start_at_epoch: Absolute Unix timestamp the client should
            ``sleep`` until before invoking ``fortio load``. ``None``
            disables the barrier and the container runs fortio
            immediately.

    Returns:
        The fully rendered Job manifest as a multi-line YAML string.
    """
    job_name = fortio_client_job_name(node, workload, iteration)
    # Write the result document to a file and ``cat`` it back on stdout
    # AFTER fortio exits. fortio's logger writes JSON-formatted records
    # to stderr during the run; ``kubectl logs`` merges stdout and
    # stderr line-by-line, so writing the result JSON to stdout would
    # let those stderr log lines interleave inside the pretty-printed
    # result document and break parsing. With this two-step shell
    # wrapper, the result JSON arrives as one contiguous block at the
    # end of the merged log.
    fortio_argv: list[str] = [
        "fortio",
        "load",
        "-qps",
        str(qps),
        "-c",
        str(connections),
        "-t",
        f"{duration}s",
        "-json",
        _FORTIO_RESULT_FILE,
    ]
    if extra_args:
        fortio_argv.extend(extra_args)
    fortio_argv.append(f"http://fortio-server-{target}:{_FORTIO_HTTP_PORT}/")
    prologue = _barrier_prologue(start_at_epoch)
    script = (
        f"set -e ; {prologue}{shlex.join(fortio_argv)} ; "
        f"cat {shlex.quote(_FORTIO_RESULT_FILE)}"
    )
    args_yaml = json.dumps([script])

    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  labels:
    app.kubernetes.io/name: fortio-client
    app.kubernetes.io/instance: {job_name}
    app.kubernetes.io/component: {workload}
spec:
  template:
    metadata:
      labels:
        app.kubernetes.io/name: fortio-client
        app.kubernetes.io/instance: {job_name}
        app.kubernetes.io/component: {workload}
    spec:
      nodeSelector:
        kubernetes.io/hostname: "{node}"
      tolerations:
        - operator: "Exists"
          effect: "NoSchedule"
      containers:
        - name: fortio-client
          image: nicolaka/netshoot:v0.15
          command: ["sh", "-c"]
          args: {args_yaml}
          resources:
            requests:
              memory: "64Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
      restartPolicy: Never
  backoffLimit: 1
"""
