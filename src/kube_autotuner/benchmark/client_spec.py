"""Programmatic iperf3 client Job YAML builder."""

from __future__ import annotations

import json
import shlex
from typing import Literal


def _barrier_prologue(start_at_epoch: int | None) -> str:
    """Return the sleep-until-epoch shell prologue, or ``""`` when disabled.

    Emits no log output. ``kubectl logs`` merges stdout and stderr
    line-by-line; iperf3's JSON stdout is parsed with bare
    :func:`json.loads`, so a stray ``echo`` would corrupt the happy
    path. Operators can reconstruct drift between clients from each
    result's ``start.timestamp.timesecs``.

    Args:
        start_at_epoch: Absolute Unix timestamp the client should begin
            executing the benchmark tool at. ``None`` disables the
            barrier and the function returns ``""``.

    Returns:
        A shell fragment to splice between ``set -e`` and the
        ``exec iperf3 ...`` line, or ``""`` when no barrier is needed.
    """
    if start_at_epoch is None:
        return ""
    epoch = int(start_at_epoch)
    return (
        f"NOW=$(date +%s)\n"
        f"DELTA=$(( {epoch} - NOW ))\n"
        f'if [ "$DELTA" -gt 0 ]; then sleep "$DELTA"; fi\n'
    )


def build_client_yaml(  # noqa: PLR0913, PLR0917 - iperf3 client flag surface
    node: str,
    target: str,
    port: int,
    duration: int,
    omit: int,
    parallel: int,
    mode: Literal["tcp", "udp"],
    extra_args: list[str] | None = None,
    *,
    start_at_epoch: int | None = None,
) -> str:
    """Build a Job YAML that runs iperf3 in client mode against a given port.

    The Job runs its command under ``sh -c`` so an optional
    ``start_at_epoch`` barrier can sleep until a shared wall-clock
    target before ``exec``-ing iperf3. ``exec`` replaces the shell,
    so iperf3 becomes PID 1 inside the container and receives SIGTERM
    directly on Job deletion.

    Args:
        node: Kubernetes node name to pin the client pod to via
            ``nodeSelector``. Also embedded in the Job's name.
        target: Server node name; the client connects to the
            ``iperf3-server-<target>`` Service.
        port: Port on the server Service the client targets.
        duration: ``-t`` seconds to run.
        omit: ``-O`` seconds of warmup to omit from stats.
        parallel: ``-P`` number of parallel streams.
        mode: ``"tcp"`` (default) or ``"udp"``; ``"udp"`` appends
            ``-u -b 0`` to lift iperf3's per-stream 1 Mbit/sec UDP
            default. Callers can override ``-b`` via ``extra_args``.
        extra_args: Additional iperf3 flags appended after the controlled
            arguments. Reserved-flag enforcement happens in the
            experiment config layer, not here.
        start_at_epoch: Absolute Unix timestamp the client should
            ``sleep`` until before exec'ing iperf3. ``None`` disables
            the barrier and the container starts iperf3 immediately.

    Returns:
        The fully rendered Job manifest as a multi-line YAML string.
    """
    job_name = f"iperf3-client-{node}-p{port}"
    args: list[str] = [
        "-c",
        f"iperf3-server-{target}",
        "-p",
        str(port),
        "-t",
        str(duration),
        "-O",
        str(omit),
        "-P",
        str(parallel),
        "--get-server-output",
        "--json",
    ]
    if mode == "udp":
        args.extend(["-u", "-b", "0"])
    if extra_args:
        args.extend(extra_args)
    iperf3_cmd = shlex.join(["iperf3", *args])
    prologue = _barrier_prologue(start_at_epoch)
    script = f"set -e\n{prologue}exec {iperf3_cmd}\n"
    args_yaml = json.dumps([script])

    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  labels:
    app.kubernetes.io/name: iperf3-client
    app.kubernetes.io/instance: {job_name}
spec:
  template:
    metadata:
      labels:
        app.kubernetes.io/name: iperf3-client
        app.kubernetes.io/instance: {job_name}
    spec:
      nodeSelector:
        kubernetes.io/hostname: "{node}"
      tolerations:
        - operator: "Exists"
          effect: "NoSchedule"
      containers:
        - name: iperf3-client
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
