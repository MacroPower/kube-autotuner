"""Programmatic iperf3 client Job YAML builder."""

from __future__ import annotations

import json
from typing import Literal


def build_client_yaml(  # noqa: PLR0913, PLR0917 - iperf3 client flag surface
    node: str,
    target: str,
    port: int,
    duration: int,
    omit: int,
    parallel: int,
    mode: Literal["tcp", "udp"],
    window: str | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """Build a Job YAML that runs iperf3 in client mode against a given port.

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
        window: Optional ``-w`` TCP window size argument.
        extra_args: Additional iperf3 flags appended after the controlled
            arguments. Reserved-flag enforcement happens in the
            experiment config layer, not here.

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
    if window:
        args.extend(["-w", window])
    if extra_args:
        args.extend(extra_args)
    args_yaml = json.dumps(args)

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
          command: ["iperf3"]
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
