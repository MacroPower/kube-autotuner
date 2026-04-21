"""Programmatic fortio client Job YAML builder."""

from __future__ import annotations

import json
from typing import Literal

_FORTIO_HTTP_PORT = 8080

Workload = Literal["saturation", "fixed_qps"]


def build_fortio_client_yaml(  # noqa: PLR0913, PLR0917 - fortio load flag surface
    node: str,
    target: str,
    iteration: int,
    workload: Workload,
    qps: int,
    connections: int,
    duration: int,
    extra_args: list[str] | None = None,
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

    Returns:
        The fully rendered Job manifest as a multi-line YAML string.
    """
    job_name = f"fortio-client-{node}-{workload}-i{iteration}"
    args: list[str] = [
        "load",
        "-qps",
        str(qps),
        "-c",
        str(connections),
        "-t",
        f"{duration}s",
        "-json",
        "-",
    ]
    if extra_args:
        args.extend(extra_args)
    args.append(f"http://fortio-server-{target}:{_FORTIO_HTTP_PORT}/")
    args_yaml = json.dumps(args)

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
          command: ["fortio"]
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
