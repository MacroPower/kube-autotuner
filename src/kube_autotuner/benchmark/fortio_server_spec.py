"""Programmatic fortio server Deployment + Service YAML builder."""

from __future__ import annotations

import json

_FORTIO_HTTP_PORT = 8080


def build_fortio_server_yaml(
    node: str,
    ip_family_policy: str,
    extra_args: list[str] | None = None,
) -> str:
    """Build multi-document YAML (Deployment + Service) for a fortio server.

    The ``nicolaka/netshoot`` image already ships ``fortio`` on ``PATH``
    so the container command is ``fortio server -http-port :<port>``.
    The server is pinned to ``node`` to match the iperf3 server shape,
    which keeps the latency sub-stages reading from the same
    target-node kernel/NIC path as the bandwidth sub-stage.

    Args:
        node: Kubernetes node name to pin the server pod to.
        ip_family_policy: ``spec.ipFamilyPolicy`` on the Service (e.g.
            ``"SingleStack"`` or ``"RequireDualStack"``).
        extra_args: Additional fortio server flags appended to the
            container's args list. Reserved-flag enforcement happens in
            the experiment config layer, not here.

    Returns:
        A multi-document YAML string with ``Deployment`` followed by
        ``Service``.
    """
    args = ["server", "-http-port", f":{_FORTIO_HTTP_PORT}"]
    if extra_args:
        args.extend(extra_args)
    args_yaml = json.dumps(args)

    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: fortio-server-{node}
  labels:
    app.kubernetes.io/name: fortio-server
    app.kubernetes.io/instance: fortio-server-{node}
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: fortio-server
      app.kubernetes.io/instance: fortio-server-{node}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: fortio-server
        app.kubernetes.io/instance: fortio-server-{node}
    spec:
      nodeSelector:
        kubernetes.io/hostname: "{node}"
      tolerations:
        - operator: "Exists"
          effect: "NoSchedule"
      containers:
        - name: fortio-server
          image: nicolaka/netshoot:v0.15
          command: ["fortio"]
          args: {args_yaml}
          ports:
            - containerPort: {_FORTIO_HTTP_PORT}
              protocol: TCP
          resources:
            requests:
              memory: "64Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: fortio-server-{node}
  labels:
    app.kubernetes.io/name: fortio-server
    app.kubernetes.io/instance: fortio-server-{node}
spec:
  type: ClusterIP
  ipFamilyPolicy: {ip_family_policy}
  selector:
    app.kubernetes.io/name: fortio-server
    app.kubernetes.io/instance: fortio-server-{node}
  ports:
    - name: http
      port: {_FORTIO_HTTP_PORT}
      targetPort: {_FORTIO_HTTP_PORT}
      protocol: TCP
"""
