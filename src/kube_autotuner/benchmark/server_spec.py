"""Programmatic multi-port iperf3 server Deployment + Service YAML builder."""

from __future__ import annotations

import json


def _container_block(port: int, extra_args: list[str] | None = None) -> str:
    """Render one container entry for a single iperf3 server port.

    iperf3 in ``-s`` mode listens on both TCP and UDP for the same
    port number with a single process, so we expose both
    ``containerPort`` entries -- Kubernetes Services route protocols
    independently, and the UDP route is required for the ``bw-udp``
    bandwidth stage.

    Returns:
        The container block as indented multi-line YAML.
    """
    args = ["-s", "-p", str(port)]
    if extra_args:
        args.extend(extra_args)
    args_yaml = json.dumps(args)
    return f"""        - name: iperf3-server-{port}
          image: nicolaka/netshoot:v0.15
          command: ["iperf3"]
          args: {args_yaml}
          ports:
            - containerPort: {port}
              protocol: TCP
            - containerPort: {port}
              protocol: UDP
          resources:
            requests:
              memory: "64Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"\
"""


def _service_port_block(port: int) -> str:
    """Render the Service port entries for a single iperf3 server port.

    Emits both a TCP and a UDP entry on the same port number so the
    ``bw-tcp`` and ``bw-udp`` bandwidth stages can each route to the
    server pod. kube-proxy programs the two protocols independently
    on its side, so the Service object must declare them both
    explicitly.

    Returns:
        The Service port blocks as indented multi-line YAML.
    """
    return f"""    - name: iperf-{port}-tcp
      port: {port}
      targetPort: {port}
      protocol: TCP
    - name: iperf-{port}-udp
      port: {port}
      targetPort: {port}
      protocol: UDP\
"""


def build_server_yaml(
    node: str,
    ports: list[int],
    ip_family_policy: str,
    extra_args: list[str] | None = None,
) -> str:
    """Build multi-document YAML (Deployment + Service) for a multi-port iperf3 server.

    Each port gets its own container and Service port. Container names
    are unique (``iperf3-server-<port>``) so a single Pod can host all of
    them.

    Args:
        node: Kubernetes node name to pin the server pod to.
        ports: Ports to expose; one container and one Service port per
            entry. Must be non-empty.
        ip_family_policy: ``spec.ipFamilyPolicy`` on the Service (e.g.
            ``"SingleStack"`` or ``"RequireDualStack"``).
        extra_args: Additional iperf3 flags appended to every container's
            args list.

    Returns:
        A multi-document YAML string with ``Deployment`` followed by
        ``Service``.

    Raises:
        ValueError: When ``ports`` is empty.
    """
    if not ports:
        msg = "ports must be non-empty"
        raise ValueError(msg)

    containers = "\n".join(_container_block(p, extra_args) for p in ports)
    service_ports = "\n".join(_service_port_block(p) for p in ports)

    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: iperf3-server-{node}
  labels:
    app.kubernetes.io/name: iperf3-server
    app.kubernetes.io/instance: iperf3-server-{node}
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: iperf3-server
      app.kubernetes.io/instance: iperf3-server-{node}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: iperf3-server
        app.kubernetes.io/instance: iperf3-server-{node}
    spec:
      nodeSelector:
        kubernetes.io/hostname: "{node}"
      tolerations:
        - operator: "Exists"
          effect: "NoSchedule"
      containers:
{containers}
---
apiVersion: v1
kind: Service
metadata:
  name: iperf3-server-{node}
  labels:
    app.kubernetes.io/name: iperf3-server
    app.kubernetes.io/instance: iperf3-server-{node}
spec:
  type: ClusterIP
  ipFamilyPolicy: {ip_family_policy}
  selector:
    app.kubernetes.io/name: iperf3-server
    app.kubernetes.io/instance: iperf3-server-{node}
  ports:
{service_ports}
"""
