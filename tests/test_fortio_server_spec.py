"""Tests for :mod:`kube_autotuner.benchmark.fortio_server_spec`."""

from __future__ import annotations

import yaml

from kube_autotuner.benchmark.fortio_server_spec import build_fortio_server_yaml


def _split(body: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(body) if d]


def test_yaml_has_deployment_and_service():
    body = build_fortio_server_yaml(
        node="kmain08",
        ip_family_policy="SingleStack",
    )
    docs = _split(body)
    kinds = sorted(d["kind"] for d in docs)
    assert kinds == ["Deployment", "Service"]


def test_deployment_pins_to_node():
    body = build_fortio_server_yaml(
        node="kmain08",
        ip_family_policy="SingleStack",
    )
    dep = next(d for d in _split(body) if d["kind"] == "Deployment")
    spec = dep["spec"]["template"]["spec"]
    assert spec["nodeSelector"]["kubernetes.io/hostname"] == "kmain08"
    assert spec["tolerations"][0]["operator"] == "Exists"


def test_command_and_default_port():
    body = build_fortio_server_yaml(
        node="kmain08",
        ip_family_policy="SingleStack",
    )
    dep = next(d for d in _split(body) if d["kind"] == "Deployment")
    container = dep["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["fortio"]
    args = container["args"]
    assert args[:3] == ["server", "-http-port", ":8080"]


def test_extra_args_appended():
    body = build_fortio_server_yaml(
        node="kmain08",
        ip_family_policy="SingleStack",
        extra_args=["-echo-debug-path", "/debug"],
    )
    dep = next(d for d in _split(body) if d["kind"] == "Deployment")
    args = dep["spec"]["template"]["spec"]["containers"][0]["args"]
    assert args[-2:] == ["-echo-debug-path", "/debug"]


def test_service_ip_family_policy():
    body = build_fortio_server_yaml(
        node="kmain08",
        ip_family_policy="RequireDualStack",
    )
    svc = next(d for d in _split(body) if d["kind"] == "Service")
    assert svc["spec"]["ipFamilyPolicy"] == "RequireDualStack"
    assert svc["spec"]["ports"][0]["port"] == 8080


def test_labels():
    body = build_fortio_server_yaml(
        node="kmain08",
        ip_family_policy="SingleStack",
    )
    dep = next(d for d in _split(body) if d["kind"] == "Deployment")
    assert dep["metadata"]["labels"]["app.kubernetes.io/name"] == "fortio-server"
