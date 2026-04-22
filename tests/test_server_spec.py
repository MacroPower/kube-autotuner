"""Unit tests for :mod:`kube_autotuner.benchmark.server_spec`."""

from __future__ import annotations

import pytest
import yaml

from kube_autotuner.benchmark.server_spec import build_server_yaml


def test_build_single_port():
    rendered = build_server_yaml("kmain07", [5201], "RequireDualStack")
    docs = list(yaml.safe_load_all(rendered))
    assert len(docs) == 2

    dep = docs[0]
    assert dep["kind"] == "Deployment"
    assert dep["metadata"]["name"] == "iperf3-server-kmain07"
    node_sel = dep["spec"]["template"]["spec"]["nodeSelector"]
    assert node_sel["kubernetes.io/hostname"] == "kmain07"

    containers = dep["spec"]["template"]["spec"]["containers"]
    assert len(containers) == 1
    assert containers[0]["name"] == "iperf3-server-5201"
    assert containers[0]["image"].startswith("nicolaka/netshoot")
    assert containers[0]["args"] == ["-s", "-p", "5201"]
    # Each port exposes both TCP and UDP so the bw-udp stage can reach it.
    assert [p["protocol"] for p in containers[0]["ports"]] == ["TCP", "UDP"]
    assert all(p["containerPort"] == 5201 for p in containers[0]["ports"])

    svc = docs[1]
    assert svc["kind"] == "Service"
    assert svc["spec"]["ipFamilyPolicy"] == "RequireDualStack"
    ports = svc["spec"]["ports"]
    # One TCP entry and one UDP entry per declared port.
    assert len(ports) == 2
    assert [p["name"] for p in ports] == ["iperf-5201-tcp", "iperf-5201-udp"]
    assert all(p["port"] == 5201 for p in ports)
    assert [p["protocol"] for p in ports] == ["TCP", "UDP"]


def test_build_multi_port():
    rendered = build_server_yaml("kmain08", [5201, 5202], "SingleStack")
    docs = list(yaml.safe_load_all(rendered))
    dep = docs[0]
    containers = dep["spec"]["template"]["spec"]["containers"]
    assert len(containers) == 2
    names = [c["name"] for c in containers]
    assert names == ["iperf3-server-5201", "iperf3-server-5202"]
    assert containers[0]["args"] == ["-s", "-p", "5201"]
    assert containers[1]["args"] == ["-s", "-p", "5202"]
    # Container names must be unique per Kubernetes API.
    assert len(names) == len(set(names))

    svc = docs[1]
    assert svc["spec"]["ipFamilyPolicy"] == "SingleStack"
    ports = svc["spec"]["ports"]
    # Each declared port contributes a TCP and a UDP Service entry.
    assert len(ports) == 4
    assert [p["name"] for p in ports] == [
        "iperf-5201-tcp",
        "iperf-5201-udp",
        "iperf-5202-tcp",
        "iperf-5202-udp",
    ]
    assert [p["port"] for p in ports] == [5201, 5201, 5202, 5202]
    assert [p["protocol"] for p in ports] == ["TCP", "UDP", "TCP", "UDP"]


def test_build_requires_ports():
    with pytest.raises(ValueError, match="ports must be non-empty"):
        build_server_yaml("kmain07", [], "RequireDualStack")


def test_build_extra_args_appended_to_every_container():
    rendered = build_server_yaml(
        "kmain08",
        [5201, 5202],
        "RequireDualStack",
        extra_args=["--forceflush"],
    )
    docs = list(yaml.safe_load_all(rendered))
    containers = docs[0]["spec"]["template"]["spec"]["containers"]
    assert containers[0]["args"] == ["-s", "-p", "5201", "--forceflush"]
    assert containers[1]["args"] == ["-s", "-p", "5202", "--forceflush"]
