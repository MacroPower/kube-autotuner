"""Tests for :mod:`kube_autotuner.benchmark.fortio_client_spec`."""

from __future__ import annotations

import yaml

from kube_autotuner.benchmark.fortio_client_spec import build_fortio_client_yaml


def _parse(body: str) -> dict:
    docs = list(yaml.safe_load_all(body))
    assert len(docs) == 1
    return docs[0]


def test_job_name_includes_node_workload_iteration():
    yaml_ = build_fortio_client_yaml(
        node="kmain07",
        target="kmain08",
        iteration=2,
        workload="saturation",
        qps=0,
        connections=4,
        duration=30,
    )
    doc = _parse(yaml_)
    assert doc["kind"] == "Job"
    assert doc["metadata"]["name"] == "fortio-client-kmain07-saturation-i2"


def test_saturation_uses_qps_zero():
    yaml_ = build_fortio_client_yaml(
        node="c1",
        target="t",
        iteration=0,
        workload="saturation",
        qps=0,
        connections=8,
        duration=10,
    )
    doc = _parse(yaml_)
    args = doc["spec"]["template"]["spec"]["containers"][0]["args"]
    assert args[0] == "load"
    assert "-qps" in args
    assert args[args.index("-qps") + 1] == "0"
    assert args[args.index("-c") + 1] == "8"
    assert args[args.index("-t") + 1] == "10s"
    assert args[args.index("-json") + 1] == "-"
    assert args[-1] == "http://fortio-server-t:8080/"


def test_fixed_qps_uses_configured_rate():
    yaml_ = build_fortio_client_yaml(
        node="c1",
        target="t",
        iteration=0,
        workload="fixed_qps",
        qps=1500,
        connections=4,
        duration=30,
    )
    doc = _parse(yaml_)
    args = doc["spec"]["template"]["spec"]["containers"][0]["args"]
    assert args[args.index("-qps") + 1] == "1500"


def test_extra_args_appended_before_url():
    yaml_ = build_fortio_client_yaml(
        node="c1",
        target="t",
        iteration=0,
        workload="fixed_qps",
        qps=100,
        connections=4,
        duration=5,
        extra_args=["-payload-size", "128"],
    )
    doc = _parse(yaml_)
    args = doc["spec"]["template"]["spec"]["containers"][0]["args"]
    # Extra args appear before the URL and after the controlled args.
    assert "-payload-size" in args
    assert args.index("-payload-size") > args.index("-json")
    assert args[-1].startswith("http://")


def test_labels_include_component_workload():
    yaml_ = build_fortio_client_yaml(
        node="c1",
        target="t",
        iteration=0,
        workload="saturation",
        qps=0,
        connections=4,
        duration=5,
    )
    doc = _parse(yaml_)
    labels = doc["metadata"]["labels"]
    assert labels["app.kubernetes.io/name"] == "fortio-client"
    assert labels["app.kubernetes.io/component"] == "saturation"


def test_node_selector_and_tolerations():
    yaml_ = build_fortio_client_yaml(
        node="c1",
        target="t",
        iteration=0,
        workload="fixed_qps",
        qps=100,
        connections=4,
        duration=5,
    )
    doc = _parse(yaml_)
    pod_spec = doc["spec"]["template"]["spec"]
    assert pod_spec["nodeSelector"]["kubernetes.io/hostname"] == "c1"
    assert pod_spec["tolerations"][0]["operator"] == "Exists"
