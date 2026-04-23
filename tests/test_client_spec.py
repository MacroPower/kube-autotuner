"""Unit tests for :mod:`kube_autotuner.benchmark.client_spec`."""

from __future__ import annotations

from typing import Any

import yaml

from kube_autotuner.benchmark.client_spec import build_client_yaml


def _parse(rendered: str) -> dict[str, Any]:
    return yaml.safe_load(rendered)


def test_build_tcp_default_port():
    doc = _parse(
        build_client_yaml(
            node="kmain08",
            target="kmain07",
            port=5201,
            duration=30,
            omit=5,
            parallel=16,
            mode="tcp",
        )
    )
    assert doc["kind"] == "Job"
    assert doc["metadata"]["name"] == "iperf3-client-kmain08-p5201"
    node_sel = doc["spec"]["template"]["spec"]["nodeSelector"]
    assert node_sel["kubernetes.io/hostname"] == "kmain08"

    container = doc["spec"]["template"]["spec"]["containers"][0]
    assert container["image"].startswith("nicolaka/netshoot")
    assert container["command"] == ["iperf3"]
    args = container["args"]
    assert "-c" in args
    assert "iperf3-server-kmain07" in args
    assert "-p" in args
    assert "5201" in args
    assert "--json" in args
    assert "--get-server-output" in args
    assert "-u" not in args
    assert "-b" not in args


def test_build_custom_port():
    doc = _parse(
        build_client_yaml(
            node="kmain09",
            target="kmain07",
            port=5202,
            duration=30,
            omit=5,
            parallel=16,
            mode="tcp",
        )
    )
    assert doc["metadata"]["name"] == "iperf3-client-kmain09-p5202"
    args = doc["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "5202" in args


def test_build_udp_and_window():
    doc = _parse(
        build_client_yaml(
            node="kmain08",
            target="kmain07",
            port=5201,
            duration=30,
            omit=5,
            parallel=16,
            mode="udp",
            window="256K",
        )
    )
    args = doc["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "-u" in args
    assert "-w" in args
    assert "256K" in args
    assert args.index("-b") == args.index("-u") + 1
    assert args[args.index("-b") + 1] == "0"


def test_build_extra_args_appended_after_defaults():
    doc = _parse(
        build_client_yaml(
            node="kmain08",
            target="kmain07",
            port=5201,
            duration=30,
            omit=5,
            parallel=16,
            mode="tcp",
            extra_args=["--bidir", "-Z"],
        )
    )
    args = doc["spec"]["template"]["spec"]["containers"][0]["args"]
    # Defaults still present and in original order.
    assert args.index("-c") < args.index("--json")
    # Extras appended at the end.
    assert args[-2:] == ["--bidir", "-Z"]


def test_build_extra_args_preserved_with_udp():
    doc = _parse(
        build_client_yaml(
            node="kmain08",
            target="kmain07",
            port=5201,
            duration=30,
            omit=5,
            parallel=16,
            mode="udp",
            extra_args=["-Z"],
        )
    )
    args = doc["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "-u" in args
    assert "-Z" in args
    assert args.index("-u") < args.index("-Z")
    assert args.index("-b") < args.index("-Z")


def test_build_udp_bitrate_override():
    doc = _parse(
        build_client_yaml(
            node="kmain08",
            target="kmain07",
            port=5201,
            duration=30,
            omit=5,
            parallel=16,
            mode="udp",
            extra_args=["-b", "500M"],
        )
    )
    args = doc["spec"]["template"]["spec"]["containers"][0]["args"]
    b_indices = [i for i, a in enumerate(args) if a == "-b"]
    assert len(b_indices) == 2
    assert args[b_indices[0] + 1] == "0"
    assert args[b_indices[1] + 1] == "500M"
    assert b_indices[0] < b_indices[1]
