"""Tests for :mod:`kube_autotuner.benchmark.fortio_client_spec`."""

from __future__ import annotations

import shlex

import yaml

from kube_autotuner.benchmark.fortio_client_spec import (
    build_fortio_client_yaml,
    fortio_client_job_name,
)


def _parse(body: str) -> dict:
    docs = list(yaml.safe_load_all(body))
    assert len(docs) == 1
    return docs[0]


def _container(doc: dict) -> dict:
    return doc["spec"]["template"]["spec"]["containers"][0]


def _fortio_argv(doc: dict) -> list[str]:
    """Tokenise the fortio invocation out of the shell wrapper script.

    Returns:
        The ``fortio`` argv slice up to (but not including) the
        terminating ``;`` that separates the load command from the
        trailing ``cat`` step.
    """
    args = _container(doc)["args"]
    assert len(args) == 1, "expected a single shell-script arg"
    tokens = shlex.split(args[0])
    fortio_idx = tokens.index("fortio")
    end_marker = tokens.index(";", fortio_idx)
    return tokens[fortio_idx:end_marker]


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
    argv = _fortio_argv(_parse(yaml_))
    assert argv[:2] == ["fortio", "load"]
    assert argv[argv.index("-qps") + 1] == "0"
    assert argv[argv.index("-c") + 1] == "8"
    assert argv[argv.index("-t") + 1] == "10s"
    # `-json` writes to a file (not stdout) so the result document
    # can be cat'd back as one contiguous block by the wrapper script.
    assert argv[argv.index("-json") + 1].endswith(".json")
    assert argv[-1] == "http://fortio-server-t:8080/"


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
    argv = _fortio_argv(_parse(yaml_))
    assert argv[argv.index("-qps") + 1] == "1500"


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
    argv = _fortio_argv(_parse(yaml_))
    assert "-payload-size" in argv
    assert argv.index("-payload-size") > argv.index("-json")
    assert argv[-1].startswith("http://")


def test_wrapper_writes_result_to_file_then_cats_it():
    """Fortio's stderr can interleave with stdout in ``kubectl logs``.

    Writing the result JSON to a file and ``cat``ing it afterward keeps
    the result document contiguous at the end of the merged log.
    """
    yaml_ = build_fortio_client_yaml(
        node="c1",
        target="t",
        iteration=0,
        workload="saturation",
        qps=0,
        connections=4,
        duration=5,
    )
    container = _container(_parse(yaml_))
    assert container["command"] == ["sh", "-c"]
    script = container["args"][0]
    # ``set -e`` ensures a fortio failure short-circuits before cat,
    # so the Job goes Failed (rather than producing a partial log).
    assert script.startswith("set -e ;")
    tokens = shlex.split(script)
    cat_idx = tokens.index("cat")
    fortio_json_idx = tokens.index("-json")
    result_path = tokens[fortio_json_idx + 1]
    # cat consumes the same file fortio wrote, and runs after fortio.
    assert tokens[cat_idx + 1] == result_path
    assert cat_idx > fortio_json_idx


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


def test_fixed_qps_job_name_is_rfc1123_compliant():
    """Underscore in the ``fixed_qps`` literal must be slugified for k8s names."""
    yaml_ = build_fortio_client_yaml(
        node="kmain04",
        target="kmain06",
        iteration=0,
        workload="fixed_qps",
        qps=1500,
        connections=4,
        duration=30,
    )
    doc = _parse(yaml_)
    name = doc["metadata"]["name"]
    assert name == "fortio-client-kmain04-fixed-qps-i0"
    assert "_" not in name
    # Labels still carry the original literal so downstream selectors and
    # parsers don't have to translate the slug back.
    assert doc["metadata"]["labels"]["app.kubernetes.io/component"] == "fixed_qps"


def test_fortio_client_job_name_helper_matches_builder():
    """The standalone helper agrees with the YAML builder so callers stay in sync."""
    yaml_ = build_fortio_client_yaml(
        node="kmain04",
        target="kmain06",
        iteration=2,
        workload="fixed_qps",
        qps=1500,
        connections=4,
        duration=30,
    )
    expected = fortio_client_job_name("kmain04", "fixed_qps", 2)
    assert _parse(yaml_)["metadata"]["name"] == expected
    assert expected == "fortio-client-kmain04-fixed-qps-i2"


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
