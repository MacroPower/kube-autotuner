from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from kube_autotuner.k8s.client import Kubectl, KubectlError


def _mock_run(output: str):
    """Return a patcher that makes Kubectl._run return the given output."""
    return patch.object(Kubectl, "_run", return_value=output)


def test_top_pod_parses_output():
    kubectl = Kubectl()
    with _mock_run("iperf3-server-kmain08-abc12   250m   45Mi"):
        result = kubectl.top_pod("iperf3-server-kmain08-abc12", "default")
    assert result == {"cpu": "250m", "memory": "45Mi"}


def test_top_pod_empty_output():
    kubectl = Kubectl()
    with _mock_run(""):
        result = kubectl.top_pod("missing-pod", "default")
    assert result == {}


def test_get_pod_name():
    kubectl = Kubectl()
    with _mock_run("iperf3-server-kmain08-abc12"):
        name = kubectl.get_pod_name("app=iperf3-server", "default")
    assert name == "iperf3-server-kmain08-abc12"


def test_get_node_zone():
    kubectl = Kubectl()
    with _mock_run("az01"):
        zone = kubectl.get_node_zone("kmain07")
    assert zone == "az01"


def test_get_node_zone_empty():
    kubectl = Kubectl()
    with _mock_run(""):
        zone = kubectl.get_node_zone("kmain07")
    assert zone == ""


def test_get_node_internal_ip():
    kubectl = Kubectl()
    with _mock_run("10.5.0.2\n"):
        ip = kubectl.get_node_internal_ip("kmain07")
    assert ip == "10.5.0.2"


def test_get_node_internal_ip_empty():
    kubectl = Kubectl()
    with _mock_run(""):
        ip = kubectl.get_node_internal_ip("kmain07")
    assert ip == ""


def test_get_json():
    lease_data = {"metadata": {"name": "test"}, "spec": {"holderIdentity": "me"}}
    kubectl = Kubectl()
    with _mock_run(json.dumps(lease_data)):
        result = kubectl.get_json("lease", "test", "default")
    assert result == lease_data


def test_get_json_not_found():
    kubectl = Kubectl()
    with patch.object(
        Kubectl,
        "_run",
        side_effect=KubectlError(
            ["kubectl"], 1, 'Error from server (NotFound): leases "x" not found'
        ),
    ):
        result = kubectl.get_json("lease", "x", "default")
    assert result is None


def test_get_json_other_error():
    kubectl = Kubectl()
    with (
        patch.object(
            Kubectl,
            "_run",
            side_effect=KubectlError(["kubectl"], 1, "connection refused"),
        ),
        pytest.raises(KubectlError),
    ):
        kubectl.get_json("lease", "x", "default")


def test_create_success():
    kubectl = Kubectl()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type(
            "R", (), {"returncode": 0, "stdout": "", "stderr": ""}
        )()
        kubectl.create("apiVersion: v1\nkind: Lease", "default")
    args = mock_run.call_args[0][0]
    assert "create" in args


def test_create_already_exists():
    kubectl = Kubectl()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type(
            "R",
            (),
            {"returncode": 1, "stdout": "", "stderr": "Error: AlreadyExists"},
        )()
        with pytest.raises(KubectlError, match="AlreadyExists"):
            kubectl.create("apiVersion: v1\nkind: Lease", "default")


def test_replace_success():
    kubectl = Kubectl()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type(
            "R", (), {"returncode": 0, "stdout": "", "stderr": ""}
        )()
        kubectl.replace("apiVersion: v1\nkind: Lease", "default")
    args = mock_run.call_args[0][0]
    assert "replace" in args


def test_replace_conflict():
    kubectl = Kubectl()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type(
            "R",
            (),
            {"returncode": 1, "stdout": "", "stderr": "Error: Conflict"},
        )()
        with pytest.raises(KubectlError, match="Conflict"):
            kubectl.replace("apiVersion: v1\nkind: Lease", "default")


def test_kubectl_error_str_includes_stdout():
    err = KubectlError(
        ["kubectl", "apply"],
        1,
        "stderr text",
        stdout="stdout text",
    )
    text = str(err)
    assert "stderr text" in text
    assert "stdout text" in text
