"""Session-scoped fixtures for Talos Docker integration tests."""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING, Any
import uuid

import pytest

from kube_autotuner.k8s.client import Kubectl
from kube_autotuner.sysctl.setter import make_sysctl_setter_from_env

if TYPE_CHECKING:
    from pathlib import Path

CLUSTER_NAME = "kube-autotuner-test"
TALOSCTL = "talosctl"


def _cluster_running() -> bool:
    """Check if the test cluster is reachable.

    Prefers talosctl (no sudo needed) and falls back to sudo docker inspect
    when talosctl is unavailable, times out, or cannot confirm a live server.
    """
    try:
        talos = subprocess.run(
            ["talosctl", "-n", "10.5.0.2", "version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        # Require a Server section to distinguish a reachable Talos API from
        # a client-only response (talosctl returns rc=0 even when the server
        # is unreachable, printing only the client block).
        if (
            talos.returncode == 0
            and "Server:" in talos.stdout
            and "Tag:" in talos.stdout.split("Server:", 1)[1]
        ):
            return True
    except subprocess.TimeoutExpired, FileNotFoundError:
        pass
    docker = subprocess.run(
        ["sudo", "-n", "docker", "inspect", f"{CLUSTER_NAME}-controlplane-1"],
        capture_output=True,
        check=False,
    )
    return docker.returncode == 0


@pytest.fixture(scope="session")
def talos_cluster(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Connect to an existing Talos Docker cluster.

    The cluster must be created beforehand (see ``tests/integration/README.md``
    for the ``talosctl cluster create`` invocation).
    """
    if not _cluster_running():
        pytest.exit(
            f"Talos cluster '{CLUSTER_NAME}' is not running. "
            "See tests/integration/README.md for bring-up instructions.",
            returncode=1,
        )

    # Extract kubeconfig from the control plane node.
    # Docker provisioner uses 10.5.0.0/24 subnet; CP gets 10.5.0.2.
    kube_dir = tmp_path_factory.mktemp("kubeconfig")
    kubeconfig_path = str(kube_dir / "kubeconfig")

    # Discover CP IP from the Docker container.
    cp_ip_result = subprocess.run(
        [
            "sudo",
            "-n",
            "docker",
            "inspect",
            "-f",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            f"{CLUSTER_NAME}-controlplane-1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    cp_ip = cp_ip_result.stdout.strip() if cp_ip_result.returncode == 0 else "10.5.0.2"

    subprocess.run(
        [TALOSCTL, "kubeconfig", "--force", "--nodes", cp_ip, kubeconfig_path],
        check=True,
    )

    # Discover nodes.
    env = {**os.environ, "KUBECONFIG": kubeconfig_path}
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "nodes",
            "-o",
            "jsonpath={.items[*].metadata.name}",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    nodes = result.stdout.strip().split()

    # Classify CP vs worker by label.
    cp_nodes: list[str] = []
    worker_nodes: list[str] = []
    for node in nodes:
        label_result = subprocess.run(
            [
                "kubectl",
                "get",
                "node",
                node,
                "-o",
                "jsonpath={.metadata.labels.node-role\\.kubernetes\\.io/control-plane}",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if label_result.stdout.strip():
            cp_nodes.append(node)
        else:
            worker_nodes.append(node)

    return {
        "name": CLUSTER_NAME,
        "kubeconfig": kubeconfig_path,
        "controlplane_nodes": cp_nodes,
        "worker_nodes": worker_nodes,
        "all_nodes": nodes,
    }


@pytest.fixture(scope="session")
def kubeconfig_env(talos_cluster: dict[str, Any]) -> Any:
    """Set KUBECONFIG env var for the test session."""
    old = os.environ.get("KUBECONFIG")
    os.environ["KUBECONFIG"] = talos_cluster["kubeconfig"]
    yield talos_cluster["kubeconfig"]
    if old is not None:
        os.environ["KUBECONFIG"] = old
    else:
        os.environ.pop("KUBECONFIG", None)


@pytest.fixture(scope="session")
def kubectl(kubeconfig_env: str) -> Kubectl:  # noqa: ARG001 - activates env
    """Kubectl instance configured for the test cluster."""
    return Kubectl()


@pytest.fixture(scope="session")
def node_names(talos_cluster: dict[str, Any]) -> dict[str, str]:
    """Source and target node names from the test cluster."""
    if len(talos_cluster["worker_nodes"]) >= 2:
        return {
            "source": talos_cluster["worker_nodes"][0],
            "target": talos_cluster["worker_nodes"][1],
        }
    return {
        "source": talos_cluster["controlplane_nodes"][0],
        "target": talos_cluster["worker_nodes"][0],
    }


@pytest.fixture
def sysctls_available(
    kubectl: Kubectl, node_names: dict[str, str], test_namespace: str
) -> None:
    """Fail the test if sysctl read fails, unless KUBE_AUTOTUNER_ALLOW_SYSCTL_SKIP=1."""
    setter = make_sysctl_setter_from_env(
        node=node_names["target"],
        namespace=test_namespace,
        kubectl=kubectl,
    )
    try:
        setter.get(["net.core.rmem_max"])
    except Exception as e:
        if os.environ.get("KUBE_AUTOTUNER_ALLOW_SYSCTL_SKIP") == "1":
            pytest.skip(f"sysctl unavailable (skip allowed): {e}")
        raise


def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001 - hook signature
    items: list[pytest.Item],
) -> None:
    """Skip tests marked requires_real_sysctl_write when the opt-in env var is set.

    Writing host sysctls from a privileged pod is blocked on Talos Docker
    (EACCES on /proc/sys/net/core/* because the pod's userns does not own
    init_net). Tests that rely on this path are marked and skipped in that
    environment via KUBE_AUTOTUNER_ALLOW_SYSCTL_SKIP=1.
    """
    if os.environ.get("KUBE_AUTOTUNER_ALLOW_SYSCTL_SKIP") != "1":
        return
    skip = pytest.mark.skip(
        reason=(
            "sysctl write unavailable in this cluster "
            "(KUBE_AUTOTUNER_ALLOW_SYSCTL_SKIP=1)"
        )
    )
    for item in items:
        if "requires_real_sysctl_write" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fake_sysctl_state(tmp_path: Path) -> Path:
    """Per-test JSON state file for the fake sysctl backend."""
    return tmp_path / "fake_sysctl_state.json"


@pytest.fixture
def fake_sysctl_env(monkeypatch: pytest.MonkeyPatch, fake_sysctl_state: Path) -> Any:
    """Route sysctl calls through FakeSysctlBackend for the test's duration."""
    monkeypatch.setenv("KUBE_AUTOTUNER_SYSCTL_BACKEND", "fake")
    monkeypatch.setenv("KUBE_AUTOTUNER_SYSCTL_FAKE_STATE", str(fake_sysctl_state))
    return fake_sysctl_state


@pytest.fixture
def test_namespace(kubectl: Kubectl) -> Any:
    """Create an ephemeral namespace for a single test, clean up after."""
    ns = f"kube-autotuner-test-{uuid.uuid4().hex[:8]}"
    ns_yaml = (
        "apiVersion: v1\n"
        "kind: Namespace\n"
        "metadata:\n"
        f"  name: {ns}\n"
        "  labels:\n"
        "    pod-security.kubernetes.io/enforce: privileged\n"
    )
    kubectl.create(ns_yaml, "default")
    yield ns
    kubectl.delete("namespace", ns, "default")
