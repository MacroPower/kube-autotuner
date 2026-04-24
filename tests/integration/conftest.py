"""Session-scoped fixtures for Talos Docker integration tests."""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING, Any
import uuid

from kubernetes import client as _k8s, config as _k8s_config
import pytest

from kube_autotuner.k8s.client import K8sClient
from kube_autotuner.sysctl.setter import make_sysctl_setter_from_env

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
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

    # Discover nodes via the typed API.
    _k8s_config.load_kube_config(config_file=kubeconfig_path)
    core = _k8s.CoreV1Api()
    node_items = core.list_node().items

    cp_nodes: list[str] = []
    worker_nodes: list[str] = []
    for node in node_items:
        labels = node.metadata.labels or {}
        if labels.get("node-role.kubernetes.io/control-plane") is not None:
            cp_nodes.append(node.metadata.name)
        else:
            worker_nodes.append(node.metadata.name)

    return {
        "name": CLUSTER_NAME,
        "kubeconfig": kubeconfig_path,
        "controlplane_nodes": cp_nodes,
        "worker_nodes": worker_nodes,
        "all_nodes": [*cp_nodes, *worker_nodes],
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
def k8s_client(kubeconfig_env: str) -> K8sClient:  # noqa: ARG001 - activates env
    """K8sClient instance configured for the test cluster."""
    return K8sClient()


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
    k8s_client: K8sClient, node_names: dict[str, str], test_namespace: str
) -> None:
    """Fail the test if sysctl read fails, unless KUBE_AUTOTUNER_ALLOW_SYSCTL_SKIP=1."""
    setter = make_sysctl_setter_from_env(
        node=node_names["target"],
        namespace=test_namespace,
        client=k8s_client,
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
def write_experiment_yaml(
    tmp_path: Path,
) -> Callable[..., Path]:
    """Return a factory that materializes experiment YAML in ``tmp_path``.

    Defaults to ``SingleStack`` and ``1g`` because the Talos Docker test
    cluster has no dual-stack CNI and the tests never assert against
    another hardware class. ``trial_sysctls`` makes the config valid
    for the ``trial`` subcommand; ``optimize`` makes it valid for
    ``optimize``; omitting both leaves a ``baseline``-shaped config.
    """

    def _write(  # noqa: PLR0913 - an experiment has many knobs
        *,
        node_names: Mapping[str, str],
        namespace: str,
        output: Path,
        hardware_class: str = "1g",
        ip_family_policy: str = "SingleStack",
        duration: int = 5,
        iterations: int = 1,
        trial_sysctls: Mapping[str, str] | None = None,
        optimize: Mapping[str, int] | None = None,
        filename: str = "experiment.yaml",
    ) -> Path:
        lines = [
            "nodes:",
            f"  sources: [{node_names['source']}]",
            f"  target: {node_names['target']}",
            f"  hardwareClass: {hardware_class}",
            f"  namespace: {namespace}",
            f"  ipFamilyPolicy: {ip_family_policy}",
            "benchmark:",
            f"  duration: {duration}",
            f"  iterations: {iterations}",
        ]
        if trial_sysctls is not None:
            lines.extend(("trial:", "  sysctls:"))
            lines.extend(
                f"    {key}: {value!r}" for key, value in trial_sysctls.items()
            )
        if optimize is not None:
            lines.append("optimize:")
            lines.extend(f"  {key}: {value}" for key, value in optimize.items())
        lines.append(f"output: {output}")
        config_path = tmp_path / filename
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return config_path

    return _write


@pytest.fixture
def test_namespace(k8s_client: K8sClient) -> Any:
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
    k8s_client.create(ns_yaml, "default")
    yield ns
    k8s_client.delete("namespace", ns, "default")
