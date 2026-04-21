from __future__ import annotations

import yaml

from kube_autotuner.k8s.templates import render_template


def test_render_lease():
    variables = {
        "LEASE_NAME": "kube-autotuner-lock-kmain07",
        "LEASE_NAMESPACE": "default",
        "HOLDER_ID": "kube-autotuner-abc12345",
        "LEASE_TTL": "900",
        "ACQUIRE_TIME": "2026-04-17T00:00:00.000000Z",
        "RENEW_TIME": "2026-04-17T00:00:00.000000Z",
        "RESOURCE_VERSION_LINE": "",
    }
    rendered = render_template("lease.yaml", variables)
    doc = yaml.safe_load(rendered)
    assert doc["apiVersion"] == "coordination.k8s.io/v1"
    assert doc["kind"] == "Lease"
    assert doc["metadata"]["name"] == "kube-autotuner-lock-kmain07"
    assert doc["metadata"]["namespace"] == "default"
    assert doc["spec"]["holderIdentity"] == "kube-autotuner-abc12345"
    assert doc["spec"]["leaseDurationSeconds"] == 900
    assert "resourceVersion" not in doc["metadata"]


def test_render_lease_with_resource_version():
    variables = {
        "LEASE_NAME": "kube-autotuner-lock-kmain07",
        "LEASE_NAMESPACE": "default",
        "HOLDER_ID": "kube-autotuner-abc12345",
        "LEASE_TTL": "900",
        "ACQUIRE_TIME": "2026-04-17T00:00:00.000000Z",
        "RENEW_TIME": "2026-04-17T00:00:00.000000Z",
        "RESOURCE_VERSION_LINE": 'resourceVersion: "12345"',
    }
    rendered = render_template("lease.yaml", variables)
    doc = yaml.safe_load(rendered)
    assert doc["metadata"]["resourceVersion"] == "12345"


def test_render_sysctl_setter():
    variables = {
        "NODE": "kmain07",
        "POD_NAME": "sysctl-set-kmain07",
        "SYSCTL_COMMANDS": "sysctl -w net.core.rmem_max=67108864",
    }
    rendered = render_template("sysctl_setter.yaml", variables)
    doc = yaml.safe_load(rendered)
    assert doc["kind"] == "Pod"
    assert doc["metadata"]["name"] == "sysctl-set-kmain07"
    assert doc["spec"]["nodeName"] == "kmain07"
    assert doc["spec"]["hostNetwork"] is True
    assert "hostPID" not in doc["spec"]
    container = doc["spec"]["containers"][0]
    assert container["image"].startswith("nicolaka/netshoot")
    assert container["securityContext"]["privileged"] is True
    assert "sysctl -w net.core.rmem_max=67108864" in " ".join(
        str(a) for a in container["command"]
    )
