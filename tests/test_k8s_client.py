"""Unit tests for :class:`kube_autotuner.k8s.client.K8sClient`.

The tests bypass ``K8sClient.__init__`` to avoid loading a kubeconfig
and inject ``MagicMock`` typed-API handles directly.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from kubernetes.client.exceptions import ApiException
import pytest

from kube_autotuner.k8s.client import (
    FIELD_MANAGER,
    JobFailedConditionError,
    K8sApiError,
    K8sClient,
)


def _client_with_mocks() -> K8sClient:
    c = K8sClient.__new__(K8sClient)
    c.api_client = MagicMock()
    c.core_v1 = MagicMock()
    c.apps_v1 = MagicMock()
    c.batch_v1 = MagicMock()
    c.coord_v1 = MagicMock()
    c.custom_objects = MagicMock()
    return c


def _api_exception(status: int, reason: str, message: str = "") -> ApiException:
    exc = ApiException(status=status, reason=reason)
    exc.body = (
        f'{{"kind":"Status","reason":"{reason}","message":"{message or reason}"}}'
    )
    return exc


# ---- get_pod_name / node lookups --------------------------------------


def test_get_pod_name_returns_first_item():
    c = _client_with_mocks()
    pod = SimpleNamespace(metadata=SimpleNamespace(name="p-1"))
    c.core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[pod])
    assert c.get_pod_name("app=x", "default") == "p-1"


def test_get_pod_name_empty_list_returns_empty_string():
    c = _client_with_mocks()
    c.core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[])
    # Must not raise — callers rely on the "" return shape during shutdown.
    assert c.get_pod_name("app=x", "default") == ""


def test_get_node_zone_reads_label():
    c = _client_with_mocks()
    c.core_v1.read_node.return_value = SimpleNamespace(
        metadata=SimpleNamespace(labels={"topology.kubernetes.io/zone": "az01"}),
    )
    assert c.get_node_zone("n-1") == "az01"


def test_get_node_zone_empty_when_label_missing():
    c = _client_with_mocks()
    c.core_v1.read_node.return_value = SimpleNamespace(
        metadata=SimpleNamespace(labels={}),
    )
    assert c.get_node_zone("n-1") == ""


def test_get_node_internal_ip_picks_internal_address():
    c = _client_with_mocks()
    c.core_v1.read_node.return_value = SimpleNamespace(
        status=SimpleNamespace(
            addresses=[
                SimpleNamespace(type="ExternalIP", address="10.0.0.1"),
                SimpleNamespace(type="InternalIP", address="10.5.0.3"),
            ],
        ),
    )
    assert c.get_node_internal_ip("n-1") == "10.5.0.3"


def test_get_node_internal_ip_empty_when_missing():
    c = _client_with_mocks()
    c.core_v1.read_node.return_value = SimpleNamespace(
        status=SimpleNamespace(addresses=[]),
    )
    assert c.get_node_internal_ip("n-1") == ""


# ---- get_json (lease) --------------------------------------------------


def test_get_json_returns_dict_with_camelcase_keys(monkeypatch):
    c = _client_with_mocks()
    lease_obj = object()
    c.coord_v1.read_namespaced_lease.return_value = lease_obj
    sanitized = {
        "metadata": {"name": "L", "resourceVersion": "12"},
        "spec": {
            "holderIdentity": "me",
            "leaseDurationSeconds": 900,
            "renewTime": "2026-04-17T00:00:00.000000Z",
        },
    }
    # Patch the on-the-fly ApiClient sanitizer used inside get_json.
    monkeypatch.setattr(
        "kube_autotuner.k8s.client._k8s.ApiClient",
        lambda: SimpleNamespace(
            sanitize_for_serialization=lambda _obj: sanitized,
        ),
    )
    result = c.get_json("lease", "L", "default")
    assert result == sanitized


def test_get_json_returns_none_on_404():
    c = _client_with_mocks()
    c.coord_v1.read_namespaced_lease.side_effect = _api_exception(404, "NotFound")
    assert c.get_json("lease", "missing", "default") is None


def test_get_json_raises_on_other_errors():
    c = _client_with_mocks()
    c.coord_v1.read_namespaced_lease.side_effect = _api_exception(500, "ServerError")
    with pytest.raises(K8sApiError) as excinfo:
        c.get_json("lease", "x", "default")
    assert excinfo.value.status == 500
    assert excinfo.value.reason == "ServerError"


# ---- apply -------------------------------------------------------------


_DEPLOY_YAML = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: server
spec:
  replicas: 1
"""
_SERVICE_YAML = """\
apiVersion: v1
kind: Service
metadata:
  name: srv
spec: {}
"""
_POD_YAML = """\
apiVersion: v1
kind: Pod
metadata:
  name: p
spec: {}
"""
_JOB_YAML = """\
apiVersion: batch/v1
kind: Job
metadata:
  name: j
spec: {}
"""


def test_apply_multidoc_dispatches_to_each_kind():
    c = _client_with_mocks()
    combined = f"{_DEPLOY_YAML}\n---\n{_SERVICE_YAML}"
    c.apply(combined, "ns")

    c.apps_v1.patch_namespaced_deployment.assert_called_once()
    dep_kwargs = c.apps_v1.patch_namespaced_deployment.call_args.kwargs
    assert dep_kwargs["field_manager"] == FIELD_MANAGER
    assert dep_kwargs["force"] is True

    c.core_v1.patch_namespaced_service.assert_called_once()
    svc_kwargs = c.core_v1.patch_namespaced_service.call_args.kwargs
    assert svc_kwargs["field_manager"] == FIELD_MANAGER
    assert svc_kwargs["force"] is True


def test_apply_pod_single_doc():
    c = _client_with_mocks()
    c.apply(_POD_YAML, "ns")
    c.core_v1.patch_namespaced_pod.assert_called_once()


def test_apply_job_single_doc():
    c = _client_with_mocks()
    c.apply(_JOB_YAML, "ns")
    c.batch_v1.patch_namespaced_job.assert_called_once()


def test_apply_unknown_kind_raises_value_error():
    c = _client_with_mocks()
    weird = "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: cm\ndata: {}\n"
    with pytest.raises(ValueError, match="unsupported kind"):
        c.apply(weird, "ns")


# ---- delete ------------------------------------------------------------


def test_delete_swallows_404_when_ignore_not_found():
    c = _client_with_mocks()
    c.core_v1.delete_namespaced_pod.side_effect = _api_exception(404, "NotFound")
    c.delete("pod", "p", "ns")  # must not raise


def test_delete_raises_when_not_ignoring():
    c = _client_with_mocks()
    c.core_v1.delete_namespaced_pod.side_effect = _api_exception(404, "NotFound")
    with pytest.raises(K8sApiError):
        c.delete("pod", "p", "ns", ignore_not_found=False)


def test_delete_surfaces_non_404_errors():
    c = _client_with_mocks()
    c.core_v1.delete_namespaced_pod.side_effect = _api_exception(500, "ServerError")
    with pytest.raises(K8sApiError):
        c.delete("pod", "p", "ns")


# ---- wait --------------------------------------------------------------


class _FakeWatch:
    """Canned watch stream used to drive K8sClient.wait predicates."""

    def __init__(self, events: list[dict]) -> None:
        self.events = events

    def stream(self, *_args, **_kwargs):
        yield from self.events

    def stop(self) -> None:
        return


def test_wait_matches_condition_case_insensitive(monkeypatch):
    c = _client_with_mocks()
    pod = {
        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
    }
    monkeypatch.setattr(
        "kube_autotuner.k8s.client._k8s_watch.Watch",
        lambda: _FakeWatch([{"object": pod}]),
    )
    # Lowercase predicate type still matches the Ready condition.
    c.wait("pod", "p", "condition=ready", "ns", timeout=5)


def test_wait_matches_jsonpath_phase(monkeypatch):
    c = _client_with_mocks()
    pod = {"status": {"phase": "Succeeded"}}
    monkeypatch.setattr(
        "kube_autotuner.k8s.client._k8s_watch.Watch",
        lambda: _FakeWatch([{"object": pod}]),
    )
    c.wait("pod", "p", "jsonpath={.status.phase}=Succeeded", "ns", timeout=5)


def test_wait_raises_timeout(monkeypatch):
    c = _client_with_mocks()
    monkeypatch.setattr(
        "kube_autotuner.k8s.client._k8s_watch.Watch",
        lambda: _FakeWatch([]),
    )
    # Simulate immediate deadline expiry.
    monkeypatch.setattr(time, "monotonic", lambda: 1e18)
    with pytest.raises(K8sApiError) as excinfo:
        c.wait("pod", "p", "condition=ready", "ns", timeout=0)
    assert excinfo.value.reason == "Timeout"


def test_wait_failure_condition_raises_without_timeout(monkeypatch):
    """Job ``Failed=True`` short-circuits the wait rather than timing out."""
    c = _client_with_mocks()
    job = {
        "status": {
            "conditions": [
                {
                    "type": "Failed",
                    "status": "True",
                    "reason": "BackoffLimitExceeded",
                    "message": "Job has reached the specified backoff limit",
                },
            ],
        },
    }
    monkeypatch.setattr(
        "kube_autotuner.k8s.client._k8s_watch.Watch",
        lambda: _FakeWatch([{"object": job}]),
    )
    with pytest.raises(JobFailedConditionError) as exc:
        c.wait(
            "job",
            "j",
            "condition=complete",
            "ns",
            timeout=5,
            failure_condition="condition=failed",
        )
    assert exc.value.job == "ns/j"
    assert any(
        row["type"] == "Failed" and row["reason"] == "BackoffLimitExceeded"
        for row in exc.value.conditions
    )


def test_wait_same_event_success_wins_over_failure(monkeypatch):
    """When Complete and Failed both fire on one event, success wins."""
    c = _client_with_mocks()
    job = {
        "status": {
            "conditions": [
                {"type": "Complete", "status": "True"},
                {"type": "Failed", "status": "True", "reason": "x"},
            ],
        },
    }
    monkeypatch.setattr(
        "kube_autotuner.k8s.client._k8s_watch.Watch",
        lambda: _FakeWatch([{"object": job}]),
    )
    # Should return normally; no raise.
    c.wait(
        "job",
        "j",
        "condition=complete",
        "ns",
        timeout=5,
        failure_condition="condition=failed",
    )


def test_wait_backcompat_without_failure_condition(monkeypatch):
    """Existing callers that omit ``failure_condition`` still work."""
    c = _client_with_mocks()
    job = {"status": {"conditions": [{"type": "Complete", "status": "True"}]}}
    monkeypatch.setattr(
        "kube_autotuner.k8s.client._k8s_watch.Watch",
        lambda: _FakeWatch([{"object": job}]),
    )
    c.wait("job", "j", "condition=complete", "ns", timeout=5)


# ---- rollout_status ----------------------------------------------------


def test_rollout_status_blocks_until_ready(monkeypatch):
    c = _client_with_mocks()
    healthy = {
        "metadata": {"generation": 2},
        "spec": {"replicas": 3},
        "status": {
            "observedGeneration": 2,
            "replicas": 3,
            "updatedReplicas": 3,
            "readyReplicas": 3,
            "unavailableReplicas": 0,
        },
    }
    not_ready = {
        "metadata": {"generation": 2},
        "spec": {"replicas": 3},
        "status": {"observedGeneration": 1},
    }
    monkeypatch.setattr(
        "kube_autotuner.k8s.client._k8s_watch.Watch",
        lambda: _FakeWatch([{"object": not_ready}, {"object": healthy}]),
    )
    c.rollout_status("deployment", "d", "ns", timeout=5)


def test_rollout_status_raises_on_timeout(monkeypatch):
    c = _client_with_mocks()
    monkeypatch.setattr(
        "kube_autotuner.k8s.client._k8s_watch.Watch",
        lambda: _FakeWatch([]),
    )
    monkeypatch.setattr(time, "monotonic", lambda: 1e18)
    with pytest.raises(K8sApiError) as excinfo:
        c.rollout_status("deployment", "d", "ns", timeout=0)
    assert excinfo.value.reason == "Timeout"


# ---- logs (job → pod dispatch) -----------------------------------------


def _log_response(body: bytes) -> MagicMock:
    """Return a stand-in ``urllib3.HTTPResponse`` for log reads.

    Matches the object returned by ``read_namespaced_pod_log`` when
    invoked with ``_preload_content=False``.
    """
    resp = MagicMock()
    resp.read.return_value = body
    resp.release_conn.return_value = None
    return resp


def test_logs_job_prefers_modern_label():
    c = _client_with_mocks()
    modern_pod = SimpleNamespace(metadata=SimpleNamespace(name="p-modern"))

    def _list(_ns, label_selector):
        if "batch.kubernetes.io/job-name" in label_selector:
            return SimpleNamespace(items=[modern_pod])
        return SimpleNamespace(items=[])

    c.core_v1.list_namespaced_pod.side_effect = _list
    c.core_v1.read_namespaced_pod_log.return_value = _log_response(b"hello")
    assert c.logs("job", "j", "ns") == "hello"
    c.core_v1.read_namespaced_pod_log.assert_called_once_with(
        "p-modern", "ns", _preload_content=False
    )


def test_logs_job_falls_back_to_legacy_label():
    c = _client_with_mocks()
    legacy_pod = SimpleNamespace(metadata=SimpleNamespace(name="p-legacy"))

    def _list(_ns, label_selector):
        if "batch.kubernetes.io/job-name" in label_selector:
            return SimpleNamespace(items=[])
        return SimpleNamespace(items=[legacy_pod])

    c.core_v1.list_namespaced_pod.side_effect = _list
    c.core_v1.read_namespaced_pod_log.return_value = _log_response(b"legacy")
    assert c.logs("job", "j", "ns") == "legacy"


def test_logs_returns_json_body_verbatim():
    # Regression: the default ``_preload_content=True`` path coerces a
    # JSON body into ``str(dict)`` (Python repr with single quotes).
    c = _client_with_mocks()
    pod = SimpleNamespace(metadata=SimpleNamespace(name="p-json"))
    c.core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[pod])
    c.core_v1.read_namespaced_pod_log.return_value = _log_response(b'{"k": 1}')

    out = c.logs("job", "j", "ns")
    assert out == '{"k": 1}'
    assert json.loads(out) == {"k": 1}


def _pod(name: str, phase: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name),
        status=SimpleNamespace(phase=phase) if phase is not None else None,
    )


def test_logs_job_prefers_succeeded_pod_over_failed_attempt():
    """Picking the Succeeded retry hands back the pod whose log holds results."""
    c = _client_with_mocks()
    failed = _pod("p-attempt-1", phase="Failed")
    succeeded = _pod("p-attempt-2", phase="Succeeded")
    # Listing order isn't guaranteed; put the Failed pod first to prove
    # we pick by phase, not by index.
    c.core_v1.list_namespaced_pod.return_value = SimpleNamespace(
        items=[failed, succeeded]
    )
    c.core_v1.read_namespaced_pod_log.return_value = _log_response(b"result")
    assert c.logs("job", "j", "ns") == "result"
    c.core_v1.read_namespaced_pod_log.assert_called_once_with(
        "p-attempt-2", "ns", _preload_content=False
    )


def test_logs_job_falls_back_to_first_pod_when_none_succeeded():
    c = _client_with_mocks()
    pending = _pod("p-only", phase="Pending")
    c.core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[pending])
    c.core_v1.read_namespaced_pod_log.return_value = _log_response(b"partial")
    assert c.logs("job", "j", "ns") == "partial"
    c.core_v1.read_namespaced_pod_log.assert_called_once_with(
        "p-only", "ns", _preload_content=False
    )


def test_logs_job_raises_not_found_when_no_pod():
    c = _client_with_mocks()
    c.core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[])
    with pytest.raises(K8sApiError) as excinfo:
        c.logs("job", "j", "ns")
    assert excinfo.value.reason == "NotFound"


# ---- error translation -------------------------------------------------


def test_api_exception_already_exists_translates_to_reason():
    c = _client_with_mocks()
    c.coord_v1.create_namespaced_lease.side_effect = _api_exception(
        409, "AlreadyExists", "lease already exists"
    )
    body = (
        "apiVersion: coordination.k8s.io/v1\n"
        "kind: Lease\n"
        "metadata:\n  name: L\n"
        "spec: {}\n"
    )
    with pytest.raises(K8sApiError) as excinfo:
        c.create(body, "default")
    assert excinfo.value.status == 409
    assert excinfo.value.reason == "AlreadyExists"


def test_api_exception_conflict_translates_to_reason():
    c = _client_with_mocks()
    c.coord_v1.replace_namespaced_lease.side_effect = _api_exception(
        409, "Conflict", "resourceVersion conflict"
    )
    body = (
        "apiVersion: coordination.k8s.io/v1\n"
        "kind: Lease\n"
        'metadata:\n  name: L\n  resourceVersion: "42"\n'
        "spec: {}\n"
    )
    with pytest.raises(K8sApiError) as excinfo:
        c.replace(body, "default")
    assert excinfo.value.reason == "Conflict"


def test_api_exception_not_found_translates_to_reason():
    c = _client_with_mocks()
    c.core_v1.read_node.side_effect = _api_exception(404, "NotFound", "nope")
    with pytest.raises(K8sApiError) as excinfo:
        c.get_node_zone("n-missing")
    assert excinfo.value.reason == "NotFound"
    assert excinfo.value.status == 404
