"""Typed-API Kubernetes client for the rest of the package.

Every method wraps a call on a :mod:`kubernetes.client` typed API and
translates :class:`kubernetes.client.exceptions.ApiException` into
:class:`K8sApiError`, a domain-specific exception carrying a structured
``reason`` string (``"NotFound"``, ``"AlreadyExists"``, ``"Conflict"``,
...) that callers match on instead of stderr substrings.

:meth:`K8sClient.apply` uses **server-side apply** uniformly — a single
``PATCH`` with ``Content-Type: application/apply-patch+yaml``,
``fieldManager="kube-autotuner"``, ``force=True``. That collapses the
first-run/re-run distinction for Deployments, Services, Pods, and Jobs.
:class:`coordination.k8s.io/v1 Lease` is handled separately via
:meth:`create` / :meth:`replace` because the lease CAS path relies on
``AlreadyExists`` / ``Conflict`` signalling.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any, NoReturn

from kubernetes import client as _k8s, config as _k8s_config, watch as _k8s_watch
from kubernetes.client.exceptions import ApiException
import yaml

from kube_autotuner.benchmark.parser import parse_k8s_memory

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


FIELD_MANAGER = "kube-autotuner"
_APPLY_CONTENT_TYPE = "application/apply-patch+yaml"
_HTTP_NOT_FOUND = 404


class K8sApiError(Exception):
    """Raised when a Kubernetes API call fails.

    Attributes:
        op: Short description of the attempted operation, e.g.
            ``"apply pod/foo"``.
        status: HTTP status code returned by the API server
            (``0`` for non-HTTP failures such as timeouts surfaced
            locally).
        reason: Structured ``reason`` from the API server's Status
            response (``"NotFound"``, ``"AlreadyExists"``,
            ``"Conflict"``, ``"Timeout"`` for local timeouts). This is
            the field callers branch on.
        message: Human-readable message from the API server, or the
            local failure string for timeouts.
    """

    def __init__(
        self,
        *,
        op: str,
        status: int,
        reason: str,
        message: str,
    ) -> None:
        """Store the structured fields and format the exception message.

        Args:
            op: Short operation description.
            status: HTTP status code.
            reason: Structured reason string.
            message: Human-readable message.
        """
        self.op = op
        self.status = status
        self.reason = reason
        self.message = message
        super().__init__(
            f"{op} failed (status={status}, reason={reason!r}): {message}",
        )


def _raise(op: str, e: ApiException) -> NoReturn:
    """Translate :class:`ApiException` into :class:`K8sApiError`.

    Args:
        op: Short operation description used in the formatted message.
        e: The underlying :class:`ApiException`.

    Raises:
        K8sApiError: Always. The original exception is attached as the
            cause.
    """
    body: dict[str, Any] = {}
    if e.body:
        try:
            body = json.loads(e.body)
        except ValueError, TypeError:
            body = {}
    raise K8sApiError(
        op=op,
        status=int(e.status or 0),
        reason=str(body.get("reason") or e.reason or ""),
        message=str(body.get("message") or e),
    ) from e


class K8sClient:
    """Typed Kubernetes API client surface used throughout the package.

    Constructs typed API handles (``core_v1``, ``apps_v1``, ``batch_v1``,
    ``coord_v1``, ``custom_objects``) off a single :class:`ApiClient`
    loaded from the ambient kubeconfig (or in-cluster config when the
    ``KUBERNETES_SERVICE_HOST`` environment variable is set).
    """

    def __init__(self, config_file: Path | None = None) -> None:
        """Load kubeconfig and construct the typed API handles.

        Args:
            config_file: Optional kubeconfig path. Ignored when running
                in-cluster (``KUBERNETES_SERVICE_HOST`` set).

        Raises:
            kubernetes.config.config_exception.ConfigException: No
                in-cluster environment and no resolvable kubeconfig.
                Propagated so the CLI boundary can surface the failure
                verbatim.
        """
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            _k8s_config.load_incluster_config()
        else:
            _k8s_config.load_kube_config(
                config_file=str(config_file) if config_file else None,
            )
        self.api_client = _k8s.ApiClient()
        self.core_v1 = _k8s.CoreV1Api(self.api_client)
        self.apps_v1 = _k8s.AppsV1Api(self.api_client)
        self.batch_v1 = _k8s.BatchV1Api(self.api_client)
        self.coord_v1 = _k8s.CoordinationV1Api(self.api_client)
        self.custom_objects = _k8s.CustomObjectsApi(self.api_client)

    # ---- apply / create / replace -------------------------------------

    def apply(self, yaml_str: str, namespace: str) -> None:
        """Server-side-apply every document in ``yaml_str``.

        Supported kinds: ``Pod``, ``Deployment``, ``Service``, ``Job``.
        ``Lease`` is deliberately routed through :meth:`create` /
        :meth:`replace` because its AlreadyExists/Conflict signals drive
        the lease CAS path.

        Args:
            yaml_str: One or more YAML documents separated by ``---``.
            namespace: Target namespace for every document.

        Raises:
            ValueError: A document uses an unsupported ``kind``.
            K8sApiError: The API call failed.
        """
        for doc in yaml.safe_load_all(yaml_str):
            if not doc:
                continue
            self._apply_one(doc, namespace)

    def _apply_one(self, body: dict[str, Any], namespace: str) -> None:
        """Server-side-apply one parsed document.

        Args:
            body: Parsed YAML document (must carry ``kind``,
                ``metadata.name``).
            namespace: Target namespace.

        Raises:
            ValueError: Unsupported ``kind``.
            K8sApiError: The API call failed.
        """
        kind = body.get("kind", "")
        name = body.get("metadata", {}).get("name", "")
        op = f"apply {kind.lower()}/{name}"
        kwargs: dict[str, Any] = {
            "field_manager": FIELD_MANAGER,
            "force": True,
        }
        # Server-side apply requires Content-Type application/apply-patch+yaml;
        # the typed APIs auto-select the first content type in their list
        # (JSON Patch), so override the api_client's header selector for the
        # duration of this call.
        with _apply_content_type(self.api_client):
            try:
                if kind == "Pod":
                    self.core_v1.patch_namespaced_pod(name, namespace, body, **kwargs)
                elif kind == "Service":
                    self.core_v1.patch_namespaced_service(
                        name, namespace, body, **kwargs
                    )
                elif kind == "Deployment":
                    self.apps_v1.patch_namespaced_deployment(
                        name, namespace, body, **kwargs
                    )
                elif kind == "Job":
                    self.batch_v1.patch_namespaced_job(name, namespace, body, **kwargs)
                else:
                    msg = f"apply: unsupported kind {kind!r}"
                    raise ValueError(msg)
            except ApiException as e:
                _raise(op, e)

    def create(self, yaml_str: str, namespace: str) -> None:
        """Create a single resource (typically a Lease).

        Args:
            yaml_str: Single-document YAML.
            namespace: Target namespace.

        Raises:
            ValueError: Multi-document YAML, or unsupported ``kind``.
            K8sApiError: The API call failed (``reason="AlreadyExists"``
                is the signal the lease fast path relies on).
        """
        docs = [d for d in yaml.safe_load_all(yaml_str) if d]
        if len(docs) != 1:
            msg = f"create: expected single-document YAML, got {len(docs)}"
            raise ValueError(msg)
        body = docs[0]
        kind = body.get("kind", "")
        name = body.get("metadata", {}).get("name", "")
        op = f"create {kind.lower()}/{name}"
        try:
            if kind == "Lease":
                self.coord_v1.create_namespaced_lease(namespace, body)
            elif kind == "Namespace":
                self.core_v1.create_namespace(body)
            elif kind == "Pod":
                self.core_v1.create_namespaced_pod(namespace, body)
            elif kind == "Job":
                self.batch_v1.create_namespaced_job(namespace, body)
            else:
                msg = f"create: unsupported kind {kind!r}"
                raise ValueError(msg)
        except ApiException as e:
            _raise(op, e)

    def replace(self, yaml_str: str, namespace: str) -> None:
        """Replace an existing resource (typically a Lease) with optimistic CC.

        The rendered body must embed ``metadata.resourceVersion`` for
        the API server's compare-and-swap; callers that observe
        ``reason="Conflict"`` retry.

        Args:
            yaml_str: Single-document YAML.
            namespace: Target namespace.

        Raises:
            ValueError: Multi-document YAML, or unsupported ``kind``.
            K8sApiError: The API call failed.
        """
        docs = [d for d in yaml.safe_load_all(yaml_str) if d]
        if len(docs) != 1:
            msg = f"replace: expected single-document YAML, got {len(docs)}"
            raise ValueError(msg)
        body = docs[0]
        kind = body.get("kind", "")
        name = body.get("metadata", {}).get("name", "")
        op = f"replace {kind.lower()}/{name}"
        try:
            if kind == "Lease":
                self.coord_v1.replace_namespaced_lease(name, namespace, body)
            else:
                msg = f"replace: unsupported kind {kind!r}"
                raise ValueError(msg)
        except ApiException as e:
            _raise(op, e)

    # ---- delete -------------------------------------------------------

    def delete(
        self,
        resource_type: str,
        name: str,
        namespace: str,
        *,
        ignore_not_found: bool = True,
    ) -> None:
        """Delete a single named resource.

        Args:
            resource_type: Resource kind (``"pod"``, ``"job"``,
                ``"deployment"``, ``"service"``, ``"lease"``,
                ``"namespace"``, ``"configmap"``).
            name: Object name.
            namespace: Target namespace (ignored for ``"namespace"``).
            ignore_not_found: When ``True`` (default), swallow 404s.

        Raises:
            ValueError: Unsupported ``resource_type``.
            K8sApiError: The API call failed and the error was not a
                swallowed 404.
        """
        op = f"delete {resource_type}/{name}"
        try:
            if resource_type == "pod":
                self.core_v1.delete_namespaced_pod(name, namespace)
            elif resource_type == "service":
                self.core_v1.delete_namespaced_service(name, namespace)
            elif resource_type == "configmap":
                self.core_v1.delete_namespaced_config_map(name, namespace)
            elif resource_type == "deployment":
                self.apps_v1.delete_namespaced_deployment(name, namespace)
            elif resource_type == "job":
                self.batch_v1.delete_namespaced_job(
                    name, namespace, propagation_policy="Background"
                )
            elif resource_type == "lease":
                self.coord_v1.delete_namespaced_lease(name, namespace)
            elif resource_type == "namespace":
                self.core_v1.delete_namespace(name)
            else:
                msg = f"delete: unsupported resource_type {resource_type!r}"
                raise ValueError(msg)
        except ApiException as e:
            if ignore_not_found and e.status == _HTTP_NOT_FOUND:
                return
            _raise(op, e)

    def delete_by_label(self, resource_type: str, label: str, namespace: str) -> None:
        """Delete every ``resource_type`` matching ``label`` (``"k=v"``).

        Args:
            resource_type: Resource kind (``"pod"``, ``"job"``,
                ``"deployment"``, ``"service"``).
            label: Label selector, e.g. ``"app.kubernetes.io/name=foo"``.
            namespace: Target namespace.

        Raises:
            ValueError: Unsupported ``resource_type``.
            K8sApiError: The API call failed.
        """
        op = f"delete {resource_type} -l {label}"
        try:
            if resource_type == "pod":
                self.core_v1.delete_collection_namespaced_pod(
                    namespace, label_selector=label
                )
            elif resource_type == "service":
                # CoreV1Api has no delete_collection_namespaced_service; list
                # then delete each.
                listing = self.core_v1.list_namespaced_service(
                    namespace, label_selector=label
                )
                for svc in listing.items:
                    self.core_v1.delete_namespaced_service(svc.metadata.name, namespace)
            elif resource_type == "deployment":
                self.apps_v1.delete_collection_namespaced_deployment(
                    namespace, label_selector=label
                )
            elif resource_type == "job":
                self.batch_v1.delete_collection_namespaced_job(
                    namespace,
                    label_selector=label,
                    propagation_policy="Background",
                )
            else:
                msg = f"delete_by_label: unsupported resource_type {resource_type!r}"
                raise ValueError(msg)
        except ApiException as e:
            _raise(op, e)

    # ---- wait / rollout_status ----------------------------------------

    def wait(
        self,
        resource_type: str,
        name: str,
        condition: str,
        namespace: str,
        timeout: int = 120,
    ) -> None:
        """Block until a predicate on ``resource_type/name`` holds.

        Two predicate forms are supported, matching the shapes used by
        our callers:

        * ``condition=<Type>`` — wait until
          ``status.conditions[?(@.type == Type)].status == "True"`` (the
          type match is case-insensitive).
        * ``jsonpath={.status.phase}=<value>`` — wait until
          ``status.phase == value`` exactly.

        Args:
            resource_type: ``"pod"`` or ``"job"``.
            name: Object name.
            condition: Predicate string (see above).
            namespace: Target namespace.
            timeout: Seconds before raising.

        Raises:
            ValueError: Unsupported predicate form or ``resource_type``.
            K8sApiError: ``reason="Timeout"`` on elapsed timeout, or a
                forwarded API error from the watch stream.
        """
        predicate = _parse_wait_predicate(condition)
        list_fn, field_selector = self._wait_list_fn(resource_type, name, namespace)
        deadline = time.monotonic() + timeout
        watcher = _k8s_watch.Watch()
        try:
            while True:
                remaining = max(1, int(deadline - time.monotonic()))
                try:
                    for event in watcher.stream(
                        list_fn,
                        namespace=namespace,
                        field_selector=field_selector,
                        timeout_seconds=remaining,
                    ):
                        obj = event.get("object")
                        if obj is not None and predicate(_object_to_dict(obj)):
                            watcher.stop()
                            return
                except ApiException as e:
                    _raise(f"wait {resource_type}/{name}", e)
                if time.monotonic() >= deadline:
                    raise K8sApiError(
                        op=f"wait {resource_type}/{name}",
                        status=0,
                        reason="Timeout",
                        message=f"timed out after {timeout}s waiting for {condition}",
                    )
        finally:
            watcher.stop()

    def _wait_list_fn(
        self,
        resource_type: str,
        name: str,
        namespace: str,  # noqa: ARG002 - surfaced via closure in caller
    ) -> tuple[Any, str]:
        """Return ``(list_fn, field_selector)`` for :meth:`wait`.

        Args:
            resource_type: ``"pod"`` or ``"job"``.
            name: Object name embedded in the field selector.
            namespace: Target namespace. Passed through to the list
                call at the caller.

        Returns:
            ``(list_fn, field_selector)``.

        Raises:
            ValueError: Unsupported ``resource_type``.
        """
        field_selector = f"metadata.name={name}"
        if resource_type == "pod":
            return self.core_v1.list_namespaced_pod, field_selector
        if resource_type == "job":
            return self.batch_v1.list_namespaced_job, field_selector
        msg = f"wait: unsupported resource_type {resource_type!r}"
        raise ValueError(msg)

    def rollout_status(
        self,
        resource_type: str,
        name: str,
        namespace: str,
        timeout: int = 120,
    ) -> None:
        """Block until a Deployment rollout reports completion.

        The rollout is complete when ``observedGeneration >=
        generation``, the desired-replica counts match, and no pods
        are unavailable.

        Args:
            resource_type: Must be ``"deployment"``.
            name: Deployment name.
            namespace: Target namespace.
            timeout: Seconds before raising.

        Raises:
            ValueError: ``resource_type`` is not ``"deployment"``.
            K8sApiError: ``reason="Timeout"`` on elapsed timeout, or a
                forwarded API error.
        """
        if resource_type != "deployment":
            msg = f"rollout_status: unsupported resource_type {resource_type!r}"
            raise ValueError(msg)
        deadline = time.monotonic() + timeout
        watcher = _k8s_watch.Watch()
        try:
            while True:
                remaining = max(1, int(deadline - time.monotonic()))
                try:
                    for event in watcher.stream(
                        self.apps_v1.list_namespaced_deployment,
                        namespace=namespace,
                        field_selector=f"metadata.name={name}",
                        timeout_seconds=remaining,
                    ):
                        obj = event.get("object")
                        if obj is not None and _deployment_ready(obj):
                            watcher.stop()
                            return
                except ApiException as e:
                    _raise(f"rollout_status deployment/{name}", e)
                if time.monotonic() >= deadline:
                    raise K8sApiError(
                        op=f"rollout_status deployment/{name}",
                        status=0,
                        reason="Timeout",
                        message=f"timed out after {timeout}s waiting for rollout",
                    )
        finally:
            watcher.stop()

    # ---- logs ---------------------------------------------------------

    def logs(self, resource_type: str, name: str, namespace: str) -> str:
        """Return the logs for a pod or the first pod of a Job.

        For ``"job"``, the modern ``batch.kubernetes.io/job-name`` label
        (k8s ≥ 1.27) is tried first and the legacy ``job-name`` is
        consulted as a fallback.

        Args:
            resource_type: ``"pod"`` or ``"job"``.
            name: Pod or Job name.
            namespace: Target namespace.

        Returns:
            The decoded log output.

        Raises:
            ValueError: Unsupported ``resource_type``.
            K8sApiError: ``reason="NotFound"`` when no matching pod
                exists for a Job, or a forwarded API error from the
                read call.
        """
        op = f"logs {resource_type}/{name}"
        if resource_type == "pod":
            try:
                return self._read_pod_log(name, namespace)
            except ApiException as e:
                _raise(op, e)
        if resource_type == "job":
            pod_name = self._job_pod_name(name, namespace)
            if not pod_name:
                raise K8sApiError(
                    op=op,
                    status=_HTTP_NOT_FOUND,
                    reason="NotFound",
                    message=f"no pod found for job {name!r}",
                )
            try:
                return self._read_pod_log(pod_name, namespace)
            except ApiException as e:
                _raise(op, e)
        msg = f"logs: unsupported resource_type {resource_type!r}"
        raise ValueError(msg)

    def _read_pod_log(self, pod_name: str, namespace: str) -> str:
        """Return the raw log body for ``pod_name``, bypassing OpenAPI decode.

        ``read_namespaced_pod_log``'s default ``_preload_content=True``
        runs ``ApiClient.deserialize``, which calls ``json.loads`` on
        the body and then coerces the resulting object to the declared
        ``str`` return type via ``str(...)`` — corrupting any log that
        is itself valid JSON (e.g. ``iperf3 --json``) into a Python
        dict repr. ``_preload_content=False`` returns the underlying
        ``urllib3.HTTPResponse`` so we can decode the socket bytes
        directly.

        Args:
            pod_name: Pod name.
            namespace: Target namespace.

        Returns:
            The decoded UTF-8 log body.

        Raises:
            ApiException: Forwarded from the initial request or the
                lazy body read. The caller owns translation to
                :class:`K8sApiError`.
        """
        resp = self.core_v1.read_namespaced_pod_log(
            pod_name, namespace, _preload_content=False
        )
        try:
            return resp.read().decode("utf-8")
        finally:
            resp.release_conn()

    def _job_pod_name(self, job_name: str, namespace: str) -> str:
        """Return the name of a pod backing ``job_name``.

        Tries the modern ``batch.kubernetes.io/job-name`` label first,
        falls back to the legacy ``job-name`` label.

        With ``backoffLimit > 0`` a Job can spawn multiple pods (one per
        retry); ``list_namespaced_pod`` makes no ordering guarantee,
        so naively returning ``items[0]`` may hand back a Failed
        attempt's logs even when the Job ultimately Succeeded. Prefer a
        ``Succeeded`` pod when one is present so callers reading
        results-bearing logs (e.g. the fortio runner) get the attempt
        whose stdout actually contains the result document.

        Args:
            job_name: Job name.
            namespace: Target namespace.

        Returns:
            The pod name, or an empty string if none found.
        """
        for selector in (
            f"batch.kubernetes.io/job-name={job_name}",
            f"job-name={job_name}",
        ):
            try:
                listing = self.core_v1.list_namespaced_pod(
                    namespace, label_selector=selector
                )
            except ApiException:
                continue
            items = list(listing.items)
            if not items:
                continue
            for pod in items:
                phase = getattr(getattr(pod, "status", None), "phase", None)
                if phase == "Succeeded":
                    return str(pod.metadata.name)
            return str(items[0].metadata.name)
        return ""

    # ---- metrics ------------------------------------------------------

    def top_pod(self, name: str, namespace: str) -> dict[str, str]:
        """Return summed CPU/memory across every container of a pod.

        Args:
            name: Pod name.
            namespace: Target namespace.

        Returns:
            ``{"cpu": "<n>m", "memory": "<n>Ki"}`` aggregated across
            containers, or ``{}`` when the metrics-server has no data
            for the pod (404).
        """
        try:
            resp = self.custom_objects.get_namespaced_custom_object(
                "metrics.k8s.io",
                "v1beta1",
                namespace,
                "pods",
                name,
            )
        except ApiException as e:
            if e.status == _HTTP_NOT_FOUND:
                return {}
            _raise(f"top pod/{name}", e)
        containers = resp.get("containers", [])
        if not containers:
            return {}
        cpu_total_milli = sum(_parse_cpu_milli(c["usage"]["cpu"]) for c in containers)
        mem_total_bytes = sum(
            parse_k8s_memory(c["usage"]["memory"]) for c in containers
        )
        return {
            "cpu": f"{cpu_total_milli}m",
            "memory": f"{mem_total_bytes // 1024}Ki",
        }

    def top_node(self, name: str) -> dict[str, str]:
        """Return summed CPU/memory usage for a single node.

        Args:
            name: Node name.

        Returns:
            ``{"cpu": "<n>m", "memory": "<n>Ki"}`` sourced from
            ``metrics.k8s.io/v1beta1 nodes/<name>``, or ``{}`` when the
            metrics-server has no data for the node (404).
        """
        try:
            resp = self.custom_objects.get_cluster_custom_object(
                "metrics.k8s.io",
                "v1beta1",
                "nodes",
                name,
            )
        except ApiException as e:
            if e.status == _HTTP_NOT_FOUND:
                return {}
            _raise(f"top node/{name}", e)
        usage = resp.get("usage", {})
        cpu = usage.get("cpu")
        mem = usage.get("memory")
        if cpu is None or mem is None:
            return {}
        return {"cpu": str(cpu), "memory": str(mem)}

    def list_pods_by_selector_on_node(
        self,
        label_selector: str,
        namespace: str,
        node_name: str,
    ) -> list[str]:
        """Return pod names matching ``label_selector`` on ``node_name``.

        Merges the label and field selectors server-side so the result
        is scoped to pods scheduled on the target node.

        Args:
            label_selector: Label selector (``"k=v"``).
            namespace: Target namespace.
            node_name: Node name to filter on via
                ``spec.nodeName==<node_name>``.

        Returns:
            The pod names, in list-API order. Empty list when nothing
            matches.

        Raises:
            K8sApiError: The list call itself failed.
        """
        try:
            listing = self.core_v1.list_namespaced_pod(
                namespace,
                label_selector=label_selector,
                field_selector=f"spec.nodeName={node_name}",
            )
        except ApiException as e:
            _raise(f"list pods -l {label_selector} on {node_name}", e)
        return [p.metadata.name for p in listing.items]

    def top_pod_containers(self, name: str, namespace: str) -> list[dict[str, str]]:
        """Return per-container CPU/memory usage for a pod.

        Args:
            name: Pod name.
            namespace: Target namespace.

        Returns:
            ``[{"container": str, "cpu": str, "memory": str}]``, one row
            per container. Values are the raw Kubernetes-suffixed strings
            returned by metrics-server. Empty list on 404.
        """
        try:
            resp = self.custom_objects.get_namespaced_custom_object(
                "metrics.k8s.io",
                "v1beta1",
                namespace,
                "pods",
                name,
            )
        except ApiException as e:
            if e.status == _HTTP_NOT_FOUND:
                return []
            _raise(f"top pod/{name}", e)
        return [
            {
                "container": c["name"],
                "cpu": c["usage"]["cpu"],
                "memory": c["usage"]["memory"],
            }
            for c in resp.get("containers", [])
        ]

    # ---- lookups ------------------------------------------------------

    def get_pod_name(self, label: str, namespace: str) -> str:
        """Return the name of the first pod matching ``label``.

        Args:
            label: Label selector (``"k=v"``).
            namespace: Target namespace.

        Returns:
            The first pod name, or ``""`` when the selector matches
            nothing. Never raises on empty lists.

        Raises:
            K8sApiError: The list call itself failed.
        """
        try:
            listing = self.core_v1.list_namespaced_pod(namespace, label_selector=label)
        except ApiException as e:
            _raise(f"list pods -l {label}", e)
        if not listing.items:
            return ""
        return str(listing.items[0].metadata.name)

    def get_node_zone(self, node: str) -> str:
        """Return the ``topology.kubernetes.io/zone`` label for ``node``.

        Args:
            node: Node name.

        Returns:
            The label value, or ``""`` when the label is unset.

        Raises:
            K8sApiError: The read call failed.
        """
        try:
            obj = self.core_v1.read_node(node)
        except ApiException as e:
            _raise(f"get node/{node}", e)
        labels = obj.metadata.labels or {}
        return str(labels.get("topology.kubernetes.io/zone", ""))

    def get_node_internal_ip(self, node: str) -> str:
        """Return the node's first ``InternalIP``, or ``""``.

        Args:
            node: Node name.

        Returns:
            The address, or ``""`` when no ``InternalIP`` is present.

        Raises:
            K8sApiError: The read call failed.
        """
        try:
            obj = self.core_v1.read_node(node)
        except ApiException as e:
            _raise(f"get node/{node}", e)
        for addr in obj.status.addresses or []:
            if addr.type == "InternalIP":
                return str(addr.address)
        return ""

    def get_json(
        self, resource_type: str, name: str, namespace: str
    ) -> dict[str, Any] | None:
        """Return a resource as a dict (camelCase keys) or ``None`` on 404.

        Only ``"lease"`` is used today — callers narrow to Pydantic
        models as needed. The dict uses the API server's camelCase
        field names (``holderIdentity``, ``renewTime``,
        ``leaseDurationSeconds``, ``resourceVersion``).

        Args:
            resource_type: Resource kind. Only ``"lease"`` is
                currently supported.
            name: Object name.
            namespace: Target namespace.

        Returns:
            The resource as a camelCase dict, or ``None`` when the API
            server returns 404.

        Raises:
            ValueError: Unsupported ``resource_type``.
            K8sApiError: The read call failed with any non-404 error.
            TypeError: The serialized object is not a dict (indicates a
                regression in the client library).
        """
        if resource_type != "lease":
            msg = f"get_json: unsupported resource_type {resource_type!r}"
            raise ValueError(msg)
        op = f"get lease/{name}"
        try:
            obj = self.coord_v1.read_namespaced_lease(name, namespace)
        except ApiException as e:
            if e.status == _HTTP_NOT_FOUND:
                return None
            _raise(op, e)
        sanitized = _k8s.ApiClient().sanitize_for_serialization(obj)
        if not isinstance(sanitized, dict):
            msg = f"get_json: unexpected serialization shape for {op}"
            raise TypeError(msg)
        return sanitized


# ---- helpers ----------------------------------------------------------


class _ApplyContentTypeContext:
    """Temporarily force the api_client to select apply-patch+yaml."""

    def __init__(self, api_client: _k8s.ApiClient) -> None:
        """Store the api_client whose selector is swapped.

        Args:
            api_client: Shared :class:`ApiClient` backing the typed
                API handles.
        """
        self._api_client = api_client
        self._original: Any = None

    def __enter__(self) -> None:
        """Install the apply-patch content-type selector."""
        self._original = self._api_client.select_header_content_type

        def _select(content_types: list[str]) -> str:
            for ct in content_types:
                if ct.lower() == _APPLY_CONTENT_TYPE:
                    return _APPLY_CONTENT_TYPE
            return self._original(content_types)

        self._api_client.select_header_content_type = _select  # type: ignore[method-assign]

    def __exit__(self, *_exc: object) -> None:
        """Restore the original content-type selector."""
        self._api_client.select_header_content_type = self._original  # type: ignore[method-assign]


def _apply_content_type(api_client: _k8s.ApiClient) -> _ApplyContentTypeContext:
    """Return a context manager that forces server-side-apply content type.

    Args:
        api_client: Shared :class:`ApiClient` backing the typed APIs.

    Returns:
        A context manager to wrap a patch call.
    """
    return _ApplyContentTypeContext(api_client)


def _parse_wait_predicate(condition: str) -> Callable[[dict[str, Any]], bool]:
    """Compile a :meth:`K8sClient.wait` predicate string to a callable.

    Args:
        condition: Predicate string in one of the two forms listed on
            :meth:`K8sClient.wait`.

    Returns:
        A callable taking a parsed-object dict and returning ``True``
        when the predicate holds.

    Raises:
        ValueError: The predicate string is not recognised.
    """
    if condition.startswith("condition="):
        target = condition.removeprefix("condition=").lower()

        def _predicate(obj: dict[str, Any]) -> bool:
            for c in (obj.get("status") or {}).get("conditions") or []:
                if (c.get("type") or "").lower() == target and c.get(
                    "status"
                ) == "True":
                    return True
            return False

        return _predicate
    if condition.startswith("jsonpath={.status.phase}="):
        target_phase = condition.removeprefix("jsonpath={.status.phase}=")

        def _phase_predicate(obj: dict[str, Any]) -> bool:
            return (obj.get("status") or {}).get("phase") == target_phase

        return _phase_predicate
    msg = f"wait: unrecognised condition {condition!r}"
    raise ValueError(msg)


def _object_to_dict(obj: Any) -> dict[str, Any]:  # noqa: ANN401
    """Normalise a watch-event object to a plain dict.

    The watch stream returns either typed model instances or raw dicts
    depending on whether the response-type hint was resolvable.

    Args:
        obj: Typed model or dict from the watch stream.

    Returns:
        A JSON-serialisable dict (camelCase keys).
    """
    if isinstance(obj, dict):
        return obj
    result = _k8s.ApiClient().sanitize_for_serialization(obj)
    return result if isinstance(result, dict) else {}


def _deployment_ready(obj: Any) -> bool:  # noqa: ANN401
    """Return whether a Deployment watch event shows a completed rollout.

    Args:
        obj: Deployment watch event payload (typed model or dict).

    Returns:
        ``True`` when ``observedGeneration >= generation``, desired vs.
        updated vs. ready replica counts all match, and no pods are
        unavailable.
    """
    d = _object_to_dict(obj)
    spec = d.get("spec") or {}
    metadata = d.get("metadata") or {}
    status = d.get("status") or {}
    desired = spec.get("replicas")
    if desired is None:
        return False
    observed_gen = status.get("observedGeneration", 0) or 0
    generation = metadata.get("generation", 0) or 0
    if observed_gen < generation:
        return False
    replicas = status.get("replicas", 0) or 0
    updated = status.get("updatedReplicas", 0) or 0
    ready = status.get("readyReplicas", 0) or 0
    unavailable = status.get("unavailableReplicas", 0) or 0
    return (
        replicas == desired
        and updated == desired
        and ready == desired
        and unavailable == 0
    )


def _parse_cpu_milli(cpu: str) -> int:
    """Parse a Kubernetes CPU quantity into milli-CPU units.

    Args:
        cpu: Quantity string. Accepts the three forms metrics-server
            emits: bare integer cores (``"2"``), fractional cores
            (``"0.25"``), milli-CPU (``"250m"``), and nano-CPU
            (``"500000000n"``).

    Returns:
        Integer milli-CPU.

    Raises:
        ValueError: The string does not match any known form.
    """
    s = cpu.strip()
    if s.endswith("n"):
        return int(s[:-1]) // 1_000_000
    if s.endswith("u"):
        return int(s[:-1]) // 1_000
    if s.endswith("m"):
        return int(s[:-1])
    return round(float(s) * 1000)
