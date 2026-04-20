"""Thin subprocess wrapper around the ``kubectl`` binary.

Every :class:`Kubectl` method layers on top of
:func:`kube_autotuner.subproc.run_tool` — the one sanctioned subprocess
entrypoint in the repo. The wrapper normalises error handling
(:class:`KubectlError` carries ``cmd``, ``returncode``, ``stdout`` and
``stderr``) and provides ergonomic helpers for the lookups the rest of
the package needs (``top pod``, ``get node``, JSON fetch, apply /
create / replace / delete).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from kube_autotuner.subproc import run_tool

if TYPE_CHECKING:
    from collections.abc import Sequence


class KubectlError(Exception):
    """Raised when a ``kubectl`` subprocess exits non-zero.

    Attributes:
        cmd: The full argv that was invoked (``["kubectl", ...]``).
        returncode: The process's exit code.
        stdout: Captured ``stdout`` — retained alongside ``stderr`` so
            failures like ``kubectl apply`` that print diagnostic info to
            ``stdout`` stay debuggable.
        stderr: Captured ``stderr``.
    """

    def __init__(
        self,
        cmd: Sequence[str],
        returncode: int,
        stderr: str,
        stdout: str = "",
    ) -> None:
        """Store the command context and format a rich message.

        Args:
            cmd: Argv of the failed invocation.
            returncode: Child process exit code.
            stderr: Captured ``stderr``.
            stdout: Captured ``stdout``. Defaults to an empty string so
                call sites that only have ``stderr`` on hand stay
                ergonomic.
        """
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        msg = f"kubectl failed (rc={returncode}): stderr={stderr!r} stdout={stdout!r}"
        super().__init__(msg)


_TOP_POD_FIELDS = 3
_TOP_POD_CONTAINER_FIELDS = 4


class Kubectl:
    """Thin subprocess wrapper around ``kubectl``.

    All shell-outs route through :func:`kube_autotuner.subproc.run_tool`;
    the wrapper does no shell quoting and invokes the binary via
    ``PATH``. Methods return decoded strings (``stdout``) or parsed JSON
    and raise :class:`KubectlError` on non-zero exit.
    """

    @staticmethod
    def _run(args: Sequence[str]) -> str:
        """Run ``kubectl <args>`` and return decoded ``stdout``.

        Args:
            args: Arguments appended after ``kubectl``.

        Returns:
            The child's ``stdout`` as a ``str``.

        Raises:
            KubectlError: On any non-zero exit.
        """
        result = run_tool("kubectl", args)
        if result.returncode != 0:
            raise KubectlError(
                ["kubectl", *args],
                result.returncode,
                result.stderr,
                result.stdout,
            )
        return result.stdout

    @staticmethod
    def _run_with_input(args: Sequence[str], yaml_str: str) -> None:
        """Run ``kubectl <args>`` feeding ``yaml_str`` on ``stdin``.

        Used by :meth:`apply`, :meth:`create`, and :meth:`replace` which
        all pipe a rendered manifest to ``kubectl -f -``.

        Args:
            args: Arguments appended after ``kubectl``.
            yaml_str: Manifest body forwarded to ``stdin``.

        Raises:
            KubectlError: On any non-zero exit.
        """
        result = run_tool("kubectl", args, input_=yaml_str)
        if result.returncode != 0:
            raise KubectlError(
                ["kubectl", *args],
                result.returncode,
                result.stderr,
                result.stdout,
            )

    def apply(self, yaml_str: str, namespace: str) -> None:
        """Run ``kubectl apply -n <namespace> -f -`` with ``yaml_str`` on stdin."""
        self._run_with_input(["apply", "-n", namespace, "-f", "-"], yaml_str)

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
            resource_type: Kubernetes resource kind (``"pod"``,
                ``"lease"``, ...).
            name: Object name.
            namespace: Target namespace.
            ignore_not_found: When ``True``, append
                ``--ignore-not-found`` so a missing object is not an
                error.
        """
        args = ["delete", resource_type, name, "-n", namespace]
        if ignore_not_found:
            args.append("--ignore-not-found")
        self._run(args)

    def delete_by_label(self, resource_type: str, label: str, namespace: str) -> None:
        """Delete all resources of ``resource_type`` matching ``label``."""
        self._run([
            "delete",
            resource_type,
            "-l",
            label,
            "-n",
            namespace,
            "--ignore-not-found",
        ])

    def wait(
        self,
        resource_type: str,
        name: str,
        condition: str,
        namespace: str,
        timeout: int = 120,
    ) -> None:
        """Block until ``resource_type/name`` reaches ``condition``."""
        self._run([
            "wait",
            f"{resource_type}/{name}",
            f"--for={condition}",
            f"--timeout={timeout}s",
            "-n",
            namespace,
        ])

    def logs(self, resource_type: str, name: str, namespace: str) -> str:
        """Return ``kubectl logs`` output for ``resource_type/name``."""
        return self._run(["logs", f"{resource_type}/{name}", "-n", namespace])

    def rollout_status(
        self,
        resource_type: str,
        name: str,
        namespace: str,
        timeout: int = 120,
    ) -> None:
        """Block until ``kubectl rollout status`` reports completion."""
        self._run([
            "rollout",
            "status",
            f"{resource_type}/{name}",
            "-n",
            namespace,
            f"--timeout={timeout}s",
        ])

    def top_pod(self, name: str, namespace: str) -> dict[str, str]:
        """Return CPU/memory usage for a pod.

        Args:
            name: Pod name.
            namespace: Pod namespace.

        Returns:
            ``{"cpu": ..., "memory": ...}`` parsed from ``kubectl top pod
            --no-headers`` output, or ``{}`` if the output has fewer than
            three whitespace-separated fields (empty / unexpected shape).
        """
        output = self._run([
            "top",
            "pod",
            name,
            "-n",
            namespace,
            "--no-headers",
        ])
        parts = output.strip().split()
        if len(parts) >= _TOP_POD_FIELDS:
            return {"cpu": parts[1], "memory": parts[2]}
        return {}

    def top_pod_containers(self, name: str, namespace: str) -> list[dict[str, str]]:
        """Return per-container CPU/memory usage for a pod.

        Args:
            name: Pod name.
            namespace: Pod namespace.

        Returns:
            A list of ``{"container": ..., "cpu": ..., "memory": ...}``
            rows, one per container. Rows with fewer than four
            whitespace-separated fields are skipped.
        """
        output = self._run([
            "top",
            "pod",
            name,
            "-n",
            namespace,
            "--containers",
            "--no-headers",
        ])
        rows: list[dict[str, str]] = []
        for line in output.strip().splitlines():
            parts = line.split()
            if len(parts) >= _TOP_POD_CONTAINER_FIELDS:
                rows.append({
                    "container": parts[1],
                    "cpu": parts[2],
                    "memory": parts[3],
                })
        return rows

    def get_pod_name(self, label: str, namespace: str) -> str:
        """Return the name of the first pod matching ``label``."""
        return self._run([
            "get",
            "pods",
            "-l",
            label,
            "-n",
            namespace,
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ]).strip()

    def get_node_zone(self, node: str) -> str:
        """Return the ``topology.kubernetes.io/zone`` label for ``node``.

        Returns an empty string if the label is unset.
        """
        return self._run([
            "get",
            "node",
            node,
            "-o",
            r"jsonpath={.metadata.labels.topology\.kubernetes\.io/zone}",
        ]).strip()

    def get_node_internal_ip(self, node: str) -> str:
        """Return ``node``'s ``InternalIP`` address, or ``""`` if absent."""
        return self._run([
            "get",
            "node",
            node,
            "-o",
            'jsonpath={.status.addresses[?(@.type=="InternalIP")].address}',
        ]).strip()

    def get_json(
        self, resource_type: str, name: str, namespace: str
    ) -> dict[str, Any] | None:
        """Return a resource as parsed JSON, or ``None`` on ``NotFound``.

        Args:
            resource_type: Kubernetes resource kind.
            name: Object name.
            namespace: Target namespace.

        Returns:
            The decoded JSON object, or ``None`` if ``kubectl`` reports
            the resource is missing. Values are typed ``Any`` because the
            Kubernetes object schema is heterogeneous (spec/metadata/
            status each carry their own shapes) — callers narrow as
            needed.

        Raises:
            KubectlError: For any error other than ``NotFound``.
        """
        try:
            output = self._run([
                "get",
                resource_type,
                name,
                "-n",
                namespace,
                "-o",
                "json",
            ])
        except KubectlError as e:
            if "NotFound" in e.stderr:
                return None
            raise
        parsed: dict[str, Any] = json.loads(output)
        return parsed

    def create(self, yaml_str: str, namespace: str) -> None:
        """Run ``kubectl create -n <namespace> -f -``.

        Raises :class:`KubectlError` on any non-zero exit; callers that
        need to tolerate ``AlreadyExists`` inspect ``e.stderr``.
        """
        self._run_with_input(["create", "-n", namespace, "-f", "-"], yaml_str)

    def replace(self, yaml_str: str, namespace: str) -> None:
        """Run ``kubectl replace -n <namespace> -f -``.

        Raises :class:`KubectlError` on any non-zero exit; callers that
        need to tolerate ``Conflict`` inspect ``e.stderr``.
        """
        self._run_with_input(["replace", "-n", namespace, "-f", "-"], yaml_str)
