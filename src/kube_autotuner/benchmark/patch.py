"""Apply kustomize patches to rendered Kubernetes YAML.

Shells out to the ``kustomize`` binary for full-fidelity patch semantics
(strategic merge by ``patchMergeKey``, JSON6902, ``$patch`` directives,
``labelSelector`` / ``annotationSelector`` targets). Every subprocess
goes through :func:`kube_autotuner.subproc.run_tool` -- the repo's
single sanctioned entrypoint -- so hygiene is uniform with the
``kubectl`` and ``talosctl`` call sites.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

import yaml

from kube_autotuner.experiment import ExperimentConfigError
from kube_autotuner.subproc import run_tool

if TYPE_CHECKING:
    from kube_autotuner.experiment import Patch, PatchTarget


def apply_patches(yaml_text: str, patches: list[Patch]) -> str:
    """Apply a list of kustomize-style patches to a multi-doc YAML string.

    Args:
        yaml_text: Base YAML (one or more documents) to feed kustomize.
        patches: Patches to apply; when empty, ``yaml_text`` is returned
            unchanged and ``kustomize`` is not invoked.

    Returns:
        The rendered YAML produced by ``kustomize build``.

    Raises:
        ExperimentConfigError: When ``kustomize`` is missing from ``PATH``
            or exits non-zero. The error message carries stderr so
            callers can surface kustomize diagnostics.
    """
    if not patches:
        return yaml_text

    with tempfile.TemporaryDirectory(prefix="kube-autotuner-kustomize-") as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "base.yaml").write_text(yaml_text, encoding="utf-8")
        (tmpdir / "kustomization.yaml").write_text(
            _build_kustomization(patches),
            encoding="utf-8",
        )
        try:
            result = run_tool("kustomize", ["build", str(tmpdir)], check=False)
        except FileNotFoundError as e:
            msg = "`kustomize` binary not found on PATH"
            raise ExperimentConfigError(msg) from e
        if result.returncode != 0:
            msg = f"kustomize build failed (rc={result.returncode}):\n{result.stderr}"
            raise ExperimentConfigError(msg)
        return result.stdout


# Sensible apiVersion defaults so users don't have to spell one out for
# the strategic-merge placeholder header. Only consulted when the user's
# Patch target supplies a ``kind`` but no group/version.
_DEFAULT_API_VERSION: dict[str, str] = {
    "Deployment": "apps/v1",
    "StatefulSet": "apps/v1",
    "DaemonSet": "apps/v1",
    "ReplicaSet": "apps/v1",
    "Job": "batch/v1",
    "CronJob": "batch/v1",
    "Pod": "v1",
    "Service": "v1",
    "ConfigMap": "v1",
    "Secret": "v1",
    "ServiceAccount": "v1",
    "Namespace": "v1",
}


def _build_kustomization(patches: list[Patch]) -> str:
    """Serialize our Patch objects into a ``kustomization.yaml``.

    Returns:
        The rendered ``kustomization.yaml`` contents.
    """
    entries: list[dict[str, Any]] = []
    for p in patches:
        entry: dict[str, Any] = {"patch": _patch_body_to_str(p.patch, p.target)}
        target = p.target.model_dump(exclude_none=True)
        if target:
            entry["target"] = target
        entries.append(entry)
    doc = {
        "apiVersion": "kustomize.config.k8s.io/v1beta1",
        "kind": "Kustomization",
        "resources": ["base.yaml"],
        "patches": entries,
    }
    return yaml.safe_dump(doc, sort_keys=False)


def _patch_body_to_str(
    body: list[dict[str, Any]] | dict[str, Any] | str,
    target: PatchTarget,
) -> str:
    """Serialize a :class:`Patch` body to the string form kustomize expects.

    Strategic Merge Patch bodies must carry resource identity
    (``apiVersion`` / ``kind`` / ``metadata.name``) for kustomize's
    parser. When the user supplies a dict body without these headers,
    synthesize them from the patch ``target`` selector.

    Returns:
        The patch body serialized as a single YAML or JSON Patch string.
    """
    if isinstance(body, str):
        return body
    if isinstance(body, list):
        return yaml.safe_dump(body, sort_keys=False)
    body = _ensure_smp_headers(dict(body), target)
    return yaml.safe_dump(body, sort_keys=False)


def _ensure_smp_headers(
    body: dict[str, Any],
    target: PatchTarget,
) -> dict[str, Any]:
    """Return a copy of ``body`` with SMP headers prepended if missing.

    Never mutates the input. ``apiVersion`` / ``kind`` are inferred from
    ``target`` when absent; ``metadata.name`` is synthesized as
    ``"_patch"`` because kustomize requires a name but the value is
    irrelevant for target-scoped merges.

    Returns:
        A new dict with the headers prepended and the original body
        contents preserved.
    """
    out: dict[str, Any] = {}
    if "apiVersion" not in body:
        out["apiVersion"] = _infer_api_version(target)
    if "kind" not in body:
        out["kind"] = target.kind or "Deployment"
    out.update(body)
    md = out.get("metadata")
    if not isinstance(md, dict):
        out["metadata"] = {"name": "_patch"}
    elif "name" not in md:
        out["metadata"] = {**md, "name": "_patch"}
    return out


def _infer_api_version(target: PatchTarget) -> str:
    """Infer an ``apiVersion`` string from a :class:`PatchTarget`.

    Returns:
        The resolved ``apiVersion`` ("``group/version``" or bare
        version), defaulting to ``"v1"`` when the target provides no
        identifying fields.
    """
    if target.group and target.version:
        return f"{target.group}/{target.version}"
    if target.version and not target.group:
        return target.version
    if target.kind and target.kind in _DEFAULT_API_VERSION:
        return _DEFAULT_API_VERSION[target.kind]
    return "v1"
