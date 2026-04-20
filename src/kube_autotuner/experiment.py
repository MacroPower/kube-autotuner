"""Experiment types consumed by the benchmark layer.

This module exposes the types that :mod:`kube_autotuner.benchmark`
imports at runtime:

* :class:`ExperimentConfigError` -- raised by
  :func:`kube_autotuner.benchmark.patch.apply_patches` when ``kustomize``
  cannot be invoked or refuses the generated ``kustomization.yaml``.
* :class:`PatchTarget` / :class:`Patch` -- target-and-body shapes that the
  benchmark patch layer renders into ``kustomization.yaml`` entries.
* :class:`IperfArgs` / :class:`IperfSection` -- optional ``extra_args``
  passthrough for iperf3 client and server pods, defaulted by
  :class:`kube_autotuner.benchmark.runner.BenchmarkRunner`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentConfigError(Exception):
    """YAML parse, schema validation, or preflight failure.

    The benchmark layer raises this for ``kustomize`` failures so that
    a single exception type covers both preflight validation and live
    benchmark runs.
    """


class PatchTarget(BaseModel):
    """Kustomize patch target selector.

    Mirrors the subset of the kustomize ``patches[*].target`` schema that
    the autotuner exposes to users. ``extra="forbid"`` so typos in user
    YAML surface as validation errors rather than silent no-ops.
    """

    model_config = ConfigDict(extra="forbid")

    group: str | None = None
    version: str | None = None
    kind: str | None = None
    name: str | None = None
    namespace: str | None = None
    labelSelector: str | None = None  # noqa: N815 - kustomize field name
    annotationSelector: str | None = None  # noqa: N815 - kustomize field name


class Patch(BaseModel):
    """A single kustomize patch with its target selector.

    ``patch`` accepts the three shapes kustomize does:

    * ``list[dict]`` -- a JSON6902 patch (RFC 6902 operation objects).
    * ``dict`` -- a Strategic Merge Patch body.
    * ``str`` -- a pre-rendered patch string.

    ``strict`` is stored on the model for use by preflight dry-rendering.
    """

    model_config = ConfigDict(extra="forbid")

    target: PatchTarget = Field(default_factory=PatchTarget)
    patch: list[dict[str, Any]] | dict[str, Any] | str
    strict: bool = True

    @model_validator(mode="after")
    def _smp_needs_kind(self) -> Patch:
        """Require a resolvable ``kind`` for strategic-merge patches.

        When a dict-body patch uses a ``labelSelector``-only target,
        kustomize applies the merge to every matching resource regardless
        of kind; without a ``kind`` in either the target or the body we
        cannot synthesize a sensible SMP header and we risk silent
        corruption across kinds. Require one or the other.

        Returns:
            ``self`` when the invariant holds.

        Raises:
            ValueError: Strategic-merge body lacks both a ``target.kind``
                and an explicit ``kind:`` entry in the body.
        """
        if not isinstance(self.patch, dict):
            return self
        if self.target.kind:
            return self
        body_kind = self.patch.get("kind")
        if isinstance(body_kind, str) and body_kind:
            return self
        msg = (
            "strategic-merge patches (dict body) require `target.kind` "
            "or an explicit `kind:` in the patch body"
        )
        raise ValueError(msg)


class IperfArgs(BaseModel):
    """Extra command-line arguments for an iperf3 invocation."""

    model_config = ConfigDict(extra="forbid")

    extra_args: list[str] = Field(default_factory=list)


class IperfSection(BaseModel):
    """Per-role ``extra_args`` for iperf3 client and server pods."""

    model_config = ConfigDict(extra="forbid")

    client: IperfArgs = Field(default_factory=IperfArgs)
    server: IperfArgs = Field(default_factory=IperfArgs)
