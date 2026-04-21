"""YAML-driven experiment configuration with preflight checks.

Defines :class:`ExperimentConfig`, the Pydantic model backing
``experiment.yaml`` files that drive ``kube-autotuner`` runs, plus the
preflight harness that validates a config against local tooling and the
live cluster before a run begins.

Preflight checks are modelled as :class:`PreflightResult` records;
:meth:`ExperimentConfig.preflight` runs every check and returns the full
list so callers can display every problem at once instead of failing on
the first. :class:`ExperimentConfigError` is reserved for YAML parse and
schema validation failures surfaced by :meth:`ExperimentConfig.from_yaml`.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
from typing import TYPE_CHECKING, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from pydantic.alias_generators import to_camel
import yaml

from kube_autotuner.models import BenchmarkConfig, NodePair, ParamSpace, SysctlParam
from kube_autotuner.subproc import run_tool
from kube_autotuner.sysctl.params import PARAM_CATEGORIES, PARAM_SPACE

if TYPE_CHECKING:
    from kube_autotuner.k8s.client import K8sClient

Mode = Literal["baseline", "trial", "optimize"]


# iperf3 flags the tool controls. Users cannot pass these through
# ``extra_args`` because either :class:`BenchmarkConfig` owns them or
# the JSON parser depends on the output format. Matching is whole-token,
# not substring, so ``--windowsize-hint`` is not confused with ``--window``.
CLIENT_FLAG_DENYLIST: frozenset[str] = frozenset({
    "-c",
    "--client",
    "-p",
    "--port",
    "-t",
    "--time",
    "-O",
    "--omit",
    "-P",
    "--parallel",
    "-w",
    "--window",
    "-u",
    "--udp",
    "-J",
    "--json",
    "-B",
    "--bind",
    "-f",
    "--format",
    "--get-server-output",
    "--logfile",
})

SERVER_FLAG_DENYLIST: frozenset[str] = frozenset({
    "-s",
    "--server",
    "-p",
    "--port",
    "-B",
    "--bind",
    "-f",
    "--format",
    "--logfile",
})

# fortio flags the tool controls. ``-qps`` / ``-c`` / ``-t`` and ``-n``
# are owned by :class:`FortioSection`; ``-json`` and the trailing URL
# shape are required by the parser and runner; ``-H``, ``-http1.0``,
# ``-stdclient`` and ``-quiet`` change the request shape or output
# format in ways that break the parser.
FORTIO_CLIENT_FLAG_DENYLIST: frozenset[str] = frozenset({
    "-qps",
    "-c",
    "-t",
    "-n",
    "-json",
    "-url",
    "-H",
    "-http1.0",
    "-stdclient",
    "-quiet",
})

FORTIO_SERVER_FLAG_DENYLIST: frozenset[str] = frozenset({
    "-http-port",
})

SYSCTL_NAME_RE = re.compile(r"^[a-z][a-z0-9._]+$")


class ExperimentConfigError(Exception):
    """YAML parse or schema validation failure.

    Raised by :meth:`ExperimentConfig.from_yaml` for malformed YAML and
    Pydantic validation errors. Preflight failures are reported via
    :class:`PreflightResult` instead of this exception, so callers can
    collect every problem before deciding whether to abort.
    """


class PreflightResult(BaseModel):
    """Outcome of a single preflight check.

    Attributes:
        name: Short identifier for the check (e.g. ``"denylists"``).
        passed: ``True`` when the check succeeded; ``False`` when the
            config violates the invariant the check enforces.
        detail: Human-readable context. On success this is often empty or
            a short "skipped" note; on failure it carries the diagnostic
            message the operator needs to fix the config.
    """

    name: str
    passed: bool
    detail: str = ""


class PatchTarget(BaseModel):
    """Kustomize patch target selector.

    Mirrors the subset of the kustomize ``patches[*].target`` schema that
    ``kube-autotuner`` exposes to users. ``extra="forbid"`` so typos in
    user YAML surface as validation errors rather than silent no-ops.
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

    ``strict`` is honoured by :meth:`ExperimentConfig.preflight`'s
    dry-render check, which fails when a strict patch matches zero
    resources in the rendered output.
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

    model_config = ConfigDict(
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    extra_args: list[str] = Field(default_factory=list)


class IperfSection(BaseModel):
    """Per-role ``extra_args`` and retry budget for iperf3 client Jobs.

    ``max_attempts`` is the number of full Job lifecycles the benchmark
    runner may try per client per iteration before giving up and
    raising. It is independent of the pod-level ``backoffLimit: 1`` in
    the Job manifest; that still controls how many times the Kubernetes
    Job controller retries a failed pod within one Job, while
    ``max_attempts`` gates how many times the runner rebuilds the Job
    from scratch. Worst-case wall time per client is
    ``max_attempts * _CLIENT_WAIT_TIMEOUT_SECONDS`` (default 3 * 180s =
    9 minutes) when every attempt hits the watch deadline.
    """

    model_config = ConfigDict(extra="forbid")

    client: IperfArgs = Field(default_factory=IperfArgs)
    server: IperfArgs = Field(default_factory=IperfArgs)
    max_attempts: int = Field(default=3, ge=1)


class FortioArgs(BaseModel):
    """Extra command-line arguments for a fortio invocation."""

    model_config = ConfigDict(
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    extra_args: list[str] = Field(default_factory=list)


class FortioSection(BaseModel):
    """Per-role ``extra_args`` and run shape for the fortio sub-stages.

    ``duration`` is intentionally independent of
    :attr:`BenchmarkConfig.duration` so operators can keep fortio runs
    short without shortening the iperf3 bandwidth window. ``fixed_qps``
    drives the latency percentile sub-stage; ``connections`` is
    forwarded to fortio's ``-c`` flag for both sub-stages.

    ``max_attempts`` is the number of full Job lifecycles the benchmark
    runner may try per client per iteration before giving up and
    raising. See :class:`IperfSection` for the relationship with the
    pod-level ``backoffLimit`` and the worst-case wall-time
    calculation.
    """

    model_config = ConfigDict(
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    client: FortioArgs = Field(default_factory=FortioArgs)
    server: FortioArgs = Field(default_factory=FortioArgs)
    fixed_qps: int = Field(default=1000, ge=1)
    connections: int = Field(default=4, ge=1)
    duration: int = Field(default=30, ge=1)
    max_attempts: int = Field(default=3, ge=1)


class CniSection(BaseModel):
    """Selector for CNI pods to track on the target node."""

    model_config = ConfigDict(
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    enabled: bool = True
    namespace: str = "kube-system"
    label_selector: str = "k8s-app=cilium"


class NodesSection(BaseModel):
    """Source/target node topology for an experiment."""

    model_config = ConfigDict(
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    sources: list[str] = Field(min_length=1)
    target: str
    hardware_class: str = Field(default="10g", min_length=1)
    namespace: str = "default"
    ip_family_policy: str = "RequireDualStack"


class OptimizeSection(BaseModel):
    """Ax Bayesian-loop configuration for ``mode=optimize``."""

    model_config = ConfigDict(
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    n_trials: int = Field(default=50, ge=1)
    n_sobol: int = Field(default=15, ge=1)
    apply_source: bool = False
    param_space: list[SysctlParam] | None = None


class TrialSection(BaseModel):
    """Fixed sysctl set for ``mode=trial``."""

    model_config = ConfigDict(extra="forbid")

    sysctls: dict[str, str | int] = Field(default_factory=dict)


Metric = Literal[
    "throughput",
    "cpu",
    "retransmit_rate",
    "node_memory",
    "cni_memory",
    "rps",
    "latency_p50",
    "latency_p90",
    "latency_p99",
]
Direction = Literal["maximize", "minimize"]

_CONSTRAINT_RE = re.compile(
    r"^\s*(?P<metric>[a-z_0-9]+)\s*(?P<op><=|>=|==)\s*(?P<value>[+-]?"
    r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$",
)

_DEFAULT_CONSTRAINTS: list[str] = [
    "throughput >= 1e6",
    "cpu <= 200",
    "retransmit_rate <= 1e-6",
    "node_memory <= 1e10",
    # requests/sec; only the saturation sub-stage feeds ``rps``, so
    # this floor only fails on fortio server crash (zero achieved
    # QPS). Not intended as a performance gate.
    "rps >= 100",
    # milliseconds; from the fixed_qps sub-stage only.
    "latency_p99 <= 1000",
]

_DEFAULT_WEIGHTS: dict[str, float] = {
    "cpu": 0.15,
    "node_memory": 0.15,
    "retransmit_rate": 0.3,
    "latency_p90": 0.1,
    "latency_p99": 0.15,
}


class ParetoObjective(BaseModel):
    """One metric/direction pair in the Pareto objective set."""

    model_config = ConfigDict(
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    metric: Metric
    direction: Direction


def _default_pareto() -> list[ParetoObjective]:
    """Return the default Pareto objective list.

    Returns:
        Nine objectives: throughput (max), cpu (min),
        retransmit_rate (min), node_memory (min), cni_memory (min),
        rps (max), latency_p50/p90/p99 (min).
    """
    return [
        ParetoObjective(metric="throughput", direction="maximize"),
        ParetoObjective(metric="cpu", direction="minimize"),
        ParetoObjective(metric="retransmit_rate", direction="minimize"),
        ParetoObjective(metric="node_memory", direction="minimize"),
        ParetoObjective(metric="cni_memory", direction="minimize"),
        ParetoObjective(metric="rps", direction="maximize"),
        ParetoObjective(metric="latency_p50", direction="minimize"),
        ParetoObjective(metric="latency_p90", direction="minimize"),
        ParetoObjective(metric="latency_p99", direction="minimize"),
    ]


class ObjectivesSection(BaseModel):
    """Pareto objectives, outcome constraints, and recommendation weights.

    Drives both the live Ax optimization loop and post-hoc analysis.
    ``pareto`` selects which metrics form the Pareto frontier and
    whether each is maximized or minimized. ``constraints`` are
    forwarded verbatim to Ax as outcome constraints.
    ``recommendation_weights`` (YAML key ``recommendationWeights``)
    scales minimize-direction metrics in the shared scoring formula
    implemented by :func:`kube_autotuner.scoring.score_rows` -- the
    same formula drives both the live ``Best so far`` panel and the
    post-hoc :func:`kube_autotuner.analysis.recommend_configs`
    ranking. Weights on maximize-direction metrics are rejected
    because the gain term is always ``+1.0 * norm`` in the normalized
    score.

    The default weights are
    ``{cpu: 0.15, node_memory: 0.15, retransmit_rate: 0.3,
    latency_p90: 0.1, latency_p99: 0.15}``; ``latency_p50`` is left
    unweighted so the mean-latency axis enters the Pareto set without
    dominating the recommendation score.
    """

    model_config = ConfigDict(
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    pareto: list[ParetoObjective] = Field(
        default_factory=_default_pareto,
        min_length=1,
    )
    constraints: list[str] = Field(default_factory=lambda: list(_DEFAULT_CONSTRAINTS))
    recommendation_weights: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_WEIGHTS),
    )

    @model_validator(mode="after")
    def _validate_objectives(self) -> ObjectivesSection:
        """Cross-field validation for objectives, weights, and constraints.

        Returns:
            ``self`` when every invariant holds.

        Raises:
            ValueError: A weight targets an unknown metric, a
                maximize-direction metric, or a negative value; or a
                constraint does not parse as ``"<metric> <op> <float>"``
                against the four known metrics.
        """
        directions: dict[str, str] = {obj.metric: obj.direction for obj in self.pareto}
        for metric, weight in self.recommendation_weights.items():
            if weight < 0:
                msg = (
                    f"recommendation_weights[{metric!r}]={weight} must be non-negative"
                )
                raise ValueError(msg)
            direction = directions.get(metric)
            if direction is None:
                msg = (
                    f"recommendation_weights key {metric!r} is not in pareto objectives"
                )
                raise ValueError(msg)
            if direction != "minimize":
                msg = (
                    f"recommendation_weights[{metric!r}] targets a "
                    "maximize-direction metric; weights are only valid "
                    "on minimize-direction metrics"
                )
                raise ValueError(msg)
        known = {
            "throughput",
            "cpu",
            "retransmit_rate",
            "node_memory",
            "cni_memory",
            "rps",
            "latency_p50",
            "latency_p90",
            "latency_p99",
        }
        for constraint in self.constraints:
            match = _CONSTRAINT_RE.match(constraint)
            if match is None:
                msg = (
                    f"constraint {constraint!r} does not match "
                    "'<metric> <=|>=|== <float>'"
                )
                raise ValueError(msg)
            metric = match.group("metric")
            if metric not in known:
                msg = (
                    f"constraint {constraint!r} references unknown metric "
                    f"{metric!r}; expected one of {sorted(known)}"
                )
                raise ValueError(msg)
        return self


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    mode: Mode
    nodes: NodesSection
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    optimize: OptimizeSection | None = None
    trial: TrialSection | None = None
    iperf: IperfSection = Field(default_factory=IperfSection)
    fortio: FortioSection = Field(default_factory=FortioSection)
    cni: CniSection = Field(default_factory=CniSection)
    patches: list[Patch] = Field(default_factory=list)
    objectives: ObjectivesSection = Field(default_factory=ObjectivesSection)
    output: str = "results.jsonl"

    @model_validator(mode="after")
    def _mode_requirements(self) -> ExperimentConfig:
        """Enforce mode-specific structural invariants.

        * ``mode=optimize`` requires an ``optimize:`` section.
        * ``mode=trial`` requires a ``trial:`` section with at least one
          entry in ``sysctls``.
        * ``optimize.n_sobol`` must not exceed ``optimize.n_trials``.

        Returns:
            ``self`` when every invariant holds.

        Raises:
            ValueError: One of the invariants above is violated.
        """
        if self.mode == "optimize" and self.optimize is None:
            msg = "mode=optimize requires `optimize:` section"
            raise ValueError(msg)
        if self.mode == "trial" and (self.trial is None or not self.trial.sysctls):
            msg = (
                "mode=trial requires a `trial:` section with a non-empty `sysctls:` map"
            )
            raise ValueError(msg)
        if self.optimize and self.optimize.n_sobol > self.optimize.n_trials:
            msg = "optimize.n_sobol must be <= optimize.n_trials"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        """Load and validate an experiment config from a YAML file.

        Args:
            path: Path to a single-document YAML file.

        Returns:
            The validated :class:`ExperimentConfig`.

        Raises:
            ExperimentConfigError: YAML parse error, multi-document YAML,
                or Pydantic schema validation failure.
        """
        try:
            docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
        except yaml.YAMLError as e:
            msg = f"invalid YAML at {path}: {e}"
            raise ExperimentConfigError(msg) from e
        if len(docs) > 1:
            msg = f"{path}: multi-document YAML not supported (got {len(docs)} docs)"
            raise ExperimentConfigError(msg)
        data = docs[0] if docs else {}
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            msg = f"invalid config at {path}:\n{e}"
            raise ExperimentConfigError(msg) from e

    def to_node_pair(self) -> NodePair:
        """Project ``nodes:`` into a :class:`NodePair` for the runner.

        Returns:
            A :class:`NodePair` whose ``source`` is the first entry in
            ``nodes.sources`` and whose ``extra_sources`` holds the rest.
        """
        primary, *extras = self.nodes.sources
        return NodePair(
            source=primary,
            target=self.nodes.target,
            hardware_class=self.nodes.hardware_class,
            namespace=self.nodes.namespace,
            ip_family_policy=self.nodes.ip_family_policy,
            extra_sources=extras,
        )

    def effective_param_space(self) -> ParamSpace:
        """Return the active search space for the experiment.

        Returns:
            ``optimize.param_space`` when the config supplies one. For
            the canonical default, UDP-category params are stripped when
            ``benchmark.modes`` does not include ``"udp"``, since the
            clients never generate UDP traffic and those dimensions are
            noise. A user-supplied override bypasses the mode gate.
        """
        if self.optimize and self.optimize.param_space:
            return ParamSpace(params=self.optimize.param_space)
        if "udp" in self.benchmark.modes:
            return PARAM_SPACE
        udp_names = set(PARAM_CATEGORIES.get("udp", []))
        return ParamSpace(
            params=[p for p in PARAM_SPACE.params if p.name not in udp_names],
        )

    def preflight(self, client: K8sClient) -> list[PreflightResult]:
        """Run every preflight check and return the results.

        Order: pure-data checks first, then local tool shell-outs, then
        cluster calls. The checks are independent so all results are
        collected; callers decide whether to fail fast by inspecting the
        returned list.

        Args:
            client: :class:`K8sClient` used for the node-exists cluster
                probe.

        Returns:
            One :class:`PreflightResult` per check, in execution order.
        """
        return [
            self._check_denylists(),
            self._check_sysctl_names(),
            self._check_patches_shape(),
            self._check_kustomize_available(),
            self._dry_render_patches(),
            self._check_output_path(),
            self._check_nodes_exist(client),
        ]

    def _check_denylists(self) -> PreflightResult:
        """Verify no reserved iperf3 or fortio flags appear in user ``extra_args``.

        Returns:
            A passing result when every entry in the iperf3 and fortio
            client/server ``extra_args`` lists avoids the denylists,
            otherwise a failing result naming the first offending
            flag.
        """
        name = "denylists"
        for tok in self.iperf.client.extra_args:
            if tok in CLIENT_FLAG_DENYLIST:
                detail = (
                    f"iperf.client.extra_args contains reserved flag {tok!r}; "
                    f"these are controlled by benchmark config or parser "
                    f"invariants: {sorted(CLIENT_FLAG_DENYLIST)}"
                )
                return PreflightResult(name=name, passed=False, detail=detail)
        for tok in self.iperf.server.extra_args:
            if tok in SERVER_FLAG_DENYLIST:
                detail = (
                    f"iperf.server.extra_args contains reserved flag {tok!r}; "
                    f"these are controlled by the server invariants: "
                    f"{sorted(SERVER_FLAG_DENYLIST)}"
                )
                return PreflightResult(name=name, passed=False, detail=detail)
        for tok in self.fortio.client.extra_args:
            if tok in FORTIO_CLIENT_FLAG_DENYLIST:
                detail = (
                    f"fortio.client.extra_args contains reserved flag {tok!r}; "
                    f"these are controlled by fortio config or parser "
                    f"invariants: {sorted(FORTIO_CLIENT_FLAG_DENYLIST)}"
                )
                return PreflightResult(name=name, passed=False, detail=detail)
        for tok in self.fortio.server.extra_args:
            if tok in FORTIO_SERVER_FLAG_DENYLIST:
                detail = (
                    f"fortio.server.extra_args contains reserved flag {tok!r}; "
                    f"these are controlled by the fortio server invariants: "
                    f"{sorted(FORTIO_SERVER_FLAG_DENYLIST)}"
                )
                return PreflightResult(name=name, passed=False, detail=detail)
        return PreflightResult(name=name, passed=True)

    def _check_sysctl_names(self) -> PreflightResult:
        """Verify every sysctl name conforms to :data:`SYSCTL_NAME_RE`.

        Returns:
            A passing result when every sysctl referenced in
            ``optimize.param_space`` or ``trial.sysctls`` is syntactically
            valid, otherwise a failing result naming the first violation.
        """
        name = "sysctl-names"
        names: list[str] = []
        if self.optimize and self.optimize.param_space:
            names.extend(p.name for p in self.optimize.param_space)
        if self.trial and self.trial.sysctls:
            names.extend(self.trial.sysctls.keys())
        for sysctl_name in names:
            if not SYSCTL_NAME_RE.match(sysctl_name):
                detail = (
                    f"invalid sysctl name {sysctl_name!r}: must match "
                    f"{SYSCTL_NAME_RE.pattern}"
                )
                return PreflightResult(name=name, passed=False, detail=detail)
        return PreflightResult(name=name, passed=True)

    def _check_patches_shape(self) -> PreflightResult:
        """Reject patches that try to override namespace handling.

        Returns:
            A passing result when no patch sets ``target.namespace``,
            ``metadata.namespace``, or a JSON6902 op against
            ``/metadata/namespace``, otherwise a failing result naming
            the first offending patch.
        """
        name = "patches-shape"
        for i, p in enumerate(self.patches):
            if p.target.namespace is not None:
                detail = (
                    f"patch[{i}] sets target.namespace; namespace is "
                    f"controlled by `nodes.namespace`"
                )
                return PreflightResult(name=name, passed=False, detail=detail)
            if isinstance(p.patch, dict):
                md = p.patch.get("metadata")
                if isinstance(md, dict) and "namespace" in md:
                    detail = (
                        f"patch[{i}] sets metadata.namespace; namespace is "
                        f"controlled by `nodes.namespace` and passed on "
                        f"every apply call"
                    )
                    return PreflightResult(name=name, passed=False, detail=detail)
            elif isinstance(p.patch, list):
                bad = _first_namespace_op(p.patch)
                if bad is not None:
                    detail = (
                        f"patch[{i}].ops[{bad}] targets /metadata/namespace; "
                        f"namespace is controlled by `nodes.namespace`"
                    )
                    return PreflightResult(name=name, passed=False, detail=detail)
        return PreflightResult(name=name, passed=True)

    def _check_kustomize_available(self) -> PreflightResult:
        """Verify ``kustomize`` is on ``PATH`` and responds to ``version``.

        The check is skipped when no patches are configured. When patches
        are configured, the probe routes through
        :func:`kube_autotuner.subproc.run_tool` -- the package's single
        sanctioned subprocess entrypoint.

        Returns:
            A passing result when the binary is usable (or not needed),
            otherwise a failing result carrying the invocation error.
        """
        name = "kustomize-available"
        if not self.patches:
            return PreflightResult(
                name=name,
                passed=True,
                detail="skipped: no patches configured",
            )
        if shutil.which("kustomize") is None:
            detail = (
                "`kustomize` binary not found on PATH; required when "
                "`patches:` is set. Install from https://kustomize.io/"
            )
            return PreflightResult(name=name, passed=False, detail=detail)
        try:
            result = run_tool("kustomize", ["version"], check=False)
        except OSError as e:
            detail = f"`kustomize version` failed: {e}"
            return PreflightResult(name=name, passed=False, detail=detail)
        if result.returncode != 0:
            detail = (
                f"`kustomize version` exited rc={result.returncode}: "
                f"{result.stderr.strip()}"
            )
            return PreflightResult(name=name, passed=False, detail=detail)
        return PreflightResult(name=name, passed=True)

    def _check_nodes_exist(self, client: K8sClient) -> PreflightResult:
        """Verify every referenced node is reachable by the API server.

        Args:
            client: :class:`K8sClient` used to probe each node via a
                topology-zone fetch.

        Returns:
            A passing result when every node in ``nodes.target`` and
            ``nodes.sources`` responds, otherwise a failing result naming
            the first unreachable node and the API error that surfaced
            it.
        """
        name = "nodes-exist"
        nodes = [self.nodes.target, *self.nodes.sources]
        for node in nodes:
            try:
                client.get_node_zone(node)
            except Exception as e:  # noqa: BLE001 - surface any client failure
                detail = f"node {node!r} not found or unreachable: {e}"
                return PreflightResult(name=name, passed=False, detail=detail)
        return PreflightResult(name=name, passed=True)

    def _check_output_path(self) -> PreflightResult:
        """Ensure the output path's parent directory exists and is writable.

        Creates missing parents via :meth:`pathlib.Path.mkdir` so the
        JSONL writer does not fail mid-run. The output file itself is
        never touched -- a later preflight failure must not leak an
        empty results file.

        Returns:
            A passing result when the parent exists (or was created) and
            is writable by the current process, otherwise a failing
            result describing the filesystem error.
        """
        name = "output-path"
        out = Path(self.output)
        try:
            if out.parent and not out.parent.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            detail = f"output path {self.output!r} not writable: {e}"
            return PreflightResult(name=name, passed=False, detail=detail)
        parent = out.parent if str(out.parent) else Path()
        if not os.access(parent, os.W_OK):
            detail = (
                f"output path {self.output!r} not writable: parent "
                f"directory {parent!s} is not writable"
            )
            return PreflightResult(name=name, passed=False, detail=detail)
        return PreflightResult(name=name, passed=True)

    def _dry_render_patches(self) -> PreflightResult:
        """Dry-render a representative manifest with patches applied.

        Validates (1) every patch produces valid K8s objects, and (2)
        each strict patch actually matches at least one resource --
        kustomize silently no-ops otherwise, so each strict patch is
        rendered individually and compared against the base.

        Returns:
            A passing result when every patch renders cleanly, otherwise
            a failing result describing the first rendering problem.
        """
        name = "dry-render-patches"
        if not self.patches:
            return PreflightResult(
                name=name,
                passed=True,
                detail="skipped: no patches configured",
            )

        # Local imports: the benchmark package imports ``Patch`` /
        # ``ExperimentConfigError`` from this module, so the dependency
        # must stay one-way at import time.
        from kube_autotuner.benchmark.client_spec import (  # noqa: PLC0415
            build_client_yaml,
        )
        from kube_autotuner.benchmark.fortio_client_spec import (  # noqa: PLC0415
            build_fortio_client_yaml,
        )
        from kube_autotuner.benchmark.fortio_server_spec import (  # noqa: PLC0415
            build_fortio_server_yaml,
        )
        from kube_autotuner.benchmark.patch import apply_patches  # noqa: PLC0415
        from kube_autotuner.benchmark.server_spec import (  # noqa: PLC0415
            build_server_yaml,
        )

        primary = self.nodes.sources[0]
        client_yaml = build_client_yaml(
            node=primary,
            target=self.nodes.target,
            port=5201,
            duration=self.benchmark.duration,
            omit=self.benchmark.omit,
            parallel=self.benchmark.parallel,
            mode="tcp",
            window=self.benchmark.window,
            extra_args=self.iperf.client.extra_args,
        )
        server_yaml = build_server_yaml(
            node=self.nodes.target,
            ports=[5201],
            ip_family_policy=self.nodes.ip_family_policy,
            extra_args=self.iperf.server.extra_args,
        )
        fortio_server_yaml = build_fortio_server_yaml(
            node=self.nodes.target,
            ip_family_policy=self.nodes.ip_family_policy,
            extra_args=self.fortio.server.extra_args,
        )
        fortio_client_yaml = build_fortio_client_yaml(
            node=primary,
            target=self.nodes.target,
            iteration=0,
            workload="fixed_qps",
            qps=self.fortio.fixed_qps,
            connections=self.fortio.connections,
            duration=self.fortio.duration,
            extra_args=self.fortio.client.extra_args,
        )
        combined = (
            f"{client_yaml}\n---\n{server_yaml}\n---\n"
            f"{fortio_server_yaml}\n---\n{fortio_client_yaml}"
        )
        base_docs = list(yaml.safe_load_all(combined))

        try:
            rendered = apply_patches(combined, self.patches)
        except ExperimentConfigError as e:
            return PreflightResult(name=name, passed=False, detail=str(e))

        docs = [d for d in yaml.safe_load_all(rendered) if d]
        shape_failure = _validate_rendered_docs(docs)
        if shape_failure is not None:
            return PreflightResult(name=name, passed=False, detail=shape_failure)

        for i, patch in enumerate(self.patches):
            if not patch.strict:
                continue
            try:
                single = apply_patches(combined, [patch])
            except ExperimentConfigError as e:
                return PreflightResult(name=name, passed=False, detail=str(e))
            single_docs = list(yaml.safe_load_all(single))
            if _yaml_docs_equal(single_docs, base_docs):
                detail = (
                    f"patch[{i}] target "
                    f"{patch.target.model_dump(exclude_none=True)} matched "
                    f"zero resources in the dry-render; set `strict: false` "
                    f"to allow this"
                )
                return PreflightResult(name=name, passed=False, detail=detail)
        return PreflightResult(name=name, passed=True)


def _first_namespace_op(ops: list[dict[str, Any]]) -> int | None:
    """Return the index of the first JSON6902 op touching metadata.namespace.

    Args:
        ops: JSON6902 operation objects from a :class:`Patch` body.

    Returns:
        The index of the offending op, or ``None`` when no op targets
        ``/metadata/namespace``.
    """
    for j, op in enumerate(ops):
        path = op.get("path", "") if isinstance(op, dict) else ""
        if isinstance(path, str) and path.startswith("/metadata/namespace"):
            return j
    return None


def _validate_rendered_docs(docs: list[Any]) -> str | None:
    """Confirm every dry-rendered doc carries K8s-required identity fields.

    Args:
        docs: YAML documents produced by ``kustomize build``.

    Returns:
        A diagnostic string describing the first invalid doc, or
        ``None`` when every doc is well-formed.
    """
    for d in docs:
        if not isinstance(d, dict):
            return f"dry-render produced non-object YAML doc: {d!r}"
        for req in ("apiVersion", "kind"):
            if req not in d:
                return f"dry-render produced doc missing {req!r}: {d.get('kind', d)}"
        metadata = d.get("metadata")
        if not isinstance(metadata, dict) or "name" not in metadata:
            return f"dry-render produced {d.get('kind')} without metadata.name"
    return None


# Resolve the forward reference inside
# :class:`kube_autotuner.models.ResumeMetadata` now that
# :class:`ObjectivesSection` exists. ``models`` declares the field under
# ``TYPE_CHECKING`` to avoid a circular import; importing this module
# is the single sanctioned trigger for the rebuild.
from kube_autotuner.models import _ensure_resume_metadata_built  # noqa: E402

_ensure_resume_metadata_built()


def _yaml_docs_equal(a: list[Any], b: list[Any]) -> bool:
    """Return whether two lists of YAML docs are equal as multisets.

    Kustomize re-sorts output docs by kind/name, so the comparison is an
    unordered multiset over full-content equality.

    Args:
        a: Rendered YAML documents.
        b: Baseline YAML documents.

    Returns:
        ``True`` when ``a`` and ``b`` contain the same documents
        (ignoring order), otherwise ``False``.
    """
    if len(a) != len(b):
        return False
    remaining = list(b)
    for doc in a:
        for i, other in enumerate(remaining):
            if doc == other:
                remaining.pop(i)
                break
        else:
            return False
    return True
