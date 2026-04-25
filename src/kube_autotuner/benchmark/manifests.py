"""Thin wrappers that combine ``build_*_yaml`` builders with ``apply_patches``.

Each wrapper takes the primitives the runner already has on hand (node
identity, ports, the Pydantic section model) and returns the rendered
YAML text. The boundary is ``str`` in / ``str`` out because
:func:`kube_autotuner.benchmark.patch.apply_patches` pipes through
``kustomize`` which is YAML-text in / YAML-text out -- a ``dict``
boundary would force an unnecessary ``yaml.safe_load`` / ``yaml.dump``
round-trip and lose document order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from kube_autotuner.benchmark.client_spec import build_client_yaml
from kube_autotuner.benchmark.fortio_client_spec import build_fortio_client_yaml
from kube_autotuner.benchmark.fortio_server_spec import build_fortio_server_yaml
from kube_autotuner.benchmark.patch import apply_patches
from kube_autotuner.benchmark.server_spec import build_server_yaml

if TYPE_CHECKING:
    from kube_autotuner.benchmark.fortio_client_spec import Workload
    from kube_autotuner.experiment import FortioSection, IperfSection, Patch


def render_iperf_server(
    *,
    node: str,
    ip_family_policy: str,
    ports: list[int],
    iperf_args: IperfSection,
    patches: list[Patch],
) -> str:
    """Render the iperf3 server Deployment + Service YAML with patches applied.

    Args:
        node: Kubernetes node name to pin the server pod to.
        ip_family_policy: ``spec.ipFamilyPolicy`` on the Service.
        ports: Ports to expose on the server.
        iperf_args: Iperf section model. Only ``server.extra_args`` is
            consumed; the section is taken whole for symmetry with the
            client wrapper so all four ``render_*`` wrappers share the
            same shape.
        patches: Kustomize patches applied to the rendered manifest.

    Returns:
        The rendered multi-document YAML string.
    """
    yaml_text = build_server_yaml(
        node=node,
        ports=ports,
        ip_family_policy=ip_family_policy,
        extra_args=iperf_args.server.extra_args,
    )
    return apply_patches(yaml_text, patches)


def render_iperf_client(
    *,
    source_node: str,
    target_node: str,
    port: int,
    mode: Literal["tcp", "udp"],
    iperf_args: IperfSection,
    patches: list[Patch],
    start_at_epoch: int | None,
) -> str:
    """Render an iperf3 client Job YAML with patches applied.

    Args:
        source_node: Client node; pinned via ``nodeSelector``.
        target_node: Server node; the client connects to the
            ``iperf3-server-<target_node>`` Service.
        port: Server-side port for this client.
        mode: ``"tcp"`` or ``"udp"``.
        iperf_args: Iperf section model. Consumed: ``duration``,
            ``omit``, ``parallel``, ``client.extra_args``.
        patches: Kustomize patches applied to the rendered Job.
        start_at_epoch: Absolute Unix timestamp the client should sleep
            until before exec'ing iperf3. ``None`` disables the barrier.

    Returns:
        The rendered Job YAML string.
    """
    yaml_text = build_client_yaml(
        node=source_node,
        target=target_node,
        port=port,
        duration=iperf_args.duration,
        omit=iperf_args.omit,
        parallel=iperf_args.parallel,
        mode=mode,
        extra_args=iperf_args.client.extra_args,
        start_at_epoch=start_at_epoch,
    )
    return apply_patches(yaml_text, patches)


def render_fortio_server(
    *,
    node: str,
    ip_family_policy: str,
    fortio_args: FortioSection,
    patches: list[Patch],
) -> str:
    """Render the fortio server Deployment + Service YAML with patches applied.

    Args:
        node: Kubernetes node name to pin the server pod to.
        ip_family_policy: ``spec.ipFamilyPolicy`` on the Service.
        fortio_args: Fortio section model. Only ``server.extra_args`` is
            consumed; the section is taken whole for symmetry with the
            client wrapper.
        patches: Kustomize patches applied to the rendered manifest.

    Returns:
        The rendered multi-document YAML string.
    """
    yaml_text = build_fortio_server_yaml(
        node=node,
        ip_family_policy=ip_family_policy,
        extra_args=fortio_args.server.extra_args,
    )
    return apply_patches(yaml_text, patches)


def render_fortio_client(
    *,
    source_node: str,
    target_node: str,
    iteration: int,
    workload: Workload,
    fortio_args: FortioSection,
    patches: list[Patch],
    start_at_epoch: int | None,
) -> str:
    """Render a fortio client Job YAML with patches applied.

    Derives ``qps`` from ``workload`` so the runner does not have to:
    saturation runs use ``-qps 0`` (drives max rate); fixed-QPS runs
    use ``fortio_args.fixed_qps``.

    Args:
        source_node: Client node; pinned via ``nodeSelector``.
        target_node: Server node; the client connects to the
            ``fortio-server-<target_node>`` Service.
        iteration: Zero-based iteration index; embedded in the Job
            name for cross-iteration uniqueness.
        workload: ``"saturation"`` or ``"fixed_qps"``.
        fortio_args: Fortio section model. Consumed: ``fixed_qps``,
            ``connections``, ``duration``, ``client.extra_args``.
        patches: Kustomize patches applied to the rendered Job.
        start_at_epoch: Absolute Unix timestamp the client should sleep
            until before invoking ``fortio load``. ``None`` disables the
            barrier.

    Returns:
        The rendered Job YAML string.
    """
    qps = 0 if workload == "saturation" else fortio_args.fixed_qps
    yaml_text = build_fortio_client_yaml(
        node=source_node,
        target=target_node,
        iteration=iteration,
        workload=workload,
        qps=qps,
        connections=fortio_args.connections,
        duration=fortio_args.duration,
        extra_args=fortio_args.client.extra_args,
        start_at_epoch=start_at_epoch,
    )
    return apply_patches(yaml_text, patches)
