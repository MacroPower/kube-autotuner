"""Parameter search space definitions for the sysctl tuning surface.

The canonical search space is assembled and validated by
:func:`build_param_space`. A module-level :data:`PARAM_SPACE` alias is
kept for consumers that prefer a ready-made constant.
"""

from __future__ import annotations

from kube_autotuner.models import ParamSpace, SysctlParam

# Full search space, organised by category.

_TCP_BUFFER_PARAMS: list[SysctlParam] = [
    SysctlParam(
        name="net.core.rmem_max",
        values=[212992, 4194304, 16777216, 67108864],
        param_type="int",
    ),
    SysctlParam(
        name="net.core.wmem_max",
        values=[212992, 4194304, 16777216, 67108864],
        param_type="int",
    ),
    SysctlParam(
        name="net.core.rmem_default",
        values=[212992, 1048576, 16777216],
        param_type="int",
    ),
    SysctlParam(
        name="net.core.wmem_default",
        values=[212992, 1048576, 16777216],
        param_type="int",
    ),
    SysctlParam(
        name="net.ipv4.tcp_rmem",
        values=[
            "4096 87380 6291456",
            "4096 131072 16777216",
            "4096 87380 33554432",
        ],
        param_type="choice",
    ),
    SysctlParam(
        name="net.ipv4.tcp_wmem",
        values=[
            "4096 16384 4194304",
            "4096 65536 16777216",
            "4096 65536 33554432",
        ],
        param_type="choice",
    ),
    SysctlParam(
        name="net.ipv4.tcp_mem",
        # Kernel default is auto-sized; represented here as the common
        # three-integer form since sysctl -w requires three integers.
        values=["393216 524288 786432", "786432 1048576 1572864"],
        param_type="choice",
    ),
    SysctlParam(
        name="net.ipv4.udp_mem",
        values=["393216 524288 786432", "786432 1048576 1572864"],
        param_type="choice",
    ),
    SysctlParam(
        name="net.core.optmem_max",
        values=[131072, 524288, 16777216],
        param_type="int",
    ),
]

_CONGESTION_PARAMS: list[SysctlParam] = [
    SysctlParam(
        name="net.ipv4.tcp_congestion_control",
        values=["cubic", "bbr"],
        param_type="choice",
    ),
    SysctlParam(
        name="net.core.default_qdisc",
        values=["pfifo_fast", "fq", "fq_codel"],
        param_type="choice",
    ),
    SysctlParam(
        name="net.ipv4.tcp_fastopen",
        values=[0, 3],
        param_type="choice",
    ),
    SysctlParam(
        name="net.ipv4.tcp_slow_start_after_idle",
        values=[0, 1],
        param_type="choice",
    ),
    SysctlParam(
        name="net.ipv4.tcp_no_metrics_save",
        values=[0, 1],
        param_type="choice",
    ),
    SysctlParam(
        name="net.ipv4.tcp_mtu_probing",
        values=[0, 1],
        param_type="choice",
    ),
    SysctlParam(
        name="net.ipv4.tcp_ecn",
        values=[0, 1, 2],
        param_type="choice",
    ),
    SysctlParam(
        name="net.ipv4.tcp_limit_output_bytes",
        values=[262144, 4194304],
        param_type="int",
    ),
]

_NAPI_PARAMS: list[SysctlParam] = [
    SysctlParam(
        name="net.core.netdev_max_backlog",
        values=[1000, 5000, 30000, 250000],
        param_type="int",
    ),
    SysctlParam(
        name="net.core.netdev_budget",
        values=[300, 600, 1000, 2000],
        param_type="int",
    ),
    SysctlParam(
        name="net.core.dev_weight",
        values=[64, 128, 256, 600],
        param_type="int",
    ),
    SysctlParam(
        name="net.core.gro_normal_batch",
        values=[8, 16, 32],
        param_type="int",
    ),
]

_BUSY_POLL_PARAMS: list[SysctlParam] = [
    SysctlParam(
        name="net.core.busy_poll",
        values=[0, 50, 100],
        param_type="int",
    ),
    SysctlParam(
        name="net.core.busy_read",
        values=[0, 50, 100],
        param_type="int",
    ),
]

_MEMORY_PARAMS: list[SysctlParam] = [
    SysctlParam(
        name="vm.min_free_kbytes",
        values=[65536, 131072, 262144],
        param_type="int",
    ),
    SysctlParam(
        name="vm.swappiness",
        values=[1, 10, 60],
        param_type="int",
    ),
]

_CONNECTION_PARAMS: list[SysctlParam] = [
    SysctlParam(
        name="net.core.somaxconn",
        values=[128, 4096],
        param_type="int",
    ),
    SysctlParam(
        name="net.ipv4.tcp_max_syn_backlog",
        values=[1024, 4096, 65536],
        param_type="int",
    ),
    SysctlParam(
        name="net.ipv4.tcp_tw_reuse",
        values=[0, 1],
        param_type="choice",
    ),
    SysctlParam(
        name="net.ipv4.tcp_fin_timeout",
        values=[15, 60],
        param_type="int",
    ),
]

_UDP_PARAMS: list[SysctlParam] = [
    SysctlParam(
        name="net.ipv4.udp_rmem_min",
        values=[4096, 65536],
        param_type="int",
    ),
]


_MIN_INT_VALUES = 2


def _validate_params(params: list[SysctlParam]) -> None:
    """Validate that every :class:`SysctlParam` is well-formed.

    Args:
        params: Parameters to check.

    Raises:
        ValueError: If a parameter has an empty ``values`` list, or if an
            ``int``-typed parameter has fewer than two numeric values or a
            zero-width range.
    """
    for p in params:
        if not p.values:
            msg = f"SysctlParam {p.name!r} has empty values list"
            raise ValueError(msg)
        if p.param_type == "int":
            numeric = [v for v in p.values if isinstance(v, int)]
            if len(numeric) < _MIN_INT_VALUES or min(numeric) >= max(numeric):
                msg = (
                    f"SysctlParam {p.name!r} is typed 'int' but its values "
                    f"do not span a range (min < max required): {p.values!r}"
                )
                raise ValueError(msg)


def build_param_space(params: list[SysctlParam] | None = None) -> ParamSpace:
    """Assemble and validate the canonical sysctl search space.

    Args:
        params: Optional override for the default parameter set. When
            omitted, the canonical categories declared in this module are
            concatenated. Tests pass deliberately malformed lists to
            exercise the validator.

    Returns:
        A validated :class:`ParamSpace` containing ``params`` in
        declaration order. :func:`_validate_params` raises
        :class:`ValueError` for malformed input.
    """
    effective = (
        params
        if params is not None
        else (
            _TCP_BUFFER_PARAMS
            + _CONGESTION_PARAMS
            + _NAPI_PARAMS
            + _BUSY_POLL_PARAMS
            + _MEMORY_PARAMS
            + _CONNECTION_PARAMS
            + _UDP_PARAMS
        )
    )
    _validate_params(effective)
    return ParamSpace(params=effective)


PARAM_SPACE: ParamSpace = build_param_space()

PARAM_CATEGORIES: dict[str, list[str]] = {
    "tcp_buffer": [p.name for p in _TCP_BUFFER_PARAMS],
    "congestion": [p.name for p in _CONGESTION_PARAMS],
    "napi": [p.name for p in _NAPI_PARAMS],
    "busy_poll": [p.name for p in _BUSY_POLL_PARAMS],
    "memory": [p.name for p in _MEMORY_PARAMS],
    "connection": [p.name for p in _CONNECTION_PARAMS],
    "udp": [p.name for p in _UDP_PARAMS],
}

PARAM_TO_CATEGORY: dict[str, str] = {
    param: cat for cat, params in PARAM_CATEGORIES.items() for param in params
}

# Current production sysctl values for 10G nodes.
# Excludes vm.nr_hugepages which is not in the tuning space.
DEFAULT_SYSCTLS_10G: dict[str, str | int] = {
    "net.core.rmem_max": 67108864,
    "net.core.wmem_max": 67108864,
    "net.ipv4.tcp_mtu_probing": 1,
    "net.ipv4.tcp_rmem": "4096 87380 33554432",
    "net.ipv4.tcp_wmem": "4096 65536 33554432",
}
