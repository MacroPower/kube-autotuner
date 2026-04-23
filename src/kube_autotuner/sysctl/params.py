"""Parameter search space definitions for the sysctl tuning surface.

The canonical search space is assembled and validated by
:func:`build_param_space`. A module-level :data:`PARAM_SPACE` alias is
kept for consumers that prefer a ready-made constant.

Each :class:`SysctlParam` below carries a short comment summarising how
changing that knob is expected to move the configured objectives
(tcp_throughput, udp_throughput, tcp_retransmit_rate, udp_loss_rate,
udp_jitter, rps, latency_p50/p90/p99). These are expectations used to
read results, not guarantees; the optimizer discovers the actual
response surface.
"""

from __future__ import annotations

from kube_autotuner.models import ParamSpace, SysctlParam

# Full search space, organised by category.

_TCP_BUFFER_PARAMS: list[SysctlParam] = [
    # System-wide SO_RCVBUF ceiling. Low rungs cap the TCP receive
    # window on fat pipes and throttle throughput on long-BDP paths;
    # high rungs trade node memory headroom under many flows.
    SysctlParam(
        name="net.core.rmem_max",
        values=[212992, 4194304, 16777216, 67108864],
        param_type="int",
    ),
    # System-wide SO_SNDBUF ceiling. Mirror of rmem_max on the send
    # path: low rungs cap the send window and throughput, high rungs
    # raise memory pressure.
    SysctlParam(
        name="net.core.wmem_max",
        values=[212992, 4194304, 16777216, 67108864],
        param_type="int",
    ),
    # Autotuning triple (min default max) for TCP receive buffers.
    # Governs the peak receive window. Under-sized max caps iperf3
    # throughput on high-BDP links and can raise tcp_retransmit_rate
    # when the window collapses; over-sized max raises node memory
    # under many concurrent flows.
    SysctlParam(
        name="net.ipv4.tcp_rmem",
        values=[
            "4096 87380 6291456",
            "4096 131072 16777216",
            "4096 87380 33554432",
        ],
        param_type="choice",
    ),
    # Autotuning triple for TCP send buffers. Same trade-off as
    # tcp_rmem on the send path: throughput vs. memory footprint.
    SysctlParam(
        name="net.ipv4.tcp_wmem",
        values=[
            "4096 16384 4194304",
            "4096 65536 16777216",
            "4096 65536 33554432",
        ],
        param_type="choice",
    ),
    # Global TCP memory pressure thresholds in pages (min pressure max).
    # Too small and the kernel prunes/throttles under load (throughput
    # drops, tcp_retransmit_rate rises); too large and node memory climbs.
    # Represented as the canonical three-integer string form.
    SysctlParam(
        name="net.ipv4.tcp_mem",
        values=["393216 524288 786432", "786432 1048576 1572864"],
        param_type="choice",
    ),
]

_CONGESTION_PARAMS: list[SysctlParam] = [
    # cubic vs. bbr. BBR drives higher throughput on shallow-buffered
    # paths and generally cuts tcp_retransmit_rate, but can move p99
    # latency in either direction depending on the fabric. One of the
    # largest single levers in the whole space.
    SysctlParam(
        name="net.ipv4.tcp_congestion_control",
        values=["cubic", "bbr"],
        param_type="choice",
    ),
    # Egress queueing discipline. fq enables BBR pacing (so it
    # couples with tcp_congestion_control); fq_codel actively fights
    # bufferbloat, which shows up as lower fortio p99 under load.
    # pfifo_fast is the legacy FIFO baseline.
    SysctlParam(
        name="net.core.default_qdisc",
        values=["pfifo_fast", "fq", "fq_codel"],
        param_type="choice",
    ),
    # Path-MTU probing. Flat under Talos Docker (shared bridge,
    # uniform MTU). Live on production clusters running VXLAN
    # (Flannel, Calico-VXLAN), GENEVE (Antrea, OVN), or WireGuard
    # overlay CNIs across nodes with mismatched MTUs or firewalls
    # that drop ICMP Frag Needed. When it matters, it cuts
    # tcp_retransmit_rate and latency spikes from fragmentation loss.
    # Benchmark-flat / production-live: kept despite flat response
    # so the production default (1) is anchored via the seeded prior
    # and the dimension stays available for cluster-specific tuning.
    SysctlParam(
        name="net.ipv4.tcp_mtu_probing",
        values=[0, 1],
        param_type="choice",
    ),
    # Explicit Congestion Notification. 0 disabled; 1 accept-and-
    # request (both directions, so iperf3 / fortio clients will
    # negotiate ECN on outbound connections). Rung 2 (responder-only)
    # is omitted because the clients initiate every flow in this
    # harness, making it equivalent to 0. On fabrics that honour ECN
    # marks (DCTCP, BBRv2+, most modern DC switches), 1 substitutes
    # marks for drops and lowers tcp_retransmit_rate without hurting
    # throughput; on fabrics that don't (Talos Docker, most
    # internet-facing paths), it's a no-op. Benchmark-flat under the
    # default backend / production-live for DC deployments: kept.
    SysctlParam(
        name="net.ipv4.tcp_ecn",
        values=[0, 1],
        param_type="choice",
    ),
    # Disable the idle-to-slow-start reset. When 0 (kernel default),
    # a connection that has been idle longer than one RTO gets its
    # cwnd clamped back to the initial window on the next send,
    # spiking p99 latency on the post-idle request. Flat at the
    # default fortio shape (1000 QPS -> ~1 ms gaps, well below any
    # RTO), live on low-QPS 1G clusters running sparse API traffic
    # where inter-request gaps exceed the RTO. Benchmark-flat /
    # production-live: kept for 1G deployment coverage.
    SysctlParam(
        name="net.ipv4.tcp_slow_start_after_idle",
        values=[0, 1],
        param_type="choice",
    ),
    # Kernel-side coalescing of small write()/send() calls. Default
    # 1. Flat for apps that issue full-response writes (Go net/http,
    # fortio, iperf3); live for apps doing many small writes per
    # response (Python WSGI stacks, legacy C servers) where 0 can
    # cut p99 at the cost of PPS efficiency. Benchmark-flat /
    # production-live for a subset of apps: kept.
    SysctlParam(
        name="net.ipv4.tcp_autocorking",
        values=[0, 1],
        param_type="choice",
    ),
    # Per-socket cap on bytes queued in the qdisc/TX ring. Low values
    # cut bufferbloat and improve p99 latency; high values give the
    # send path more runway for throughput at the cost of tail
    # latency. Direct lever on the throughput vs. p99 trade-off.
    SysctlParam(
        name="net.ipv4.tcp_limit_output_bytes",
        values=[262144, 4194304],
        param_type="int",
    ),
]

_NAPI_PARAMS: list[SysctlParam] = [
    # Per-CPU input queue length before drops under RX bursts. Low
    # rungs cause packet drops on bursty workloads (visible as
    # tcp_retransmit_rate spikes); very high rungs add queuing latency
    # once the system is already overloaded.
    SysctlParam(
        name="net.core.netdev_max_backlog",
        values=[1000, 5000, 30000, 250000],
        param_type="int",
    ),
    # Max packets per NAPI poll cycle. Higher = better throughput on
    # bulk traffic (iperf3), potentially at the cost of softirq
    # fairness and higher cpu under the bulk sub-stage.
    SysctlParam(
        name="net.core.netdev_budget",
        values=[300, 600, 1000, 2000],
        param_type="int",
    ),
    # Packets GRO coalesces before flushing up the stack. Larger
    # batches cut per-packet overhead (cpu, throughput) on bulk
    # flows; small batches reduce in-stack latency for small
    # request/response traffic.
    SysctlParam(
        name="net.core.gro_normal_batch",
        values=[8, 16, 32],
        param_type="int",
    ),
]

_MEMORY_PARAMS: list[SysctlParam] = [
    # Floor on free pages the kernel keeps reserved. Too low and
    # allocations stall under pressure (packet drops, p99 spikes,
    # tcp_retransmit_rate); too high wastes headroom and inflates
    # node memory.
    SysctlParam(
        name="vm.min_free_kbytes",
        values=[65536, 131072, 262144],
        param_type="int",
    ),
]

_CONNECTION_PARAMS: list[SysctlParam] = [
    # listen() accept queue cap. Under-sized values drop SYNs during
    # fortio saturation, showing up as rps regressions and
    # connection-establishment latency spikes.
    SysctlParam(
        name="net.core.somaxconn",
        values=[128, 4096, 16384, 65535],
        param_type="int",
    ),
    # Per-listener SYN queue. Complements somaxconn earlier in the
    # handshake; under-sized values cause SYN drops under heavy
    # new-connection load.
    SysctlParam(
        name="net.ipv4.tcp_max_syn_backlog",
        values=[1024, 4096, 65536],
        param_type="int",
    ),
    # Allows outbound reuse of TIME_WAIT sockets. Mitigates ephemeral
    # port exhaustion under fortio connection churn; couples with
    # ip_local_port_range and tcp_fin_timeout.
    SysctlParam(
        name="net.ipv4.tcp_tw_reuse",
        values=[0, 1],
        param_type="choice",
    ),
    # FIN-WAIT-2 timeout. Shorter values reclaim sockets faster under
    # high connection-establishment churn (fortio saturation), at the
    # cost of occasional late-packet rejection.
    SysctlParam(
        name="net.ipv4.tcp_fin_timeout",
        values=[15, 60],
        param_type="int",
    ),
    # Ceiling on concurrent TIME_WAIT entries. Overflow triggers the
    # `TCP: time wait bucket table overflow` dmesg warning and forced
    # eviction of the oldest TW entry. fortio fixed_qps p99 (the only
    # sub-stage feeding latency percentiles) can move if saturation
    # churn pushes the table into eviction and the residual pressure
    # carries into the following fixed_qps stage. At fortio.connections=4
    # and the current default shape the lowest rung is not reachable;
    # this dimension only bites if connection churn is scaled up
    # (e.g. envoy egress at 10k+ new connections/s). Benchmark-flat
    # at current shape / production-live on busy clusters: kept.
    SysctlParam(
        name="net.ipv4.tcp_max_tw_buckets",
        values=[65536, 262144, 1048576],
        param_type="int",
    ),
    # Cap on unsent bytes in the TCP send queue. The kernel default
    # is UINT_MAX (4294967295), i.e. effectively unlimited; the
    # sysctl handler is proc_douintvec over an unsigned int, so
    # negative writes are rejected at apply time. Smaller positive
    # rungs cut head-of-line blocking and bufferbloat inside the
    # host stack, lowering fortio fixed_qps p99 without usually
    # hurting iperf3 throughput.
    SysctlParam(
        name="net.ipv4.tcp_notsent_lowat",
        values=[131072, 262144, 4294967295],
        param_type="int",
    ),
    # Ephemeral port pool size. With benchmark.parallel iperf3
    # streams plus fortio saturation from the same source pod,
    # narrow ranges exhaust ports and cap rps. Expressed as a
    # "low high" integer tuple since sysctl -w reads two integers.
    SysctlParam(
        name="net.ipv4.ip_local_port_range",
        values=["32768 60999", "15000 65535", "1024 65535"],
        param_type="choice",
    ),
]

_UDP_PARAMS: list[SysctlParam] = [
    # Per-socket receive-buffer floor for UDP. Too small and bursts
    # overflow the socket queue, producing datagram loss (visible as
    # udp_loss_rate / udp_jitter spikes).
    SysctlParam(
        name="net.ipv4.udp_rmem_min",
        values=[4096, 65536],
        param_type="int",
    ),
    # Global UDP memory pressure thresholds (pages). Too small and
    # the kernel prunes under load; too large and node memory climbs.
    SysctlParam(
        name="net.ipv4.udp_mem",
        values=["393216 524288 786432", "786432 1048576 1572864"],
        param_type="choice",
    ),
]

_CONNTRACK_PARAMS: list[SysctlParam] = [
    # Conntrack table capacity. On nodes using iptables/ipvs
    # kube-proxy or a netfilter-based CNI, every new flow through a
    # Service IP or SNAT consumes an entry. Overflow caps rps on the
    # saturation sub-stage; residual conntrack pressure carried into
    # the following fixed_qps sub-stage can also move p99. Rungs are
    # kept narrow because SysctlParam has no way to couple a paired
    # buckets write, so we rely on the node's boot-time buckets value
    # (typically ~max/4) and avoid max values that would blow the
    # chain-depth ratio past ~8x. At the default fortio shape this
    # dimension mostly matters for production guidance rather than
    # moving trial metrics -- overloaded conntrack is one of the most
    # common real-world K8s networking incidents. Note: Cilium in
    # kube-proxy-replacement (eBPF) mode bypasses netfilter conntrack
    # for service traffic, so this is a no-op for that subset; still
    # live for pod-to-external and pod-to-pod non-service flows.
    # Benchmark-flat / production-live: kept.
    SysctlParam(
        name="net.netfilter.nf_conntrack_max",
        values=[131072, 262144, 1048576],
        param_type="int",
    ),
    # How long ESTABLISHED entries persist when idle (default 432000s
    # / 5 days). Shorter values reduce conntrack table pressure
    # (helps nf_conntrack_max headroom) at the cost of reaping
    # idle-but-alive flows sooner. Benchmark cannot observe the
    # timeout firing (6-min trial vs 5-day default); live on
    # long-uptime production clusters where dead entries accumulate
    # from short-lived HTTP connections, cronjobs, probe traffic.
    # Benchmark-flat / production-live: kept.
    SysctlParam(
        name="net.netfilter.nf_conntrack_tcp_timeout_established",
        values=[600, 3600, 86400, 432000],
        param_type="int",
    ),
    # TIME_WAIT tail inside conntrack. Shorter values free table slots
    # faster under fortio connection churn; when the saturation
    # sub-stage leaves lingering conntrack pressure, the effect
    # surfaces on the following fixed_qps sub-stage's p99 (latency is
    # never sampled during saturation). Too short risks conntrack
    # losing track of strays from the same 5-tuple on lossy paths,
    # though intra-cluster loss makes that caveat mostly theoretical.
    SysctlParam(
        name="net.netfilter.nf_conntrack_tcp_timeout_time_wait",
        values=[30, 60, 120, 300],
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
            + _MEMORY_PARAMS
            + _CONNECTION_PARAMS
            + _UDP_PARAMS
            + _CONNTRACK_PARAMS
        )
    )
    _validate_params(effective)
    return ParamSpace(params=effective)


PARAM_SPACE: ParamSpace = build_param_space()

PARAM_CATEGORIES: dict[str, list[str]] = {
    "tcp_buffer": [p.name for p in _TCP_BUFFER_PARAMS],
    "congestion": [p.name for p in _CONGESTION_PARAMS],
    "napi": [p.name for p in _NAPI_PARAMS],
    "memory": [p.name for p in _MEMORY_PARAMS],
    "connection": [p.name for p in _CONNECTION_PARAMS],
    "udp": [p.name for p in _UDP_PARAMS],
    "conntrack": [p.name for p in _CONNTRACK_PARAMS],
}

PARAM_TO_CATEGORY: dict[str, str] = {
    param: cat for cat, params in PARAM_CATEGORIES.items() for param in params
}

# Production-reasonable defaults covering every knob in the search
# space. Seeded into the optimizer via ``_seed_prior_trials`` so the
# GP has a concrete known-good anchor on benchmark-flat dimensions
# (tcp_mtu_probing, tcp_ecn, nf_conntrack_*, tcp_max_tw_buckets,
# tcp_slow_start_after_idle, tcp_autocorking). Without this anchor
# the optimizer recommends Sobol-random values on flat axes.
#
# Every value here must match one of the rungs declared above --
# Ax rejects seeded points outside the choice set.
#
# Targeted at 10G intra-DC as the "most common" deployment; 1G edge
# nodes survive these values, 100G long-BDP clusters will want a
# follow-up expert profile once the search space extends past 64 MB
# buffer ceilings.
RECOMMENDED_DEFAULTS: dict[str, str | int] = {
    # tcp_buffer
    "net.core.rmem_max": 67108864,
    "net.core.wmem_max": 67108864,
    "net.ipv4.tcp_rmem": "4096 87380 33554432",
    "net.ipv4.tcp_wmem": "4096 65536 33554432",
    "net.ipv4.tcp_mem": "786432 1048576 1572864",
    # congestion
    "net.ipv4.tcp_congestion_control": "bbr",
    "net.core.default_qdisc": "fq",
    "net.ipv4.tcp_mtu_probing": 1,
    "net.ipv4.tcp_ecn": 1,
    "net.ipv4.tcp_slow_start_after_idle": 0,
    "net.ipv4.tcp_autocorking": 1,
    "net.ipv4.tcp_limit_output_bytes": 262144,
    # napi
    "net.core.netdev_max_backlog": 5000,
    "net.core.netdev_budget": 600,
    "net.core.gro_normal_batch": 8,
    # memory
    "vm.min_free_kbytes": 131072,
    # connection
    "net.core.somaxconn": 4096,
    "net.ipv4.tcp_max_syn_backlog": 4096,
    "net.ipv4.tcp_tw_reuse": 1,
    "net.ipv4.tcp_fin_timeout": 60,
    "net.ipv4.tcp_max_tw_buckets": 262144,
    "net.ipv4.tcp_notsent_lowat": 4294967295,
    "net.ipv4.ip_local_port_range": "15000 65535",
    # udp
    "net.ipv4.udp_rmem_min": 65536,
    "net.ipv4.udp_mem": "786432 1048576 1572864",
    # conntrack
    "net.netfilter.nf_conntrack_max": 262144,
    "net.netfilter.nf_conntrack_tcp_timeout_established": 86400,
    "net.netfilter.nf_conntrack_tcp_timeout_time_wait": 120,
}


def _validate_recommended_defaults() -> None:
    """Check :data:`RECOMMENDED_DEFAULTS` covers the space and hits rungs.

    The dict is seeded into Ax as a prior trial; Ax rejects a seeded
    point whose value is not in the param's choice set, and silently
    omits a knob missing from the seed. This catches both at import.

    Raises:
        ValueError: If any knob is missing from ``RECOMMENDED_DEFAULTS``
            or carries a value not among its declared rungs.
    """
    names = {p.name for p in PARAM_SPACE.params}
    missing = names - RECOMMENDED_DEFAULTS.keys()
    extra = RECOMMENDED_DEFAULTS.keys() - names
    if missing or extra:
        msg = (
            f"RECOMMENDED_DEFAULTS must cover every PARAM_SPACE knob: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
        raise ValueError(msg)
    for p in PARAM_SPACE.params:
        value = RECOMMENDED_DEFAULTS[p.name]
        if value not in p.values:
            msg = (
                f"RECOMMENDED_DEFAULTS[{p.name!r}] = {value!r} is not in "
                f"the declared rung set {p.values!r}"
            )
            raise ValueError(msg)


_validate_recommended_defaults()
