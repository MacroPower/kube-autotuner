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

from kube_autotuner.models import MemoryCost, ParamSpace, SysctlParam

# Full search space, organised by category.

_TCP_BUFFER_PARAMS: list[SysctlParam] = [
    # System-wide SO_RCVBUF ceiling. Low rungs cap the TCP receive
    # window on fat pipes and throttle throughput on long-BDP paths;
    # high rungs trade node memory headroom under many flows. Top rung
    # 128 MiB covers 100G long-BDP paths (BDP at 100 Gbps with ~10 ms
    # RTT exceeds 64 MiB). Memory cost == rung value (per-socket
    # SO_RCVBUF ceiling).
    SysctlParam(
        name="net.core.rmem_max",
        values=[212992, 67108864, 134217728],
        param_type="int",
        memory_cost=MemoryCost(kind="identity"),
    ),
    # System-wide SO_SNDBUF ceiling. Mirror of rmem_max on the send
    # path: low rungs cap the send window and throughput, high rungs
    # raise memory pressure. Top rung 128 MiB covers 100G long-BDP.
    # Memory cost == rung value (per-socket).
    SysctlParam(
        name="net.core.wmem_max",
        values=[212992, 67108864, 134217728],
        param_type="int",
        memory_cost=MemoryCost(kind="identity"),
    ),
    # Default SO_RCVBUF when an app does not call setsockopt. Raising
    # the default lifts the receive window for every socket on the
    # node, including infrastructure traffic that never opts in via
    # setsockopt. Distinct from rmem_max (the cap an app can opt up to)
    # and from tcp_rmem's middle field (the autotuning starting point
    # for TCP specifically). Memory cost == rung value (per-socket
    # SO_RCVBUF default).
    SysctlParam(
        name="net.core.rmem_default",
        values=[212992, 16777216, 33554432],
        param_type="int",
        memory_cost=MemoryCost(kind="identity"),
    ),
    # Default SO_SNDBUF when an app does not call setsockopt. Mirror
    # of rmem_default on the send path. Memory cost == rung value
    # (per-socket SO_SNDBUF default).
    SysctlParam(
        name="net.core.wmem_default",
        values=[212992, 16777216, 33554432],
        param_type="int",
        memory_cost=MemoryCost(kind="identity"),
    ),
    # Autotuning triple (min default max) for TCP receive buffers.
    # Governs the peak receive window. Under-sized max caps iperf3
    # throughput on high-BDP links and can raise tcp_retransmit_rate
    # when the window collapses; over-sized max raises node memory
    # under many concurrent flows. Top triple's 64 MiB max-field gives
    # autotuning enough headroom on 100G long-BDP paths. Memory cost
    # == max field of the triple.
    SysctlParam(
        name="net.ipv4.tcp_rmem",
        values=[
            "4096 131072 6291456",
            "4096 131072 33554432",
            "4096 131072 67108864",
        ],
        param_type="choice",
        memory_cost=MemoryCost(kind="triple_max"),
    ),
    # Autotuning triple for TCP send buffers. Same trade-off as
    # tcp_rmem on the send path: throughput vs. memory footprint. Top
    # triple's 64 MiB max-field covers 100G long-BDP. Memory cost ==
    # max field of the triple.
    SysctlParam(
        name="net.ipv4.tcp_wmem",
        values=[
            "4096 16384 4194304",
            "4096 65536 33554432",
            "4096 65536 67108864",
        ],
        param_type="choice",
        memory_cost=MemoryCost(kind="triple_max"),
    ),
    # Global TCP memory pressure thresholds in pages (min pressure max).
    # Too small and the kernel prunes/throttles under load (throughput
    # drops, tcp_retransmit_rate rises); too large and node memory climbs.
    # Represented as the canonical three-integer string form. Triples
    # whose max-field crosses ~100 GiB of pages are reachable only
    # through an `optimize.paramSpace` override -- including them in
    # the canonical rung set would inflate triple_max_pages cost by
    # an order of magnitude over the next rung and dominate
    # recommendation ranking on every run. Memory cost == max-field
    # pages x 4096 bytes/page.
    SysctlParam(
        name="net.ipv4.tcp_mem",
        values=[
            "196608 262144 393216",
            "393216 524288 786432",
            "786432 1048576 1572864",
        ],
        param_type="choice",
        memory_cost=MemoryCost(kind="triple_max_pages"),
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
    # Explicit Congestion Notification. 0 disables it; 1 accepts and
    # requests ECN in both directions, so iperf3 / fortio clients
    # negotiate ECN on outbound connections; 2 is accept-only and has
    # been the kernel default since Linux 4.1 (the host honours ECN
    # marks on incoming flows but does not request ECN on outgoing
    # ones). The benchmark clients initiate every flow, so rung 2
    # behaves like 0 under this harness; the seed therefore lands on
    # the kernel default. Rung 1 stays in the search space for
    # fabrics that honour ECN marks (DCTCP, BBRv2+, most modern DC
    # switches), where 1 substitutes marks for drops and lowers
    # tcp_retransmit_rate without hurting throughput. On fabrics
    # that don't (Talos Docker, most internet-facing paths), it's a
    # no-op. Benchmark-flat under the default backend, production-
    # live for DC deployments, so we keep it.
    SysctlParam(
        name="net.ipv4.tcp_ecn",
        values=[0, 1, 2],
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
        values=[1048576, 4194304],
        param_type="int",
    ),
    # Pacing rate multiplier during slow start, percentage of the
    # cwnd-derived rate. Kernel default 200 (pace at 2x to allow
    # ramp-up). Only takes effect when the egress qdisc supports
    # pacing (fq / fq_codel) -- couples directly with
    # net.core.default_qdisc. The 300 rung accelerates ramp-up on
    # clean fat pipes at the cost of more aggressive bursts that can
    # raise tcp_retransmit_rate on policer-shaped egress paths.
    SysctlParam(
        name="net.ipv4.tcp_pacing_ss_ratio",
        values=[100, 200, 300],
        param_type="int",
    ),
    # Pacing rate multiplier during congestion avoidance, percentage
    # form. Kernel default 120 (pace 20% above the cwnd-derived rate
    # to leave probing headroom). Same qdisc coupling as
    # tcp_pacing_ss_ratio. The 150 and 200 rungs probe harder on
    # clean fat pipes for higher throughput at the cost of policer
    # drops on shaped egress paths.
    SysctlParam(
        name="net.ipv4.tcp_pacing_ca_ratio",
        values=[120, 150, 200],
        param_type="int",
    ),
    # RFC 7323 PAWS / RTT-measurement timestamps. Kernel default 1
    # (enabled). Setting this to 0 only makes sense behind a middlebox
    # (L4 load balancer, some NATs) that strips TCP timestamps from
    # inbound packets -- in that case PAWS protection is already
    # broken on the inbound path and the local cost of leaving
    # timestamps on is wasted bytes per segment plus useless RTT
    # measurement. Pair with `tcp_tw_reuse=2` when timestamps are
    # stripped, since that combination lets a stripping middlebox
    # still reuse TIME_WAIT sockets safely. This is a binary 0/1
    # sysctl in mainline kernels; do not add a `2` rung.
    SysctlParam(
        name="net.ipv4.tcp_timestamps",
        values=[0, 1],
        param_type="choice",
    ),
    # Forward-RTO recovery (F-RTO) behaviour. Kernel default 2 (the
    # SACK-enhanced variant from RFC 5682 -- distinguishes spurious
    # retransmissions from genuine loss when the original ACK ordering
    # implies a delay rather than a drop). 0 disables F-RTO entirely;
    # 1 is the basic non-SACK variant. Wired DC paths see
    # spurious-retx detection adding latency without payoff (0 is the
    # right pick); the kernel default 2 wins on lossy/jittery paths
    # (Wi-Fi, mobile, long-haul). Benchmark-flat for clean Talos
    # Docker; live for policy tuning.
    SysctlParam(
        name="net.ipv4.tcp_frto",
        values=[0, 1, 2],
        param_type="choice",
    ),
]

_NAPI_PARAMS: list[SysctlParam] = [
    # Per-CPU input queue length before drops under RX bursts. Low
    # rungs cause packet drops on bursty workloads (visible as
    # tcp_retransmit_rate spikes); very high rungs add queuing latency
    # once the system is already overloaded. Memory cost: entries x
    # 256 bytes (sk_buff head ~232 B on x86_64 plus per-slot pointer
    # slack).
    SysctlParam(
        name="net.core.netdev_max_backlog",
        values=[1000, 32768, 250000],
        param_type="int",
        memory_cost=MemoryCost(kind="per_entry", per_entry_bytes=256),
    ),
    # Max packets per NAPI poll cycle. Higher = better throughput on
    # bulk traffic (iperf3), potentially at the cost of softirq
    # fairness and higher cpu under the bulk sub-stage.
    SysctlParam(
        name="net.core.netdev_budget",
        values=[300, 1000, 2000],
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
    # RX busy-poll budget in microseconds. Kernel default 0 (disabled).
    # Latency-sensitive workloads can spin on the NIC in poll mode
    # for up to this many microseconds before falling back to the
    # softirq path, cutting wakeup latency on the hot path.
    # SO_BUSY_POLL caveat: the effect requires every participating
    # socket to opt in via setsockopt(SO_BUSY_POLL, ...); apps that
    # don't enable it pay no cost on this dimension but also see no
    # benefit, so the kernel default 0 is the seeded prior and the
    # optimizer reaches non-zero rungs only when an opted-in app
    # shows a measurable win.
    SysctlParam(
        name="net.core.busy_poll",
        values=[0, 50, 100],
        param_type="int",
    ),
    # System-wide default for the per-socket SO_BUSY_POLL budget,
    # paired with net.core.busy_poll on the read path. Same
    # SO_BUSY_POLL caveat applies: effective only for sockets that
    # opt in via setsockopt; default 0 is a no-op for everyone else.
    SysctlParam(
        name="net.core.busy_read",
        values=[0, 50, 100],
        param_type="int",
    ),
    # Per-socket ancillary buffer ceiling (control messages and
    # auxiliary sk_buffs). Kernel default 20480 bytes. Apps that
    # heavily use SCM_RIGHTS, IP_PKTINFO, or large cmsg payloads
    # exhaust the default and start dropping packets at the socket
    # boundary. The 1 MiB rung covers DPDK-style high-throughput
    # envelopes. Memory cost == rung value (per-socket ancillary
    # buffer cap).
    SysctlParam(
        name="net.core.optmem_max",
        values=[65535, 131072, 1048576],
        param_type="int",
        memory_cost=MemoryCost(kind="identity"),
    ),
]

_MEMORY_PARAMS: list[SysctlParam] = [
    # Floor on free pages the kernel keeps reserved. Too low and
    # allocations stall under pressure (packet drops, p99 spikes,
    # tcp_retransmit_rate); too high wastes headroom and inflates
    # node memory. Memory cost == rung x 1024 bytes/KiB.
    SysctlParam(
        name="vm.min_free_kbytes",
        values=[16384, 131072, 262144],
        param_type="int",
        memory_cost=MemoryCost(kind="kib"),
    ),
]

_CONNECTION_PARAMS: list[SysctlParam] = [
    # listen() accept queue cap. Under-sized values drop SYNs during
    # fortio saturation, showing up as rps regressions and
    # connection-establishment latency spikes. Memory cost: entries
    # x 256 bytes (same sk_buff-head upper bound used by the backlog
    # params; request_sock is slightly smaller but rounding up is
    # safe).
    SysctlParam(
        name="net.core.somaxconn",
        values=[4096, 32768, 65535],
        param_type="int",
        memory_cost=MemoryCost(kind="per_entry", per_entry_bytes=256),
    ),
    # Per-listener SYN queue. Complements somaxconn earlier in the
    # handshake; under-sized values cause SYN drops under heavy
    # new-connection load. Memory cost: entries x 256 bytes.
    SysctlParam(
        name="net.ipv4.tcp_max_syn_backlog",
        values=[1024, 4096, 65536],
        param_type="int",
        memory_cost=MemoryCost(kind="per_entry", per_entry_bytes=256),
    ),
    # Allows outbound reuse of TIME_WAIT sockets. Mitigates ephemeral
    # port exhaustion under fortio connection churn; couples with
    # ip_local_port_range and tcp_fin_timeout. 0 disables reuse, 1
    # enables it for all destinations, and 2 (the kernel default
    # since Linux 4.12) restricts reuse to loopback so the host-local
    # case stays fast while non-loopback flows keep TIME_WAIT
    # semantics.
    SysctlParam(
        name="net.ipv4.tcp_tw_reuse",
        values=[0, 1, 2],
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
    # Memory cost: entries x 144 bytes (tcp_timewait_sock size).
    SysctlParam(
        name="net.ipv4.tcp_max_tw_buckets",
        values=[65536, 262144, 1048576],
        param_type="int",
        memory_cost=MemoryCost(kind="per_entry", per_entry_bytes=144),
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
    # Ephemeral port pool size. With iperf.parallel iperf3
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
        values=[4096, 16384, 65536],
        param_type="int",
    ),
    # Per-socket send-buffer floor for UDP. Mirror of udp_rmem_min on
    # the send path: too small and bursty senders see EAGAIN /
    # head-of-line stalls in the socket queue. A symmetric envelope
    # with udp_rmem_min is the usual pairing.
    SysctlParam(
        name="net.ipv4.udp_wmem_min",
        values=[4096, 16384, 65536],
        param_type="int",
    ),
    # Global UDP memory pressure thresholds (pages). Too small and
    # the kernel prunes under load; too large and node memory climbs.
    # As with tcp_mem, triples whose max-field crosses ~100 GiB of
    # pages are reachable only through an `optimize.paramSpace`
    # override -- the canonical rung set deliberately omits them so
    # triple_max_pages cost stays balanced. Memory cost == max-field
    # pages x 4096 bytes/page.
    SysctlParam(
        name="net.ipv4.udp_mem",
        values=["393216 524288 786432", "786432 1048576 1572864"],
        param_type="choice",
        memory_cost=MemoryCost(kind="triple_max_pages"),
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
    # Memory cost: entries x 320 bytes (nf_conn is ~304 bytes on
    # modern kernels; round up for hash-table slot + overhead).
    SysctlParam(
        name="net.netfilter.nf_conntrack_max",
        values=[131072, 262144, 1048576],
        param_type="int",
        memory_cost=MemoryCost(kind="per_entry", per_entry_bytes=320),
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
        values=[3600, 86400, 432000],
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
        values=[60, 120, 240],
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

# Stock Ubuntu 24.04 sysctl defaults covering every knob in the search
# space. ``_attach_recommended_defaults`` seeds these into the
# optimizer so the GP has a concrete observation on benchmark-flat
# dimensions (tcp_mtu_probing, tcp_ecn, nf_conntrack_*,
# tcp_max_tw_buckets, tcp_slow_start_after_idle, tcp_autocorking).
# Without the anchor, Sobol picks random rungs on those axes and the
# GP regresses noise.
#
# Sourced from Linux 6.8 (the kernel Ubuntu 24.04 GA ships) plus
# systemd 255's ``/usr/lib/sysctl.d/50-default.conf``, which overrides
# the kernel-literal ``default_qdisc`` to ``fq_codel``. RAM-dependent
# kernel defaults (``tcp_mem``, ``udp_mem``, ``vm.min_free_kbytes``,
# the ``tcp_rmem``/``tcp_wmem`` max fields, ``tcp_max_tw_buckets``,
# ``tcp_max_syn_backlog``, ``nf_conntrack_max``) are evaluated for a
# 16 GB node with ``nr_free_buffer_pages == totalram_pages``. Real
# installs may report values a few percent lower (lowmem reserves) or
# off by a power-of-two for hash-table-derived knobs; the dict is the
# formula sample, not a measured snapshot. That way
# ``baseline_comparison()`` in ``report/analysis.py`` reports uplift
# over the no-tuning reference point a user would have if they
# touched nothing.
#
# Every value here must match one of the rungs declared above. Ax
# rejects seeded points outside the choice set, and the
# ``_validate_recommended_defaults()`` drift guard catches both
# missing knobs and off-rung values at import time.
#
# Extreme ``tcp_mem`` / ``udp_mem`` page-count triples (max-fields
# above ~100 GiB) stay out of the canonical rung set on purpose.
# Their magnitude would inflate the ``triple_max_pages`` cost term
# by an order of magnitude over the next-highest rung and dominate
# recommendation ranking on every run. Users who need those literal
# page counts can override ``optimize.paramSpace`` (see ``examples/``).
RECOMMENDED_DEFAULTS: dict[str, str | int] = {
    # tcp_buffer
    "net.core.rmem_max": 212992,
    "net.core.wmem_max": 212992,
    "net.core.rmem_default": 212992,
    "net.core.wmem_default": 212992,
    "net.ipv4.tcp_rmem": "4096 131072 6291456",
    "net.ipv4.tcp_wmem": "4096 16384 4194304",
    "net.ipv4.tcp_mem": "196608 262144 393216",
    # congestion
    "net.ipv4.tcp_congestion_control": "cubic",
    "net.core.default_qdisc": "fq_codel",
    "net.ipv4.tcp_mtu_probing": 0,
    "net.ipv4.tcp_ecn": 2,
    "net.ipv4.tcp_slow_start_after_idle": 1,
    "net.ipv4.tcp_autocorking": 1,
    "net.ipv4.tcp_limit_output_bytes": 1048576,
    "net.ipv4.tcp_pacing_ss_ratio": 200,
    "net.ipv4.tcp_pacing_ca_ratio": 120,
    "net.ipv4.tcp_timestamps": 1,
    "net.ipv4.tcp_frto": 2,
    # napi
    "net.core.netdev_max_backlog": 1000,
    "net.core.netdev_budget": 300,
    "net.core.gro_normal_batch": 8,
    "net.core.busy_poll": 0,
    "net.core.busy_read": 0,
    "net.core.optmem_max": 131072,
    # memory
    "vm.min_free_kbytes": 16384,
    # connection
    "net.core.somaxconn": 4096,
    "net.ipv4.tcp_max_syn_backlog": 4096,
    "net.ipv4.tcp_tw_reuse": 2,
    "net.ipv4.tcp_fin_timeout": 60,
    "net.ipv4.tcp_max_tw_buckets": 262144,
    "net.ipv4.tcp_notsent_lowat": 4294967295,
    "net.ipv4.ip_local_port_range": "32768 60999",
    # udp
    "net.ipv4.udp_rmem_min": 4096,
    "net.ipv4.udp_wmem_min": 4096,
    "net.ipv4.udp_mem": "393216 524288 786432",
    # conntrack
    "net.netfilter.nf_conntrack_max": 262144,
    "net.netfilter.nf_conntrack_tcp_timeout_established": 432000,
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
