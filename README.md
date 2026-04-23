# kube-autotuner

Benchmark-driven kernel parameter tuning for Kubernetes nodes.

## Overview

`kube-autotuner` searches for Linux sysctl values that maximise network
performance on the nodes of a Kubernetes cluster. Each trial measures
the target node along two axes: iperf3 for bulk throughput, and fortio
for request/response latency and achieved RPS. An Ax-platform
multi-objective Bayesian optimizer proposes the next configuration,
the tool applies it, and the benchmark runs again. The loop converges
on a Pareto-optimal set of trade-offs between throughput, TCP
retransmit rate, UDP jitter, RPS under saturation, and p50/p90/p99
latency under a fixed offered load.

## Why this is hard

- **Expensive trials.** A single iteration expands into four
  sequential sub-stages: an iperf3 TCP bandwidth fan-out
  (`duration=30s` + `omit=5s` warmup), an iperf3 UDP bandwidth
  fan-out (same budget, source of `udp_jitter` (stored in seconds),
  UDP throughput, and UDP packet-loss rate; also the residual pressure
  that exercises the UDP-tuning dimensions), a fortio saturation run
  (`-qps 0`, `fortio.duration=30s`), and a fortio fixed-QPS run at
  `fortio.fixedQps`. The sequence repeats `iterations` times before
  setup and teardown. A 50-trial run takes hours. Grid search over
  the default space is infeasible by orders of magnitude.
- **Noisy measurements.** iperf3 throughput has real run-to-run
  variance. The optimizer collapses each metric into a `(mean, SEM)`
  pair and accounts for the fact that samples are correlated when
  multiple clients share one server. The surrogate is fitting a noisy
  response, not a deterministic function; iteration counts exist partly
  to buy it confidence.
- **High-dimensional, discrete search.** The default space has 27
  parameters (see the next section) with well over 10^10 combinations
  under the rung convention. It is discrete and categorical, so the
  optimizer treats it as a mixed integer/categorical problem.
  Gradient-based acquisition still runs under the hood, but on a
  relaxed surrogate, not on the (non-differentiable) iperf3 objective
  itself.
- **Parameters interact.** The buffer family mixes types in the default
  space: `net.core.rmem_max` / `net.core.wmem_max` are `int` rungs that
  cap the window, while `net.ipv4.tcp_rmem` / `net.ipv4.tcp_wmem` are
  `choice` parameters carrying the three-tuple strings the kernel
  actually reads, and one bounds the other.
  `net.ipv4.tcp_congestion_control` (cubic/bbr) and
  `net.core.default_qdisc` (pfifo_fast/fq/fq_codel) couple on the egress
  path. A setting that helps in isolation can regress in combination.
  That is the kind of structure a GP surrogate can pick up.
- **Objectives conflict.** Throughput-maximising configurations often
  raise retransmit rate; configurations that win on raw Gbps frequently
  degrade tail latency under request/response workloads, so
  bandwidth-only tuning hides regressions. The tool returns a
  **Pareto front** rather than a single winner. Default outcome
  constraints (a throughput floor, a retransmit-rate ceiling, an RPS
  floor, jitter and latency ceilings) are forwarded to Ax as hard
  constraints; the exact values live in the `objectives` block shown
  under [Quick start](#quick-start).
- **Results are hardware- and topology-dependent.** The tool stratifies
  results by `hardwareClass` (a free-form label you choose, e.g.
  `graviton4` or `epyc-9454p`) and topology (intra-AZ / inter-AZ).
  Tunings found on one NIC class or AZ pair do not transfer, so the
  tool re-runs against the live cluster rather than shipping a static
  recommendation.

Ax fits this shape: sample-efficient Bayesian optimization with native
multi-objective support, Sobol warm-up (`optimize.nSobol`, default 15)
to explore, then a GP surrogate that absorbs the remaining trials and
trades off throughput, retransmit rate, jitter, RPS, and latency
percentiles in one pass.

## How it works

1. You write an `experiment.yaml`.
2. `kube-autotuner run` resolves the sysctl backend and runs preflight
   checks against the live cluster.
3. An iperf3 server Deployment and a fortio server Deployment land on
   the target node. Each iteration then drives four sub-stages
   sequentially: an iperf3 TCP bandwidth fan-out, an iperf3 UDP
   bandwidth fan-out (sole source of `jitter_ms`), a fortio
   saturation fan-out (RPS), and a fortio fixed-QPS fan-out
   (latency percentiles). Sub-stages run one at a time so fortio
   never contends with iperf3 for NIC, CPU, or CNI state; within
   each sub-stage the source nodes still fan out in parallel.
4. In `optimize` mode the Ax loop proposes trials, applies sysctls,
   benchmarks, and appends one JSONL record per trial.
5. Every run writes the resolved `objectives` block alongside the JSONL
   as `<output>.meta.json`, so `kube-autotuner analyze` picks up the
   same frontier and recommendation weights without re-specifying them
   and turns the JSONL into Pareto plots, parameter importance scores,
   and a ranked list of configurations.

## Parameter space

When `optimize.paramSpace` is omitted, the tool searches a canonical
default: 27 sysctls across seven categories. Two parameter types are
accepted: `int` (a numeric range) and `choice` (an explicit value list
of strings or ints). The canonical default quantises every integer
parameter to a handful of representative rungs rather than covering the
full `[min, max]` integer range; that's a convention of the default,
not a type-system invariant, so a user-supplied `paramSpace` is free to
hand Ax a wide integer range instead. Under the rung convention the
default space has well over 10^10 combinations; treating integer
parameters as full ranges (as the YAML doc comment
`values = [min, max]` implies for custom params) makes it larger still.
Either way, exhaustive search is not an option.

| Category           | Count | Examples                                                                               |
|--------------------|-------|----------------------------------------------------------------------------------------|
| TCP buffers        | 5     | `net.core.rmem_max`, `net.ipv4.tcp_rmem`, `net.ipv4.tcp_mem`                           |
| Congestion control | 6     | `net.ipv4.tcp_congestion_control`, `net.core.default_qdisc`                            |
| NAPI / softirq     | 3     | `net.core.netdev_budget`, `net.core.netdev_max_backlog`                                |
| VM / memory        | 1     | `vm.min_free_kbytes`                                                                   |
| Connection         | 7     | `net.core.somaxconn`, `net.ipv4.tcp_max_tw_buckets`, `net.ipv4.ip_local_port_range`    |
| UDP                | 2     | `net.ipv4.udp_rmem_min`, `net.ipv4.udp_mem`                                            |
| Conntrack          | 3     | `net.netfilter.nf_conntrack_max`, `net.netfilter.nf_conntrack_tcp_timeout_established` |

UDP-category params are always part of the default search space: every
iteration runs both a TCP and a UDP iperf3 bandwidth stage, so
`udp_throughput`, `udp_loss_rate`, and `udp_jitter` (stored in seconds)
-- plus the residual kernel pressure that these knobs control -- are
always observable. All three are first-class Pareto objectives by
default; see [Metric catalog](#metric-catalog). Conntrack tuning
assumes the `nf_conntrack` kernel module is loaded on the target node;
on nodes without `/proc/sys/net/netfilter` these writes will fail at
apply time.

Two shapes of customisation:

- Override the whole default set by supplying `optimize.paramSpace` in
  `experiment.yaml` (see the example under [Quick start](#quick-start)).
  Both `int` (range) and `choice` (explicit list) types are accepted;
  `int` params must have `min < max`.
- Use `mode: trial` with a fixed `sysctls:` map to benchmark one
  configuration without invoking the optimizer.

## Install

The package is not yet published to PyPI. Install from source:

```sh
pip install "git+https://github.com/<owner>/kube-autotuner.git"
# or, from a clone:
uv pip install -e .
```

Python >= 3.14 is required. For a managed uv + Nix dev environment, see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Quick start

A full `experiment.yaml` covering every field the loader accepts. Every
section except `mode` and `nodes` is optional; the defaults shown below
are what you get when you omit them.

```yaml
# mode selects the command branch: baseline | trial | optimize.
# `optimize:` is required when mode=optimize; `trial:` when mode=trial.
mode: optimize

nodes:
  sources: [nodeA]               # >=1 source node; first entry is primary
  target: nodeB
  hardwareClass: 10g             # free-form label; stratifies results
  namespace: default
  ipFamilyPolicy: RequireDualStack

benchmark:
  duration: 30                   # seconds of measurement per iteration
  omit: 5                        # warmup seconds discarded from stats
  iterations: 3
  parallel: 16                   # iperf3 -P streams per client
  window: null                   # iperf3 -w hint; e.g. "256K" to pin it

# Required when mode: optimize. Ax Bayesian loop knobs plus the search
# space. Omit paramSpace to use the built-in canonical sysctl set.
optimize:
  nTrials: 50
  nSobol: 15                     # quasi-random exploration trials; <= nTrials
  applySource: false             # also write best sysctls on source nodes
  paramSpace:
    - name: net.core.rmem_max
      paramType: int             # integer range: values = [min, max]
      values: [4194304, 67108864]
    - name: net.ipv4.tcp_congestion_control
      paramType: choice          # discrete set; values are strings or ints
      values: [cubic, bbr]

# Required when mode: trial. Apply a fixed sysctl set for one benchmark.
# trial:
#   sysctls:
#     net.core.rmem_max: 67108864
#     net.ipv4.tcp_congestion_control: bbr

# Extra iperf3 flags per role. Flags the tool itself owns (-t, -P, -w,
# -u, -J, --bind, etc.) are rejected at preflight.
iperf:
  client:
    extraArgs: ["--bidir", "-Z"]
  server:
    extraArgs: ["--forceflush"]

# Fortio drives the latency / RPS sub-stages. `duration` is independent
# of benchmark.duration so fortio runs stay short. `fixedQps` is the
# offered load for the latency sub-stage; the saturation sub-stage
# always runs with -qps 0. Flags the tool controls (-qps, -c, -t, -n,
# -json, -url, -H, -http1.0, -stdclient, -quiet on the client;
# -http-port on the server) are rejected at preflight.
fortio:
  fixedQps: 1000
  connections: 4
  duration: 30                    # seconds, per fortio sub-stage
  client:
    extraArgs: []
  server:
    extraArgs: []

# Kustomize patches layered onto the generated client/server manifests.
# `patch:` accepts a Strategic Merge Patch body (dict), a JSON6902 op
# list, or a pre-rendered patch string. Do not set target.namespace or
# metadata.namespace -- namespace is controlled by nodes.namespace.
patches:
  - target:
      kind: Job
      name: iperf3-client        # also: group, version, labelSelector, annotationSelector
    patch:
      spec:
        template:
          spec:
            containers:
              - name: iperf3-client
                resources:
                  limits:
                    memory: "2Gi"
  - target:
      kind: Deployment
    strict: false                # default true; false lets the patch no-op
    patch:
      - op: add
        path: /spec/template/spec/hostNetwork
        value: true

objectives:
  pareto:
    - { metric: tcp_throughput, direction: maximize }
    - { metric: udp_throughput, direction: maximize }
    - { metric: tcp_retransmit_rate, direction: minimize }
    - { metric: udp_loss_rate, direction: minimize }
    - { metric: udp_jitter, direction: minimize }
    - { metric: rps, direction: maximize }
    - { metric: latency_p50, direction: minimize }
    - { metric: latency_p90, direction: minimize }
    - { metric: latency_p99, direction: minimize }
  constraints:                       # k8s-style quantity suffixes accepted on the threshold
    - "tcp_throughput >= 1M"         # bits/sec; 1M (decimal mega) = 1e6
    - "udp_throughput >= 1M"
    - "tcp_retransmit_rate <= 1u"    # retransmits per byte sent; 1u (micro) ~ 1 retx/MB
    - "udp_loss_rate <= 0.05"        # 5% UDP packet loss cap; UDP loss naturally runs
                                     # higher than TCP retransmit rate.
    - "rps >= 100"                   # requests/sec; only the saturation sub-stage feeds
                                     # rps, so this floor only fails on fortio server crash.
    - "latency_p99 <= 1000m"         # seconds; 1000m (milli) = 1.0s ceiling from the
                                     # fixed_qps sub-stage only.
  recommendationWeights:
    tcp_retransmit_rate: 0.3
    udp_loss_rate: 0.3               # mirror tcp_retransmit_rate's weight; UDP loss pushes
                                     # back on the score with the same force TCP retransmit
                                     # rate does.
    udp_jitter: 0.1                  # UDP inter-arrival jitter (ms); modest weight -- a
                                     # stability signal, not a primary optimization target.
    latency_p90: 0.1                 # tail-latency weights keep the live Best-so-far panel
    latency_p99: 0.15                # and the post-hoc recommendation from over-indexing on
                                     # raw throughput; latency_p50 stays unweighted so the
                                     # mean-latency axis enters the Pareto set without
                                     # dominating the score.

output: out/results.jsonl
```

A few rules govern how the `objectives:` block is interpreted:

- Supplying `constraints:` or `recommendationWeights:` **replaces**
  the default list wholesale rather than extending it.
- Constraint thresholds accept the k8s quantity grammar: binary IEC
  (`Ki`, `Mi`, `Gi`, `Ti`, `Pi`, `Ei`), decimal SI (`n`, `u`, `m`, `k`,
  `M`, `G`, `T`, `P`, `E`), and decimal exponents (`1e6`, `1E-9`).
  Values are normalized to bare floats at load time, so a config
  written as `"throughput >= 1Gi"` is stored and forwarded to Ax as
  `"throughput >= 1073741824"`.
- Weights are only valid on minimize-direction metrics and must
  reference a metric present in `pareto`.
- Every iteration runs both iperf3 bandwidth stages (TCP then UDP)
  and both fortio sub-stages, so every metric below is always
  observable. Sidecar metadata files (`<results.jsonl>.meta.json`)
  from earlier versions that set `benchmark.modes: [tcp]` still
  load: the `modes:` key is silently dropped at parse time, so a
  resume will rerun against the new always-both semantics.

## Metric catalog

Valid `pareto.metric` values and their sources:

| Metric                | Direction | Source sub-stage         | Notes                                                                                                  |
|-----------------------|-----------|--------------------------|--------------------------------------------------------------------------------------------------------|
| `tcp_throughput`      | maximize  | iperf3 bw-tcp            | Bits per second. Summed across source clients per iteration, averaged across iterations.              |
| `udp_throughput`      | maximize  | iperf3 bw-udp            | Bits per second. Same aggregation as `tcp_throughput`.                                                |
| `tcp_retransmit_rate` | minimize  | iperf3 bw-tcp            | Retransmits per byte sent; scale-invariant, so high-throughput/high-loss does not win on raw count.   |
| `udp_loss_rate`       | minimize  | iperf3 bw-udp            | Lost packets per packet sent; per-iteration ratio-of-sums then averaged. UDP analog of `tcp_retransmit_rate`. |
| `udp_jitter`          | minimize  | iperf3 bw-udp            | Seconds (stored); displayed as milliseconds. Mean UDP inter-arrival jitter; tail-stability signal that TCP-only runs cannot observe. |
| `rps`                 | maximize  | fortio saturation        | Achieved QPS under `-qps 0`. Fixed-QPS RPS would clamp to the offered load, so it is not a source.    |
| `latency_p50`         | minimize  | fortio fixed-QPS         | Seconds (stored); displayed as milliseconds. Measured under the configured `fortio.fixedQps` offered load. |
| `latency_p90`         | minimize  | fortio fixed-QPS         | Seconds (stored); displayed as milliseconds. See `latency_p50`.                                       |
| `latency_p99`         | minimize  | fortio fixed-QPS         | Seconds (stored); displayed as milliseconds. Comparable across trials because offered load is stable. |

Splitting latency and RPS across two fortio sub-stages avoids the
"overloaded system has high latency because it's overloaded" confound:
RPS is only meaningful under saturation, and latency percentiles are
only comparable across trials under a stable offered load.

See [`tests/fixtures/experiment_example.yaml`](tests/fixtures/experiment_example.yaml)
for the canonical executable fixture.

Run it:

```sh
# YAML-driven (recommended): one config drives baseline, trial, or optimize.
kube-autotuner run --config experiment.yaml

# Or ad-hoc, no YAML:
kube-autotuner baseline --source nodeA --target nodeB --output out/results.jsonl
kube-autotuner analyze --input out/results.jsonl --output-dir out/analysis
```

## Commands

| Command          | Purpose                                                      |
|------------------|--------------------------------------------------------------|
| `baseline`       | iperf3 with the current sysctls; reference measurement.      |
| `trial`          | One benchmark with a fixed `--param key=value` set.          |
| `optimize`       | Ax Bayesian tuning loop (requires `[optimize]`).             |
| `run`            | Dispatch the mode declared in `experiment.yaml`.             |
| `analyze`        | Pareto, importance, recommendations (requires `[analysis]`). |
| `sysctl get/set` | Low-level read/write against the selected backend.           |

Per-command flags: `kube-autotuner <command> --help`.

## Backends

Three sysctl backends handle the write path, so the same loop runs
against different cluster shapes:

- **real**: schedules a privileged pod and writes via `sysctl -w` in the
  host init namespace. Works on any Kubernetes distribution that allows
  privileged pods.
- **talos**: patches machineconfig via `talosctl patch mc --mode=no-reboot`
  and polls `/proc/sys` until the new values show up.
- **fake**: JSON-file state, for tests and local iteration.
