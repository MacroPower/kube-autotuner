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
- **High-dimensional, discrete search.** The default space has 28
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
2. `kube-autotuner baseline|trial|optimize <experiment.yaml>` resolves
   the sysctl backend and runs preflight checks against the live
   cluster.
3. An iperf3 server Deployment and a fortio server Deployment land on
   the target node. Each iteration then drives four sub-stages
   sequentially: an iperf3 TCP bandwidth fan-out, an iperf3 UDP
   bandwidth fan-out (sole source of `jitter_ms`), a fortio
   saturation fan-out (RPS), and a fortio fixed-QPS fan-out
   (latency percentiles). Sub-stages run one at a time so fortio
   never contends with iperf3 for NIC, CPU, or CNI state; within
   each sub-stage the source nodes still fan out in parallel.
4. In `optimize` the Ax loop proposes trials, applies sysctls,
   benchmarks, and appends one zstd-compressed Parquet file per trial
   into the `--output` directory.
5. Every run writes the resolved `objectives` block alongside the
   trial dataset as `<output>/_meta.json`, so `kube-autotuner analyze`
   picks up the same frontier and recommendation weights without
   re-specifying them and turns the dataset into Pareto plots,
   parameter importance scores, and a ranked list of configurations.

## Parameter space

When `optimize.paramSpace` is omitted, the tool searches a canonical
default: 28 sysctls across seven categories. Two parameter types are
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

| Category           | Count | Examples                                                                                |
| ------------------ | ----- | --------------------------------------------------------------------------------------- |
| TCP buffers        | 5     | `net.core.rmem_max`, `net.ipv4.tcp_rmem`, `net.ipv4.tcp_mem`                            |
| Congestion control | 7     | `net.ipv4.tcp_congestion_control`, `net.core.default_qdisc`, `net.ipv4.tcp_autocorking` |
| NAPI / softirq     | 3     | `net.core.netdev_budget`, `net.core.netdev_max_backlog`                                 |
| VM / memory        | 1     | `vm.min_free_kbytes`                                                                    |
| Connection         | 7     | `net.core.somaxconn`, `net.ipv4.tcp_max_tw_buckets`, `net.ipv4.ip_local_port_range`     |
| UDP                | 2     | `net.ipv4.udp_rmem_min`, `net.ipv4.udp_mem`                                             |
| Conntrack          | 3     | `net.netfilter.nf_conntrack_max`, `net.netfilter.nf_conntrack_tcp_timeout_established`  |

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
- Use the `trial` subcommand on a YAML with a `trial:` section
  (`sysctls:` map) to benchmark one configuration without invoking
  the optimizer.

## Install

The package is not yet published to PyPI. Install from source:

```sh
pip install "git+https://github.com/macropower/kube-autotuner.git"
# or, from a clone:
uv pip install -e .
```

Python >= 3.14 is required. For a managed uv + Nix dev environment, see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Quick start

A full `experiment.yaml` covering every field the loader accepts. Every
section except `nodes` is optional; the defaults shown below are what
you get when you omit them. The subcommand picks the execution flow
(`baseline | trial | optimize`); the YAML must carry an `optimize:`
section to run `optimize`, and a `trial:` section to run `trial`.

```yaml
output: out/results

nodes:
  sources: [nodeA]
  target: nodeB
  hardwareClass: 10g # Arbitrary label used to stratify results.
  namespace: default
  ipFamilyPolicy: RequireDualStack

benchmark:
  iterations: 3 # Iterations per trial.
  # Wall-clock barrier that aligns multi-client stages on a shared start epoch.
  # Note that this relies on NTP-synced nodes; a pod that misses the window starts
  # late rather than blocking the run. Set to 0 to disable.
  syncWindowSeconds: 15
  # Benchmark sub-stages to run per iteration. Omitting a stage skips its
  # wall-clock cost and prunes its metrics from objectives at load time.
  stages: [bw-tcp, bw-udp, fortio-sat, fortio-fixed]
  # Record per-iteration host-state snapshots (conntrack, sockets, slab) on
  # each TrialResult.
  collectHostState: false

# Ax Bayesian loop knobs plus the search space.
optimize:
  nTrials: 50
  # Quasi-random exploration trials; <= nTrials.
  nSobol: 15
  # Write best sysctls on source nodes (default: destination only).
  applySource: false
  # Optional parameter overrides; omit to use the built-in canonical sysctl set.
  paramSpace:
    - name: net.core.rmem_max
      # integer range: values = [min, max]
      paramType: int
      values: [4194304, 67108864]
      # Optional. Drives memoryCostWeight at recommendation time only.
      # The `kind` selects how the rung value maps to bytes:
      #   * identity (rung is bytes)
      #   * triple_max (max field of the space-separated triple)
      #   * triple_max_pages (same x 4096)
      #   * kib (rung in KiB)
      #   * per_entry (perEntryBytes sets the per-entry size)
      memoryCost:
        kind: identity
    - name: net.ipv4.tcp_congestion_control
      # Discrete set; values are strings or ints.
      paramType: choice
      values: [cubic, bbr]
  # After the Bayesian loop, re-run the top-K recommended configs this many
  # extra times to confirm the win is real and not a noise artifact.
  # Set to 0 to disable the verification phase.
  verificationTrials: 0
  # Number of top-ranked configs to verify when verificationTrials > 0.
  verificationTopK: 3

# Required for the `trial` subcommand. Apply a fixed sysctl set for one
# benchmark.
# trial:
#   sysctls:
#     net.core.rmem_max: 67108864
#     net.ipv4.tcp_congestion_control: bbr

# Iperf3 drives two bandwidth sub-stages per iteration, each ~`duration`
# seconds of wall time:
#   * `bw-tcp`: `iperf3`    -> `tcp_throughput|retransmit_rate`.
#   * `bw-udp`: `iperf3 -u` -> `udp_throughput|loss_rate|jitter`.
iperf:
  duration: 30  # Seconds of measurement per iteration (iperf3 -t).
  omit: 5       # Warmup seconds to discard from stats (iperf3 -O).
  parallel: 16  # Streams per client (iperf3 -P).
  client:
    extraArgs: ["--bidir", "-Z"]
  server:
    extraArgs: ["--forceflush"]
  # Job retry budget per client per iteration. Independent of the pod-level
  # backoffLimit baked into the manifest: that controls pod retries inside
  # one Job, this controls how many times the runner rebuilds the Job from
  # scratch. Worst-case wall time per client is maxAttempts * 180s. Note
  # the snake_case key: IperfSection does not register a camelCase alias.
  maxAttempts: 3

# Fortio drives two request/response sub-stages per iteration, each ~`duration`
# seconds of wall time:
#   * `fortio-sat`:   `fortio load -qps 0`          -> `rps`.
#   * `fortio-fixed`: `fortio load -qps <fixedQps>` -> `latency_p50|p90|p99`.
fortio:
  duration: 30     # Seconds of measurement per iteration (fortio -t <n>s).
  fixedQps: 1000   # Offered QPS for the fixed sub-stage (fortio -qps).
  connections: 4   # Fortio connections for both sub-stages (fortio -c).
  client:
    extraArgs: []
  server:
    extraArgs: []
  # Job retry budget per client per iteration; see `iperf.maxAttempts`.
  maxAttempts: 3

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
  # Every Pareto objective needs an explicit threshold so Ax's hypervolume
  # geometry stays well-defined. Note that supplying any constraints will
  # replace the default list. Accepts the k8s quantity grammar:
  #   * Binary IEC (`Ki`, `Mi`, `Gi`, `Ti`, `Pi`, `Ei`)
  #   * Decimal SI (`n`, `u`, `m`, `k`, `M`, `G`, `T`, `P`, `E`)
  #   * Decimal exponents (`1e6`, `1E-9`)
  constraints:
    # Throughput bits/sec.
    - "tcp_throughput >= 1M"
    - "udp_throughput >= 1M"
    # Retransmits per GB sent; 1000 retx/GB ~ 1 retx/MB.
    - "tcp_retransmit_rate <= 1000"
    # 5% UDP packet loss cap.
    - "udp_loss_rate <= 0.05"
    # Seconds; 10ms jitter ceiling.
    - "udp_jitter <= 10m"
    - "rps >= 100"
    # Seconds; mean-latency loose cap from the fixed_qps sub-stage.
    # Loose caps keep hypervolume informative without dominating the
    # recommendation score.
    - "latency_p50 <= 100m"
    - "latency_p90 <= 500m"
    - "latency_p99 <= 1000m"
  # Weights apply to every metric listed in `pareto` (both maximize and minimize
  # directions) and must reference a metric present in `pareto`. Defaults depend
  # on direction: an omitted maximize-metric weight defaults to `1.0` (the
  # metric contributes its full +norm), and an omitted minimize-metric weight
  # defaults to `0.0` (the metric participates in frontier selection but does
  # not bias the recommendation score). Raising a maximize weight above `1.0`
  # biases the recommendation toward that metric; setting any weight to `0.0`
  # disables the metric's contribution entirely.
  recommendationWeights:
    tcp_retransmit_rate: 0.3
    udp_loss_rate: 0.3
    udp_jitter: 0.1
    latency_p50: 0.1
    latency_p90: 0.2
    latency_p99: 0.3
  # Gently penalise configs that burn kernel/CNI memory on over-sized
  # buffers, conntrack entries, and backlog queues. Derived statically
  # from the selected rungs, summed across sysctls, min-max normalised
  # alongside the other minimize terms. Set to 0.0 to disable. Applies
  # only at recommendation-ranking time; Ax exploration stays untouched.
  memoryCostWeight: 0.1

# Kustomize patches layered onto the generated client/server manifests.
# Accepts a Strategic Merge Patch body (dict), a JSON6902 op list,
# or a pre-rendered patch string.
patches:
  - target:
      kind: Job
      name: iperf3-client
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
    strict: false # Allow the patch to no-op.
    patch:
      - op: add
        path: /spec/template/spec/hostNetwork
        value: true
```

## Metric catalog

Valid `pareto.metric` values and their sources:

| Metric                | Direction | Source sub-stage  | Notes                                                                                    |
| --------------------- | --------- | ----------------- | ---------------------------------------------------------------------------------------- |
| `tcp_throughput`      | maximize  | iperf3 bw-tcp     | Bits per second. Summed across source clients per iteration, averaged across iterations. |
| `udp_throughput`      | maximize  | iperf3 bw-udp     | Bits per second. Same aggregation as `tcp_throughput`.                                   |
| `tcp_retransmit_rate` | minimize  | iperf3 bw-tcp     | Retransmits per GB sent (1.0 ~ one retransmit per gigabyte).                             |
| `udp_loss_rate`       | minimize  | iperf3 bw-udp     | Lost packets per packet sent; per-iteration ratio-of-sums then averaged.                 |
| `udp_jitter`          | minimize  | iperf3 bw-udp     | Seconds. Mean UDP inter-arrival jitter.                                                  |
| `rps`                 | maximize  | fortio saturation | Achieved QPS under maximum load.                                                         |
| `latency_p50`         | minimize  | fortio fixed-QPS  | Seconds. Measured under the configured `fortio.fixedQps` offered load.                   |
| `latency_p90`         | minimize  | fortio fixed-QPS  | Seconds. See `latency_p50`.                                                              |
| `latency_p99`         | minimize  | fortio fixed-QPS  | Seconds. See `latency_p50`.                                                              |

## Commands

| Command                      | Purpose                                                      |
| ---------------------------- | ------------------------------------------------------------ |
| `baseline <experiment.yaml>` | iperf3 with the current sysctls; reference measurement.      |
| `trial <experiment.yaml>`    | One benchmark with the YAML's `trial.sysctls` map applied.   |
| `optimize <experiment.yaml>` | Ax Bayesian tuning loop (requires `[optimize]`).             |
| `analyze <results/>`         | Pareto, importance, recommendations (requires `[analysis]`). |
| `sysctl get/set`             | Low-level read/write against the selected backend.           |

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
