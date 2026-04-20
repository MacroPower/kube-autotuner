# kube-autotuner

Benchmark-driven kernel parameter tuning for Kubernetes nodes.

## Overview

`kube-autotuner` searches for Linux sysctl values that maximise network
performance on the nodes of a Kubernetes cluster. iperf3 between two
nodes is the measurement. An Ax-platform multi-objective Bayesian
optimizer proposes the next configuration, the tool applies it, and the
benchmark runs again. The loop repeats until it converges on a
Pareto-optimal set of trade-offs between throughput, retransmits, CPU,
and memory.

## How it works

1. You write an `experiment.yaml`.
2. `kube-autotuner run` resolves the sysctl backend and runs preflight
   checks against the live cluster.
3. An iperf3 server Deployment lands on the target node and iperf3 client
   Jobs on each source node; the runner collects JSON results.
4. In `optimize` mode the Ax loop proposes trials, applies sysctls,
   benchmarks, and appends one JSONL record per trial.
5. Every run writes the resolved `objectives` block alongside the JSONL
   as `<output>.meta.json`, so `kube-autotuner analyze` picks up the
   same frontier and recommendation weights without re-specifying them
   and turns the JSONL into Pareto plots, parameter importance scores,
   and a ranked list of configurations.

## Install

The package is not yet published to PyPI. Install from source:

```sh
pip install "git+https://github.com/<owner>/kube-autotuner.git"
# or, from a clone:
uv pip install -e .
```

Python ≥ 3.14 is required. For a managed uv + Nix dev environment, see
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
  hardwareClass: 10g             # 1g | 10g
  namespace: default
  ipFamilyPolicy: RequireDualStack

benchmark:
  duration: 30                   # seconds of measurement per iteration
  omit: 5                        # warmup seconds discarded from stats
  iterations: 3
  parallel: 16                   # iperf3 -P streams per client
  window: null                   # iperf3 -w hint; e.g. "256K" to pin it
  modes: [tcp]                   # any of: tcp, udp

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
    - { metric: throughput, direction: maximize }
    - { metric: cpu, direction: minimize }
    - { metric: retransmits, direction: minimize }
    - { metric: memory, direction: minimize }
  constraints:
    - "throughput >= 1e6"
    - "cpu <= 200"
    - "retransmits <= 1e6"
    - "memory <= 1e10"
  recommendationWeights:
    cpu: 0.15
    retransmits: 0.3
    memory: 0.15

output: out/results.jsonl
```

Supplying `constraints:` or `recommendationWeights:` replaces the
default list wholesale rather than extending it. Weights are only valid
on minimize-direction metrics and must reference a metric present in
`pareto`. Valid metric names: `throughput`, `cpu`, `memory`,
`retransmits`.

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
