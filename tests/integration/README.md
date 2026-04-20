# Integration tests

These run against a live Talos cluster (Docker by default). The suite is wired
up in `conftest.py` and gated by the `integration` pytest marker, which is
deselected by default via `addopts = -m "not integration"`; `task test` always
skips them and the lefthook pre-push hook never invokes them.

## Running

```sh
task test:integration
```

Bring a Talos Docker cluster up before invoking the task. The conftest
expects the cluster to be named `kube-autotuner-test`:

```sh
sudo -E talosctl cluster create docker \
  --name kube-autotuner-test \
  --workers 1
```

Tear down with `sudo -E talosctl cluster destroy docker --name kube-autotuner-test`
when you're done.

Both tasks set `KUBE_AUTOTUNER_SYSCTL_BACKEND=talos` by default so sysctl
writes route through `talosctl patch mc` and the optimize/trial suites
exercise the real tuning path.

## Backend selection

The sysctl backend is selected by `KUBE_AUTOTUNER_SYSCTL_BACKEND`:

- **`talos`** (default for integration tests): writes via
  `talosctl patch mc` (YAML strategic-merge patch against
  `machine.sysctls`) followed by `talosctl reboot` to force the
  boot-time sysctl apply. Reads via `talosctl read /proc/sys/...`.
  Works on both Talos Docker and bare-metal Talos. Each `apply()`
  takes ~30 s on Docker because the node is rebooted. **Requires**
  `talosctl` in `PATH` and a valid `TALOSCONFIG` (the cluster-create
  step configures this automatically for Docker clusters; bare-metal
  callers already have one). Optionally override the Talos API
  endpoint with `KUBE_AUTOTUNER_TALOS_ENDPOINT`; by default the
  backend resolves the node's `InternalIP` via kubectl.
- **`real`**: privileged-pod + `sysctl -w` path. Works on bare-metal
  Talos where the pod runs in the init user namespace. Fails on Talos
  Docker because `/proc/sys/net/core/*` is exposed read-only to the pod
  (the pod's user namespace does not own `init_net`). Set
  `KUBE_AUTOTUNER_ALLOW_SYSCTL_SKIP=1` to skip the write-requiring
  tests in that environment.
- **`fake`**: JSON-file-backed in-memory state for unit tests. Never
  runs host commands.

Tests that require a backend capable of writing host sysctls are marked
`@pytest.mark.requires_real_sysctl_write`:

- `test_sysctl_e2e.py::test_sysctl_set_applies_and_sysctl_get_reflects_it`
- `test_sysctl_setter.py::test_apply_and_verify`
- `test_sysctl_setter.py::test_snapshot_and_restore`

`KUBE_AUTOTUNER_SYSCTL_BACKEND=talos` (the default) qualifies as such a
backend, so these tests run. Setting `KUBE_AUTOTUNER_ALLOW_SYSCTL_SKIP=1`
still forces them to skip.

### Why reboots?

Talos's runtime sysctl reconciler accepts `machine.sysctls` changes via
`talosctl patch mc` and persists them to both the on-disk and in-memory
machine configs. On Talos Docker, however, the controller does not
reliably push those changes into `/proc/sys` at runtime -- the patch
completes `rc=0` and `machine.sysctls` shows the new value, but the
kernel sysctl is unchanged. Rebooting the node forces the boot-time
sysctl apply, which always works. This is the only mechanism that gives
us runtime tuning on both Talos Docker and bare-metal Talos. The cost is
one reboot per `apply()` (~30 s on Docker).

### `machine.sysctls` persistence

The `talos` backend writes into the node's machine config under
`machine.sysctls`. `restore()` overwrites the values but does not remove
keys. On an ephemeral Docker cluster this is irrelevant (the cluster is
destroyed after tests). **On bare-metal Talos**, running `optimize` then
`restore` leaves the machine config populated with whatever the final
restored values were -- including snapshot values that originally came
from kernel defaults. Clean up manually by emitting a `remove` patch on
`/machine/sysctls` or editing the node's machine config.

## Exercising the privileged-pod backend on bare metal

To exercise the `real` (privileged-pod) backend on a bare-metal cluster
where writes succeed:

```sh
export KUBECONFIG=~/.kube/config
KUBE_AUTOTUNER_SYSCTL_BACKEND=real uv run pytest tests/integration/ \
  -m integration -v
```

## Optional: seeding read values deterministically

Read tests only assert that the returned values are numeric, so kernel
defaults are sufficient. With the `talos` backend, read determinism is
usually unnecessary -- applies and reads are round-tripped through the
same Talos API. If a more specific assertion is required in a local
experiment, recreate the Talos Docker cluster with a sysctls patch:

```sh
sudo -E talosctl cluster destroy docker --name kube-autotuner-test
sudo -E talosctl cluster create docker --name kube-autotuner-test --workers 1 \
  --config-patch '[{"op":"add","path":"/machine/sysctls","value":{"net.core.rmem_max":"16777216","net.core.wmem_max":"16777216"}}]'
```

The seed is applied at boot by `machined` (which runs in the init userns
and can write sysctls). The values land in `init_net`, which is where
both `talosctl read /proc/sys/...` and the `hostNetwork` privileged pod
read from.
