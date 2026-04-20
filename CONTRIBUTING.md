# Contributing

## Development environment

All commands below (including `git push`, which triggers the lefthook
pre-push gate) assume you are inside the Nix devshell. With direnv this is
automatic after the first `direnv allow`; otherwise run `nix develop` before
anything else:

```sh
direnv allow    # first time only
# or
nix develop
```

The devshell's entry hook runs `task bootstrap`, which syncs the uv
virtualenv and regenerates shell completions as needed. Install git hooks
once with:

```sh
task hooks:install
```

## Common commands

```sh
task                    # list all available tasks
task test
task run -- tune my-node
```

## Integration tests

`task test:integration` runs the integration suite (`tests/integration/`,
marked `@pytest.mark.integration`) against a live Talos/k8s cluster. The
default `task test` and the lefthook pre-push hook **do not** run these
tests — `pyproject.toml` sets `-m "not integration"` in `addopts` so
cluster-free pushes stay fast.

Bring a Talos Docker cluster up first; see `tests/integration/README.md`
for the `talosctl cluster create docker --name kube-autotuner-test`
invocation and the `KUBE_AUTOTUNER_SYSCTL_BACKEND` / `KUBE_AUTOTUNER_ALLOW_SYSCTL_SKIP`
environment knobs.

## Dependency groups

The project splits dependencies into one runtime set and three PEP 735
dev groups:

- `dev` (mandatory): `pytest`, `pytest-cov`, `pytest-timeout`,
  `types-PyYAML`. Installed by `uv sync` / `task bootstrap`.
- `optimize` (optional): `ax-platform` for the Bayesian optimizer.
  Install with `uv sync --group optimize` when you need to run
  `kube-autotuner optimize` or `tests/test_optimizer.py` against the
  real Ax engine — tests gated on Ax skip cleanly without it.
- `analysis` (optional): `pandas`, `plotly`, `scikit-learn` for the
  post-run reporting tools. Install with `uv sync --group analysis`.

The optional groups are deliberately *not* nested inside `dev` so
`task bootstrap` and the lefthook pre-push gate stay cheap. Run
`uv sync --all-groups` when you want every group at once.

## Shell completions

`task bootstrap` writes bash, zsh, and fish completion scripts into
`.venv/completions/` using the standard XDG layout. The devshell exports
`XDG_DATA_DIRS` and `FPATH` so each shell's native discovery picks them up
on the next shell start. zsh requires `compinit` to run after `FPATH` is
set; most direnv-driven zsh setups handle this.

If auto-discovery does not work in your setup, install completions
permanently to your shell's rc via Typer's built-in:

```sh
kube-autotuner --install-completion
```
