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
