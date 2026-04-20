# kube-autotuner — agent guidance

Guidance for agents (Claude Code, Codex, etc.) working in this repo. Human
onboarding lives in `CONTRIBUTING.md`; `AGENTS.md` is a symlink to this file.

## Language, frameworks, style

- **Python 3.14 floor.** `requires-python = ">=3.14"`; `target-version = "py314"`
  in ruff. Use modern typing (`X | None`, `list[X]`, `dict[K, V]`).
- **Typer CLI.** All command options use the
  `Annotated[..., typer.Option(...)]` idiom — not the older positional
  `typer.Option` default-value form. Sub-apps (`app.add_typer(...)`) are the
  chosen shape for nested commands; do not also expose flat kebab-case aliases
  for the same commands.
- **Pydantic v2** for every data model (runtime validation, not just typing).
  `Optional[X]` is `X | None`. `List[...]` / `Dict[...]` are lowercase
  generics. Do not blanket-apply `ConfigDict(frozen=True)` — set it per model
  only where no caller mutates fields.
- **Google-style docstrings.** Ruff enforces via
  `[tool.ruff.lint.pydocstyle] convention = "google"`. Every public symbol
  needs one; `D205`/`D415` expect a summary line terminated with a period.
- **Absolute imports only.** `[tool.ruff.lint.flake8-tidy-imports]
  ban-relative-imports = "all"`. Always `from kube_autotuner.X import ...`, never
  `from .X import ...`.

## Quality gates

Three commands form the contract for every commit:

```
task lint:check
task typecheck
task test
```

- **ruff.** `select = ["ALL"]`, `preview = true`. The repo's own ignores live
  in `[tool.ruff.lint]` and `[tool.ruff.lint.per-file-ignores]`; adding new
  ignores is a judgement call that must be justified in the commit message.
  Notable per-file-ignores already in place:
  - `tests/**/*.py`: `S101`, `D1`, `PLR2004`, `SLF001`, `ANN`, `INP001`.
  - `**/__init__.py`: `D104`, `F401`.
  - `**/cli.py`: `FBT` (Typer boolean flags are the canonical CLI shape).
- **ty.** `[tool.ty.rules] all = "error"` — strict across `src/` and `tests/`.
  Tests get a narrow override loosening `possibly-unresolved-reference` and
  `unused-ignore-comment` to `warn`. Untyped third-party modules need stubs
  (prefer `types-*` dev deps) or a narrow `[[tool.ty.overrides]]`.
- **lefthook.** `task hooks:install` wires:
  - pre-commit: `ruff check --fix` then `ruff format` on staged files.
  - pre-push: `task test` and `task typecheck`.
  Pre-push does **not** run integration tests; `addopts` carries
  `-m "not integration"` so cluster-free pushes stay fast.
- **Pytest markers.** `--strict-markers` is on. Two markers are registered in
  `pyproject.toml`:
  - `integration` — requires a Talos Docker cluster.
  - `requires_real_sysctl_write` — requires a cluster where privileged pods
    can write host sysctls (Talos Docker userns boundary blocks this).
  Any new marker must be registered before use or collection fails.
- **Coverage.** `addopts` includes `--cov=kube_autotuner --cov-report=term-missing`.
- **Timeout.** `timeout = 120` under `[tool.pytest.ini_options]` (via
  `pytest-timeout`).

## Taskfile command catalog

`Taskfile.yaml` is the single entry point. Do not invent new command names
outside this catalog without updating both the Taskfile and this document.

| Task               | Purpose                                              |
|--------------------|------------------------------------------------------|
| `task sync`        | `uv sync --quiet` the managed virtualenv.            |
| `task fmt`         | `ruff format .` across the tree.                     |
| `task lint`        | `ruff check --fix .` (auto-fix).                     |
| `task lint:check`  | CI-mode: `ruff check .` then `ruff format --check .`.|
| `task typecheck`   | `ty check`.                                          |
| `task test`        | `uv run pytest` (honours `addopts`).                 |
| `task test:integration` | `uv run pytest -m integration` (requires a live Talos/k8s cluster). |
| `task run -- ...`  | `uv run kube-autotuner ...`.                         |
| `task bootstrap`   | Sync venv + regenerate shell completions.            |
| `task completions` | Regenerate bash/zsh/fish completions under `.venv/`. |
| `task hooks:install` | `lefthook install`.                                |

`task test:integration` stands apart from the default pre-push gate: the
lefthook pre-push hook runs `task test` and `task typecheck`, which select
`-m "not integration"` via `addopts` and therefore **never** run integration
tests. Integration tests are opt-in and require a live Talos/k8s cluster — see
`tests/integration/README.md` for the `talosctl cluster create docker
--name kube-autotuner-test` bring-up step.

## Dependency groups

Dependencies are split into one runtime set and three PEP 735 dev groups.
Never nest the optional groups inside `dev` — lefthook pre-push and
`task bootstrap` must not pull ~500 MB of optional deps on every clone.

| Group     | Contents                                  | Who installs it                           |
|-----------|-------------------------------------------|-------------------------------------------|
| runtime   | `typer`, `pydantic`, `pyyaml`             | everyone, via `uv sync`.                  |
| `dev`     | `pytest`, `pytest-cov`, `pytest-timeout`  | mandatory for developers; `uv sync` pulls it by default. |
| `optimize` | `ax-platform`                            | `uv sync --group optimize`; optional.     |
| `analysis` | `pandas`, `plotly`, `scikit-learn`       | `uv sync --group analysis`; optional. |

Everything in `optimize` and `analysis` must be **lazy-imported** — either
inside a function body, or behind `if TYPE_CHECKING:` for annotation-only
references. `task completions` eagerly imports the Typer `app`, so any
module reachable from `cli.py` at import time must also import cleanly
without those groups installed.

## Repository layout

- `src/kube_autotuner/` — the package. New subpackages (`sysctl/`, `k8s/`,
  `benchmark/`) get an empty `__init__.py`; `templates/` is resource data
  and does not.
- `tests/` — flat pytest tree. `tests/integration/` lives behind the
  `integration` marker.
- `templates/*.yaml` (under the package) — Kubernetes manifests loaded via
  `importlib.resources`. Must ship in the built wheel.
