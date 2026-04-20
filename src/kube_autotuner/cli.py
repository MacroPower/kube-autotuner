"""Command-line interface for kube-autotuner.

The CLI is the single sanctioned place in the package that bridges from
environment variables to a concrete
:class:`~kube_autotuner.sysctl.backend.SysctlBackend` -- all backend
construction flows through :func:`_resolve_backend`, which is the only
caller of
:func:`~kube_autotuner.sysctl.setter.make_sysctl_setter_from_env`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import typer

from kube_autotuner import __version__, runs
from kube_autotuner.experiment import ExperimentConfig, ExperimentConfigError
from kube_autotuner.k8s.client import Kubectl
from kube_autotuner.sysctl.setter import (
    make_sysctl_setter,
    make_sysctl_setter_from_env,
)

if TYPE_CHECKING:
    from kube_autotuner.sysctl.backend import SysctlBackend
    from kube_autotuner.sysctl.setter import BackendName

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="kube-autotuner",
    help="Benchmark-driven kernel parameter tuning for Kubernetes nodes.",
    no_args_is_help=True,
)
sysctl_app = typer.Typer(
    name="sysctl",
    help="Read and write sysctls on a Kubernetes node.",
    no_args_is_help=True,
)
app.add_typer(sysctl_app, name="sysctl")


def _version_callback(value: bool) -> None:
    """Print the package version and exit when ``--version`` is supplied.

    Raises:
        typer.Exit: Always, once the version has been printed.
    """
    if value:
        typer.echo(f"kube-autotuner {__version__}")
        raise typer.Exit


@app.callback()
def main(
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="Enable debug logging."),
    ] = False,
    version: Annotated[  # noqa: ARG001 - consumed by the eager callback
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """kube-autotuner root command."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


# --- helpers ------------------------------------------------------------


def _parse_params(params: list[str]) -> dict[str, str]:
    """Parse repeated ``key=value`` strings into a dict.

    Args:
        params: Repeated ``--param`` values in ``key=value`` form.

    Returns:
        Mapping of sysctl key to string value.

    Raises:
        typer.Exit: A value is missing the ``=`` separator.
    """
    result: dict[str, str] = {}
    for p in params:
        if "=" not in p:
            typer.echo(f"Invalid param format (expected key=value): {p}", err=True)
            raise typer.Exit(code=1)
        key, val = p.split("=", 1)
        result[key] = val
    return result


def _load_experiment_yaml(path: Path) -> ExperimentConfig:
    """Load and validate an :class:`ExperimentConfig` from ``path``.

    Args:
        path: YAML file path.

    Returns:
        The validated :class:`ExperimentConfig`.

    Raises:
        typer.Exit: YAML parse or Pydantic validation failure.
    """
    try:
        return ExperimentConfig.from_yaml(path)
    except ExperimentConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=2) from e


def _apply_overrides(
    *,
    mode: Literal["baseline", "trial", "optimize"],
    sources: list[str],
    target: str,
    hardware_class: Literal["1g", "10g"],
    namespace: str,
    ip_family_policy: str,
    output: str,
    duration: int,
    iterations: int,
    udp: bool,
    sysctls: dict[str, str] | None = None,
    optimize: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Assemble a validated :class:`ExperimentConfig` from CLI overrides.

    Args:
        mode: Experiment mode.
        sources: Source node names; must be non-empty.
        target: Target node name.
        hardware_class: Hardware class selector.
        namespace: Kubernetes namespace.
        ip_family_policy: Service ``ipFamilyPolicy``.
        output: JSONL destination path.
        duration: iperf3 test duration in seconds.
        iterations: Iterations per benchmark mode.
        udp: ``True`` to append UDP alongside TCP.
        sysctls: Fixed sysctl map for ``mode="trial"``.
        optimize: ``optimize:`` section for ``mode="optimize"``.

    Returns:
        A validated :class:`ExperimentConfig`.

    Raises:
        typer.Exit: ``sources`` is empty.
    """
    if not sources:
        typer.echo("At least one --source is required.", err=True)
        raise typer.Exit(code=1)
    modes: list[str] = ["tcp", "udp"] if udp else ["tcp"]
    data: dict[str, Any] = {
        "mode": mode,
        "nodes": {
            "sources": list(sources),
            "target": target,
            "hardware_class": hardware_class,
            "namespace": namespace,
            "ip_family_policy": ip_family_policy,
        },
        "benchmark": {
            "duration": duration,
            "iterations": iterations,
            "modes": modes,
        },
        "output": output,
    }
    if sysctls is not None:
        data["trial"] = {"sysctls": sysctls}
    if optimize is not None:
        data["optimize"] = optimize
    return ExperimentConfig.model_validate(data)


def _resolve_backend(
    *,
    node: str,
    namespace: str,
    backend: str | None = None,
    kubectl: Kubectl | None = None,
    fake_state_path: Path | None = None,
    talos_endpoint: str | None = None,
) -> SysctlBackend:
    """Construct a :class:`SysctlBackend` from CLI overrides or the environment.

    The only function in the package that calls
    :func:`~kube_autotuner.sysctl.setter.make_sysctl_setter_from_env`;
    when ``backend`` is supplied explicitly the env-reading path is
    skipped in favour of
    :func:`~kube_autotuner.sysctl.setter.make_sysctl_setter`.

    Args:
        node: Kubernetes node the backend targets.
        namespace: Namespace for coordination resources.
        backend: Explicit backend selector (``"real"`` / ``"talos"`` /
            ``"fake"``); ``None`` means consult the environment.
        kubectl: Injected :class:`Kubectl` client.
        fake_state_path: JSON state file when ``backend="fake"``.
        talos_endpoint: Explicit ``talosctl -n`` target.

    Returns:
        A concrete :class:`SysctlBackend`.
    """
    if backend is None:
        return make_sysctl_setter_from_env(
            node=node,
            namespace=namespace,
            kubectl=kubectl,
            talos_endpoint=talos_endpoint,
        )
    return make_sysctl_setter(
        backend=cast("BackendName", backend),
        node=node,
        namespace=namespace,
        kubectl=kubectl,
        fake_state_path=fake_state_path,
        talos_endpoint=talos_endpoint,
    )


def _build_context(
    exp: ExperimentConfig,
    *,
    backend: str | None,
    fake_state_path: Path | None,
) -> runs.RunContext:
    """Build a :class:`~kube_autotuner.runs.RunContext` for a run mode.

    Args:
        exp: Validated experiment configuration.
        backend: Explicit backend override, or ``None`` for env-driven
            resolution.
        fake_state_path: JSON state file when ``backend="fake"``.

    Returns:
        A :class:`RunContext` with a freshly constructed
        :class:`Kubectl` and backend targeting ``exp.nodes.target``.
    """
    kubectl = Kubectl()
    sysctl_backend = _resolve_backend(
        node=exp.nodes.target,
        namespace=exp.nodes.namespace,
        backend=backend,
        kubectl=kubectl,
        fake_state_path=fake_state_path,
    )
    return runs.RunContext(
        exp=exp,
        kubectl=kubectl,
        backend=sysctl_backend,
        output=Path(exp.output),
    )


# --- subcommands ---------------------------------------------------------


@app.command()
def baseline(
    sources: Annotated[
        list[str],
        typer.Option(
            "--source",
            help="Source node hostname (repeatable for multi-client benchmarks).",
        ),
    ],
    target: Annotated[str, typer.Option("--target", help="Target node hostname.")],
    hardware_class: Annotated[
        Literal["1g", "10g"],
        typer.Option("--hardware-class", help="Hardware class (1g or 10g)."),
    ] = "10g",
    namespace: Annotated[
        str,
        typer.Option("--namespace", help="Kubernetes namespace."),
    ] = "default",
    ip_family_policy: Annotated[
        str,
        typer.Option("--ip-family-policy", help="Service ipFamilyPolicy."),
    ] = "RequireDualStack",
    output: Annotated[
        str,
        typer.Option("--output", help="Output JSONL file."),
    ] = "results.jsonl",
    duration: Annotated[
        int,
        typer.Option("--duration", help="Test duration in seconds."),
    ] = 30,
    iterations: Annotated[
        int,
        typer.Option("--iterations", help="Iterations per benchmark mode."),
    ] = 3,
    udp: Annotated[bool, typer.Option("--udp", help="Include UDP benchmarks.")] = False,
    backend: Annotated[
        str | None,
        typer.Option(
            "--backend",
            help="Sysctl backend override (real, talos, fake); defaults to env.",
        ),
    ] = None,
    fake_state_path: Annotated[
        Path | None,
        typer.Option(
            "--fake-state-path",
            help="JSON state file for the fake backend.",
        ),
    ] = None,
) -> None:
    """Run a baseline iperf3 benchmark with current sysctls."""
    exp = _apply_overrides(
        mode="baseline",
        sources=sources,
        target=target,
        hardware_class=hardware_class,
        namespace=namespace,
        ip_family_policy=ip_family_policy,
        output=output,
        duration=duration,
        iterations=iterations,
        udp=udp,
    )
    ctx = _build_context(exp, backend=backend, fake_state_path=fake_state_path)
    runs.run_baseline(ctx)


@app.command()
def trial(
    sources: Annotated[
        list[str],
        typer.Option("--source", help="Source node hostname (repeatable)."),
    ],
    target: Annotated[str, typer.Option("--target", help="Target node hostname.")],
    param: Annotated[
        list[str],
        typer.Option(
            "-p",
            "--param",
            help="Sysctl parameter as key=value (repeatable).",
        ),
    ],
    hardware_class: Annotated[
        Literal["1g", "10g"],
        typer.Option("--hardware-class", help="Hardware class (1g or 10g)."),
    ] = "10g",
    namespace: Annotated[
        str,
        typer.Option("--namespace", help="Kubernetes namespace."),
    ] = "default",
    ip_family_policy: Annotated[
        str,
        typer.Option("--ip-family-policy", help="Service ipFamilyPolicy."),
    ] = "RequireDualStack",
    output: Annotated[
        str,
        typer.Option("--output", help="Output JSONL file."),
    ] = "results.jsonl",
    duration: Annotated[
        int,
        typer.Option("--duration", help="Test duration in seconds."),
    ] = 30,
    iterations: Annotated[
        int,
        typer.Option("--iterations", help="Iterations per benchmark mode."),
    ] = 3,
    udp: Annotated[bool, typer.Option("--udp", help="Include UDP benchmarks.")] = False,
    backend: Annotated[
        str | None,
        typer.Option(
            "--backend",
            help="Sysctl backend override (real, talos, fake); defaults to env.",
        ),
    ] = None,
    fake_state_path: Annotated[
        Path | None,
        typer.Option(
            "--fake-state-path",
            help="JSON state file for the fake backend.",
        ),
    ] = None,
) -> None:
    """Run a single optimization trial: snapshot, apply, benchmark, restore."""
    params = _parse_params(param)
    exp = _apply_overrides(
        mode="trial",
        sources=sources,
        target=target,
        hardware_class=hardware_class,
        namespace=namespace,
        ip_family_policy=ip_family_policy,
        output=output,
        duration=duration,
        iterations=iterations,
        udp=udp,
        sysctls=params,
    )
    ctx = _build_context(exp, backend=backend, fake_state_path=fake_state_path)
    runs.run_trial(ctx)


@app.command()
def optimize(
    sources: Annotated[
        list[str],
        typer.Option("--source", help="Source node hostname (repeatable)."),
    ],
    target: Annotated[str, typer.Option("--target", help="Target node hostname.")],
    hardware_class: Annotated[
        Literal["1g", "10g"],
        typer.Option("--hardware-class", help="Hardware class (1g or 10g)."),
    ] = "10g",
    namespace: Annotated[
        str,
        typer.Option("--namespace", help="Kubernetes namespace."),
    ] = "default",
    ip_family_policy: Annotated[
        str,
        typer.Option("--ip-family-policy", help="Service ipFamilyPolicy."),
    ] = "RequireDualStack",
    output: Annotated[
        str,
        typer.Option("--output", help="Output file for trial results."),
    ] = "optimize_results.jsonl",
    duration: Annotated[
        int,
        typer.Option("--duration", help="Test duration in seconds."),
    ] = 30,
    iterations: Annotated[
        int,
        typer.Option("--iterations", help="Iterations per benchmark mode."),
    ] = 3,
    udp: Annotated[bool, typer.Option("--udp", help="Include UDP benchmarks.")] = False,
    n_trials: Annotated[
        int,
        typer.Option("--n-trials", help="Total optimization trials."),
    ] = 50,
    n_sobol: Annotated[
        int,
        typer.Option("--n-sobol", help="Sobol initialization trials."),
    ] = 15,
    apply_source: Annotated[
        bool,
        typer.Option(
            "--apply-source",
            help="Also apply sysctls on every client node.",
        ),
    ] = False,
    backend: Annotated[
        str | None,
        typer.Option(
            "--backend",
            help="Sysctl backend override (real, talos, fake); defaults to env.",
        ),
    ] = None,
    fake_state_path: Annotated[
        Path | None,
        typer.Option(
            "--fake-state-path",
            help="JSON state file for the fake backend.",
        ),
    ] = None,
) -> None:
    """Run the Bayesian optimization loop (requires the optimize group)."""
    exp = _apply_overrides(
        mode="optimize",
        sources=sources,
        target=target,
        hardware_class=hardware_class,
        namespace=namespace,
        ip_family_policy=ip_family_policy,
        output=output,
        duration=duration,
        iterations=iterations,
        udp=udp,
        optimize={
            "n_trials": n_trials,
            "n_sobol": n_sobol,
            "apply_source": apply_source,
        },
    )
    ctx = _build_context(exp, backend=backend, fake_state_path=fake_state_path)
    runs.run_optimize(ctx)


@app.command()
def run(
    config_path: Annotated[
        Path,
        typer.Option(
            "-c",
            "--config",
            exists=True,
            dir_okay=False,
            help="Path to experiment YAML.",
        ),
    ],
    backend: Annotated[
        str | None,
        typer.Option(
            "--backend",
            help="Sysctl backend override (real, talos, fake); defaults to env.",
        ),
    ] = None,
    fake_state_path: Annotated[
        Path | None,
        typer.Option(
            "--fake-state-path",
            help="JSON state file for the fake backend.",
        ),
    ] = None,
) -> None:
    """Run an experiment defined by a YAML config file.

    Raises:
        typer.Exit: ``exp.preflight`` reports any failure.
    """
    exp = _load_experiment_yaml(config_path)

    kubectl = Kubectl()
    preflight = exp.preflight(kubectl)
    failures = [r for r in preflight if not r.passed]
    if failures:
        for r in failures:
            typer.echo(f"preflight [{r.name}] FAIL: {r.detail}", err=True)
        raise typer.Exit(code=2)

    sysctl_backend = _resolve_backend(
        node=exp.nodes.target,
        namespace=exp.nodes.namespace,
        backend=backend,
        kubectl=kubectl,
        fake_state_path=fake_state_path,
    )
    ctx = runs.RunContext(
        exp=exp,
        kubectl=kubectl,
        backend=sysctl_backend,
        output=Path(exp.output),
    )
    if exp.mode == "baseline":
        runs.run_baseline(ctx)
    elif exp.mode == "trial":
        runs.run_trial(ctx)
    else:
        runs.run_optimize(ctx)


@sysctl_app.command("set")
def sysctl_set(
    node: Annotated[str, typer.Option("--node", help="Target node hostname.")],
    param: Annotated[
        list[str],
        typer.Option(
            "-p",
            "--param",
            help="Sysctl parameter as key=value (repeatable).",
        ),
    ],
    namespace: Annotated[
        str,
        typer.Option("--namespace", help="Kubernetes namespace."),
    ] = "default",
    backend: Annotated[
        str | None,
        typer.Option(
            "--backend",
            help="Sysctl backend override (real, talos, fake); defaults to env.",
        ),
    ] = None,
    fake_state_path: Annotated[
        Path | None,
        typer.Option(
            "--fake-state-path",
            help="JSON state file for the fake backend.",
        ),
    ] = None,
) -> None:
    """Apply sysctl values on a node via privileged pod."""
    params = _parse_params(param)
    setter = _resolve_backend(
        node=node,
        namespace=namespace,
        backend=backend,
        fake_state_path=fake_state_path,
    )
    with setter.lock():
        setter.apply(params)
    typer.echo(f"Applied {len(params)} sysctl(s) on {node}")


@sysctl_app.command("get")
def sysctl_get(
    node: Annotated[str, typer.Option("--node", help="Target node hostname.")],
    param: Annotated[
        list[str],
        typer.Option(
            "-p",
            "--param",
            help="Sysctl parameter name (repeatable).",
        ),
    ],
    namespace: Annotated[
        str,
        typer.Option("--namespace", help="Kubernetes namespace."),
    ] = "default",
    backend: Annotated[
        str | None,
        typer.Option(
            "--backend",
            help="Sysctl backend override (real, talos, fake); defaults to env.",
        ),
    ] = None,
    fake_state_path: Annotated[
        Path | None,
        typer.Option(
            "--fake-state-path",
            help="JSON state file for the fake backend.",
        ),
    ] = None,
) -> None:
    """Read current sysctl values from a node."""
    setter = _resolve_backend(
        node=node,
        namespace=namespace,
        backend=backend,
        fake_state_path=fake_state_path,
    )
    values = setter.get(list(param))
    typer.echo(json.dumps(values, indent=2))
