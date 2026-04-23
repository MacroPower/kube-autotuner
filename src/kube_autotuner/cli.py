"""Command-line interface for kube-autotuner.

The CLI is the single sanctioned place in the package that bridges from
environment variables to a concrete
:class:`~kube_autotuner.sysctl.backend.SysctlBackend` -- all backend
construction flows through :func:`_resolve_backend`, which is the only
caller of
:func:`~kube_autotuner.sysctl.setter.make_sysctl_setter_from_env`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
import typer

from kube_autotuner import __version__, runs
from kube_autotuner.experiment import ExperimentConfig, ExperimentConfigError
from kube_autotuner.k8s.client import K8sClient
from kube_autotuner.models import TrialLog
from kube_autotuner.progress import make_observer
from kube_autotuner.report import format_retransmit_rate
from kube_autotuner.sysctl.setter import (
    make_sysctl_setter,
    make_sysctl_setter_from_env,
)

if TYPE_CHECKING:
    from kube_autotuner.experiment import ObjectivesSection
    from kube_autotuner.progress import ProgressObserver
    from kube_autotuner.sysctl.backend import SysctlBackend
    from kube_autotuner.sysctl.setter import BackendName

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Shared CLI state threaded through every long-running command.

    Attributes:
        console: The single :class:`rich.console.Console` used by
            both the root logger's :class:`RichHandler` and any
            :class:`~kube_autotuner.progress.RichProgressObserver`.
            Sharing one instance is what lets log records render
            above live progress regions instead of tearing through
            them.
        progress_enabled: ``True`` when the console is a TTY and the
            user did not pass ``--no-progress``.
    """

    console: Console
    progress_enabled: bool

    def make_observer(
        self,
        objectives: ObjectivesSection | None = None,
    ) -> ProgressObserver:
        """Build a progress observer honoring the current CLI flags.

        Args:
            objectives: Pareto objectives and recommendation weights
                for the active experiment. Forwarded to the observer
                so the live ``Best so far`` panel ranks trials by
                the same weighted score used by
                :func:`kube_autotuner.analysis.recommend_configs`.
                ``None`` keeps the legacy throughput-descending
                fallback, which is used by flows that never call
                ``on_trial_complete`` (``baseline`` / ``trial``).

        Returns:
            A :class:`~kube_autotuner.progress.RichProgressObserver`
            when :attr:`progress_enabled` is ``True``; otherwise a
            :class:`~kube_autotuner.progress.NullObserver`.
        """
        return make_observer(
            enabled=self.progress_enabled,
            console=self.console,
            objectives=objectives,
        )


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
    ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="Enable debug logging."),
    ] = False,
    no_progress: Annotated[
        bool,
        typer.Option(
            "--no-progress",
            help="Disable the live progress display (logs only).",
        ),
    ] = False,
    _version: Annotated[
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
    # One stderr console shared across RichHandler and any progress
    # observer. ``Console`` auto-detects TTY; under ``CliRunner`` the
    # redirected stderr buffer reports ``is_terminal = False`` and
    # progress cleanly degrades to plain logging.
    console = Console(stderr=True)
    progress_enabled = console.is_terminal and not no_progress
    ctx.obj = AppState(console=console, progress_enabled=progress_enabled)
    if progress_enabled:
        handler: logging.Handler = RichHandler(
            console=console,
            show_path=False,
            markup=False,
            log_time_format="%H:%M:%S",
            rich_tracebacks=True,
        )
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[handler],
            force=True,
        )
    else:
        # ``force=True`` rebinds the handler to the current ``sys.stderr``
        # so repeated ``CliRunner.invoke`` calls each see the redirected
        # stream.
        logging.basicConfig(
            level=level,
            format="%(levelname)s: %(message)s",
            force=True,
        )
    if not verbose:
        # ``ax.api.client`` emits two INFO lines per Bayesian step
        # (``Generated new trial...`` and ``Trial N marked COMPLETED``)
        # that duplicate the kube-autotuner-owned per-trial summary.
        # Narrow to this child logger so future INFOs from other ax
        # submodules still surface if anyone wires a handler at that
        # level.
        logging.getLogger("ax.api.client").setLevel(logging.WARNING)
        # ``standardize_y`` logs "Outcome X is constant, within
        # tolerance." once per Bayesian generate when a metric has
        # collapsed. ``OptimizationLoop`` emits one actionable warning
        # for the same condition, so suppress the upstream duplicate.
        logging.getLogger("ax.adapter.transforms.standardize_y").setLevel(
            logging.ERROR,
        )


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
    hardware_class: str,
    namespace: str,
    ip_family_policy: str,
    output: str,
    duration: int,
    iterations: int,
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
        iterations: Iterations per benchmark (each iteration runs
            both TCP and UDP bandwidth stages plus both fortio
            sub-stages).
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
    client: K8sClient | None = None,
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
        client: Injected :class:`K8sClient`.
        fake_state_path: JSON state file when ``backend="fake"``.
        talos_endpoint: Explicit ``talosctl -n`` target.

    Returns:
        A concrete :class:`SysctlBackend`.
    """
    if backend is None:
        return make_sysctl_setter_from_env(
            node=node,
            namespace=namespace,
            client=client,
            talos_endpoint=talos_endpoint,
        )
    return make_sysctl_setter(
        backend=cast("BackendName", backend),
        node=node,
        namespace=namespace,
        client=client,
        fake_state_path=fake_state_path,
        talos_endpoint=talos_endpoint,
    )


def _build_context(
    exp: ExperimentConfig,
    *,
    backend: str | None,
    fake_state_path: Path | None,
    observer: ProgressObserver,
) -> runs.RunContext:
    """Build a :class:`~kube_autotuner.runs.RunContext` for a run mode.

    Args:
        exp: Validated experiment configuration.
        backend: Explicit backend override, or ``None`` for env-driven
            resolution.
        fake_state_path: JSON state file when ``backend="fake"``.
        observer: Progress observer for the run. Passed verbatim into
            the returned :class:`RunContext`.

    Returns:
        A :class:`RunContext` with a freshly constructed
        :class:`K8sClient` and backend targeting ``exp.nodes.target``.
    """
    client = K8sClient()
    sysctl_backend = _resolve_backend(
        node=exp.nodes.target,
        namespace=exp.nodes.namespace,
        backend=backend,
        client=client,
        fake_state_path=fake_state_path,
    )
    return runs.RunContext(
        exp=exp,
        client=client,
        backend=sysctl_backend,
        output=Path(exp.output),
        observer=observer,
    )


def _app_state(ctx: typer.Context) -> AppState:
    """Return the shared :class:`AppState` attached by the root callback."""
    return cast("AppState", ctx.obj)


# --- subcommands ---------------------------------------------------------


@app.command()
def baseline(
    ctx: typer.Context,
    sources: Annotated[
        list[str],
        typer.Option(
            "--source",
            help="Source node hostname (repeatable for multi-client benchmarks).",
        ),
    ],
    target: Annotated[str, typer.Option("--target", help="Target node hostname.")],
    hardware_class: Annotated[
        str,
        typer.Option(
            "--hardware-class",
            help="Hardware class label (free-form; used to stratify results).",
        ),
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
        typer.Option("--iterations", help="Iterations per benchmark."),
    ] = 3,
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
    )
    observer = _app_state(ctx).make_observer(objectives=exp.objectives)
    run_ctx = _build_context(
        exp,
        backend=backend,
        fake_state_path=fake_state_path,
        observer=observer,
    )
    with observer:
        runs.run_baseline(run_ctx)


@app.command()
def trial(
    ctx: typer.Context,
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
        str,
        typer.Option(
            "--hardware-class",
            help="Hardware class label (free-form; used to stratify results).",
        ),
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
        typer.Option("--iterations", help="Iterations per benchmark."),
    ] = 3,
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
        sysctls=params,
    )
    observer = _app_state(ctx).make_observer(objectives=exp.objectives)
    run_ctx = _build_context(
        exp,
        backend=backend,
        fake_state_path=fake_state_path,
        observer=observer,
    )
    with observer:
        runs.run_trial(run_ctx)


@app.command()
def optimize(
    ctx: typer.Context,
    sources: Annotated[
        list[str],
        typer.Option("--source", help="Source node hostname (repeatable)."),
    ],
    target: Annotated[str, typer.Option("--target", help="Target node hostname.")],
    hardware_class: Annotated[
        str,
        typer.Option(
            "--hardware-class",
            help="Hardware class label (free-form; used to stratify results).",
        ),
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
        typer.Option("--iterations", help="Iterations per benchmark."),
    ] = 3,
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
    fresh: Annotated[
        bool,
        typer.Option(
            "--fresh",
            help="Ignore prior results; move them aside before starting.",
        ),
    ] = False,
    verification_trials: Annotated[
        int,
        typer.Option(
            "--verification-trials",
            help="Re-runs per top config after the primary loop. 0 disables.",
        ),
    ] = 0,
    verification_top_k: Annotated[
        int,
        typer.Option(
            "--verification-top-k",
            help="Number of top configs to verify.",
        ),
    ] = 3,
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
        optimize={
            "n_trials": n_trials,
            "n_sobol": n_sobol,
            "apply_source": apply_source,
            "verification_trials": verification_trials,
            "verification_top_k": verification_top_k,
        },
    )
    observer = _app_state(ctx).make_observer(objectives=exp.objectives)
    run_ctx = _build_context(
        exp,
        backend=backend,
        fake_state_path=fake_state_path,
        observer=observer,
    )
    with observer:
        runs.run_optimize(run_ctx, fresh=fresh)


@app.command()
def run(
    ctx: typer.Context,
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
    fresh: Annotated[
        bool,
        typer.Option(
            "--fresh",
            help=(
                "Ignore prior results; move them aside before starting "
                "(optimize mode only)."
            ),
        ),
    ] = False,
) -> None:
    """Run an experiment defined by a YAML config file.

    Raises:
        typer.Exit: ``exp.preflight`` reports any failure.
    """
    exp = _load_experiment_yaml(config_path)

    client = K8sClient()
    preflight = exp.preflight(client)
    failures = [r for r in preflight if not r.passed]
    if failures:
        for r in failures:
            typer.echo(f"preflight [{r.name}] FAIL: {r.detail}", err=True)
        raise typer.Exit(code=2)

    sysctl_backend = _resolve_backend(
        node=exp.nodes.target,
        namespace=exp.nodes.namespace,
        backend=backend,
        client=client,
        fake_state_path=fake_state_path,
    )
    observer = _app_state(ctx).make_observer(objectives=exp.objectives)
    run_ctx = runs.RunContext(
        exp=exp,
        client=client,
        backend=sysctl_backend,
        output=Path(exp.output),
        observer=observer,
    )
    with observer:
        if exp.mode == "baseline":
            if fresh:
                logger.info("--fresh has no effect in mode=%s", exp.mode)
            runs.run_baseline(run_ctx)
        elif exp.mode == "trial":
            if fresh:
                logger.info("--fresh has no effect in mode=%s", exp.mode)
            runs.run_trial(run_ctx)
        else:
            runs.run_optimize(run_ctx, fresh=fresh)


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


# --- analyze subcommand --------------------------------------------------


@app.command()
def analyze(
    ctx: typer.Context,
    input_file: Annotated[
        Path,
        typer.Option(
            "-i",
            "--input",
            exists=True,
            dir_okay=False,
            help="Path to the JSONL trial data file.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output-dir",
            help="Directory for HTML plots and JSON report.",
        ),
    ] = Path("analysis_output"),
    hardware_class: Annotated[
        str | None,
        typer.Option(
            "--hardware-class",
            help="Filter to a single hardware class label (default: analyze all).",
        ),
    ] = None,
    top_n: Annotated[
        int,
        typer.Option("--top-n", help="Number of recommended configs."),
    ] = 3,
    topology: Annotated[
        Literal["intra-az", "inter-az"] | None,
        typer.Option(
            "--topology",
            help="Filter to a topology (default: analyze all).",
        ),
    ] = None,
) -> None:
    """Analyze trial data: Pareto frontier, parameter importance, recommendations.

    Raises:
        typer.Exit: Input JSONL is empty or the requested hardware
            class has no trials.
    """
    from kube_autotuner import analysis, plots, report  # noqa: PLC0415
    from kube_autotuner.experiment import ObjectivesSection  # noqa: PLC0415

    trials = TrialLog.load(input_file)
    if not trials:
        typer.echo(f"No trials found in {input_file}", err=True)
        raise typer.Exit(code=1)

    meta = TrialLog.load_resume_metadata(input_file)
    objectives = meta.objectives if meta is not None else ObjectivesSection()

    if hardware_class is not None:
        classes = [hardware_class]
    else:
        classes = sorted({t.node_pair.hardware_class for t in trials})
    if not classes:
        typer.echo("No hardware classes found in data", err=True)
        raise typer.Exit(code=1)

    sections: list[dict[str, Any]] = []
    state = _app_state(ctx)
    progress = Progress(
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("[dim]elapsed[/dim]"),
        TimeElapsedColumn(),
        console=state.console,
        disable=not state.progress_enabled,
        transient=True,
    )
    with progress:
        task_id = progress.add_task("Analyzing hardware classes", total=len(classes))
        for hw in classes:
            progress.update(task_id, description=f"Analyzing {hw}")
            section = _analyze_one_class(
                trials,
                hardware_class=hw,
                topology=topology,
                top_n=top_n,
                output_dir=output_dir,
                analysis=analysis,
                plots=plots,
                explicit_class=hardware_class is not None,
                objectives=objectives,
            )
            if section is not None:
                sections.append(section)
            progress.advance(task_id)

    if sections:
        index_path = report.write_index_html(output_dir, sections)
        typer.echo(f"\nCombined report: {index_path}")


def _format_top_recommendation(r: dict[str, Any]) -> str:
    """Format the top recommendation line for CLI stdout.

    Args:
        r: A single recommendation dict as returned by
            :func:`kube_autotuner.analysis.recommend_configs`.

    Returns:
        A comma-separated summary covering every measured metric: all
        metrics whose value is ``None`` (e.g. ``mean_cni_memory`` when
        CNI is disabled) are skipped.
    """
    parts: list[str] = [
        f"{r['mean_throughput'] / 1e6:.1f} Mbps",
        f"{r['mean_cpu']:.1f}% CPU",
    ]
    nmem = r["mean_node_memory"]
    if nmem is not None:
        parts.append(f"node {nmem / 1024 / 1024:.0f} MiB")
    cmem = r["mean_cni_memory"]
    if cmem is not None:
        parts.append(f"cni {cmem / 1024 / 1024:.0f} MiB")
    parts.append(f"{format_retransmit_rate(r['retransmit_rate'])} retx/MB")
    jit = r.get("mean_jitter_ms")
    if jit is not None:
        parts.append(f"{jit:.3f} ms jitter")
    rps = r.get("mean_rps")
    if rps is not None:
        parts.append(f"{rps:,.1f} rps")
    for key, label in (
        ("mean_latency_p50_ms", "p50"),
        ("mean_latency_p90_ms", "p90"),
        ("mean_latency_p99_ms", "p99"),
    ):
        v = r.get(key)
        if v is not None:
            parts.append(f"{label} {v:.1f} ms")
    return ", ".join(parts)


def _analyze_one_class(
    trials: list[Any],
    *,
    hardware_class: str,
    topology: str | None,
    top_n: int,
    output_dir: Path,
    analysis: Any,  # noqa: ANN401
    plots: Any,  # noqa: ANN401
    explicit_class: bool,
    objectives: ObjectivesSection,
) -> dict[str, Any] | None:
    """Produce analysis output for a single hardware class.

    Writes per-figure HTML, ``recommendations.json``, and
    ``importance.json`` under ``output_dir / hardware_class / ...`` and
    returns a section dict suitable for
    :func:`kube_autotuner.report.write_index_html`.

    Args:
        trials: The full trial list (unfiltered).
        hardware_class: The hardware-class label to analyse.
        topology: Optional topology filter.
        top_n: Number of recommendations to emit.
        output_dir: Root output directory.
        analysis: The lazy-imported ``kube_autotuner.analysis`` module.
        plots: The lazy-imported ``kube_autotuner.plots`` module.
        explicit_class: ``True`` when the user supplied
            ``--hardware-class``; in that case an empty result is a
            hard error (no other class will be tried), otherwise we
            log and skip.
        objectives: Effective Pareto objectives and recommendation
            weights for this run.

    Returns:
        A section dict describing the analysis, or ``None`` when the
        class produced no trials and the caller supplied no explicit
        filter.

    Raises:
        typer.Exit: The user supplied an explicit
            ``--hardware-class`` filter but no trials matched.
    """
    hw_trials = [t for t in trials if t.node_pair.hardware_class == hardware_class]
    if topology is not None:
        hw_trials = [t for t in hw_trials if t.topology == topology]

    cardinalities = {len(t.node_pair.all_sources) for t in hw_trials}
    if len(cardinalities) > 1:
        typer.echo(
            f"WARNING: hardware class '{hardware_class}' mixes trials with different "
            f"client counts {sorted(cardinalities)}; throughput is not "
            f"directly comparable across cardinalities.",
            err=True,
        )

    df, _ = analysis.trials_to_dataframe(
        trials,
        hardware_class=hardware_class,
        topology=topology,
    )
    if df.empty:
        if explicit_class:
            typer.echo(
                f"No trials for hardware class '{hardware_class}'",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(
            f"No trials for hardware class '{hardware_class}', skipping",
            err=True,
        )
        return None

    tuple_objectives = [
        (analysis.METRIC_TO_DF_COLUMN[obj.metric], obj.direction)
        for obj in objectives.pareto
    ]
    front = analysis.pareto_front(df, objectives=tuple_objectives)
    pareto_mask = df["trial_id"].isin(front["trial_id"])
    # Importance is computed per potential target metric so the browser
    # can answer "which sysctls drive throughput vs p99 vs retx rate?".
    # parameter_importance silently returns an empty frame for targets
    # with too few finite samples or no variance, so this loop never
    # needs a try/except.
    importance_by_target: dict[str, Any] = {}
    for col in analysis.METRIC_TO_DF_COLUMN.values():
        if col not in df.columns or not df[col].notna().any():
            continue
        frame = analysis.parameter_importance(df, target=col)
        if not frame.empty:
            importance_by_target[col] = frame
    importance = importance_by_target.get(
        "mean_throughput",
        analysis.parameter_importance(df),
    )
    # One Pareto+score computation per class: ``recommend_configs`` is
    # a thin wrapper, so call ``pareto_recommendation_rows`` directly
    # and slice top-N here instead of running the pipeline twice.
    pareto_rows = analysis.pareto_recommendation_rows(
        trials,
        hardware_class,
        topology,
        objectives=objectives.pareto,
        weights=objectives.recommendation_weights,
    )
    recs: list[dict[str, Any]] = [
        {
            "rank": i + 1,
            **{k: row[k] for k in row if k != "score"},
            "score": round(row["score"], 4),
        }
        for i, row in enumerate(pareto_rows[:top_n])
    ]

    hw_dir = output_dir / hardware_class
    hw_dir.mkdir(parents=True, exist_ok=True)

    figures = _write_figures(
        df=df,
        front=front,
        pareto_mask=pareto_mask,
        hw_dir=hw_dir,
        plots=plots,
    )

    (hw_dir / "recommendations.json").write_text(
        json.dumps(recs, indent=2) + "\n",
    )
    (hw_dir / "importance.json").write_text(
        importance.to_json(orient="records", indent=2) + "\n",
    )

    typer.echo(
        f"\n=== {hardware_class} ({len(df)} trials, {len(front)} Pareto-optimal) ===",
    )
    typer.echo(f"Output: {hw_dir}")
    if recs:
        typer.echo(f"Top recommendation: {_format_top_recommendation(recs[0])}")

    return {
        "hardware_class": hardware_class,
        "trial_count": len(df),
        "pareto_count": len(front),
        "topology": topology,
        "recommendations": recs,
        "pareto_rows": pareto_rows,
        "objectives": [obj.model_dump(mode="json") for obj in objectives.pareto],
        "default_weights": dict(objectives.recommendation_weights),
        "top_n": top_n,
        "importance": importance,
        "importance_by_target": importance_by_target,
        "figures": figures,
    }


def _write_figures(
    *,
    df: Any,  # noqa: ANN401
    front: Any,  # noqa: ANN401
    pareto_mask: Any,  # noqa: ANN401
    hw_dir: Path,
    plots: Any,  # noqa: ANN401
) -> list[tuple[str, Any]]:
    """Render every per-hardware-class figure and write it to disk.

    Args:
        df: The per-class DataFrame.
        front: The Pareto frontier DataFrame.
        pareto_mask: Boolean mask marking Pareto-optimal rows of
            ``df``.
        hw_dir: Per-hardware-class output directory.
        plots: The lazy-imported ``kube_autotuner.plots`` module.

    Returns:
        A list of ``(label, figure)`` tuples ready to hand to
        :func:`kube_autotuner.report.write_index_html`.
    """
    scatter_fig = plots.plot_pareto_scatter_matrix(df, pareto_mask)
    scatter_fig.write_html(str(hw_dir / "pareto_scatter_matrix.html"))
    figures: list[tuple[str, Any]] = [
        ("Objective space (scatter matrix)", scatter_fig),
    ]

    def _has_data(col: str) -> bool:
        return col in df.columns and bool(df[col].notna().any())

    pair_candidates = [
        ("mean_throughput", "mean_cpu"),
        ("mean_throughput", "mean_node_memory"),
        ("mean_throughput", "mean_cni_memory"),
        ("mean_throughput", "retransmit_rate"),
        ("mean_throughput", "mean_jitter_ms"),
        ("mean_cpu", "mean_node_memory"),
        ("mean_cpu", "mean_cni_memory"),
        ("mean_cpu", "retransmit_rate"),
        ("mean_cpu", "mean_jitter_ms"),
    ]
    for x, y in pair_candidates:
        if not (_has_data(x) and _has_data(y)):
            continue
        fig = plots.plot_pareto_2d(df, front, x, y)
        fig.write_html(str(hw_dir / f"pareto_{x}_vs_{y}.html"))
        figures.append((f"Pareto: {x} vs {y}", fig))
    return figures
