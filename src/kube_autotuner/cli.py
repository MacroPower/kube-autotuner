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
from kube_autotuner.models import ALL_STAGES, metrics_for_stages
from kube_autotuner.progress import make_observer
from kube_autotuner.report import format_retransmit_rate
from kube_autotuner.sysctl.setter import (
    make_sysctl_setter,
    make_sysctl_setter_from_env,
)
from kube_autotuner.trial_log import TrialLog
from kube_autotuner.units import format_duration

if TYPE_CHECKING:
    from kube_autotuner.experiment import ObjectivesSection
    from kube_autotuner.models import StageName
    from kube_autotuner.progress import ProgressObserver
    from kube_autotuner.sysctl.backend import SysctlBackend
    from kube_autotuner.sysctl.setter import BackendName


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
        stages: frozenset[StageName] | None = None,
    ) -> ProgressObserver:
        """Build a progress observer honoring the current CLI flags.

        Args:
            objectives: Pareto objectives and recommendation weights
                for the active experiment. Forwarded to the observer
                so the live ``Best so far`` panel ranks trials by
                the same weighted score used by
                :func:`kube_autotuner.analysis.recommend_configs`.
                ``None`` keeps the throughput-descending fallback,
                which is used by flows that never call
                ``on_trial_complete`` (``baseline`` / ``trial``).
            stages: Enabled benchmark sub-stages, forwarded so the
                Stage bar sizes to ``100%`` at the end of the last
                enabled stage and the ``Best so far`` table hides
                columns whose metrics are produced only by disabled
                stages. ``None`` (the default) falls back to the
                full stage set; callers pass ``exp.benchmark.stages``
                when they have an :class:`ExperimentConfig` on hand.

        Returns:
            A :class:`~kube_autotuner.progress.RichProgressObserver`
            when :attr:`progress_enabled` is ``True``; otherwise a
            :class:`~kube_autotuner.progress.NullObserver`.
        """
        kwargs: dict[str, Any] = {
            "enabled": self.progress_enabled,
            "console": self.console,
            "objectives": objectives,
        }
        if stages is not None:
            kwargs["stages"] = stages
        return make_observer(**kwargs)


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


def _run_experiment(
    ctx: typer.Context,
    *,
    config_path: Path,
    backend: str | None,
    fake_state_path: Path | None,
    mode: Literal["baseline", "trial", "optimize"],
    fresh: bool = False,
) -> None:
    """Load ``config_path``, run preflight, and dispatch into ``runs``.

    Args:
        ctx: Typer context carrying the shared :class:`AppState`.
        config_path: Path to the experiment YAML.
        backend: Explicit backend override, or ``None`` for env-driven
            resolution.
        fake_state_path: JSON state file when ``backend="fake"``.
        mode: Which ``runs`` entry point to dispatch into.
        fresh: Forwarded to :func:`runs.run_optimize`. Ignored for
            ``baseline`` and ``trial``.

    Raises:
        typer.Exit: The YAML lacks the section required by ``mode``,
            or :meth:`ExperimentConfig.preflight` reports any failure.
    """
    exp = _load_experiment_yaml(config_path)

    if mode == "optimize" and exp.optimize is None:
        typer.echo(
            "`optimize` requires an `optimize:` section in the YAML.",
            err=True,
        )
        raise typer.Exit(code=2)
    if mode == "trial" and exp.trial is None:
        typer.echo(
            "`trial` requires a `trial:` section in the YAML.",
            err=True,
        )
        raise typer.Exit(code=2)

    client = K8sClient()
    failures = [r for r in exp.preflight(client) if not r.passed]
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
    observer = _app_state(ctx).make_observer(
        objectives=exp.objectives,
        stages=exp.benchmark.stages,
    )
    run_ctx = runs.RunContext(
        exp=exp,
        client=client,
        backend=sysctl_backend,
        output=Path(exp.output),
        observer=observer,
        collect_host_state=exp.benchmark.collect_host_state,
    )
    with observer:
        if mode == "baseline":
            runs.run_baseline(run_ctx)
        elif mode == "trial":
            runs.run_trial(run_ctx)
        else:
            runs.run_optimize(run_ctx, fresh=fresh)


def _app_state(ctx: typer.Context) -> AppState:
    """Return the shared :class:`AppState` attached by the root callback."""
    return cast("AppState", ctx.obj)


# --- subcommands ---------------------------------------------------------


@app.command()
def baseline(
    ctx: typer.Context,
    config_path: Annotated[
        Path,
        typer.Argument(
            ...,
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
    """Run a baseline iperf3 benchmark with current sysctls."""
    _run_experiment(
        ctx,
        config_path=config_path,
        backend=backend,
        fake_state_path=fake_state_path,
        mode="baseline",
    )


@app.command()
def trial(
    ctx: typer.Context,
    config_path: Annotated[
        Path,
        typer.Argument(
            ...,
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
    """Run a single trial: snapshot, apply, benchmark, restore."""
    _run_experiment(
        ctx,
        config_path=config_path,
        backend=backend,
        fake_state_path=fake_state_path,
        mode="trial",
    )


@app.command()
def optimize(
    ctx: typer.Context,
    config_path: Annotated[
        Path,
        typer.Argument(
            ...,
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
            help="Ignore prior results; move them aside before starting.",
        ),
    ] = False,
) -> None:
    """Run the Bayesian optimization loop (requires the optimize group)."""
    _run_experiment(
        ctx,
        config_path=config_path,
        backend=backend,
        fake_state_path=fake_state_path,
        mode="optimize",
        fresh=fresh,
    )


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
        typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="Path to the trial dataset directory.",
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
        typer.Exit: The input trial dataset is empty or the requested
            hardware class has no trials.
    """
    from kube_autotuner import analysis, report  # noqa: PLC0415
    from kube_autotuner.experiment import ObjectivesSection  # noqa: PLC0415

    trials = TrialLog.load(input_file)
    if not trials:
        typer.echo(f"No trials found in {input_file}", err=True)
        raise typer.Exit(code=1)

    meta = TrialLog.load_resume_metadata(input_file)
    objectives = meta.objectives if meta is not None else ObjectivesSection()
    stages = meta.benchmark.stages if meta is not None else ALL_STAGES
    resume_metadata = meta

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
                explicit_class=hardware_class is not None,
                objectives=objectives,
                stages=stages,
                resume_metadata=resume_metadata,
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
        A comma-separated summary covering every measured metric:
        metrics whose value is ``None`` are skipped.
    """
    parts: list[str] = []
    tcp_tp = r.get("mean_tcp_throughput")
    if tcp_tp is not None:
        parts.append(f"{tcp_tp / 1e6:.1f} Mbps TCP")
    udp_tp = r.get("mean_udp_throughput")
    if udp_tp is not None:
        parts.append(f"{udp_tp / 1e6:.1f} Mbps UDP")
    retx = r.get("tcp_retransmit_rate")
    if retx is not None:
        parts.append(f"{format_retransmit_rate(retx)} TCP retx/GB")
    udp_loss = r.get("udp_loss_rate")
    if udp_loss is not None:
        parts.append(f"{udp_loss * 100:.2f}% UDP loss")
    jit = r.get("mean_udp_jitter")
    if jit is not None:
        parts.append(f"{format_duration(jit)} UDP jitter")
    rps = r.get("mean_rps")
    if rps is not None:
        parts.append(f"{rps:,.1f} rps")
    for key, label in (
        ("mean_latency_p50", "p50"),
        ("mean_latency_p90", "p90"),
        ("mean_latency_p99", "p99"),
    ):
        v = r.get(key)
        if v is not None:
            parts.append(f"{label} {format_duration(v)}")
    return ", ".join(parts)


def _analyze_one_class(  # noqa: PLR0914 - threads many helpers into one section dict
    trials: list[Any],
    *,
    hardware_class: str,
    topology: str | None,
    top_n: int,
    output_dir: Path,
    analysis: Any,  # noqa: ANN401
    explicit_class: bool,
    objectives: ObjectivesSection,
    stages: frozenset[StageName],
    resume_metadata: Any = None,  # noqa: ANN401
) -> dict[str, Any] | None:
    """Produce analysis output for a single hardware class.

    Writes ``recommendations.json`` and ``importance.json`` under
    ``output_dir / hardware_class / ...`` and returns a section dict
    suitable for :func:`kube_autotuner.report.write_index_html`.

    Args:
        trials: The full trial list (unfiltered).
        hardware_class: The hardware-class label to analyse.
        topology: Optional topology filter.
        top_n: Number of recommendations to emit.
        output_dir: Root output directory.
        analysis: The lazy-imported ``kube_autotuner.analysis`` module.
        explicit_class: ``True`` when the user supplied
            ``--hardware-class``; in that case an empty result is a
            hard error (no other class will be tried), otherwise we
            log and skip.
        objectives: Effective Pareto objectives and recommendation
            weights for this run.
        stages: Enabled benchmark sub-stages for this run. Forwarded
            to :func:`_build_axis_payload` so the parallel-coordinates
            chart drops axes whose metrics are only produced by
            disabled stages.
        resume_metadata: Optional sidecar
            :class:`~kube_autotuner.models.ResumeMetadata`; forwarded to
            :func:`analysis.trajectory_rows` and
            :func:`analysis.section_metadata`.

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
    importance = importance_by_target.get("mean_tcp_throughput")
    if importance is None:
        importance = analysis.parameter_importance(df)
    # One Pareto+score computation per class: ``recommend_configs`` is
    # a thin wrapper, so call ``pareto_recommendation_rows`` directly
    # and slice top-N here instead of running the pipeline twice.
    pareto_rows = analysis.pareto_recommendation_rows(
        trials,
        hardware_class,
        topology,
        objectives=objectives.pareto,
        weights=objectives.recommendation_weights,
        memory_cost_weight=objectives.memory_cost_weight,
    )
    _null_disabled_stage_metrics(pareto_rows, stages)
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

    all_rows, axis_columns = _build_axis_payload(
        df,
        pareto_mask,
        stages=stages,
        trials=hw_trials,
    )

    verif_stats = analysis.verification_stats(hw_trials)
    for row in pareto_rows:
        row["stability_badge"] = analysis.stability_badge(
            verif_stats.get(row["trial_id"]),
        )

    objective_dicts = [obj.model_dump(mode="json") for obj in objectives.pareto]
    top_row = pareto_rows[0] if pareto_rows else None
    baseline = analysis.baseline_comparison(hw_trials, objective_dicts, top_row)
    trajectory = analysis.trajectory_rows(
        hw_trials,
        objective_dicts,
        resume_metadata,
    )
    metadata = analysis.section_metadata(hw_trials, resume_metadata)
    correlation = analysis.sysctl_correlation_matrix(df, importance_by_target)
    category_rollup = analysis.category_importance_rollup(importance_by_target)
    host_issues = analysis.host_state_issues(hw_trials)

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
        "objectives": objective_dicts,
        "default_weights": dict(objectives.recommendation_weights),
        "memory_cost_weight": objectives.memory_cost_weight,
        "top_n": top_n,
        "importance": importance,
        "importance_by_target": importance_by_target,
        "all_rows": all_rows,
        "axis_columns": axis_columns,
        "host_state": analysis.host_state_series(
            hw_trials,
            hardware_class,
            topology,
        ),
        "baseline_comparison": baseline,
        "verification_stats": verif_stats,
        "trajectory_rows": trajectory,
        "metadata": metadata,
        "correlation_matrix": correlation,
        "importance_category_rollup": category_rollup,
        "host_state_issues": host_issues,
    }


def _null_disabled_stage_metrics(
    rows: list[dict[str, Any]],
    stages: frozenset[StageName],
) -> None:
    """Replace disabled-stage metric values with ``None`` in-place.

    The :class:`~kube_autotuner.models.TrialResult` accessors return
    ``0.0`` rather than ``NaN`` when a stage did not run, so a downstream
    ``!== null`` filter in the browser keeps disabled-stage columns
    visible on value. Nullifying them here makes
    ``recommendations.json`` and the ranked table match the stage-aware
    axis chart.

    Args:
        rows: Recommendation rows produced by
            :func:`kube_autotuner.analysis.pareto_recommendation_rows`.
            Mutated in-place.
        stages: Enabled benchmark sub-stages.
    """
    from kube_autotuner.scoring import METRIC_TO_DF_COLUMN  # noqa: PLC0415

    relevant = {METRIC_TO_DF_COLUMN[m] for m in metrics_for_stages(stages)}
    disabled = [c for c in METRIC_TO_DF_COLUMN.values() if c not in relevant]
    for row in rows:
        for col in disabled:
            if col in row:
                row[col] = None


_AXIS_METRIC_COLUMNS: tuple[str, ...] = (
    "mean_tcp_throughput",
    "mean_udp_throughput",
    "tcp_retransmit_rate",
    "udp_loss_rate",
    "mean_udp_jitter",
    "mean_rps",
    "mean_latency_p50",
    "mean_latency_p90",
    "mean_latency_p99",
)


def _build_axis_payload(
    df: Any,  # noqa: ANN401
    pareto_mask: Any,  # noqa: ANN401
    *,
    stages: frozenset[StageName],
    trials: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return JSON-safe per-trial rows plus the chart's axis columns.

    Mirrors the NaN-to-None coercion used by
    :func:`kube_autotuner.analysis.pareto_recommendation_rows` so the
    browser-side ``json.dumps(allow_nan=False)`` in
    :func:`kube_autotuner.report._embed_json` cannot choke. Also
    guards against infinities (``math.isfinite``), which ``pd.isna``
    does not catch.

    Args:
        df: The per-class DataFrame from
            :func:`kube_autotuner.analysis.trials_to_dataframe`.
        pareto_mask: Boolean Series aligned with ``df``; ``True``
            marks Pareto-optimal rows.
        stages: Enabled benchmark sub-stages. Metric columns whose
            raw metric is not produced by one of these stages are
            excluded from the payload, matching the objective
            pruning applied at config-load time.
        trials: Optional list of :class:`TrialResult` aligned with
            ``df`` rows by ``trial_id``; used to read the ``phase``
            label attached to each row. ``None`` renders as
            ``"unknown"``.

    Returns:
        ``(all_rows, axis_columns)`` where ``all_rows`` has one dict
        per df row with ``trial_id`` (str), ``pareto`` (bool),
        ``phase`` (str), every axis column, and every
        ``<col>_std`` column, and ``axis_columns`` lists those metric
        columns produced by an enabled stage that also carry at
        least one non-null value in ``df``.
    """
    import math  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    from kube_autotuner.scoring import METRIC_TO_DF_COLUMN  # noqa: PLC0415

    relevant_df_cols = {METRIC_TO_DF_COLUMN[m] for m in metrics_for_stages(stages)}
    axis_columns = [
        c
        for c in _AXIS_METRIC_COLUMNS
        if c in df.columns and c in relevant_df_cols and df[c].notna().any()
    ]

    phase_by_trial: dict[str, str] = {}
    if trials:
        for t in trials:
            phase_by_trial[t.trial_id] = t.phase or "unknown"

    rows: list[dict[str, Any]] = []
    records = df.to_dict(orient="records")
    for record, is_pareto in zip(records, pareto_mask, strict=True):
        tid = str(record["trial_id"])
        out: dict[str, Any] = {
            "trial_id": tid,
            "pareto": bool(is_pareto),
            "phase": phase_by_trial.get(tid, "unknown"),
        }
        for col in axis_columns:
            v = record.get(col)
            if v is None or pd.isna(v) or not math.isfinite(float(v)):
                out[col] = None
            else:
                out[col] = float(v)
            std_key = f"{col}_std"
            sv = record.get(std_key)
            if sv is None or pd.isna(sv) or not math.isfinite(float(sv)):
                out[std_key] = None
            else:
                out[std_key] = float(sv)
        rows.append(out)
    return rows, axis_columns
