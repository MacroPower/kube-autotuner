"""Command-line interface for kube-autotuner."""

from typing import Annotated

import typer

from kube_autotuner import __version__

app = typer.Typer(
    name="kube-autotuner",
    help="Benchmark-driven kernel parameter tuning for Kubernetes nodes.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"kube-autotuner {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
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


@app.command()
def tune(
    node: Annotated[str, typer.Argument(help="Kubernetes node name to tune.")],
) -> None:
    """Run a tuning pass against NODE (placeholder)."""
    typer.echo(f"Would tune node: {node}")
