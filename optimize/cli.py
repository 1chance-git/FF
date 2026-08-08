"""Optimization framework CLI: hyperopt, grid search, walk-forward, reporting.

Built on `click`, matching `hermes.cli`'s style and conventions (this
framework is meant to be operated alongside Hermes, not instead of it —
`optimize hyperopt`/`optimize backtest-report` shell out to the same
kind of Freqtrade subprocess `hermes` manages the bot process with).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pandas as pd
from rich.console import Console

from optimize.hyperopt_launcher import HyperoptConfig, HyperoptLauncher
from optimize.reporting import compute_performance_report, print_performance_report

console = Console()


@click.group()
def cli() -> None:
    """Optimization framework: hyperopt, parameter search, walk-forward, reporting."""


@cli.command()
@click.option(
    "-c",
    "--config",
    "config_files",
    multiple=True,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--strategy", required=True)
@click.option("--epochs", default=100, show_default=True, type=int)
@click.option("--spaces", multiple=True, default=("buy", "sell"), show_default=True)
@click.option("--hyperopt-loss", default="StatArbHyperOptLoss", show_default=True)
@click.option("--timerange", default=None)
@click.option("--jobs", default=None, type=int)
@click.option("--random-state", default=None, type=int)
@click.option("--timeout", default=None, type=float)
def hyperopt(
    config_files: tuple[Path, ...],
    strategy: str,
    epochs: int,
    spaces: tuple[str, ...],
    hyperopt_loss: str,
    timerange: str | None,
    jobs: int | None,
    random_state: int | None,
    timeout: float | None,
) -> None:
    """Launch a Freqtrade hyperopt run."""
    config = HyperoptConfig(
        strategy=strategy,
        config_files=config_files,
        epochs=epochs,
        spaces=spaces,
        hyperopt_loss=hyperopt_loss,
        timerange=timerange,
        jobs=jobs,
        random_state=random_state,
    )
    console.print(f"[bold]Launching hyperopt[/bold] for [cyan]{strategy}[/cyan] ({epochs} epochs)...")
    result = HyperoptLauncher().run(config, timeout_seconds=timeout)

    if result.succeeded:
        console.print(f"[bold green]Hyperopt succeeded[/bold green] in {result.duration_seconds:.1f}s")
    else:
        console.print(
            f"[bold red]Hyperopt failed[/bold red] "
            f"(exit code {result.exit_code}) after {result.duration_seconds:.1f}s"
        )
        console.print(result.stderr[-2000:])
    sys.exit(result.exit_code)


@cli.command("report")
@click.option(
    "--trades-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="JSON file containing a list of trade records (Freqtrade's standard trade dict shape).",
)
@click.option("--starting-balance", required=True, type=float)
@click.option("--title", default="Performance report")
def report_command(trades_file: Path, starting_balance: float, title: str) -> None:
    """Render a performance report from a JSON file of trades."""
    trades = pd.DataFrame(json.loads(trades_file.read_text(encoding="utf-8")))
    performance_report = compute_performance_report(trades, starting_balance)
    print_performance_report(performance_report, title=title, console=console)


def main() -> None:
    """Entry point for `python -m optimize` / the `optimize` console script."""
    cli()


if __name__ == "__main__":
    main()
