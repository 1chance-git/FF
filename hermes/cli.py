"""Hermes CLI: operational commands for the Freqtrade bot this project runs.

Built on `click` (mature, the de facto standard Python CLI framework —
reused rather than hand-rolling argument parsing) for command dispatch,
and `rich` for CLI-friendly output: colored status text, and tables for
anything with more than one row (health checks, process status).

Every command also logs through :mod:`hermes.logging_config`, so the
same invocation produces both a human-readable terminal transcript and
(if configured) a structured JSON log line per event — the two output
channels documented in that module.

Commands
--------
* ``hermes health`` — run health checks against a running bot's API.
* ``hermes backtest`` — launch a backtest via :mod:`hermes.backtest`.
* ``hermes start`` / ``stop`` / ``restart`` / ``status`` — process
  lifecycle via :mod:`hermes.process`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from hermes.backtest import BacktestConfig, BacktestLauncher
from hermes.health import HealthChecker, HealthStatus
from hermes.logging_config import LoggingConfig, configure_logging, get_logger
from hermes.process import BotProcessManager, ProcessConfig

logger = get_logger(__name__)
console = Console()

_STATUS_STYLE = {
    HealthStatus.HEALTHY: "bold green",
    HealthStatus.DEGRADED: "bold yellow",
    HealthStatus.UNHEALTHY: "bold red",
}

_DEFAULT_PID_FILE = Path("user_data/hermes.pid")


def _configure_from_options(json_log_file: Path | None, verbose: bool) -> None:
    configure_logging(
        LoggingConfig(
            level="DEBUG" if verbose else "INFO",
            json_log_file=json_log_file,
            console=True,
        )
    )


@click.group()
@click.option(
    "--json-log-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Write structured JSON logs to this file, in addition to console output.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG-level logging.")
@click.pass_context
def cli(ctx: click.Context, json_log_file: Path | None, verbose: bool) -> None:
    """Hermes: operational tooling for the stat-arb trading bot."""
    ctx.ensure_object(dict)
    _configure_from_options(json_log_file, verbose)


@cli.command()
@click.option("--api-url", default="http://127.0.0.1:8080", show_default=True)
@click.option("--username", default=None, help="Freqtrade REST API username.")
@click.option("--password", default=None, help="Freqtrade REST API password.")
def health(api_url: str, username: str | None, password: str | None) -> None:
    """Run health checks against a running bot's REST API and print a report."""
    from freqtrade_client import FtRestClient

    client = FtRestClient(api_url, username, password)
    report = HealthChecker(client).run()

    table = Table(title="Hermes health check")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Message")
    for check in report.checks:
        table.add_row(check.name, check.status.value, check.message)
    console.print(table)

    style = _STATUS_STYLE[report.status]
    console.print(
        f"Overall status: [{style}]{report.status.value}[/{style}] "
        f"({report.duration_seconds:.2f}s)"
    )
    sys.exit(0 if report.is_healthy else 1)


@cli.command()
@click.option(
    "-c",
    "--config",
    "config_files",
    multiple=True,
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Freqtrade config file(s), same semantics as `freqtrade backtesting -c`.",
)
@click.option("--strategy", required=True, help="Strategy class name to backtest.")
@click.option("--timerange", default=None, help="Freqtrade timerange, e.g. 20240101-20240401.")
@click.option("--timeframe", default=None, help="Timeframe override.")
@click.option(
    "--strategy-path", default=None, type=click.Path(exists=True, path_type=Path)
)
@click.option("--timeout", default=None, type=float, help="Timeout in seconds.")
def backtest(
    config_files: tuple[Path, ...],
    strategy: str,
    timerange: str | None,
    timeframe: str | None,
    strategy_path: Path | None,
    timeout: float | None,
) -> None:
    """Launch a Freqtrade backtest and print a summary."""
    bt_config = BacktestConfig(
        strategy=strategy,
        config_files=config_files,
        timerange=timerange,
        timeframe=timeframe,
        strategy_path=strategy_path,
    )
    console.print(f"[bold]Launching backtest[/bold] for strategy [cyan]{strategy}[/cyan]...")
    result = BacktestLauncher().run(bt_config, timeout_seconds=timeout)

    if result.succeeded:
        console.print(
            f"[bold green]Backtest succeeded[/bold green] in {result.duration_seconds:.1f}s"
        )
    else:
        console.print(
            f"[bold red]Backtest failed[/bold red] "
            f"(exit code {result.exit_code}) after {result.duration_seconds:.1f}s"
        )
        console.print(result.stderr[-2000:])
    sys.exit(result.exit_code)


def _process_manager(config_files: tuple[Path, ...], strategy: str, pid_file: Path) -> BotProcessManager:
    return BotProcessManager(
        ProcessConfig(config_files=config_files, strategy=strategy, pid_file=pid_file)
    )


_config_option = click.option(
    "-c",
    "--config",
    "config_files",
    multiple=True,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
_strategy_option = click.option("--strategy", required=True)
_pid_file_option = click.option(
    "--pid-file", default=_DEFAULT_PID_FILE, type=click.Path(path_type=Path)
)


@cli.command()
@_config_option
@_strategy_option
@_pid_file_option
def start(config_files: tuple[Path, ...], strategy: str, pid_file: Path) -> None:
    """Start the bot process (no-op if already running)."""
    manager = _process_manager(config_files, strategy, pid_file)
    pid = manager.start()
    console.print(f"[bold green]Bot running[/bold green] (pid={pid})")


@cli.command()
@_config_option
@_strategy_option
@_pid_file_option
def stop(config_files: tuple[Path, ...], strategy: str, pid_file: Path) -> None:
    """Stop the bot process, if running."""
    manager = _process_manager(config_files, strategy, pid_file)
    stopped = manager.stop()
    if stopped:
        console.print("[bold green]Bot stopped[/bold green]")
    else:
        console.print("[yellow]No running bot found[/yellow]")


@cli.command()
@_config_option
@_strategy_option
@_pid_file_option
def restart(config_files: tuple[Path, ...], strategy: str, pid_file: Path) -> None:
    """Restart the bot process."""
    manager = _process_manager(config_files, strategy, pid_file)
    pid = manager.restart()
    console.print(f"[bold green]Bot restarted[/bold green] (pid={pid})")


@cli.command()
@_config_option
@_strategy_option
@_pid_file_option
def status(config_files: tuple[Path, ...], strategy: str, pid_file: Path) -> None:
    """Show the bot process' running status."""
    manager = _process_manager(config_files, strategy, pid_file)
    info = manager.status()

    table = Table(title="Bot process status")
    for key, value in info.items():
        table.add_row(key, str(value))
    console.print(table)
    sys.exit(0 if info["running"] else 1)


def main() -> None:
    """Entry point for `python -m hermes` / the `hermes` console script."""
    cli(obj={})


if __name__ == "__main__":
    main()
