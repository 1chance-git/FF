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
* ``hermes analyze`` — read-only report over recorded trade history via
  :mod:`hermes.analyzer`.
* ``hermes backtest-report`` — read-only report of an *already recorded*
  backtest's Freqtrade results, via :mod:`hermes.backtest_report`. Never
  runs a backtest; it only reads what a previous run already persisted.
* ``hermes start`` / ``stop`` / ``restart`` / ``status`` — process
  lifecycle via :mod:`hermes.process`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from hermes.analyzer import AnalysisReport
from hermes.analyzer import analyze as analyze_history
from hermes.backtest import BacktestConfig, BacktestLauncher
from hermes.backtest_report import (
    BacktestReport,
    extract_stdout,
    parse_backtest_stdout,
)
from hermes.health import HealthChecker, HealthStatus
from hermes.logging_config import LoggingConfig, configure_logging, get_logger
from hermes.memory import DEFAULT_USER_DATA_DIR, USER_DATA_DIR_ENV_VAR
from hermes.memory import BacktestResult as MemoryBacktestResult
from hermes.memory import MemoryStore, memory_db_path
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
@click.option(
    "--username",
    default=None,
    envvar="HERMES_API_USERNAME",
    help="Freqtrade REST API username. Can also be set via HERMES_API_USERNAME.",
)
@click.option(
    "--password",
    default=None,
    envvar="HERMES_API_PASSWORD",
    help=(
        "Freqtrade REST API password. Prefer HERMES_API_PASSWORD over this flag: "
        "a CLI argument is visible in shell history and `ps` output on shared systems."
    ),
)
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
@click.option(
    "--user-data-dir",
    default=None,
    envvar=USER_DATA_DIR_ENV_VAR,
    show_envvar=True,
    type=click.Path(path_type=Path),
    help=(
        "Directory holding hermes_memory.sqlite3 (the same history trades are "
        f"recorded to). Defaults to {DEFAULT_USER_DATA_DIR}; set "
        f"{USER_DATA_DIR_ENV_VAR} to point every Hermes command at persistent "
        "storage instead (see RAILWAY.md)."
    ),
)
def backtest(
    config_files: tuple[Path, ...],
    strategy: str,
    timerange: str | None,
    timeframe: str | None,
    strategy_path: Path | None,
    timeout: float | None,
    user_data_dir: Path | None,
) -> None:
    """Launch a Freqtrade backtest and print a summary."""
    bt_config = BacktestConfig(
        strategy=strategy,
        config_files=config_files,
        timerange=timerange,
        timeframe=timeframe,
        strategy_path=strategy_path,
    )

    memory_store = None
    try:
        memory_store = MemoryStore(memory_db_path(user_data_dir))
    except Exception:
        logger.exception(
            "[HERMES][MEMORY][ERROR] Failed to initialize Hermes memory store; "
            "this backtest result will not be recorded"
        )

    console.print(f"[bold]Launching backtest[/bold] for strategy [cyan]{strategy}[/cyan]...")
    result = BacktestLauncher(memory_store=memory_store).run(bt_config, timeout_seconds=timeout)

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


def render_analysis_report(report: AnalysisReport) -> str:
    """Format an `AnalysisReport` as the plain-text report `hermes analyze` prints.

    Pure formatting only — every number comes straight from `report`,
    which itself only *reads* history (see `hermes.analyzer.analyze`);
    this function computes nothing and touches no database. Kept
    outside `hermes.analyzer` so that module stays untouched by this
    command's presentation choices (e.g. rendering findings with the
    tag on its own line, unlike `Finding.render()`'s single-line form).

    Note on units: `EXPECTANCY` and `MAX DRAWDOWN` are shown in the
    account's stake currency (e.g. USDC), not as a percentage — that's
    how `hermes.analyzer` computes them (see its docstrings). Only
    `WIN RATE` is a genuine percentage.
    """
    lines = ["[HERMES][ANALYSIS]", ""]
    lines.append(f"TRADES: {report.trade_count}")
    lines.append(f"WIN RATE: {report.win_rate * 100:.1f}%")
    lines.append(f"EXPECTANCY: {report.expectancy:+.2f}")
    lines.append(f"MAX DRAWDOWN: {report.max_drawdown:.2f}")
    lines.append("")

    if report.findings:
        for finding in report.findings:
            lines.append("[OBSERVATION]")
            lines.append(finding.observation)
            lines.append("")
            if finding.hypothesis:
                lines.append("[HYPOTHESIS]")
                lines.append(finding.hypothesis)
                lines.append("")
    else:
        lines.append("No notable patterns identified yet.")
        lines.append("")

    lines.append("STATUS: OBSERVATION ONLY")
    return "\n".join(lines)


def render_backtest_report(
    record: MemoryBacktestResult, report: BacktestReport
) -> str:
    """Format one recorded backtest's Freqtrade results as plain text.

    Pure formatting only — every number comes from `report`, which was
    parsed out of the stdout Freqtrade itself printed and Hermes
    already persisted (see :mod:`hermes.backtest_report`). Nothing here
    recomputes a metric, and nothing re-runs a backtest.

    Fields Freqtrade didn't print are rendered as ``N/A`` rather than
    guessed or omitted silently, so a reader can tell "not measured"
    apart from "measured as zero".
    """

    def show(value: object, suffix: str = "") -> str:
        return "N/A" if value is None else f"{value}{suffix}"

    lines = ["[HERMES][BACKTEST REPORT]", ""]
    lines.append(f"STRATEGY: {record.strategy}")
    lines.append(f"TIMERANGE: {record.timerange or 'N/A'}")
    if record.recorded_at is not None:
        lines.append(f"RECORDED AT: {record.recorded_at.isoformat()}")

    metrics = record.metrics or {}
    lines.append(f"EXIT CODE: {show(metrics.get('exit_code'))}")
    lines.append(f"SUCCEEDED: {show(metrics.get('succeeded'))}")
    lines.append(f"TIMEFRAME: {metrics.get('timeframe') or 'N/A'}")
    lines.append("")

    if not report.parsed_anything:
        lines.append(
            "No Freqtrade results tables were found in the recorded output."
        )
        lines.append(
            "This normally means the backtest produced no trades, or failed "
            "before printing results."
        )
        lines.append("")
        lines.append("STATUS: NO RESULTS RECORDED")
        return "\n".join(lines)

    win_rate = report.win_rate_pct
    lines.append(f"TRADES: {show(report.total_trades)}")
    lines.append(f"WINS: {show(report.wins)}")
    lines.append(f"LOSSES: {show(report.losses)}")
    lines.append(f"DRAWS: {show(report.draws)}")
    lines.append(
        f"WIN RATE: {'N/A' if win_rate is None else f'{win_rate:.1f}%'}"
    )
    lines.append(f"LONG TRADES: {show(report.long_trades)}")
    lines.append(f"SHORT TRADES: {show(report.short_trades)}")
    lines.append("")

    lines.append(f"PROFIT/LOSS: {show(report.profit_total_abs)}")
    lines.append(
        "PROFIT/LOSS %: "
        + (
            "N/A"
            if report.profit_total_pct is None
            else f"{report.profit_total_pct}%"
        )
    )
    lines.append(f"PROFIT FACTOR: {show(report.profit_factor)}")
    lines.append(f"MAX DRAWDOWN: {show(report.max_drawdown)}")
    lines.append(f"AVG TRADE DURATION: {show(report.avg_trade_duration)}")
    lines.append(f"STARTING BALANCE: {show(report.starting_balance)}")
    lines.append(f"FINAL BALANCE: {show(report.final_balance)}")
    lines.append("")

    if report.pair_results:
        lines.append("PER PAIR:")
        for pair_result in report.pair_results:
            pair_win_rate = (
                "N/A"
                if pair_result.win_rate_pct is None
                else f"{pair_result.win_rate_pct:.1f}%"
            )
            lines.append(
                f"  {pair_result.pair}: {pair_result.trades} trades, "
                f"{show(pair_result.profit_total_abs)} profit, "
                f"{pair_result.wins}W/{pair_result.draws}D/{pair_result.losses}L, "
                f"win rate {pair_win_rate}"
            )
        lines.append("")

    lines.append("STATUS: OBSERVATION ONLY")
    return "\n".join(lines)


_NO_BACKTEST_RECORDS_MESSAGE = (
    "[HERMES][BACKTEST REPORT]\n"
    "No backtest results have been recorded yet.\n\n"
    "Hermes records a result every time a backtest runs through\n"
    "`hermes backtest`. If a backtest did run but nothing is recorded here,\n"
    "the history database it was written to is not the one being read now\n"
    "(for example, a container with its own short-lived filesystem)."
)


_NO_STDOUT_RECORDED_MESSAGE = (
    "[HERMES][BACKTEST REPORT]\n"
    "The recorded backtest has no captured output to report on.\n\n"
    "This record predates output capture, or stored something unreadable."
)


_INSUFFICIENT_DATA_MESSAGE = (
    "[HERMES][ANALYSIS]\nInsufficient historical data for meaningful analysis."
)


def _project_directory_not_detected_message(user_data_dir: Path) -> str:
    return (
        "[HERMES][ERROR]\n"
        "Hermes project directory not detected.\n\n"
        f'Could not find a "{user_data_dir}" folder here.\n'
        "Run this command from the project's root folder (the one that\n"
        "contains the user_data folder). For example:\n\n"
        "    cd path/to/FF\n"
        "    python -m hermes analyze"
    )


def _database_unreadable_message(db_path: Path) -> str:
    return (
        "[HERMES][ERROR]\n"
        "Could not read the Hermes trading history database.\n\n"
        f"The file at {db_path} may be corrupted or in an unexpected format.\n"
        "If this keeps happening, rename or delete that file and Hermes will\n"
        "start a fresh one the next time a trade, backtest, or restart is recorded."
    )


@cli.command()
@click.option(
    "--user-data-dir",
    default=None,
    envvar=USER_DATA_DIR_ENV_VAR,
    show_envvar=True,
    type=click.Path(path_type=Path),
    help=(
        "Directory holding hermes_memory.sqlite3 (the same history trades are "
        f"recorded to). Defaults to {DEFAULT_USER_DATA_DIR}; set "
        f"{USER_DATA_DIR_ENV_VAR} to point every Hermes command at persistent "
        "storage instead (see RAILWAY.md)."
    ),
)
def analyze(user_data_dir: Path | None) -> None:
    """Analyze recorded trade history and print a plain-language report.

    Read-only: this command only reads from the existing Hermes SQLite
    history (`hermes.memory.MemoryStore`) and runs the existing
    analyzer (`hermes.analyzer.analyze`). It never writes to that
    history, never changes strategy parameters or trading decisions,
    and never deploys, optimizes, or retrains anything.
    """
    db_path = memory_db_path(user_data_dir)
    if not db_path.parent.is_dir():
        click.echo(_project_directory_not_detected_message(db_path.parent))
        sys.exit(1)

    try:
        store = MemoryStore(db_path)
    except Exception as exc:
        # A single log line, not a full traceback: this is an expected,
        # handled outcome the user gets a plain-language message for
        # below, not an unhandled crash worth alarming a terminal
        # beginner with a stack trace.
        logger.error("[HERMES][MEMORY][ERROR] Failed to open Hermes memory database at %s: %s", db_path, exc)
        click.echo(_database_unreadable_message(db_path))
        sys.exit(1)

    report = analyze_history(store)

    if report.trade_count == 0:
        click.echo(_INSUFFICIENT_DATA_MESSAGE)
        return

    click.echo(render_analysis_report(report))


@cli.command(name="backtest-report")
@click.option(
    "--user-data-dir",
    default=None,
    envvar=USER_DATA_DIR_ENV_VAR,
    show_envvar=True,
    type=click.Path(path_type=Path),
    help=(
        "Directory holding hermes_memory.sqlite3 (the same history backtests are "
        f"recorded to). Defaults to {DEFAULT_USER_DATA_DIR}; set "
        f"{USER_DATA_DIR_ENV_VAR} to point every Hermes command at persistent "
        "storage instead (see RAILWAY.md)."
    ),
)
@click.option(
    "--strategy",
    default=None,
    help="Only consider recorded backtests for this strategy.",
)
@click.option(
    "--search-limit",
    default=50,
    show_default=True,
    type=int,
    help="How many of the most recent recorded backtests to search through.",
)
def backtest_report(
    user_data_dir: Path | None, strategy: str | None, search_limit: int
) -> None:
    """Report the results of the most recent *already recorded* backtest.

    Strictly read-only: this command never launches a backtest, never
    starts a subprocess, and never writes to Hermes' history. It reads
    the stdout that `hermes backtest` already captured and persisted,
    and reports the numbers Freqtrade itself printed — closing the gap
    where a completed backtest's results were recorded but never
    surfaced.
    """
    db_path = memory_db_path(user_data_dir)
    if not db_path.parent.is_dir():
        click.echo(_project_directory_not_detected_message(db_path.parent))
        sys.exit(1)

    try:
        store = MemoryStore(db_path)
    except Exception as exc:
        logger.error(
            "[HERMES][MEMORY][ERROR] Failed to open Hermes memory database at %s: %s",
            db_path,
            exc,
        )
        click.echo(_database_unreadable_message(db_path))
        sys.exit(1)

    records = store.get_backtest_results(limit=search_limit)
    if strategy is not None:
        records = [record for record in records if record.strategy == strategy]

    if not records:
        click.echo(_NO_BACKTEST_RECORDS_MESSAGE)
        sys.exit(1)

    # get_backtest_results returns oldest-first; the most recent run is last.
    record = records[-1]

    stdout = extract_stdout(record.metrics)
    if stdout is None:
        click.echo(_NO_STDOUT_RECORDED_MESSAGE)
        sys.exit(1)

    click.echo(render_backtest_report(record, parse_backtest_stdout(stdout)))


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
