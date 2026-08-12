"""Tests for where Hermes' SQLite memory database lives.

Infrastructure-only: these tests prove *path selection*, nothing about
trading. **No test here runs a backtest, hyperopt, or subprocess.** The
one test that exercises `hermes backtest`'s memory wiring stubs out
`BacktestLauncher` entirely, so no Freqtrade process is ever started.

Context: on Railway the persistent Volume mounts at
`/app/user_data/data`, while Hermes' default memory path resolves to
`/app/user_data/hermes_memory.sqlite3` — beside the mount, not inside
it — so a recorded backtest result vanished when the container exited.
The fix reuses the existing ``--user-data-dir`` abstraction and makes it
environment-configurable, rather than hard-coding a deployment-specific
path anywhere in Hermes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from hermes.cli import cli
from hermes.memory import (
    DEFAULT_USER_DATA_DIR,
    MEMORY_DB_FILENAME,
    USER_DATA_DIR_ENV_VAR,
    BacktestResult as MemoryBacktestResult,
    MemoryStore,
    default_user_data_dir,
    memory_db_path,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a real ambient value leak into these assertions."""
    monkeypatch.delenv(USER_DATA_DIR_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class TestDefaultUserDataDir:
    def test_local_default_is_unchanged(self) -> None:
        """The pre-existing local-development behavior must not shift."""
        assert default_user_data_dir() == Path("user_data")
        assert DEFAULT_USER_DATA_DIR == Path("user_data")

    def test_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(USER_DATA_DIR_ENV_VAR, "user_data/data")
        assert default_user_data_dir() == Path("user_data/data")

    def test_env_var_is_read_at_call_time_not_import_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert default_user_data_dir() == Path("user_data")
        monkeypatch.setenv(USER_DATA_DIR_ENV_VAR, "/mnt/volume")
        assert default_user_data_dir() == Path("/mnt/volume")

    def test_empty_env_var_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(USER_DATA_DIR_ENV_VAR, "")
        assert default_user_data_dir() == Path("user_data")


class TestMemoryDbPath:
    def test_default_matches_historical_local_path(self) -> None:
        assert memory_db_path() == Path("user_data/hermes_memory.sqlite3")
        assert memory_db_path(None) == Path("user_data/hermes_memory.sqlite3")

    def test_explicit_directory_is_used(self) -> None:
        assert memory_db_path(Path("/mnt/vol")) == Path("/mnt/vol") / MEMORY_DB_FILENAME

    def test_accepts_a_plain_string(self) -> None:
        assert memory_db_path("user_data/data") == Path(
            "user_data/data/hermes_memory.sqlite3"
        )

    def test_explicit_argument_beats_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(USER_DATA_DIR_ENV_VAR, "user_data/data")
        assert memory_db_path(Path("explicit")) == Path(
            "explicit/hermes_memory.sqlite3"
        )

    def test_env_var_places_database_inside_the_persistent_volume(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Railway case: the Volume mounts at /app/user_data/data, so the
        database must resolve to a path *inside* that directory."""
        monkeypatch.setenv(USER_DATA_DIR_ENV_VAR, "/app/user_data/data")
        resolved = memory_db_path()
        assert resolved == Path("/app/user_data/data/hermes_memory.sqlite3")
        assert Path("/app/user_data/data") in resolved.parents


# ---------------------------------------------------------------------------
# Every reader/writer resolves to the same file
# ---------------------------------------------------------------------------


class TestWriterAndReadersAgree:
    def test_backtest_launcher_and_readers_share_one_location(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`hermes backtest`'s writer must target the exact file that
        `hermes backtest-report` and `hermes analyze` later read.

        BacktestLauncher is stubbed out — no Freqtrade process is started.
        """
        persistent = tmp_path / "user_data" / "data"
        persistent.mkdir(parents=True)
        monkeypatch.setenv(USER_DATA_DIR_ENV_VAR, str(persistent))

        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        captured: dict[str, object] = {}

        class _StubLauncher:
            def __init__(self, memory_store=None):
                captured["db_path"] = memory_store.db_path

            def run(self, config, timeout_seconds=None):  # noqa: ARG002
                raise SystemExit(0)

        with patch("hermes.cli.BacktestLauncher", _StubLauncher):
            CliRunner().invoke(
                cli,
                [
                    "backtest",
                    "-c", str(config_file),
                    "--strategy", "TrendFollowCore",
                ],
            )

        expected = persistent / MEMORY_DB_FILENAME
        assert captured["db_path"] == expected

        # The readers must resolve to that same file.
        assert memory_db_path() == expected

    def test_record_written_under_env_var_is_readable_by_backtest_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end persistence contract: write to the configured location,
        then read it back through the CLI with only the env var set."""
        persistent = tmp_path / "data"
        persistent.mkdir()
        monkeypatch.setenv(USER_DATA_DIR_ENV_VAR, str(persistent))

        MemoryStore(memory_db_path()).record_backtest_result(
            MemoryBacktestResult(
                strategy="TrendFollowCore",
                timerange="20260115-20260811",
                metrics={"exit_code": 0, "succeeded": True, "stdout": "no tables here"},
            )
        )
        assert (persistent / MEMORY_DB_FILENAME).exists()

        result = CliRunner().invoke(cli, ["backtest-report"])
        assert result.exit_code == 0, result.output
        assert "TrendFollowCore" in result.output
        assert "20260115-20260811" in result.output

    def test_supervisor_resolves_the_same_way(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The supervisor writes process events to the same database, so it
        must not keep a private copy of the default."""
        monkeypatch.setenv(USER_DATA_DIR_ENV_VAR, "/app/user_data/data")
        # Mirrors hermes/supervisor.py's construction (args.user_data_dir is
        # None unless --user-data-dir was passed).
        assert memory_db_path(None) == Path(
            "/app/user_data/data/hermes_memory.sqlite3"
        )


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestCliWiring:
    @pytest.mark.parametrize("command", ["backtest", "analyze", "backtest-report"])
    def test_option_documents_the_env_var(self, command: str) -> None:
        result = CliRunner().invoke(cli, [command, "--help"])
        assert result.exit_code == 0
        assert USER_DATA_DIR_ENV_VAR in result.output

    def test_analyze_reads_the_env_var_configured_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        persistent = tmp_path / "data"
        persistent.mkdir()
        monkeypatch.setenv(USER_DATA_DIR_ENV_VAR, str(persistent))

        result = CliRunner().invoke(cli, ["analyze"])
        # No trades recorded, but it must have found the directory rather
        # than reporting the project directory as missing.
        assert result.exit_code == 0, result.output
        assert "Hermes project directory not detected" not in result.output

    def test_missing_configured_directory_is_reported_clearly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(USER_DATA_DIR_ENV_VAR, str(tmp_path / "does-not-exist"))
        result = CliRunner().invoke(cli, ["analyze"])
        assert result.exit_code == 1
        assert "Hermes project directory not detected" in result.output

    def test_explicit_flag_still_wins_over_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_dir = tmp_path / "from-env"
        env_dir.mkdir()
        flag_dir = tmp_path / "from-flag"
        flag_dir.mkdir()
        monkeypatch.setenv(USER_DATA_DIR_ENV_VAR, str(env_dir))

        MemoryStore(flag_dir / MEMORY_DB_FILENAME).record_backtest_result(
            MemoryBacktestResult(
                strategy="FromFlag", timerange="x", metrics={"stdout": "s"}
            )
        )

        result = CliRunner().invoke(
            cli, ["backtest-report", "--user-data-dir", str(flag_dir)]
        )
        assert result.exit_code == 0, result.output
        assert "FromFlag" in result.output
