"""Unit tests for hermes.export_paths."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from hermes.export_paths import (
    ExportPathError,
    default_export_directory,
    prepare_export_directory,
    validate_export_directory,
)


def test_default_export_directory_is_nested_under_root(tmp_path: Path) -> None:
    result = default_export_directory(tmp_path)
    assert result == tmp_path / "backtest_results"


def test_validate_creates_missing_directory(tmp_path: Path) -> None:
    target = tmp_path / "backtest_results"
    assert not target.exists()

    result = validate_export_directory(target, persistent_root=tmp_path)

    assert result == target.resolve()
    assert target.is_dir()


def test_validate_accepts_already_existing_directory(tmp_path: Path) -> None:
    target = tmp_path / "backtest_results"
    target.mkdir()

    result = validate_export_directory(target, persistent_root=tmp_path)

    assert result == target.resolve()


def test_validate_creates_nested_missing_parents(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "backtest_results"

    result = validate_export_directory(target, persistent_root=tmp_path)

    assert result.is_dir()


def test_validate_rejects_relative_path(tmp_path: Path) -> None:
    with pytest.raises(ExportPathError, match="absolute"):
        validate_export_directory(Path("relative/dir"), persistent_root=tmp_path)


def test_validate_rejects_directory_outside_persistent_root(tmp_path: Path) -> None:
    root = tmp_path / "volume"
    root.mkdir()
    outside = tmp_path / "ephemeral" / "backtest_results"

    with pytest.raises(ExportPathError, match="not inside"):
        validate_export_directory(outside, persistent_root=root)

    assert not outside.exists()


def test_validate_rejects_sibling_directory_that_merely_shares_a_prefix(tmp_path: Path) -> None:
    """`/volume-other` must not be treated as inside `/volume` just because
    the string "/volume" is a prefix of it."""
    root = tmp_path / "volume"
    root.mkdir()
    sibling = tmp_path / "volume-other" / "backtest_results"

    with pytest.raises(ExportPathError, match="not inside"):
        validate_export_directory(sibling, persistent_root=root)


def test_validate_accepts_persistent_root_itself(tmp_path: Path) -> None:
    result = validate_export_directory(tmp_path, persistent_root=tmp_path)
    assert result == tmp_path.resolve()


def test_validate_rejects_relative_persistent_root(tmp_path: Path) -> None:
    with pytest.raises(ExportPathError, match="persistent_root"):
        validate_export_directory(tmp_path / "x", persistent_root=Path("relative"))


def test_validate_probe_file_is_cleaned_up(tmp_path: Path) -> None:
    target = tmp_path / "backtest_results"

    validate_export_directory(target, persistent_root=tmp_path)

    leftover = list(target.glob(".hermes_export_writability_probe_*"))
    assert leftover == []


def test_validate_raises_when_directory_cannot_be_created(tmp_path: Path) -> None:
    # A file where a directory is expected can never be mkdir'd into.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    target = blocker / "backtest_results"

    with pytest.raises(ExportPathError, match="cannot create"):
        validate_export_directory(target, persistent_root=tmp_path)


def test_prepare_export_directory_uses_default_when_none_given(tmp_path: Path) -> None:
    result = prepare_export_directory(None, persistent_root=tmp_path)
    assert result == (tmp_path / "backtest_results").resolve()


def test_prepare_export_directory_honors_explicit_directory(tmp_path: Path) -> None:
    explicit = tmp_path / "custom_export_dir"
    result = prepare_export_directory(explicit, persistent_root=tmp_path)
    assert result == explicit.resolve()


def test_prepare_export_directory_still_rejects_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "volume"
    root.mkdir()
    outside = tmp_path / "elsewhere"

    with pytest.raises(ExportPathError):
        prepare_export_directory(outside, persistent_root=root)
