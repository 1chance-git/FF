"""Validated resolution of the directory Freqtrade's trade/signal export writes to.

The Railway Volume mounts at one specific path (currently
``/app/user_data/data``, via ``HERMES_USER_DATA_DIR``) and nothing outside
it survives a container restart -- which is exactly the failure mode that
made the original frozen baseline's trade-level data unrecoverable (see
``hermes/memory.py``'s module docstring for the same lesson learned about
``hermes_memory.sqlite3`` itself). This module applies that same lesson to
Freqtrade's own trade/signal export artifacts: rather than trust a caller
to pass a safe ``--export-filename``, it derives the export directory from
the *same* resolved, proven-persistent root the memory database already
lives in, and refuses to hand back (or create) any path that isn't
underneath it.

Design decisions
-----------------
* **The persistent root is "wherever the memory database's parent
  directory is," not a second, independently-configured path.** Hermes
  already has exactly one source of truth for "is this actually on the
  Volume" -- ``hermes.memory.memory_db_path()`` -- and it exists precisely
  because guessing at a plausible-looking path once caused a real
  production bug. Introducing a second, separately-configured "export
  root" would recreate that exact risk. This module takes the already
  resolved memory-database parent as its ``persistent_root`` parameter
  rather than reading `hermes.memory`'s env var itself, keeping it a pure
  function of its inputs and independently testable without monkeypatching
  environment variables.
* **Validation fails loudly and safely, never silently degrades to
  ephemeral storage.** A caller who asks for an export directory outside
  the persistent root, or a directory that can't be created/written to,
  gets an :class:`ExportPathError` -- not a plausible-looking path that
  turns out to vanish on the next container restart, and not a silent
  fallback to some other location the caller didn't ask for.
* **Writability is verified by actually creating and removing a probe
  file, not by inspecting permission bits.** `os.access` can lie in
  several real environments (root, ACLs, some container filesystems);
  actually writing a small temp file and deleting it is the only check
  that directly answers "can Freqtrade actually write here."
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

_PROBE_PREFIX = ".hermes_export_writability_probe_"


class ExportPathError(Exception):
    """Raised when a requested export directory is not safely persistent."""


def default_export_directory(persistent_root: Path) -> Path:
    """The export directory to use when a caller didn't name one.

    Nested directly under ``persistent_root`` (the memory database's own
    parent directory), so it inherits the same persistence guarantee with
    zero extra configuration.
    """
    return Path(persistent_root) / "backtest_results"


def validate_export_directory(path: Path, *, persistent_root: Path) -> Path:
    """Validate that ``path`` is a safe, writable directory inside ``persistent_root``.

    Creates ``path`` (and any missing parents) if it doesn't exist yet.
    Never creates or writes anything outside ``persistent_root``.

    Returns
    -------
    Path
        ``path``, resolved (symlinks/`.."` collapsed), once every check
        has passed.

    Raises
    ------
    ExportPathError
        If ``path`` is not absolute, does not resolve to a location
        inside ``persistent_root``, cannot be created, or cannot be
        proven writable. Raised *before* any directory is created or any
        file is written whenever the check can be made without doing so
        (the absolute-path and inside-root checks); the "can create" and
        "can write" checks necessarily attempt the operation they verify.
    """
    path = Path(path)
    root = Path(persistent_root)

    if not path.is_absolute():
        raise ExportPathError(f"export directory must be an absolute path, got: {path}")
    if not root.is_absolute():
        raise ExportPathError(f"persistent_root must be an absolute path, got: {root}")

    resolved_root = root.resolve()
    # Resolve path's parent, not path itself: path may not exist yet, and
    # `Path.resolve()` on a nonexistent path still works, but doing it via
    # the parent avoids any ambiguity about a not-yet-created final segment.
    resolved_path = (path if path.exists() else path.parent / path.name).resolve()

    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ExportPathError(
            f"export directory {path} is not inside the persistent root {root}; "
            "refusing to write anywhere that isn't proven to survive a "
            "container restart"
        ) from exc

    try:
        resolved_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportPathError(f"cannot create export directory {resolved_path}: {exc}") from exc

    probe = resolved_path / f"{_PROBE_PREFIX}{uuid.uuid4().hex}"
    try:
        probe.write_text("")
    except OSError as exc:
        raise ExportPathError(f"export directory {resolved_path} is not writable: {exc}") from exc
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass

    return resolved_path


def prepare_export_directory(
    export_directory: Path | None, *, persistent_root: Path
) -> Path:
    """Resolve and validate the export directory to use for a backtest run.

    Parameters
    ----------
    export_directory:
        Caller-requested directory, or ``None`` to use
        :func:`default_export_directory`.
    persistent_root:
        The proven-persistent directory (the memory database's parent)
        every export directory must live inside.
    """
    target = (
        Path(export_directory)
        if export_directory is not None
        else default_export_directory(persistent_root)
    )
    return validate_export_directory(target, persistent_root=persistent_root)
