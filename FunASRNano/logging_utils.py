"""Project-specific logging setup wrappers."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from FunASRNano.logging_config import setup_logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RESET_DONE: set[Path] = set()


def default_log_file() -> Path:
    raw_stem = Path(sys.argv[0] or "").stem.strip()
    safe_stem = re.sub(r"[^0-9A-Za-z._-]+", "_", raw_stem).strip("._-")
    if not safe_stem:
        safe_stem = "app"
    return PROJECT_ROOT / "log" / f"{safe_stem}.log"


def setup_project_logger(
    log_level: int | str = logging.DEBUG,
    *,
    reset_log: bool = False,
    log_file: str | Path | None = None,
) -> logging.Logger:
    resolved_log_file = Path(log_file) if log_file is not None else default_log_file()
    resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
    resolved_key = resolved_log_file.resolve()

    if reset_log and resolved_key not in _RESET_DONE:
        resolved_log_file.write_text("", encoding="utf-8")
        _RESET_DONE.add(resolved_key)

    return setup_logger(log_level=log_level, log_file=str(resolved_log_file))
