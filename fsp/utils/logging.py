"""
Central logging configuration for the FSP pipeline.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

from loguru import logger

from fsp.utils.paths import LOG_DIR

LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}"
_LEVELS_TO_STDOUT = {"TRACE", "DEBUG", "INFO", "SUCCESS"}
_PROGRESS_LINE_RE = re.compile(r"^\s*(Transcribing|Sample:)\s")


def _normalize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", label.strip())
    return cleaned.strip("-") or "run"


def build_run_label(*parts: str | None) -> str:
    tokens = [part.strip() for part in parts if part and part.strip()]
    return _normalize_label("-".join(tokens) if tokens else "run")


def build_run_log_dir(run_label: str, root: Path | None = None) -> Path:
    base_dir = root or (LOG_DIR / "logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = base_dir / f"{_normalize_label(run_label)}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def write_captured_output(log_path: Path, streams: Iterable[tuple[str, str]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        for name, content in streams:
            handle.write(f"===== {name} =====\n")
            text = sanitize_captured_output(content).strip()
            if text:
                handle.write(text)
                handle.write("\n")
            else:
                handle.write("<empty>\n")


def sanitize_captured_output(content: str) -> str:
    normalized_lines: list[str] = []
    for raw_line in content.replace("\r", "\n").splitlines():
        if _PROGRESS_LINE_RE.match(raw_line):
            continue
        normalized_lines.append(raw_line)
    return "\n".join(normalized_lines)


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    *,
    log_level: str = "INFO",
    run_label: str | None = None,
    log_dir: Path | None = None,
) -> Path | None:
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    logger.remove()
    level = log_level.upper()

    logger.add(
        sys.stdout,
        level=level,
        format=LOG_FORMAT,
        filter=lambda record: record["level"].name in _LEVELS_TO_STDOUT,
        colorize=False,
    )
    logger.add(
        sys.stderr,
        level=level,
        format=LOG_FORMAT,
        filter=lambda record: record["level"].no >= logging.WARNING,
        colorize=False,
    )

    app_log_path: Path | None = None
    if run_label:
        target_dir = log_dir or build_run_log_dir(run_label)
        app_log_path = target_dir / "pipeline.log"
        logger.add(
            app_log_path,
            level=level,
            format=LOG_FORMAT,
            colorize=False,
        )

    intercept = InterceptHandler()
    logging.captureWarnings(True)
    root_logger = logging.getLogger()
    root_logger.handlers = [intercept]
    root_logger.setLevel(logging.NOTSET)

    for noisy_logger in ("py.warnings", "transformers", "nemo", "hydra", "urllib3"):
        noisy = logging.getLogger(noisy_logger)
        noisy.handlers = [intercept]
        noisy.propagate = False

    return app_log_path


__all__ = [
    "build_run_label",
    "build_run_log_dir",
    "sanitize_captured_output",
    "setup_logging",
    "write_captured_output",
]
