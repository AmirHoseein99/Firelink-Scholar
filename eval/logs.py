"""Logging setup for the evaluation harness.

The CLI calls :func:`configure_logging` once; library code only creates
module-level loggers under the ``eval`` namespace. Logs go to stderr and,
optionally, to a file. Configuring the namespace (not the root logger) keeps
the harness from changing logging for callers that embed it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER_NAME = "eval"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def configure_logging(
    *,
    level: str | int | None = None,
    verbose: bool = False,
    quiet: bool = False,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure the ``eval`` logger namespace and return it.

    Precedence: explicit ``level``, then ``verbose`` (DEBUG), then ``quiet``
    (WARNING), then INFO. Reconfiguring replaces handlers installed by earlier
    calls, so ``main`` can be invoked more than once in one process.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if isinstance(level, str):
        level = level.upper()
    if level is None:
        level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug(
        "Logging at %s%s",
        logging.getLevelName(logger.level),
        f"; file {Path(log_file)}" if log_file is not None else "",
    )
    return logger
