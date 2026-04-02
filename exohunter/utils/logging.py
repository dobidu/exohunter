"""Structured logging configuration for ExoHunter.

Provides a single ``get_logger`` factory that returns a standard-library
logger pre-configured with the project's format and level.  Every module
should obtain its logger via::

    from exohunter.utils.logging import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys

from exohunter import config


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance with a stream handler attached.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when get_logger is called more than
    # once for the same module (e.g. during interactive development).
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt=config.LOG_FORMAT,
            datefmt=config.LOG_DATE_FORMAT,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(config.LOG_LEVEL)
    return logger
