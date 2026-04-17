"""Logging configuration for ToolBlind using rich."""

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

console = Console()

_configured = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure rich-based logging for the entire toolblind package."""
    global _configured
    logger = logging.getLogger("toolblind")

    if _configured:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(message)s", datefmt="[%X]")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    _configured = True
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a child logger under the toolblind namespace."""
    base = logging.getLogger("toolblind")
    if not _configured:
        setup_logging()
    if name:
        return base.getChild(name)
    return base
