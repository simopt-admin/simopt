"""Logging utilities."""

from collections.abc import Callable
from typing import Protocol

import structlog


class Logger(Protocol):
    """Protocol for a logger."""

    def debug(self, *args: object, **kwargs: object) -> None:
        """Log a debug event."""
        ...


class _NullLogger:
    """A logger that does nothing."""

    def __getattr__(self, name: str) -> Callable:
        return self.noop

    def noop(self, *args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        return


null_logger = structlog.wrap_logger(_NullLogger())
