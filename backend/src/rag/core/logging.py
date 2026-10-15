"""Console logging with rich. One logger per module: `log = get_logger(__name__)`."""

import logging

from rich.logging import RichHandler

_configured = False


def setup_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level="WARNING",  # third-party libraries: warnings only
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("rag").setLevel(level)  # our own code
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
