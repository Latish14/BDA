"""
Logging utilities for the pipeline.
"""
import logging
import sys
import time
from contextlib import contextmanager


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a standard format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Ensure utf-8 encoding on Windows console if supported
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s  %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def timer(label: str, logger: logging.Logger = None):
    """Context manager that logs elapsed time for a labelled block."""
    _log = logger or get_logger("timer")
    start = time.time()
    _log.info(">> Starting: %s", label)
    try:
        yield
    finally:
        elapsed = time.time() - start
        _log.info("[OK] Finished: %s (%.2f s)", label, elapsed)

