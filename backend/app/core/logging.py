"""
Intelligent RAG System — Structured Logging Setup
==================================================
Uses structlog for structured, readable log output.
Structured logging means each log line is a key-value dictionary,
making it easy to filter and analyze in production tools.
"""

import logging
import structlog


def setup_logging(debug: bool = True) -> None:
    """
    Configure structlog for the application.
    
    In development (debug=True): pretty colored console output.
    In production (debug=False): JSON output for log aggregators.
    """
    log_level = logging.DEBUG if debug else logging.INFO

    # Configure standard Python logging first
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
    )

    # Structlog processors — each transforms the log event in sequence
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
    ]

    if debug:
        # Human-readable console output during development
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        # JSON output for production log collectors
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structlog logger."""
    return structlog.get_logger(name)
