import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structured JSON logging for all of Tidus.

    In development, logs are pretty-printed with colours.
    In production, logs are emitted as JSON for log aggregators.

    Example:
        configure_logging("DEBUG")
        log = structlog.get_logger("tidus.router")
        log.info("model_selected", model_id="claude-haiku-4-5", cost_usd=0.0012)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer()
        if level.upper() == "DEBUG"
        else structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)
