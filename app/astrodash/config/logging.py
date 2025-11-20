import logging
import logging.config
from logging.handlers import RotatingFileHandler
import os
import json
from typing import Optional
from astrodash.config.settings import get_settings, Settings

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with consistent naming convention.

    Args:
        name: Logger name. If None, uses the calling module's name.

    Returns:
        Configured logger instance.
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')

    return logging.getLogger(name)

def init_logging(config: Optional[Settings] = None) -> None:
    """
    Initialize logging configuration for the application.

    Args:
        config: Settings object containing logging configuration. If None, uses default settings.

    Raises:
        OSError: If log directory cannot be created or is not writable.
        ValueError: If logging configuration is invalid.
    """
    try:
        config = config or get_settings()
        LOG_DIR = config.log_dir
        LOG_FILE = os.path.join(LOG_DIR, "app.log")

        # Ensure log directory exists
        os.makedirs(LOG_DIR, exist_ok=True)

        # Verify log directory is writable
        if not os.access(LOG_DIR, os.W_OK):
            raise OSError(f"Log directory {LOG_DIR} is not writable")

        LOGGING_CONFIG = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "json": {
                    "()": "astrodash.config.logging.JsonFormatter"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": config.log_level,
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "json",
                    "filename": LOG_FILE,
                    "maxBytes": 10 * 1024 * 1024,  # 10MB
                    "backupCount": 5,
                    "level": config.log_level,
                },
            },
            "root": {
                "handlers": ["console", "file"],
                "level": config.log_level,
            },
            "loggers": {
                # Configure specific loggers for better control
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "uvicorn.access": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "fastapi": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
            }
        }

        logging.config.dictConfig(LOGGING_CONFIG)

        # Log successful initialization
        logger = get_logger(__name__)
        logger.info(f"Logging initialized successfully. Log level: {config.log_level}, Log file: {LOG_FILE}")

    except Exception as e:
        # Fallback to basic logging if configuration fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(),
            ]
        )
        logging.error(f"Failed to initialize logging configuration: {e}")
        raise

class JsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False)
