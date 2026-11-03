import os
import logging
import sys
import contextvars

# Global context var to hold the current user query
current_query = contextvars.ContextVar('current_query', default='')

# Reconfigure stdout/stderr once for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class QueryFilter(logging.Filter):
    def filter(self, record):
        query = current_query.get()
        if query:
            short_query = query[:50] + "..." if len(query) > 50 else query
            record.query = f"[Query: {short_query}] "
        else:
            record.query = ""
        return True


_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(query)s%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
_q_filter = QueryFilter()

# Shared console handler
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_formatter)
_console_handler.addFilter(_q_filter)

# Shared file handler
from core.config import DATA_DIR
_log_path = os.path.join(DATA_DIR, "fallback_errors.log")
try:
    _file_handler = logging.FileHandler(_log_path, encoding="utf-8", delay=True)
    _file_handler.setLevel(logging.WARNING)
    _file_handler.setFormatter(_formatter)
    _file_handler.addFilter(_q_filter)
except Exception:
    _file_handler = None


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(_console_handler)
        if _file_handler:
            logger.addHandler(_file_handler)

    return logger
