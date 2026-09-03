import logging
import sys
import os
import contextvars

# Global context var to hold the current user query
current_query = contextvars.ContextVar('current_query', default='')

class QueryFilter(logging.Filter):
    def filter(self, record):
        query = current_query.get()
        if query:
            # truncate query if it's too long to keep logs clean
            short_query = query[:50] + "..." if len(query) > 50 else query
            record.query = f"[Query: {short_query}] "
        else:
            record.query = ""
        return True

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:  # avoid duplicate handlers if called multiple times
        logger.setLevel(logging.INFO)

        # Added full date (%Y-%m-%d) and %(query)s
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(query)s%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        q_filter = QueryFilter()
        
        # Console handler (INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(q_filter)
        logger.addHandler(console_handler)
        
        # File handler for fallbacks and errors (WARNING and above)
        file_handler = logging.FileHandler("fallback_errors.log", encoding="utf-8")
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(q_filter)
        logger.addHandler(file_handler)

    return logger