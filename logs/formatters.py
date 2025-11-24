"""
"""
from __future__ import annotations

import os
from datetime import datetime

import logging
from logging import LogRecord


# ---------------========== Base Formatter ==========--------------- #

class BaseFormatter(logging.Formatter):
    """
    Shared functionality...
    """
    def format_timestamp(self, created: float) -> str:
        """
        Return the datetime format each formatter should share
        for a consistent time representation.
        """
        return (
            datetime.utcfromtimestamp(created)
            .isoformat(timespec="milliseconds") + "Z"
        )



# ---------------========== Json Formatting ==========--------------- #

class JsonFormatter(BaseFormatter):
    __slots__, _pid = (), os.getpid()

    def format(self, record: LogRecord) -> str:
        payload = {
            "timestamp" : self.format_timestamp(record.created),
            "level"     : record.levelname,
            "name"      : record.name,
            "message"   : record.getMessage(),
            "module"    : record.module,
            "function"  : record.funcName,
            "line"      : record.lineno,
            "pid"       : self._pid,
            "thread"    : record.threadName,
        }

        # Include some Extra fields
        for key, value in record.__dict__.items():
            if key not in payload and k not in  (
                "args", "msg", "exc_info", "exc_text",
                "stack_info", "lineno", "levelno", "pathname",
            ): payload[key] = value
        
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        
        return orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE).decode()



# ---------------========== Console Formatter ==========--------------- #

class ConsoleFormatter(BaseFormatter):
    __slots__, COLORS = (), {
        "DEBUG"     : "\033[96m",
        "INFO"      : "\033[94m",
        "WARNING"   : "\033[93m",
        "ERROR"     : "\033[91m",
        "CRITICAL"  : "\033[1;41m",
        "RESET"     : "\033[0m",
    }

    def format(self, record: LogRecord) -> str:
        color = self.COLORS.get(record. levelname, "")
        reset = self.COLORS["RESET"]

        # Single line message for clean console
        message = record.getMessage().replace("\n", "\\n")
        
        return f"{color}{record.levelname:<8}{reset} | {record.name:<18} | {message}"
        
