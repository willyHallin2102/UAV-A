"""
    logs/formatters.py
"""
from __future__ import annotations

import os
import logging
import orjson
import traceback

from datetime import datetime, timezone
from logging import LogRecord

# ============================================================
#       Base Formatting
# ============================================================

class BaseFormatting(logging.Formatter):
    """
        Includes time stamp for consistency of additional 
        formatters to be added later
    """
    def format_timestamp(self, created:float) -> str:
        return (datetime
            .fromtimestamp(created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00.00", "Z"))
    
    def format_exception(self, exc_info):
        """Format exception traceback"""
        return traceback.format_exception(*exc_info) if exc_info else None


# ============================================================
#       JSON Formatting
# ============================================================

class JsonFormatter(BaseFormatting):
    """
    Json Formatter is the instance maintaining and define the 
    layout and structure of each stored .log file. This 
    includes the features that is to be included as well as in 
    what order these are written in.
    """
    STANDARD_ATTRIBUTES = frozenset(logging.makeLogRecord({}).__dict__)
    
    def format(self, record: LogRecord) -> str:
        payload = {
            "timestamp" : self.format_timestamp(record.created),
            "severity"  : record.levelname,
            "logger"    : record.name,
            "message"   : record.getMessage(),
            "module"    : record.module,
            "function"  : record.funcName,
            "line"      : record.lineno,
            "process"   : record.process,
            "thread"    : record.threadName,
        }

        # Add structured extra layers 
        for key, value in record.__dict__.items():
            if key not in self.STANDARD_ATTRIBUTES and key not in payload:
                payload[key] = value
        
        # Exception handling - FIXED HERE
        if record.exc_info:
            exception_type, exception_value, exception_traceback = record.exc_info
            payload["exception"] = {
                "type"      : exception_type.__name__,
                "message"   : str(exception_value),
                "traceback" : self.format_exception(record.exc_info)
            }

        return orjson.dumps(payload, option=(
            orjson.OPT_APPEND_NEWLINE | orjson.OPT_NON_STR_KEYS
        ), default=str,).decode()



# ============================================================
#       Console Formatting
# ============================================================

class ConsoleFormatter(BaseFormatting):
    """
    Console formatter is managing the layout of the real-time
    feedback messages in the console, specifically the message
    with a check of the message severity. Based on the
    debugging level of being presented at all, coloring used
    as preemptive purpose of level of severity.
    """
    COLORS = {
        "DEBUG"     : "\033[96m",
        "INFO"      : "\033[94m",
        "WARNING"   : "\033[93m",
        "ERROR"     : "\033[91m",
        "CRITICAL"  : "\033[1;41m",
        "RESET"     : "\033[0m",
    }

    def format(self, record: LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]

        name = record.name[:18].ljust(18)
        message = record.getMessage().replace("\n", "\\n")

        line = f"{record.levelname:<8} | {name} | {message}"
        return f"{color}{line}{reset}" if color else line
