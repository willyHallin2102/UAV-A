"""
    logs/logger.py
    --------------
    Implements a logger instance object which handles feedback, logging results, 
    console console during runtime. It support multi-thread in concurrent which 
    could cause unfortunate errors, although supports asynchronous logging 
    through a shared queue and  listener, ensuring minimal blocking and improved 
    performance in concurrent applications.
"""
from __future__ import annotations

import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

import sys, os, time, uuid
import orjson
import atexit

from queue import Queue
from pathlib import Path
from datetime import datetime
from threading import Lock
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union
from enum import Enum



# ---------------========== Log Level Enum ==========--------------- #

class LogLevel(Enum):
    """
    Defines the standard logging severity levels used across the system. This
    enumeration is defined through the integer value of severity level defined 
    in `logging`.
    Levels (in increasing order of severity):
    -----------------------------------------
        - DEBUG:    Detailed diagnostics information, used for testing.
        - INFO: General operation events that confirm the application is 
                working as intended.
        - WARNING:  Indicating potential issues or problems to potential 
                    unexpected behaviour.
        - ERROR:    Serious issues that prevent parts of the application

            Serious issue that prevented part of the application from performing
            a function. Requires investigation, but the system may still continue.
        - CRITICAL:
            Severe error that may cause the entire application or system to fail.
            Immediate attention is typically required.
    """
    DEBUG       : int = logging.DEBUG
    INFO        : int = logging.INFO
    WARNING     : int = logging.WARNING
    ERROR       : int = logging.ERROR
    CRITICAL    : int = logging.CRITICAL



# ---------------========== JSON Formatter ==========--------------- #

class JsonFormatter(logging.Formatter):
    """
    Class manages the appearance and layout of the stored JSON files, what
    is being stored and in what order. Storage is conducted by the `orjson`
    using binary for faster load of json files.
    """
    __slots__ = ()
    _pid = os.getpid()
    def format(self, record: logging.LogRecord) -> str:
        """
        This define the layout of the layout of the stored `to_disk` and in which 
        order the parameters in and what each label is holding for value.
        """
        try:
            payload = {
                "timestamp": datetime.utcfromtimestamp(record.created)
                                .isoformat(timespec="milliseconds") + "Z",
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "pid": self._pid,
                "thread": record.threadName,
            }

            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            return orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE).decode()

        except Exception as e:
            return f'{{"level":"ERROR","message":"Format fail: {e}"}}\n'


# ---------------========== Console Formatter ==========--------------- #

class ConsoleFormatter(logging.Formatter):
    """
    Define a custom visual representation of the data in the console during
    runtime, various ANI defined colour depending on which severity logging 
    is called as well as a custom alignment.
    """
    __slots__ = ()
    COLORS = {
        "DEBUG"     : "\033[96m",
        "INFO"      : "\033[94m",
        "WARNING"   : "\033[93m",
        "ERROR"     : "\033[91m",
        "CRITICAL"  : "\033[1;41m",
        "RESET"     : "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Defines the alignment and assign the appropriate ANSI color defined 
        to this severity the logger has.
        """
        c = self.COLORS.get(record.levelname, "")
        r = self.COLORS["RESET"]
        return f"{c}{record.levelname:<8}{r} | {record.name:<18} | {record.getMessage()}"



# ===================== Logger Class ===================== #

class Logger:
    __slots__ = (
        "name", "level", "json_format", "use_console", "to_disk",
        "max_bytes", "backup_count", "directory", "logger"
    )

    # Shared across *all* Logger instances
    _instances: Dict[str, "Logger"] = {}
    _instances_lock = Lock()
    _shared_queue: Optional[Queue] = None
    _queue_listener: Optional[QueueListener] = None
    _shared_handlers: list[logging.Handler] = []
    _initialized = False
    _is_shutdown = False
    _init_lock = Lock() # testing

    def __init__(self,
        name: str, level: Union[int, LogLevel]=LogLevel.INFO,
        json_format: bool=True, use_console: bool=True, to_disk: bool=True,
        max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5,
        directory: Optional[Path] = None,
    ):

        if Logger._is_shutdown:
            raise RuntimeError("Logger system already shut down")

        self.name = name
        self.level = level.value if isinstance(level, LogLevel) else int(level)

        self.json_format = json_format
        self.use_console = use_console
        self.to_disk = to_disk
        
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        self.directory = directory or (Path(__file__).parent / "logs")
        self.directory.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        self.logger.propagate = False

        self._setup_handlers()

        # New addition
        with Logger._instances_lock:
            Logger._instances[name] = self


# ---------------========== Handler Setup ==========--------------- #

    def _setup_handlers(self):
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            try: handler.close()
            except: pass

        if not self.to_disk:
            # Direct console mode (sync)
            if self.use_console:
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setFormatter(ConsoleFormatter())
                self.logger.addHandler(stream_handler)
            return

        # Async mode
        Logger._ensure_shared_infra(self)
        self.logger.addHandler(QueueHandler(Logger._shared_queue))


    @classmethod
    def _ensure_shared_infra(cls, reference: "Logger"):
        """Ensure shared queue + listener exist. Thread-safe."""
        if cls._initialized: return

        with cls._init_lock:
            if cls._initialized: return

            cls._shared_queue = Queue(-1)
            handlers = []

            # File handler
            if reference.to_disk:
                logfile = reference.directory / f"app-{datetime.now():%Y-%m-%d}.log"
                filehandler = RotatingFileHandler(
                    logfile, maxBytes=reference.max_bytes,
                    backupCount=reference.backup_count, encoding="utf-8",
                )
                filehandler.setFormatter(JsonFormatter() if reference.json_format else logging.Formatter())
                handlers.append(filehandler)

            # Console handler
            if reference.use_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(ConsoleFormatter())
                handlers.append(console_handler)

            # Create listener
            cls._queue_listener = QueueListener(cls._shared_queue, *handlers)
            cls._shared_handlers = handlers
            cls._queue_listener.start()

            cls._initialized = True
            atexit.register(cls.shutdown_all)


    # ---------------========== Logger API Methods ==========--------------- #

    def log(self, level: Union[int, LogLevel], msg: str, **extra: Any):
        lvl = level.value if isinstance(level, LogLevel) else int(level)
        try: self.logger.log(lvl, msg, extra=extra)
        except Exception: sys.stderr.write(f"[LoggerError] {msg}\n")

    def debug(self, msg, **extra):    self.log(LogLevel.DEBUG, msg, **extra)
    def info(self, msg, **extra):     self.log(LogLevel.INFO, msg, **extra)
    def warning(self, msg, **extra):  self.log(LogLevel.WARNING, msg, **extra)
    def error(self, msg, **extra):    self.log(LogLevel.ERROR, msg, **extra)
    def critical(self, msg, **extra): self.log(LogLevel.CRITICAL, msg, **extra)
    def exception(self, msg, **extra):
        self.logger.exception(msg, extra=extra, exc_info=True)


    # ---------------========== Timing ==========--------------- #

    @contextmanager
    def time_block(self, label="Execution time", level=LogLevel.INFO):
        start = time.perf_counter()
        try: yield
        finally:
            dt = time.perf_counter() - start
            self.log(level, f"{label}: {dt:.6f}s")


    # ---------------========== Class Utilities ==========--------------- #

    @classmethod
    def get_logger(cls, name: str, **kwargs) -> "Logger":
        with cls._instances_lock:
            if name not in cls._instances:
                cls._instances[name] = Logger(name, **kwargs)
            return cls._instances[name]

    @classmethod
    def shutdown_all(cls):
        if cls._is_shutdown:
            return
        cls._is_shutdown = True

        # Stop queue listener
        if cls._queue_listener:
            try: cls._queue_listener.stop()
            except: pass

        # Close shared handlers
        for h in cls._shared_handlers:
            try: h.close()
            except: pass

        # Close instance handlers
        for instance in cls._instances.values():
            for h in list(instance.logger.handlers):
                instance.logger.removeHandler(h)
                try: h.close()
                except: pass

        cls._instances.clear()
        cls._shared_handlers.clear()
        cls._shared_queue = None
        cls._queue_listener = None

        logging.shutdown()


# ---------------========== Helper functions ==========--------------- #

def get_loglevel(name: str) -> LogLevel:
    try: return LogLevel[name.upper()]
    except KeyError:
        raise ValueError(f"Invalid log level: {name}")

