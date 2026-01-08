"""
    logs/logger.py
    --------------
    Implementation of the `Logger` instance object which handles 
    feedback, logging results, console during runtime. It 
    support multi-thread in concurrent which could cause 
    unfortunate errors, although support asynchronous logging 
    through a shared queue and listener, ensuring minimal blocking 
    and improved performance in concurrent applications.
"""
from __future__ import annotations

import logging
import sys
import atexit
import time

from enum import Enum
from queue import Queue
from threading import Lock
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from logs.formatters import JsonFormatter, ConsoleFormatter

_REVERSED_ATTRIBUTES = frozenset(logging.makeLogRecord({}).__dict__)



# ============================================================
#       Log Level Enumeration
# ============================================================

class LogLevel(Enum):
    """  """
    DEBUG       = logging.DEBUG
    INFO        = logging.INFO
    WARNING     = logging.WARNING
    ERROR       = logging.ERROR
    CRITICAL    = logging.CRITICAL


def get_loglevel(level: str) -> LogLevel:
    try: return LogLevel[level.upper()]
    except KeyError: raise ValueError(f"Invalid loglevel: {level}")



# ============================================================
#       Logger Implementation
# ============================================================

class Logger:
    """
    Logger instance that manages logging the running of the 
    operations to store and display information as well as 
    for debugging purposes.
    """

    _instances: Dict[str, "Logger"] = {}
    _instances_lock = Lock()

    _queue: Optional[Queue] = None
    _listener: Optional[QueueListener] = None
    _handlers: List = []

    _initialized = False
    _init_lock = Lock()

    _log_directory: Path
    _use_console: bool
    _use_json: bool
    _to_disk: bool
    _max_bytes: int
    _backup_count: int

    @classmethod
    def configure(
        cls, *, log_directory: Union[Path,str],
        use_console: bool=True, use_json: bool=True, to_disk: bool=False,
        max_bytes: int=10*1024*1024, backup_count: int=5
    ):
        """
        Configure the Logger instance to how its operates and what
        to be used.
        """
        with cls._init_lock:
            if cls._initialized: return

            # FIX: Use the provided log_directory instead of hardcoding
            cls._log_directory = Path(log_directory)
            cls._log_directory.mkdir(parents=True, exist_ok=True)

            cls._queue, cls._handlers = Queue(), []
            
            # Only add file handler if to_disk is True
            if to_disk:
                fh = RotatingFileHandler(
                    cls._log_directory / "app.log", 
                    maxBytes=max_bytes, 
                    backupCount=backup_count, 
                    encoding="utf-8",
                )
                fh.setFormatter(JsonFormatter() if use_json else logging.Formatter())
                cls._handlers.append(fh)

            if use_console:
                ch = logging.StreamHandler(sys.stdout)
                ch.setFormatter(ConsoleFormatter())
                cls._handlers.append(ch)
            
            cls._listener = QueueListener(cls._queue, *cls._handlers, respect_handler_level=True)
            cls._listener.start()

            atexit.register(cls.shutdown)
            cls._initialized = True

    def __init__(self, name: str, level: LogLevel=LogLevel.INFO):
        """
            Initialize Logger Instance
        """
        # Initialize once, if not run configurations
        if not Logger._initialized: Logger.configure()
        assert Logger._queue is not None

        # Retrieve the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.value)
        self.logger.propagate = False

        if not any(isinstance(h, QueueHandler) for h in self.logger.handlers):
            self.logger.addHandler(QueueHandler(Logger._queue))


    # ============================================================
    #       API Methods Implementations
    # ============================================================

    def log(self, level: Optional[LogLevel,int], msg: str, **extra):
        lvl = level.value if isinstance(level, LogLevel) else int(level)
        for key in extra:
            if key in _REVERSED_ATTRIBUTES:
                raise KeyError(f"Illegal extra key:\t{key}")
        
        self.logger.log(lvl, msg, extra=extra)
    
    def debug(self, msg, **e): self.log(LogLevel.DEBUG, msg, **e)
    def info(self, msg, **e): self.log(LogLevel.INFO, msg, **e)
    def warning(self, msg, **e): self.log(LogLevel.WARNING, msg, **e)
    def error(self, msg, **e): self.log(LogLevel.ERROR, msg, **e)
    def critical(self, msg, **e): self.log(LogLevel.CRITICAL, msg, **e)

    def exception(self, msg, **extra): self.logger.exception(msg, extra=extra)


    # ============================================================
    #       Utilities Methods
    # ============================================================

    @contextmanager
    def timed(self, label: str="exec-time", level: LogLevel=LogLevel.INFO):
        start = time.perf_counter()
        try: yield
        finally:
            elapsed = time.perf_counter() - start
            self.log(level, f"{label}: {elapsed:.6f} sec")
    

    # ============================================================
    #       Retrieval and Destruction
    # ============================================================

    @classmethod
    def get_logger(cls, name: str, level: LogLevel=LogLevel.INFO) -> Logger:
        with cls._instances_lock:
            if name not in cls._instances: 
                cls._instances[name] = Logger(name, level)
            
            return cls._instances[name]
    
    @classmethod
    def shutdown(cls):
        with cls._init_lock:
            listener = cls._listener
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    # Never let logging shutdown crash the program
                    pass
                finally:
                    cls._listener = None

            for handler in cls._handlers:
                try:
                    handler.close()
                except Exception:
                    pass

            cls._handlers.clear()
            cls._queue = None
            cls._initialized = False

            logging.shutdown()
