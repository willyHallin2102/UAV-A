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

from enum import Enum
from queue import Queue
from threading import Lock
from contextlib import contextmanager
from pathlib import Path
from typing import Union
import sys, time, atexit

from logs.formatters import JsonFormatter, ConsoleFormatter



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


def get_loglevel(name: str) -> LogLevel:
    try: return LogLevel[name.upper()]
    except KeyError: raise ValueError(f"Invalid log level: {name}")


# ---------------========== Logger Implementation ==========--------------- #

class Logger:
    __slots__ = (
        "name", "level", "json_format", "use_console", "to_disk",
        "max_bytes", "backup_count", "directory", "logger",
    )

    _instances, _instances_lock = {}, Lock()
    _queue, _listener = None, None
    _handlers = []
    _initialized, _shutdown = False, False
    _init_lock = Lock()


    def __init__(self,
        name: str, level=LogLevel.INFO,
        json_format=True, use_console=True, to_disk=True,
        max_bytes=10 * 1024 * 1024, backup_count=5
    ):
        """
            Initialize Logger Instance
        """
        if Logger._shutdown:
            raise RuntimeError("Logger system already shut down")

        # Create the Directory
        self.directory = Path(__file__).parent / "logs"
        self.directory.mkdir(parents=True, exist_ok=True)

        self.name = name
        self.level = level.value if isinstance(level, LogLevel) else int(level)

        # self.json, self.console, self.disk = json_format, use_console, to_disk
        self.json_format = json_format
        self.use_console = use_console
        self.to_disk = to_disk

        self.max_bytes, self.backup_count = max_bytes, backup_count

        # Initiate the Logger Instance
        self.logger = logging.getLogger(name=self.name)
        self.logger.setLevel(level=self.level)
        self.logger.propagate = False

        # Attach Handlers
        self._setup_handlers()

        with Logger._instances_lock: Logger._instances[name] = self
    

    # ---------------========== Handler Setup ==========--------------- #

    def _setup_handlers(self):
        """
        Configure the logging handlers for this instance.

        It removes any existing handlers to prevent duplicate logging, then
        attaches either a direct console handler (for synchronous logging)
        or a QueueHandler that feeds records into the shared asynchronous
        logging infrastructure. 
        """
        # Remove old handlers avoiding duplicate messages (unnecessary)??
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            try: handler.close()
            except: pass

        # Synchronous logging (console-only)
        if not self.to_disk:
            if self.use_console:
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(ConsoleFormatter())
                self.logger.addHandler(handler)
            return

        # Ensure async infra exists
        Logger._ensure_infrastructure(self)

        # Queue handler only once
        if not any(isinstance(h, QueueHandler) for h in self.logger.handlers):
            self.logger.addHandler(QueueHandler(Logger._queue))
    
    
    @classmethod
    def _ensure_infrastructure(cls, reference):
        """
        Initialize the shared asynchronous logging infrastructure.

        Creates the global log queue, configures file/console handlers, 
        and starts the background QueueListener thread. This method is 
        safe to call multiple times and will only initialize the system once.
        """
        if cls._initialized:
            return

        with cls._init_lock:
            if cls._initialized: return

            cls._queue = Queue()
            handlers = []

            # File handler (JSON if enabled)
            if reference.to_disk:
                logfile = reference.directory / "app.log"
                fh = RotatingFileHandler(
                    filename=logfile, maxBytes=reference.max_bytes,
                    backupCount=reference.backup_count, encoding="utf-8",
                )
                fh.setFormatter(
                    JsonFormatter() if reference.json_format else logging.Formatter()
                )
                handlers.append(fh)

            # Console handler
            if reference.use_console:
                ch = logging.StreamHandler(sys.stdout)
                ch.setFormatter(ConsoleFormatter())
                handlers.append(ch)

            # Start listener
            cls._handler_list = handlers
            cls._listener = QueueListener(cls._queue, *handlers)
            cls._listener.start()

            atexit.register(cls.shutdown_all)
            cls._initialized = True
    

    # ---------------========== API Methods ==========--------------- #

    def log(self, level: Union[LogLevel, int], msg, **extra):
        """
        Log a message at the specified log level.

        Args:
        -----
        level : Logging severity for the message.
        msg : Message to log.
        **extra : Additional structured metadata to include in the log record.
        """
        lvl = level.value if isinstance(level, LogLevel) else int(level)
        try: self.logger.log(lvl, msg, extra=extra)
        except Exception: sys.stderr.write(f"[Logger-Error] {msg}\n")

    def debug(self, msg, **e)       : self.log(LogLevel.DEBUG, msg, **e)
    def info(self, msg, **e)        : self.log(LogLevel.INFO, msg, **e)
    def warning(self, msg, **e)     : self.log(LogLevel.WARNING, msg, **e)
    def error(self, msg, **e)       : self.log(LogLevel.ERROR, msg, **e)
    def critical(self, msg, **e)    : self.log(LogLevel.CRITICAL, msg, **e)
    
    def exception(self, msg, **extra):
        self.logger.exception(msg, extra=extra, exc_info=True)
    
    # Exception Timing -- NOTE:: This return message under specified loglevel,
    # NOTE: Could be confusing if assign WARNING, ERROR or CRITICAL Looks weird but works.
    @contextmanager
    def time_block(self, label: str="exe-time", level: LogLevel=LogLevel.INFO):
        """
        Context manager for measuring execution duration of a block of code.

        Parameters
        ----------
        label : str, optional
            Text label included in the timing log message.
        level : LogLevel, optional
            Log level used for reporting the timing result.

        Logs the elapsed time when the context exits.
        """
        start = time.perf_counter()
        try: yield
        finally:
            dt = time.perf_counter() - start
            self.log(level, f"{label}: {dt:.6f} seconds")
    

    # ---------------========== Utilities ==========--------------- #

    @classmethod
    def get_logger(cls, name: str, **kwargs):
        """
        Retrieve an existing logger instance by name, or create one 
        if it does not exist. Ensures that loggers are reused rather 
        than created repeatedly across the application.
        """

        with cls._instances_lock:
            if name not in cls._instances:
                cls._instances[name] = Logger(name, **kwargs)
            return cls._instances[name]


    @classmethod
    def shutdown_all(cls):
        """
        Shut down the logging system and close all active handlers.

        Stops the background listener thread, closes global handlers, 
        removes handlers from all logger instances, and calls 
        `logging.shutdown()`. This is automatically registered to 
        run at interpreter exit.
        """

        if cls._shutdown: return
        cls._shutdown = True

        if cls._listener:
            try: cls._listener.stop()
            except: pass
        
        for handler in cls._handlers:
            try: handler.close()
            except: pass

        for instance in cls._instances.values():
            for handler in list(instance.logger.handlers):
                instance.logger.removeHandler(handler)

                try: handler.close()
                except: pass
        
        logging.shutdown()
