"""
    Timer is not completed, unsure if its necessary to and add more statistical
    structure within the Timer class. Perhaps pass the function to measure under
    Callable and run it a number of times and compute statistics once last run 
    on the appended list in __exit__ with a print function to abstract the code
    from each of the test CLI scripts, having it as is for now, only slight 
    preparations have been added to later fulfill it.
"""
from __future__ import annotations

import time
import json
import functools

from dataclasses import dataclass, field
from statistics import mean, median, stdev
from contextlib import ContextDecorator
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union, overload
)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[...,Any])


@dataclass(frozen=True)
class TimingStats:
    """ Statistics for a collection of timing measurements """
    count   : int
    mean    : float
    median  : float
    min     : float
    max     : float
    stddev  : float
    total   : float
    p95     : float
    p99     : float
    times_ms: List[float] = field(repr=False)

    def __str__(self) -> str: return (
        f"TimingStats(count={self.count} | mean={self.mean:.3f} ms | "
        f"median={self.median:.3f} ms | min={self.min:.3f} ms | "
        f"max={self.max:.3f} ms | stddev={self.stddev:.3f} ms"
    )

    def to_dict(self) -> Dict[str,Any]: return {
        'count': self.count, 'mean_ms': self.mean, 'median_ms': self.median,
        'min_ms': self.min, 'max_ms': self.max, 'stddev_ms': self.stddev,
        'total_ms': self.total, 'p95_ms': self.p95, 'p99_ms': self.p99
    }

    def to_json(self) -> str: return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_times(cls, times_ms: List[float]) -> TimingStats:
        if not times_ms: return cls(0, 0, 0, 0, 0, 0, 0, 0, 0, [])
        
        sorted_times = sorted(times_ms)
        n = len(sorted_times)

        return cls(
            count = n,
            mean = mean(times_ms),
            median = median(times_ms),
            min = min(times_ms), max = max(times_ms),
            stddev = stddev(times_ms) if n > 1 else 0,
            total = sum(times_ms),
            p95 = sorted_times[int(0.95 * n)] if n >= 20 else sorted_times[-1],
            p99 = sorted_times[int(0.99 * n)] if n >= 100 else sorted_times[-1],
            times_ms=times_ms
        )



class Timer(ContextDecorator):
    """ Simple timer context manager """

    def __init__(self, name: str="", print_on_exit: bool=False):
        self.name = name
        self.print_on_exit = print_on_exit
        self.clock = time.perf_counter

        self.start_time : Optional[float]=None
        self.elapsed    : Optional[float]=None

        self.times: List[float] = []
    
    def __enter__(self):
        self.start_time = self.clock()
        return self
    
    def __exit__(self, *args):
        self.elapsed = self.clock() - self.start_time
        if self.name: print(f"{self.name}: {self.elapsed *1000:.3f} ms")
    
    @property
    def time_ms(self) -> float:
        return self.elapsed * 1000 if self.elapsed else 0.0

