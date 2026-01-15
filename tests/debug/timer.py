"""
"""
from __future__ import annotations

import time
import json

from dataclasses import dataclass, field
from statistics import mean, median, stdev
from contextlib import ContextDecorator
from typing import Any, Dict, List, Optional, Sequence


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


class Timer:
    """ Simple timer context manager """

    def __init__(self, name: str=""):
        self.name = name
        self.clock = time.perf_counter
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = self.clock()
        return self
    
    def __exit__(self, *args):
        self.elapsed = self.clock() - self.start_time
        if self.name: print(f"{self.name}: {self.elapsed *1000:.3f} ms")
    
    @property
    def time_ms(self) -> float:
        return self.elapsed * 1000 if self.elapsed else 0.0

