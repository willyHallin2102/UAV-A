"""
    debug / clock.py
    ----------------
    Benchmarking utilities for timing the functions, this is including 
    statistics for averaging as well as warmup to get the program 
    initialize first to avoid time-waste
"""
from __future__ import annotations

import time
import numpy as np

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from contextlib import contextmanager



@dataclass
class BenchmarkStats:
    """ Statistics from a benchmark run """
    n_runs: int
    times: List[float]
    warmup_runs: int = 0


    @property
    def mean(self) -> float: return float(np.mean(self.times))

    @property
    def median(self) -> float: return float(np.median(self.times))

    @property
    def std(self) -> float: return float(np.std(self.times))

    @property
    def min(self) -> float: return float(np.min(self.times))

    @property
    def max(self) -> float: return float(np.max(self.times))

    @property
    def total(self) -> float: return float(np.sum(self.times))

    def to_dict(self) -> Dict[str,float]:
        """ Convert to dictionary """
        return {
            'n_runs'    : self.n_runs,
            'means_ms'  : self.mean * 1000,
            'median_ms' : self.median * 1000,
            'std_ms'    : self.std * 1000,
            'min_ms'    : self.min * 1000,
            'max_ms'    : self.max * 1000,
            'total_ms'  : self.total * 1000
        }
    
    def __str__(self) -> str:
        return (
            f"BenchmarkStats( n={self.n_runs}) | "
            f"mean = {self.mean * 1000:.3f} ms | "
            f"std = {self.std * 1000:.3f} ms | "
            f"min = {self.min * 1000:.3f} ms | "
            f"max = {self.max * 1000:.3f} ms )"
        )



class Timer:
    """ Simple timer context manager """

    def __init__(self, name: str="", clock=time.perf_counter):
        self.name = name
        self.clock = clock
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = self.clock()
        return self
    
    def __exit__(self, *args):
        self.elapsed = self.clock() - self.start_time
        if self.name: print(f"{self.name}: {self.elapsed*1000:.3f} ms")
    
    @property
    def time_ms(self) -> float:
        """ Get elapsed time in milliseconds """
        return self.elapsed * 1000 if self.elapsed else 0.0



class FunctionBenchmark:
    """Benchmark individual functions."""
    
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
    
    def time_once(self, f: Callable, *args, **kwargs) -> float:
        """Time a single execution."""
        start = self.clock()
        f(*args, **kwargs)
        return self.clock() - start
    
    def time_with_return(self, f: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Time a single execution and return result."""
        start = self.clock()
        result = f(*args, **kwargs)
        return result, self.clock() - start
    
    def benchmark(self,
        f: Callable, n_runs: int = 10, warmup_runs: int = 3, *args, **kwargs
    ) -> BenchmarkStats:
        """
        Benchmark a function with warmup runs.
        
        Args:
            f: Function to benchmark
            n_runs: Number of timed runs
            warmup_runs: Number of untimed warmup runs
            *args, **kwargs: Arguments to pass to function
        
        Returns:
            BenchmarkStats object with timing results
        """
        # Warmup runs
        for _ in range(warmup_runs):
            f(*args, **kwargs)
        
        # Timed runs
        times = []
        for _ in range(n_runs):
            start = self.clock()
            f(*args, **kwargs)
            times.append(self.clock() - start)
        
        return BenchmarkStats(n_runs=n_runs, times=times, warmup_runs=warmup_runs)
    
    def benchmark_with_return(self,
        f: Callable, n_runs: int = 10, warmup_runs: int = 3, *args, **kwargs
    ) -> Tuple[Any, BenchmarkStats]:
        """
        Benchmark a function and return the result from the last run.
        
        Returns:
            Tuple of (result_from_last_run, BenchmarkStats)
        """
        # Warmup runs
        result = None
        for _ in range(warmup_runs):
            result = f(*args, **kwargs)
        
        # Timed runs
        times = []
        for _ in range(n_runs):
            start = self.clock()
            result = f(*args, **kwargs)
            times.append(self.clock() - start)
        
        return result, BenchmarkStats(n_runs=n_runs, times=times, warmup_runs=warmup_runs)




def benchmark_comparison(
    functions: Dict[str, Callable], n_runs: int = 10,
    warmup_runs: int = 3, *args, **kwargs
) -> Dict[str, BenchmarkStats]:
    """
    """
    benchmarker = FunctionBenchmark()
    results = {}
    
    for name, func in functions.items():
        results[name] = benchmarker.benchmark(
            func, n_runs, warmup_runs, *args, **kwargs
        )
    return results


@contextmanager
def measure_time(name: str = ""):
    """Context manager for measuring code block execution time."""
    timer = Timer(name)
    with timer: yield timer




































# """
#     debug / clock.py
#     ----------------
# """
# import time

# from dataclasses import dataclass
# from typing import Any, Callable, List, Tuple



# @dataclass
# class BenchmarkResults:
#     runs: int
#     times: List[float]

#     @property
#     def mean(self) -> float:
#         return sum(self.times) / self.runs
    
#     @property
#     def min(self) -> float:
#         return min(self.times)
    
#     @property
#     def max(self) -> float:
#         return max(self.times)
    
#     @property
#     def std(self) -> float:
#         return (sum((t - self.mean) ** 2 for t in self.times) / self.runs) ** 2
    
#     @property
#     def stderr(self) -> float:
#         return self.std / (self.runs ** 0.5)



# class FunctionBenchmark:
#     def __init__(self, clock=time.perf_counter):
#         self.clock = clock
    
#     def run(self, f: Callable, *args, **kwargs) -> float:
#         start = self.clock()
#         f(*args, **kwargs)
#         return self.clock() - start
    
#     def benchmark(self,
#         f: Callable, runs: int=10, warmup: int=10,
#         *args, **kwargs
#     ) -> BenchmarkResults:
#         """ Runs benchmark with optional warmup """
#         # Warmup runs are not measured
#         if warmup > 20: warmup = 20
#         for _ in range(warmup):
#             f(*args, **kwargs)
        
#         times: float=[]
#         for _ in range(runs):
#             start = self.clock()
#             f(*args,**kwargs)
#             times.append(self.clock() - start)
        
#         return BenchmarkResults(runs=runs, times=times)
