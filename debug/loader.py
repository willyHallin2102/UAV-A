"""
    debug/loader.py
    ---------------
    Extended CLI tester for data loading and processing functionality
"""
from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd

from data.data_processing import DataProcessor
from data.file_handler import HandlerFactory, CsvHandler
from data.loader import DataLoader, shuffle_and_split

from logs.logger import Logger, LogLevel, get_loglevel
