"""
    src / models / generators / genai.py
    ------------------------------------
    Abstract parent class of requirement for the generative artificial models
    for UAV trajectory
"""
from __future__ import annotations

import tensorflow as tf
tfk = tf.keras

from abc import ABC, abstractmethod
from pathlib import Path


class Genai(tfk.Model, ABC):
    """
    Abstract base class for all generative ai models. Provides a unified 
    interface (sample, save, load, etc)
    """
    pass

