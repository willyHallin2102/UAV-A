"""
    src/models.chanmod.py
    ---------------------
    Operates as a channel manager, that handles link state determination
    based on relative position of the UAV depending on the corresponding 
    antenna. It holds the link model manages the predictions of the link
    state, and a path model which based on the link state modelling the
    UAV trajectory using a generative model.
"""
from __future__ import annotations
import random

import numpy as np
import tensorflow as tf

from pathlib import Path
from src.config.data import DataConfig

from src.models.link import LinkStatePredictor
from src.models.path import PathModel

from logs.logger import LogLevel


# Add json_format, use_console, to_dist to the Channel Model
class ChannelModel:
    """
    Operates as channel model manager for the objects link state predictor, 
    and the path model. The link state model handles state prediction based 
    on position and environmental condition parameters. The Path model advantages
    the uav modelling trajectories.
    """
    def __init__(self,
        config: DataConfig=DataConfig(), model_type: str="vae",
        directory: Union[str, Path]="beijing", seed: int=42,
        to_disk: bool=False, use_console: bool=False, json_format: bool=True,
        loglevel: LogLevel=LogLevel.INFO
    ):
        """
            Initialize the Channel-Model Instance
        """
        # Initialize Logger Instance (perhaps not necessary, not implemented)'

        # Initialize Directory of the channel model
        # self.directory = Path(__file__).resolve().parents[2] / "models" / directory
        self.directory = Path(__file__).parent / "store" / directory
        self.directory.mkdir(parents=True, exist_ok=True)

        self._set_seed(seed)

        self.link = LinkStatePredictor(
            directory=self.directory/"link", rx_types=config.rx_types, 
            n_unit_links=config.n_unit_links, dropout_rate=config.dropout_rate,
            add_zero_los_frac=config.add_zero_los_frac, level=loglevel
        )
    
        self.path = PathModel(
                directory=self.directory/model_type.lower(), 
                model_type=model_type, rx_types=config.rx_types, 
                n_max_paths=config.n_max_paths, max_path_loss=config.max_path_loss,
                loglevel=loglevel
            )

    @staticmethod
    def _set_seed(seed: int=42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
