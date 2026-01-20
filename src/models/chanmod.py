"""
    src / models / chanmod.py
    -------------------------

"""
from __future__ import annotations

from pathlib import Path
from typing import Union
from src.models.link import LinkStatePredictor
from src.models.path import PathModel
from src.configs.data import DataConfig


class ChannelModel:

    def __init__(self,
        config: DataConfig=DataConfig(), model_type: str="vae", seed: int=42
    ):
        """
            Initialize Channel-Model Instance
        """
        self.directory = Path(__file__).resolve().parents[2]/"models"/"a"
        self.directory.mkdir(parents=True, exist_ok=True)

        self.link = LinkStatePredictor(
            directory=self.directory / "link", rx_types=config.rx_types, 
            n_unit_links=config.n_unit_links, dropout_rate=config.dropout_rate,
            seed=seed
        )

        self.path = PathModel(
            directory=self.directory / model_type.lower(),
            model_type=model_type, rx_types=config.rx_types, 
            n_max_paths=config.n_max_paths, max_path_loss=config.max_path_loss
        )

