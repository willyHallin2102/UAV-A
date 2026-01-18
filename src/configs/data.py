"""
    src / configs / data.py
    -----------------------
    Manages the environmental features of the scene the UAV is in
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import ClassVar, Final, Tuple
from enum import IntEnum
from src.configs.const import TERRESTRIAL, AERIAL


class AngleIndex(IntEnum):
    AOA_PHI     = 0
    AOA_THETA   = 1
    AOD_PHI     = 2
    AOD_THETA   = 3

    @classmethod
    @property
    def n_angles(cls) -> int: 
        return len(cls)
    
    @classmethod
    def from_string(cls, state: str) -> AngleIndex: return {
            "aoa_phi": cls.AOA_PHI, "aoa_theta": cls.AOA_THETA,
            "aod_phi": cls.AOD_PHI, "aod_theta": cls.AOD_THETA
        }[state.lower()]
    

class LinkState(IntEnum):
    NO_LINK     = 0
    NLOS        = 1
    LOS         = 2

    @classmethod
    @property
    def n_states(cls) -> int:
        return len(cls)

    @classmethod
    def from_string(cls, state: str) -> LinkState: return {
            "no_link": cls.NO_LINK, "nlos": cls.NLOS, "los": cls.LOS
        }[state.lower()]


@dataclass(frozen=True)
class DataConfig:
    """ Immutable configuration for simulation and data parameters """
    frequency       : float = 28e9
    created         : str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    description     : str = "dataset"
    rx_types        : Tuple[str, ...] = (TERRESTRIAL, AERIAL)
    max_path_loss   : float = 200.0
    tx_power_dbm    : float = 16.0
    n_max_paths     : int = 20
    n_unit_links    : Tuple[int,...] = (25,10)
    add_zero_los_frac   : float = 0.1
    dropout_rate    : float = 0.2

    def __post_init__(self):
        if self.frequency <= 0: raise ValueError("Frequency must be positive")
        if self.max_path_loss <= 0.0: 
            raise ValueError("Maximum path loss must be positive")
        if self.n_max_paths <= 0: raise ValueError("Number of max path must be positive")
        if any(n <= 0 for n in self.n_unit_links):
            raise ValueError("All unit links must be positive")
        if not 0 <= self.add_zero_los_frac <= 1:
            raise ValueError("LOS fraction must be between 0 and 1")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
    
    @property
    def wavelength(self) -> float:
        from src.configs.const import LIGHT_SPEED
        return LIGHT_SPEED / self.frequency

