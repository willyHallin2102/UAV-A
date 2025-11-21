"""
    src/config/const.py
    --------------------
    Constants referring to names of reoccurring names, this enable a collection 
    parameter name for simpler access on change or collection and consistency.
"""
from typing import Final

# ---------------========== Physical Constants ==========--------------- #

LIGHT_SPEED     : Final[float]  = 2.99792458e8  # m/s Light Speed
THERMAL_NOISE   : Final[float]  = -174.0        # dBm Thermal Noise


# ---------------========== File Name Constants ==========--------------- #

PREPROC_FN      : Final[str]    = "preproc.pkl" #
WEIGHTS_FN      : Final[str]    = "model.weights.h5"
CONFIG_FN       : Final[str]    = "model_config.json"