from typing import Final
from pathlib import Path

# ============================================================
#       Physical Constants 
# ============================================================

LIGHT_SPEED         : Final[float] = 2.99792458e8   # m/s
THERMAL_NOISE       : Final[float] = -174.0         # dBm/Hz
PLANCK_CONSTANT     : Final[float] = 6.62607015e-34 # J·s (optional, for completeness)
BOLTZMANN_CONSTANT  : Final[float] = 1.380649e-23   # J/K (optional)

TERRESTRIAL         : Final[str] = "Terrestrial"
AERIAL              : Final[str] = "Aerial"


# ============================================================
#       Filenames and Path constants
# ============================================================

DATA_DIR            : Final[Path] = Path("data")
MODEL_DIR           : Final[Path] = Path("models")
RESULTS_DIR         : Final[Path] = Path("results")

# File naming constants
PREPROC_FN          : Final[str] = "preproc.pkl"
WEIGHTS_FN          : Final[str] = "model.weights.h5"
CONFIG_FN           : Final[str] = "model_config.json"
METRICS_FN          : Final[str] = "training_metrics.json"


# ============================================================
#       Filenames and Path constants
# ============================================================

PKL_EXT             : Final[str] = ".pkl"
H5_EXT              : Final[str] = ".h5"
JSON_EXT            : Final[str] = ".json"
