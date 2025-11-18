"""
    src/config/model.py
    -------------------
    Configuration file for the model parameters as well as 
    definitions
"""
from dataclasses import dataclass
from typing import Dict, Final, Literal, Sequence, Tuple, Type


# ---------------========== Basic Configurations ==========--------------- #

@dataclass(slots=True)
class ModelConfig:
    """
    Common features for the generative artificial model, these parameters
    includes dimensions of latent space, which dropout rate, and initial 
    deviance for both kernel as well as the bias.
    """
    n_latent        : int   = 10
    min_variance    : float = 1e-4
    dropout_rate    : float = 0.20
    init_kernel     : float = 10.0
    init_bias       : float = 10.0

    def __post_init__(self):
        if self.n_latent <= 0:
            raise ValueError("n_latent must be positive")
        if self.min_variance < 0:
            raise ValueError("min_variance cannot be negative")
        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValueError("dropout_rate must be between 0.0 and 1.0")


# ---------------========== Validation of Layers ==========--------------- #

def _validate_layers(name: str, layers: Sequence[int]):
    """ Testing the validity of passed layers """
    if not layers:
        raise ValueError(f"{name} layer cannot be empty")
    if any(layer <= 0 for layer in layers):
        raise ValueError(f"All {name} layers must be positive")



# ---------------========== Variational Autoencoder ==========--------------- #

@dataclass(slots=True)
class VaeConfig(ModelConfig):
    encoder_layers  : Tuple[int, ...] = (200, 80)
    decoder_layers  : Tuple[int, ...] = (80, 200)

    beta    : float = 0.50
    beta_annealing_step : int = 100_000
    kl_warmup_steps : int   = 20

    def __post_init__(self):
        ModelConfig.__post_init__(self)

        _validate_layers("encoder", self.encoder_layers)
        _validate_layers("decoder", self.decoder_layers)

        if not (0.0 <= self.beta <= 1.0):
            raise ValueError("beta must be between 0.0 and 1.0")



# ---------------========== Configurations Getter ==========--------------- #

MODELS: Dict[str, Type[ModelConfig]] = {
    "vae": VaeConfig,
}

VALID_MODELS = tuple(MODELS.keys())
def get_config(model_type: str) -> ModelConfig:
    """
    Return the config object associated with the model type
    """
    key = model_type.lower()
    if key not in MODELS:
        raise ValueError(
            f"Unknown model '{model_type}'. "
            f"Supported types: {', '.join(VALID_MODELS)}"
        )
    return MODELS[key]()


