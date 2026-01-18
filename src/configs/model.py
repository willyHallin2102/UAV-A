"""
    src / configs / model.py
    ------------------------
    Model configuration parameters and definitions
"""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Final, Type



class ModelType(str, Enum):
    VAE     = "str"


@dataclass
class ModelConfig:
    """Base configuration for all models."""
    # Model architecture
    n_latent: int = 10
    min_variance: float = 1e-4
    dropout_rate: float = 0.20
    
    # Weight initialization
    init_kernel: float = 10.0
    init_bias: float = 10.0
    
    # Training parameters (consider separating into TrainingConfig)
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    
    # Regularization
    l2_reg: float = 1e-4
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_latent <= 0: raise ValueError("n_latent must be positive")
        if self.min_variance < 0: raise ValueError("minimum variance cannot be negative")
        
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError("Dropout rate must be within [0.0, 1.0]")
        
        if self.init_kernel <= 0 or self.init_bias <= 0:
            raise ValueError("Initialization values must be positive")
        
        if self.learning_rate <= 0: raise ValueError("Learning rate must be positive")
        if self.batch_size <= 0: raise ValueError("Batch size must be positive")
        if self.epochs <= 0: raise ValueError("Number of epochs must be positive")
        if self.l2_reg < 0: raise ValueError("L2 regularization cannot be negative")
    
    @property
    def model_type(self) -> str:
        """Return the model type string."""
        return self.__class__.__name__.replace("Config", "").lower()


def _validate_layers(name: str, layers: tuple[int, ...]) -> None:
    """Validate layer dimensions."""
    if not layers:
        raise ValueError(f"`{name}` layers cannot be empty")
    if any(layer <= 0 for layer in layers):
        raise ValueError(f"All `{name}` layers must be positive integers")


@dataclass
class VaeConfig(ModelConfig):
    """Configuration for Variational Autoencoder."""
    # Architecture
    encoder_layers: tuple[int, ...] = (200, 80)
    decoder_layers: tuple[int, ...] = (80, 200)
    
    # VAE-specific parameters
    beta: float = 0.50
    beta_annealing_step: float = 100_000
    kl_warmup_steps: int = 20
    
    # Reconstruction loss weight
    reconstruction_weight: float = 1.0
    
    def __post_init__(self):
        """Validate VAE-specific parameters."""
        super().__post_init__()
        
        _validate_layers("encoder", self.encoder_layers)
        _validate_layers("decoder", self.decoder_layers)
        
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta value must lie within 0.0 and 1.0")
        if self.beta_annealing_step <= 0:
            raise ValueError("beta_annealing_step must be positive")
        if self.kl_warmup_steps < 0:
            raise ValueError("kl_warmup_steps cannot be negative")
        if self.reconstruction_weight <= 0:
            raise ValueError("reconstruction_weight must be positive")
    
    @property
    def total_layers(self) -> int:
        return len(self.encoder_layers) + len(self.decoder_layers) + 2 



MODEL_CONFIGS: Dict[str, Type[ModelConfig]] = {
    ModelType.VAE: VaeConfig,
}

def get_config(model_type: str | ModelType) -> ModelConfig:
    if isinstance(model_type, ModelType): model_type = model_type.value
    
    model_type = model_type.lower().strip()
    try:
        model_enum = ModelType(model_type)
        config_class = MODEL_CONFIGS[model_enum]
        return config_class()

    except (KeyError, ValueError):
        # Fallback to direct string matching for backward compatibility
        for key, config_class in MODEL_CONFIGS.items():
            if key.value == model_type: return config_class()
        
        supported = ", ".join([m.value for m in MODEL_CONFIGS.keys()])
        raise ValueError(
            f"Unknown model type: `{model_type}`. Supported types: {supported}"
        ) from None
