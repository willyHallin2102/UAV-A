"""
    src/models/utilities/common.py
    ------------------------------
    Common TensorFlow/Keras utility shared across multiple modules are being
    stored in this script.
"""
from __future__ import annotations

# import warnings
from typing import Dict, Literal, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
tfk, tfkl = tf.keras, tf.keras.layers



class SortLayer(tfkl.Layer):
    """
    A custom Keras layer that sorts the first `n_sort` features along the last
    dimension for each sample.
    """
    def __init__(self, n_sort: int, **kwargs):
        """
            Initialize the Sort-Layer Instance
        """
        super().__init__(**kwargs)

        if not isinstance(n_sort, int) or n_sort <= 0:
            raise ValueError("`n_sort` is required to be a positive integer.")
        
        self.n_sort = n_sort
    

    def call(self,
        inputs: tf.Tensor, direction: Literal["ASCENDING", "DESCENDING"]="DESCENDING"
    ) -> tf.Tensor:
        """
        Sorts the first `n_sort` elements of the last dimension.

        Args:
        -----
            inputs: tensor of shape (..., features)
            direction: Sort order
        
        Returns:
        --------
            tf.Tensor with same shape as inputs , with first `n_sort`
            values sorted.
        """
        inputs = tf.convert_to_tensor(inputs)
        tf.debugging.assert_rank_at_least(inputs, 2)

        head = tf.sort(inputs[..., : self.n_sort], direction=direction)
        tail = inputs[..., self.n_sort :]

        return tf.concat([head, tail], axis=-1)
    

    def get_config(self):
        return {**super().get_config(), "n_sort": self.n_sort}
    



class SplitSortLayer(tfkl.Layer):
    """
    Like `SortLayer` although always uses descending order.
    """
    def __init__(self, n_sort: int, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(n_sort, int) or n_sort <= 0:
            raise Valueerror("`n_sort` must be a positive integer")
        
        self.n_sort = n_sort
    

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.convert_to_tensor(x)

        head = tf.sort(x[..., : self.n_sort], direction="DESCENDING")
        tail = x[..., self.n_sort:]

        return tf.concat([head, tail], axis=-1)


    def get_config(self):
        return {**super().get_config(), "n_sort": self.n_sort}





def extract_inputs(
    inputs: Union[
        Tuple[Union[np.ndarray, tf.Tensor], Union[np.ndarray, tf.Tensor]],
        Dict[str, Union[np.ndarray, tf.Tensor]],
        Sequence[Union[np.ndarray, tf.Tensor]],
        tf.Tensor,
    ]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Standardize various input formats into `(x, cond)` tensors.

    Accepted formats
    ----------------
        1. Tuple or list: (x, cond)
        2. Dict with keys "x" and "cond"
        3. Sequence of exactly two arrays/tensors

    Returns:
    --------
        (tf.Tensor, tf.Tensor)
    """
    # Case 1: Tuple or list of two elements
    if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
        x, cond = inputs

        # If x is a list/tuple of partial tensors, concatenate
        if isinstance(x, (list, tuple)):
            x = tf.concat([tf.convert_to_tensor(t) for t in x], axis=-1)
        else:
            x = tf.convert_to_tensor(x)

        return x, tf.convert_to_tensor(cond)

    # Case 2: dictionary input
    if isinstance(inputs, dict):
        if "x" not in inputs or "cond" not in inputs:
            raise ValueError("Dictionary must contain keys 'x' and 'cond'.")
        x = tf.convert_to_tensor(inputs["x"], tf.float32)
        cond = tf.convert_to_tensor(inputs["cond"], tf.float32)
        return x, cond

    raise ValueError(f"Unsupported input format: {type(inputs)}")


# ---------------------------------------------------------------------------
# Layer weight reinitialization
# ---------------------------------------------------------------------------

def set_initialization(
    model: tfk.Model,
    names: Sequence[str],
    kernel_init: float = 1.0,
    bias_init: float = 1.0,
    noise_type: str = "gaussian",
):
    """
    Reinitialize the weights of selected Dense layers in a model.

    Args:
    -----
    model:  The model containing the target layers.
    names:  Names of layers to reset.
    kernel_init : Stddev scale or uniform limit (depending on noise type).
    bias_init : Stddev or range for bias initialization.
    noise_type : {"gaussian", "uniform"}

    Notes
    -----
        - Only built Dense layers are modified.
    """

    for name in names:
        try:
            layer = model.get_layer(name)
        except Exception:
            warnings.warn(f"Layer '{name}' not found; skipping.")
            continue

        if not isinstance(layer, tfkl.Dense) or not layer.built:
            warnings.warn(
                f"Layer '{name}' is not a built Dense layer; skipping."
            )
            continue

        in_dim, out_dim = map(int, layer.kernel.shape)

        # Gaussian initialization
        if noise_type.lower() == "gaussian":
            kernel = tf.random.normal(
                (in_dim, out_dim),
                mean=0.0,
                stddev=kernel_init / np.sqrt(in_dim),
                dtype=tf.float32,
            )
            bias = tf.random.normal(
                (out_dim,), mean=0.0, stddev=bias_init, dtype=tf.float32
            )

        # Uniform initialization
        elif noise_type.lower() == "uniform":
            limit = kernel_init / np.sqrt(in_dim)
            kernel = tf.random.uniform(
                (in_dim, out_dim), -limit, limit, dtype=tf.float32
            )
            bias = tf.random.uniform(
                (out_dim,), -bias_init, bias_init, dtype=tf.float32
            )

        else:
            raise ValueError(
                f"Unsupported noise type '{noise_type}'. Use 'gaussian' or 'uniform'."
            )

        try:
            layer.set_weights([kernel.numpy(), bias.numpy()])
        except Exception as e:
            warnings.warn(f"Could not set weights for layer '{name}': {e}")
