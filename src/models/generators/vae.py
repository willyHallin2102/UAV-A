"""
"""
from __future__ import annotations

import orjson
import tensorflow as tf
tfk, tfkl = tf.keras, tf.keras.layers

from pathlib import Path
from typing import List,Sequence,Tuple,Union

from src.models.generators.genai import Genai
from src.models.utils.common import SplitSortLayer, set_initialization



class Reparameterize(tfkl.Layer):
    def call(self, x: Tuple[tf.Tensor,tf.Tensor]) -> tf.Tensor:
        mu, logvar = x
        eps = tf.random.normal(shape=tf.shape(mu), dtype=mu.dtype)
        return mu + eps * tf.exp(0.5 * logvar)
    
    def get_config(self): return super().get_config()


def reconstruction_loss(x: tf.Tensor, mu: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
    logvar = tf.clip_by_value(logvar, -10, 10)
    # Average over batch: Σ[precision · (x-µ)² + log(σ)]
    return tf.reduce_mean(0.5 * tf.reduce_sum(
        tf.exp(-logvar) * tf.square(x-mu) + logvar, axis=-1
    ))


def kl_divergence(mu: tf.Tensor, logvar: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    logvar = tf.clip_by_value(logvar, -10.0, 10.0)
    return tf.reduce_mean(weights * tf.reduce_sum(
        1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1
    ))



class Vae(Genai):

    def __init__(self,
        n_latent: int, n_data: int, n_conditions: int, 
        encoder_layout: Tuple[int,...], decoder_layers: Tuple[int,...],
        min_variance: float=1e-4, dropout_out: float=0.2, beta: float=0.5,
        beta_annealing_step: int=10_000, kl_warm_steps: int=1000,
        init_kernel: float=10.0, bias_kernel: float=10.0, n_sort: int=0
    ):
        """
            Initialize VAE instance 
        """
        super().__init__(name="vae")

        # Hyperparameters
        self.n_latent = int(n_latent)
        self.n_data = int(n_data)
        self.n_conditions = int(n_conditions)
        self.min_variance = float(min_variance)
        self.dropout_rate = float(dropout_rate)
        self.n_sort = int(n_sort)

        self.encoder_layers = tuple(int(layer) for layer in encoder_layers)
        self.decoder_layers = tuple(int(layer) for layer in decoder_layers)

        self.init_kernel = float(init_kernel)
        self.init_bias = float(init_bias)

        # Schedulers initialized and setup
        self.beta = tf.Variable(float(beta), trainable=False, dtype=tf.float32)
        self.beta_annealing_step = int(beta_annealing_step)
        self.kl_warmup_steps = tf.constant(int(kl_warmup_steps), dtype=tf.float32)
        self.current_step = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        self.sampler = Reparameterize()

        self.encoder = self._encoder()
        self.decoder = self._decoder()

    def train_step():
        print("train")
    def test_step():
        print("test")
    


    # ======================================================================
    #       Internal Model Components 
    # ======================================================================

    def _encoder(self) -> tfk.Model:
        x_in = tfkl.Input(shape=(self.n_data,), name="encoder-x")
        cond_in = tfkl.Input(shape=(self.n_conditions,), name="encoder-conditions")
        h = tfkl.Concatenate(name="encoder-input")([x_in, cond_in])

        names: List[str] = []
        for i, units in enumerate(self.encoder_layers):
            name=f"encoder-hidden-{i}"
            names.append(name)
            h = tfkl.Dense(units=units, activation="swish", name=name)(h)
            h = tfkl.BatchNormalization(name=f"{name}-batch")(h)

            if self.dropout_rate > 0.0:
                h = tfkl.Dropout(self.dropout_rate, name=f"{name}-drop")(h)
        
        z_mu = tfkl.Dense(self.n_latent, name="encoder-mu")(h)
        z_logvar = tfkl.Dense(self.n_latent, name="encoder-logvar")(h)
        encoder = tfk.Model(
            [x_in, cond_in], [z_mu, z_logvar], name="encoder"
        )
        set_initialization(encoder, names, self.init_kernel, self.init_bias)

        return encoder

    
    def _decoder(self) -> tfk.Model:
        z_in = tfkl.Input(shape=(self.n_latent,), name="decoder-z")
        cond_in = tfkl.Input(shape=(self.n_conditions,), name="decoder-conditions")
        h = tfkl.Concatenate(name="decoder-concat")([z_in, cond_in])

        names: List[str] = []
        for i, units in enumerate(self.decoder_layers):
            name = f"decoder-hidden-{i}"
            names.append(name)

            h = tfkl.Dense(units=units, activation="tanh", name=name)(h)
            h = tfkl.BatchNormalization(name=f"{name}-batch")(h)

            if self.dropout_rate > 0.0:
                h = tfkl.Dropout(self.dropout_rate, name=f"{name}-drop")(h)
        
        x_mu = tfkl.Dense(self.n_data, name="decoder-mu")(h)
        if self.n_sort > 0:
            x_mu = SplitSortLayer(self.n_sort, name="decoder-mu-sort-slice")(x_mu)
        
        x_logvar = tfkl.Dense(self.n_data, name="decoder-logvar")(h)
        x_logvar = tfkl.Lambda(
            lambda t: tf.math.log(self.min_variance + tf.nn.softplus(t)),
            name="decoder-logvar-activation"
        )(x_logvar)

        decoder = tfk.Model([z_in,cond_in], [x_mu,x_logvar], name="decoder")
        set_initialization(decoder, names, self.init_kernel, self.init_bias)

        return decoder
