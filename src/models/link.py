"""
    src/models/link.py
    ------------------
    This module implements the `LinkStatePredictor`, a neural network-based
    predictor for classifying the state of a wireless link between the UAV
    and its receiver antenna, potential outcomes are (e.g., LoS, NLoS).

    The predictor combines learned representation of relative UAV-receiver
    geometry (`dvec`) and categorical receiver type (`rx-type`) to estimate
    the link state. Preprocessing steps include one-hot-encoding of receiver 
    types, geometrical feature transformation and standard normalization.
"""
from __future__ import annotations

import orjson
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
tfk, tfkl = tf.keras, tf.keras.layers

from pathlib import Path
from typing import Dict, List, Tuple, Union
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.data import LinkState
from src.config.const import CONFIG_FN, WEIGHTS_FN, PREPROC_FN
from logs.logger import Logger, LogLevel

AUTOTUNE = tf.data.AUTOTUNE



class LinkStatePredictor:
    """
    The LinkStatePredictor is a subsection of the channel model, its purpose is
    to predict the UAV-receiver link-state based on the given distance-vector, 
    that is the vector representing the distance between the UAV and the antenna
    as well as the receiver antenna (Rx) type which either could be `terrestrial`
    or it could be `aerial`. The link-states that it could generate are either
    of `NLoS, LoS` or `NO-LINK. 
    """
    def __init__(self,
        rx_types: List[Union[str, int]], n_unit_links: Tuple[int, ...], 
        n_dimensions: int=3, add_zero_los_frac: float=0.1, dropout_rate: float=0.2,
        level: LogLevel=LogLevel.INFO, directory: Union[str, Path]="link",
        to_disk: bool=False, json_format: bool=True, use_console: bool=False, 
        seed: int=42
    ):
        """
            Initialize the Link-State Predictor
        """
        # Initialize the Logger for the link predictor
        self.logger = Logger(
            name="link-state-predictor", json_format=json_format, 
            use_console=use_console, level=level
        )

        # Initialize the Directory for the Link-Predictor
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.logger.debug("Root directory created")

        # Model Preprocessors
        self.model, self.link_scaler, self.rx_type_encoder = None, None, None

        # Parameters
        self.rx_types = list(rx_types)
        self.n_unit_links = tuple(n_unit_links)
        self.n_dimensions = int(n_dimensions)
        self.add_zero_los_frac = float(add_zero_los_frac)
        self.dropout_rate = float(dropout_rate)
        self.seed = int(seed)

        self.__version__ = 1
        self.history = None


    # ---------------========== Model Construction ==========--------------- #
    
    def build(self):
        """
        Constructs the `tf.keras.Model` object, creating a `Sequential`
        model architecture. Input dimension to the model is twice the 
        number of rx_types receiver types in the scene.
        """
        self.logger.debug("Building the model...")
        layers: list = [tfkl.Input(shape=(2 * len(self.rx_types),), name="input")]
        # self.rx_type_encoder = OneHotEncoder(
        #     sparse_output=False,
        #     handle_unknown="ignore",
        #     categories=[self.rx_types]
        # )

        for index, units in enumerate(self.n_unit_links):
            self.logger.debug(f"Adding dense layer {index}: {units}")
            
            layers.append(tfkl.Dense(
                units, activation=None, kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-5), name=f"hidden-{index}")
            )
            layers.append(tfkl.BatchNormalization())
            layers.append(tfkl.Activation("relu"))

            if self.dropout_rate > 0.0:
                layers.append(tfkl.Dropout(self.dropout_rate))
        
        layers.append(tfkl.Dense(
            LinkState.N_STATES, activation="softmax", name="output"
        ))
        self.model = tfk.models.Sequential(layers)
        self.logger.info("Model has been successfully built.")
    

    # ---------------========== Training Model ==========--------------- #

    def fit(self,
        dtr: Dict[str, np.ndarray], dts: Dict[str, np.ndarray],
        epochs: int=50, batch_size: int=512, learning_rate: float=1e-4
    ) -> tfk.callbacks.History:
        """
        Trains the `LinkStatePredictor` model on labeled data. It train
        the model using keras model operations by overriding methods.

        Args:
        -----
            dtr:    Training dataset dictionary containing `dvec`,
                    `rx_type`, and `link_state`.
            dts:    Validation dataset dictionary with the same keys
                    `dtr`, used for test_step for validating.
            epochs: int = 50, Number of full passes over the training
                    dataset.
            batch_size: int = 512, Number of samples per training
                        batch, `512` rather decent for utilization
                        of GPU, depending on hardware may need alter.
            learning_rate:  float = 1e-4, learning-rate for the applied
                            optimizer, this using Adam.
        
        Returns:
        --------
            history:    tf.keras.callbacks.History, training history
                        object with the losses/accuracy metrics.
        """
        self.logger.info(
            f"Starting training for `{epochs}` epochs "
            f"(batch-size=`{batch_size}` and learning-rate=`{learning_rate}`)"
        )

        if self.model is None:
            self.logger.info("Building model, before training")
            self.build()
        
        with self.logger.time_block("fit() training duration"):
            xtr, ytr = self._prepare_arrays(dtr, fit=True)
            xts, yts = self._prepare_arrays(dts, fit=False)

            self.model.compile(
                optimizer=tfk.optimizers.Adam(learning_rate=learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            train_ds = tf.data.Dataset.from_tensor_slices(
                (xtr, ytr)
            ).batch(batch_size).prefetch(AUTOTUNE)
            val_ds = tf.data.Dataset.from_tensor_slices(
                (xts, yts)
            ).batch(batch_size).prefetch(AUTOTUNE)
            self.history = self.model.fit(
                train_ds, epochs=epochs, validation_data=val_ds, verbose=1
            )

        self.logger.info("Training completed.")
        return self.history
    


    def predict(self, dvec: np.ndarray, rx_type: np.ndarray):
        """
        Method for predicting the link-state for given the distance difference 
        vector between Rx and Tx as well as the rx-type either being 
        terrestrial or aerial in order to return the link-state prediction.
        """
        return self.model.predict(self._transform_links(dvec, rx_type))
        
    
    # ---------------========== Saving / Loading ==========--------------- #

    def save(self):
        """
        """
        payload = {
            "version": self.__version__,
            "framework": {"tensorflow": tf.__version__},
            "config": {
                "rx_types": self.rx_types,
                "n_unit_links": self.n_unit_links,
                "n_dimensions": self.n_dimensions,
                "add_zero_los_frac": self.add_zero_los_frac,
                "dropout_rate": self.dropout_rate,
                "seed": self.seed,
            },
            "history": getattr(self, "history", None).history if hasattr(self, "history") else None,
        }

        with open(self.directory / CONFIG_FN, "wb") as fp:
            fp.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        
        # Save model weights and thereafter, Save preprocessors using pickle
        self.model.save_weights(str(self.directory / WEIGHTS_FN))
        preproc_data = {
            "link_scaler": self.link_scaler,
            "rx_encoder": self.rx_type_encoder,
        }
        with open(self.directory / PREPROC_FN, "wb") as fp:
            pickle.dump(preproc_data, fp)
        
        self.logger.info(f"Model saved to {self.directory}")
    

    def load(self):
        """
        """
        if not (self.directory / CONFIG_FN).exists():
            raise FileNotFoundError("Missing model config file")
        if not (self.directory / WEIGHTS_FN).exists():
            raise FileNotFoundError("Missing model weights file")
        if not (self.directory / PREPROC_FN).exists():
            raise FileNotFoundError("Missing preprocessors file")
        
        # Load model config
        with open(self.directory / CONFIG_FN, "rb") as fp:
            payload = orjson.loads(fp.read())
        
        # Check version compatibility
        if payload.get("version", 0) != self.__version__:
            self.logger.warning("Version mismatch, potential incompatibilities")
        
        # Load configuration
        config = payload.get("config", {})
        self.rx_types = config.get("rx_types", self.rx_types)
        self.n_unit_links = tuple(config.get("n_unit_links", self.n_unit_links))
        self.n_dimensions = int(config.get("n_dimensions", self.n_dimensions))
        self.add_zero_los_frac = float(config.get("add_zero_los_frac", self.add_zero_los_frac))
        self.dropout_rate = float(config.get("dropout_rate", self.dropout_rate))
        
        # Load preprocessors using pickle
        with open(self.directory / PREPROC_FN, "rb") as fp:
            preproc_data = pickle.load(fp)
        
        self.link_scaler = preproc_data["link_scaler"]
        self.rx_type_encoder = preproc_data["rx_encoder"]

        self.build()
        self.model.load_weights(str(self.directory / WEIGHTS_FN))

        if payload.get("history"):
            self.history = payload["history"]
            
        self.logger.info(f"Model loaded from {self.directory}")


    # ---------------========== Internal Preprocessing ==========--------------- #

    def _prepare_arrays(self,
        data: Dict[str, np.ndarray], fit: bool=False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform the dataset dictionary into model-ready arrays

        Args:
        -----
        data :  Dict[str, np.ndarray]: Input dataset with keys
                'dvec', 'rx_type', and 'link_state'.  
        fit :   bool, default=False, Whether to fit preprocessing 
                steps (True for training, False for validation/test).  

        Returns:
        --------
        x : np.ndarray. Feature matrix ready for model input.  
        y : np.ndarray, Corresponding target labels.
        """
        mode = "fit" if fit else "transform"
        self.logger.debug(f"Preparing arrays in mode: `{mode}`")

        dvec = np.asarray(data["dvec"], dtype=np.float32)
        rx_type = np.asarray(data["rx_type"])
        link_state = np.asarray(data["link_state"], dtype=np.int32)

        if fit:
            self.rx_type_encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            )
            self.link_scaler = StandardScaler()

        # Perhaps not necessary, then transform the features
        dvec, rx_type, link_state = self._add_los_zero(dvec, rx_type, link_state)
        x, y = self._transform_links(dvec, rx_type, fit=fit), link_state

        self.logger.debug(f"Prepared arrays: X={x.shape}, Y={y.shape}")
        return x, y


    def _transform_links(self,
        dvec: np.ndarray, rx_type: np.ndarray, fit: bool=False
    ) -> np.ndarray:
        """
        Convert geometric and categorical features into standardized inputs.
        
        Steps:
        ------
            1.  Compute horizontal distance `dx = sqrt(x² + y²)` and vertical 
                distance `dz`.  
            2.  Encode receiver types into one-hot vectors.  
            3.  Broadcast `dx` and `dz` across receiver encodings to produce 
                per-type features.  
            4.  Concatenate features horizontally and scale with `StandardScaler`.
        Args:
        -----
        dvec : np.ndarray, shape (N, 3) 
            Relative UAV–receiver displacement vectors.  
        rx_type : np.ndarray, shape (N,) 
            Receiver type identifiers.  
        fit : bool,
            default=False If True, fit preprocessors on input data.  

        Returns
        -------
        x : np.ndarray, shape (N, 2 * len(rx_types))
            Transformed and scaled feature matrix.
        """
        dx = np.linalg.norm(dvec[:, :2], axis=1, keepdims=True)
        dz = dvec[:, 2:3]

        if self.rx_type_encoder is None:
            raise RuntimeError(
                "Encoder not initialized. `_prepare_arrays(..., fit=True)` first."
            )

        rx_one = (
            self.rx_type_encoder.fit_transform(rx_type[:, None])
            if fit else
            self.rx_type_encoder.transform(rx_type[:, None])
        ).astype(np.float32)

        x = np.hstack([rx_one * dx, rx_one * dz])
        
        if self.link_scaler is None:
            raise RuntimeError(
                "Scaler not initialized. `_prepare_arrays(..., fit=True)` first."
            )
        
        return self.link_scaler.fit_transform(x) if fit else self.link_scaler.transform(x)
    

    def _add_los_zero(self,
        dvec: np.ndarray, rx_type: np.ndarray, link_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Augment dataset with synthetic zero-distance LOS samples. 
        This augmentation helps improve robustness by injecting 
        near-ground-truth samples that represent UAV–receiver pairs 
        with minimal displacement and guaranteed LOS.

        Args
        ----
        dvec:   np.ndarray, shape (N, 3) Original displacement 
                vectors.  
        rx_type:    np.ndarray, shape (N,), Receiver types.  
        link_state: np.ndarray, shape (N,) Ground-truth link 
                    state labels.

        Returns
        -------
        dvec_new : np.ndarray
            Augmented displacement vectors.  
        rx_type_new : np.ndarray
            Augmented receiver types.  
        link_state_new : np.ndarray
            Augmented link state labels.  
        """
        n_samples = len(dvec)
        n_add = int(n_samples * self.add_zero_los_frac)
        if n_add <= 0:
            self.logger.debug("No zero-LOS augmentation added.")
            return dvec, rx_type, link_state

        self.logger.debug(f"Adding {n_add} synthetic LOS-zero samples for augmentation.")
        index = np.random.choice(n_samples, size=n_add, replace=True)

        dvec_i = np.zeros_like(dvec[index])
        dvec_i[:, 2] = np.maximum(dvec[index, 2], 0)

        rx_type_i = rx_type[index]
        link_state_i = np.full(n_add, LinkState.LOS, dtype=link_state.dtype)

        # Concatenate efficiently
        dvec = np.concatenate([dvec, dvec_i], axis=0)
        rx_type = np.concatenate([rx_type, rx_type_i], axis=0)
        link_state = np.concatenate([link_state, link_state_i], axis=0)

        self.logger.debug(f"Data augmentation complete. New total samples: {len(dvec)}")
        return dvec, rx_type, link_state
