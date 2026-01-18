"""
    src / models / link.py
    ----------------------
    Link state predictor is a class that is used in a part of the channel-model
    whose purpose it is to predict the link-state between `No-Link` available,
    `NLOS` that there is no Line-of-Sight or `LOS` Line-of-Sight. Predictions 
    are made based on the distance vector `dvec` between the receiver antenna
    `Rx` and the transmitter antenna (UAV) `Tx` along with whereas the antenna
    is `Terrestrial` or `Aerial`.
"""
from __future__ import annotations

import orjson
import pickle
import numpy as np
import tensorflow as tf
tfk = tf.keras

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.configs.data import LinkState
from src.models.utils.preproc import serialize_preproc, deserialize_preproc
from src.configs.const import CONFIG_FN, WEIGHTS_FN, PREPROC_FN

AUTOTUNE = tf.data.AUTOTUNE


class LinkStatePredictor:

    def __init__(self,
        rx_types: List[str], n_unit_links: Tuple[int,...],
        add_zero_los_frac: float=0.1, dropout_rate: float=0.55,
        directory: Union[str,Path]="link", seed: int=42
    ):
        """
            Initialize Link State Predictor Initialized
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.model: Optional[tfk.Model]=None
        self.rx_type_encoder: Optional[OneHotEncoder]=None
        self.link_scaler: Optional[StandardScaler]=None

        self.rx_types = list(rx_types)
        self.n_unit_links = tuple(n_unit_links)
        self.add_zero_los_frac = float(add_zero_los_frac)
        self.dropout_rate = float(dropout_rate)

        self.seed = int(seed)

        self.__version__ = 1
        self.history: Optional[tfk.callbacks.History]=None
    

    # ======================================================================
    #       Model Construction
    # ======================================================================

    def build(self):
        layers = [tfk.layers.Input(shape=(2*len(self.rx_types),), name="input")]
        for i, units in enumerate(self.n_unit_links):
            layers.append(tfk.layers.Dense(
                units=units, activation=None, kernel_initializer="he_normal",
                name=f"hidden-{i}"
            ))
            layers.append(tfk.layers.Activation("relu")) # or sigmoid...
        
        layers.append(tfk.layers.Dense(
            units=LinkState.n_states, activation="softmax", name="output"
        ))
        self.model = tfk.models.Sequential(layers)
    

    # =====================================================================
    #       Model Fitting Method
    # =====================================================================

    def fit(self,
        dtr: Dict[str,np.ndarray], dts: Dict[str,np.ndarray],
        epochs: int=50, batch_size: int=512, learning_rate: float=1e-3
    ) -> tfk.callbacks.History:
        """
        """
        xtr, ytr = self._prepare_arrays(dtr, fit=True)
        xts, yts = self._prepare_arrays(dts, fit=False)
        self.model.compile(
            optimizer=tfk.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy', metrics=['accuracy']
        )

        t = tf.data.Dataset.from_tensor_slices((xtr,ytr)).batch(batch_size).prefetch(AUTOTUNE)
        v = tf.data.Dataset.from_tensor_slices((xts,yts)).batch(batch_size).prefetch(AUTOTUNE)

        self.history = self.model.fit(t, epochs=epochs, validation_data=v, verbose=1)
        return self.history

    # ======================================================================
    #       Saving / Loading Methods
    # ======================================================================

    def save(self):
        """Save the model and preprocessors."""
        from src.models.utils.preproc import serialize_preproc
        with open(self.directory / CONFIG_FN, "wb") as fp:
            fp.write(orjson.dumps({
                "version": self.__version__,"framework": {"tensorflow": tf.__version__},
                "config": {
                    "rx_types": self.rx_types,
                    "n_unit_links": list(self.n_unit_links), # list and tuple might be confused
                    "add_zero_los_frac": self.add_zero_los_frac,
                    "dropout_rate": self.dropout_rate,
                    "seed": self.seed
                },
                "history": getattr(self.history, "history", None) if self.history else None
            }, option=orjson.OPT_INDENT_2))
        
        self.model.save_weights(str(self.directory / WEIGHTS_FN))
        preproc_data = {}
        if self.link_scaler:
            preproc_data["link_scaler"] = serialize_preproc(self.link_scaler)
        if self.rx_type_encoder:
            preproc_data["rx_encoder"] = serialize_preproc(self.rx_type_encoder)
        
        with open(self.directory / PREPROC_FN, "wb") as fp:
            fp.write(orjson.dumps(preproc_data, option=orjson.OPT_INDENT_2))


    def load(self):
        """Load the model and preprocessors."""
        from src.models.utils.preproc import deserialize_preproc        
        with open(self.directory / CONFIG_FN, "rb") as fp:
            payload = orjson.loads(fp.read())
        
        if payload.get("version", 0) != self.__version__: print(
                f"Warning: Version mismatch. Model: {payload.get('version')}, "
                f"Current: {self.__version__}"
            )
        
        config = payload.get("config", {})
        self.rx_types = config.get("rx_types", self.rx_types)
        self.n_unit_links = tuple(config.get("n_unit_links", self.n_unit_links))
        self.add_zero_los_frac = float(config.get("add_zero_los_frac", self.add_zero_los_frac))
        self.dropout_rate = float(config.get("dropout_rate", self.dropout_rate))
        self.seed = int(config.get("seed", self.seed))
        
        with open(self.directory / PREPROC_FN, "rb") as fp:
            preproc_dict = orjson.loads(fp.read())
        
        self.link_scaler = deserialize_preproc(preproc_dict["link_scaler"])
        self.rx_type_encoder = deserialize_preproc(preproc_dict["rx_encoder"])
        
        self.build()
        self.model.load_weights(str(self.directory / WEIGHTS_FN))

        if payload.get("history"):
            self.history = tfk.callbacks.History()
            self.history.history = payload["history"]
        
        print(f"Model loaded from {self.directory}") # May be unnecessary later on 

    # ======================================================================
    #       Internal Methods for Link Model
    # ======================================================================

    def _prepare_arrays(self, data: Dict[str,np.ndarray], fit: bool=False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        dvec = np.asarray(data['dvec'], dtype=np.float32)
        rx_types = np.asarray(data["rx_type"])
        rx_map = {
            0: "Terrestrial", 1: "Aerial", "0": "Terrestrial", "1": "Aerial", 
            "Terrestrial": "Terrestrial", "Aerial": "Aerial",
        }
        rx_types = np.vectorize(rx_map.__getitem__)(rx_types)
        link_states = np.asarray(data["link_state"],dtype=np.int32)

        if fit:
            self.rx_type_encoder = OneHotEncoder(
                categories=[list(self.rx_types)],
                sparse_output=False,
                handle_unknown="ignore",
                dtype=np.float32
            )
            self.rx_type_encoder.fit(rx_types[:,None])
            self.link_scaler = StandardScaler()
        
        dvec, rx_types, link_states = self._add_los_zero(dvec, rx_types, link_states)
        return self._transform_links(dvec, rx_types, fit=fit), link_states
 

    def _add_los_zero(self,
        dvec: np.ndarray, rx_types: np.ndarray, link_states: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        """
        n_samples = len(dvec)
        n_add = int(n_samples * self.add_zero_los_frac)
        if n_add <= 0:
            return dvec, rx_types, link_states
        
        i = np.random.choice(n_samples, size=n_add, replace=True)
        dvec_i = np.zeros_like(dvec[i])
        dvec_i[:,2] = np.maximum(dvec[i,2], 0)

        rx_type_i = rx_types[i]
        link_state_i = link_states[i]

        return (
            np.concatenate([dvec, dvec_i], axis=0),
            np.concatenate([rx_types, rx_type_i], axis=0),
            np.concatenate([link_states, link_state_i], axis=0)
        )


    def _transform_links(self,
        dvec: np.ndarray, rx_types: np.ndarray, fit: bool=False
    ) -> np.ndarray:
        """
        """
        dr = np.linalg.norm(dvec, axis=1, keepdims=True)
        dh = dvec[:,2:3]
        if self.rx_type_encoder is None:
            raise RuntimeError("Encoder not initialized. Call `_prepare_arrays`")
        
        rx_one = self.rx_type_encoder.transform(rx_types[:,None]).astype(np.float32)
        x = np.hstack([rx_one*dr, rx_one*dh])
        if self.link_scaler is None:
            raise RuntimeError("link_scaler not initialized. Call `_prepare_arrays`")
        
        return self.link_scaler.fit_transform(x) if fit else self.link_scaler.transform(x)
