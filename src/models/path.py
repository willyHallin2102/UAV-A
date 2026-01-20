"""
    src / models / path.py
    ----------------------
    Preprocessing the path loss, angles of arrival (AoA) as well as the 
    departure (AoD), and delays within the path object. This  conduct 
    these transformations and then pass the transformed data passed to 
    the generative artificial model processed with the  actual generation 
    of UAV trajectories.
"""
from __future__ import annotations

import orjson
import numpy as np
import tensorflow as tf
tfk = tf.keras

from pathlib import Path
from typing import Final, List, Optional, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from src.maths.coords import cartesian_to_spherical,add_angles,sub_angles
from src.configs.model import get_config
from src.configs.data import LinkState, AngleIndex
from src.configs.const import CONFIG_FN, PREPROC_FN, LIGHT_SPEED



class PathModel:
    """
    A model for learning and generating wireless multi-path characteristics
    such as path loss, angle of arrival/departure, and delays. This object
    provides all the preprocessing, model building (e.g., assign the gen ai
    model), convert the dataset and pass it to the generative ai model.
        - Vae
    """
    ANGLE_SCALE: Final[float] = 180.0

    # Add seed as config??

    def __init__(self,
        directory: Union[str,Path], model_type: str, rx_types: List[str],
        n_max_paths: int, max_path_loss: float
    ):
        """
            Initialize Path - Model Instance
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.model_type = model_type.lower()
        self.config = get_config(self.model_type)
        self.model: Optional[tfk.Model] = None

        self.rx_types = list(rx_types)
        self.n_max_paths = int(n_max_paths)
        self.max_path_loss = float(max_path_loss)

        self.__version__ = 1
        self._initialize_preprocessors()

    # ======================================================================
    #       Saving / Loading Methods
    # ======================================================================
    # Versions should be controlled better
    # def save(self):
    #     from src.models.utils.preproc import serialize_preproc

    #     # self.model.save() # Generative ai save itself
    #     with open(self.directory / CONFIG_FN, "wb") as fp:
    #         fp.write(orjson.dumps({
    #             "version": self.__version__,
    #             "framework": {
    #                 "tensorflow": tf.__version__,
    #             },
    #             "config": {
    #                 "rx_types": list(self.rx_types),
    #                 "n_max_paths": self.n_max_paths,
    #                 "max_path_loss": self.max_path_loss
    #             }
    #         }, option=orjson.OPT_INDENT_2))
        
    #     with open(self.directory / PREPROC_FN) as fp:
    #         fp.write(orjson.dumps({
    #             "path_loss_scaler": serialize_preproc(self.path_loss_scaler),
    #             "condition_scaler": serialize_preproc(self.condition_scaler),
    #             "rx_encoder": serialize_preproc(self.rx_types),
    #             "delay_scaler": self.delay_scaler
    #         }, option=orjson.OPT_INDENT_2))
    

    # def load(self):
    #     from src.models.utils.preproc import deserialize_preproc
    #     with open(self.directory / CONFIG_FN, "rb") as fp:
    #         payload = orjson.loads(fp.read())
        
    #     if payload.get("version",0) != self.__version__: print(
    #             f"Warning: Version mismatch. Model: {payload.get('version')}, "
    #             f"Current: {self.__version__}"
    #         )

    #     config = payload.get("config",{})
    #     self.rx_types = config.get("rx_types",self.rx_types)
    #     self.n_max_paths = config.get("n_max_paths", self.n_max_paths)
    #     self.max_path_loss = config.get("max_path_loss", self.max_path_loss)

    #     with open(self.directory / PREPROC_FN, "rb") as fp:
    #         _payload = orjson.loads(fp.read())
        
    #     self.delay_scaler = _payload.get("delay_scaler",self.delay_scaler)
    #     self.rx_types = deserialize_preproc(_payload.get("rx_types", self.rx_types))
    #     self.path_loss_scaler = deserialize_preproc(
    #         _payload.get("path_loss_scaler", self.path_loss_scaler)
    #     )
    #     self.rx_encoder = deserialize_preproc(_payload.get("rx_encoder", self.rx_encoder))

    #     self.build()
    #     self.model.load(self.directory)

    #     print(f"Model loaded from {self.directory}")

    # ======================================================================
    #       Construction Models 
    # ======================================================================

    def build(self):
        if self.model_type == "vae":
            from src.models.generators.vae import Vae
            # print(self.config.dropout_rate)
            self.model = Vae(
                n_latent=self.config.n_latent,
                n_data=self.n_max_paths * (2 + AngleIndex.n_angles),
                n_conditions=3 + max(len(self.rx_types), 1),
                encoder_layers=self.config.encoder_layers,
                decoder_layers=self.config.decoder_layers,
                min_variance=self.config.min_variance,
                dropout_rate=self.config.dropout_rate,
                beta_annealing_step=self.config.beta_annealing_step,
                kl_warmup_steps=self.config.kl_warmup_steps,
                init_kernel=self.config.init_kernel, init_bias=self.config.init_bias,
                n_sort=self.n_max_paths
            )
        
        else:
            raise ValueError(f"Unsupported model type: `{self.model_type}`")
    

    
    def fit(self,
        dtr: Dict[str,np.ndarray], dts: Dict[str,np.ndarray],
        epochs: int=100, batch: int=512, learning_rate: float=1e-4
    ) -> tfk.callbacks.History:
        xtr = self._prepare_dataset(dtr, batch, True)
        xts = self._prepare_dataset(dts, batch, False)

        self.model.compile(
            optimizer=tfk.optimizers.Adam(learning_rate=learning_rate,clipvalue=1.0),
            run_eagerly=True
        )

        history = self.model.fit(x=xtr, validation_data=xts, epochs=epochs)
        self.history = history
        
        return history


    # ======================================================================
    #       Internal Methods for the Path Model
    # ======================================================================

    def _initialize_preprocessors(self):
        self.path_loss_scaler = MinMaxScaler()
        self.condition_scaler = StandardScaler()
        self.rx_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.delay_scale = 1.0


    def _prepare_dataset(self,d: Dict[str,np.ndarray], batch: int, fit: bool)->tf.data.Dataset:
        link_state = np.asarray(d["link_state"])
        mask = link_state != LinkState.NO_LINK
        if not np.any(mask):
            raise ValueError("No valid links found in the input data")
        
        i = np.flatnonzero(mask)
        dvec = np.asarray(d["dvec"][i], dtype=np.float32)
        rx = np.asarray(d["rx_type"][i], dtype=np.float32)
        los = (link_state[i] == LinkState.LOS).astype(np.float32)

        conds = self._transform_conditions(dvec, rx, los, fit)
        x = self._transform_data(
            dvec, np.asarray(d["nlos_pl"][i]), np.asarray(d["nlos_pl"][i]),
            np.asarray(link_state[i] == LinkState.LOS).astype(np.float32)
        )
        if conds.shape[0] != x.shape[0]: raise ValueError(
            f"Conditional/Path sample mismatch: {conds.shape[0]} vs {x.shape[0]}"
        )
        ds = tf.data.Dataset.from_tensor_slices((x, conds))
        ds = ds.batch(batch).cache().prefetch(tf.data.AUTOTUNE)

        return ds

    
    # ======================================================================
    #       Data Transformations
    # ======================================================================

    def _transform_conditions(self,
        dvec: np.ndarray, rx_types: np.ndarray, los: np.ndarray, fit: bool
    ) -> np.ndarray:
        """ """
        d3d = np.maximum(np.linalg.norm(dvec,axis=1),1.0)
        dh = dvec[:,2]

        base = np.stack([d3d,np.log10(d3d),dh,los], axis=1)
        if fit: rx_one = self.rx_encoder.fit_transform(rx_types-reshape(-1,1))
        else: rx_one = self.rx_encoder.transform(rx_types.reshape(-1,1))

        if rx_one.shape[1] > 1: rx_one[:,:-1]

        conds = np.concatenate((base,rx_one), axis=1)
        if fit: self.condition_scaler.fit_transform(conds)
        else: self.condition_scaler.transform(conds)

        return conds.astype(np.float32)
    

    def _transform_data(self,
        dvec: np.ndarray, nlos_path_loss: np.ndarray, nlos_angles: np.ndarray,
        nlos_delays: np.ndarray, fit: bool
    ) -> np.ndarray: return np.hstack([
            self._transform_path_loss(nlos_path_loss, fit=fit),
            self._transform_angles(dvec, nlos_angles),
            self._transform_delays(dvec, nlos_delays, fit=fit)
        ]).astype(np.float32, copy=False)
    

    def _inverse_transform_data(self, 
        dvec: np.ndarray, x: np.ndarray
    ) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        nmp = self.n_max_paths
        n_ang = 4 * nmp
        
        pl, ang, dly = x[:,:nmp], x[:,nmp:nmp+n_ang], x[:,nmp+n_ang:]
        return (
            self._inverse_transform_path_loss(pl),
            self._inverse_transform_angles(ang),
            self._inverse_transform_delays(dly)
        )

    # ======================================================================
    #       Path-Loss Transformations
    # ======================================================================

    def _transform_path_loss(self, nlos_path_loss: np.ndarray, fit: bool) -> np.ndarray:
        x = self.max_path_loss - nlos_path_loss[:,:self.n_max_paths]
        return self.path_loss_scaler.fit_transform(x) if fit else self.path_loss_scaler.transform(x)
    

    def _inverse_transform_path_loss(self, path_loss: np.ndarray, fit: bool) -> np.ndarray:
        x = np.clip(path_loss, 0.0, 1.0)
        x = self.path_loss_scaler.inverse_transform(x)
        x = np.fliplr(np.sort(x, axis=-1))

        return self.max_path_loss - x

    # ======================================================================
    #       Angular Transformations
    # ======================================================================

    def _transform_angles(self, dvec: np.ndarray, nlos: np.ndarray) -> np.ndarray:
        _, los_aoa_phi, los_aoa_theta = cartesian_to_spherical(-dvec)
        _, los_aod_phi, los_aod_theta = cartesian_to_spherical(dvec)

        aoa_phi_rel, aoa_theta_rel = sub_angles(
            nlos[..., AngleIndex.AOA_PHI], nlos[..., AngleIndex.AOA_THETA],
            los_aoa_phi[:,None], nlos_aoa_theta[:,None]
        )
        aod_phi_rel, aod_theta_rel = sub_angles(
            nlos[...,AngleIndex.AOD_PHI], nlos[...,AngleIndex.AOD_THETA],
            los_aod_phi[:,None], los_aod_theta[:,None]
        )

        nmp = self.n_max_paths
        out = np.empty((dvec.shape[0], 4*nmp), dtype=np.float32)
        out[:,0:nmp] = aoa_phi_rel / self.ANGLE_SCALE
        out[:,nmp:2*nmp] = aoa_theta_rel / self.ANGLE_SCALE
        out[:,2*nmp:3*nmp] = aod_phi_rel / self.ANGLE_SCALE
        out[:,3*nmp:4*nmp] = aod_theta_rel / self.ANGLE_SCALE

        return out


    def _inverse_transform_angles(self,dvec:np.ndarray,ang:np.ndarray) -> np.ndarray:
        nmp = self.n_max_paths
        aoa_phi_rel = ang[:,0:nmp] * self.ANGLE_SCALE
        aoa_theta_rel = ang[:,nmp:2*nmp] * self.ANGLE_SCALE
        aod_phi_rel = ang[:,2*nmp:3*nmp] * self.ANGLE_SCALE
        aod_theta_rel = ang[:,3*nmp:4*nmp] * self.ANGLE_SCALE

        _, los_aoa_phi, los_aoa_theta = cartesian_to_spherical(dvec)
        _, los_aod_phi, los_aod_theta = cartesian_to_spherical(-dvec)

        nlos_aoa_phi, nlos_aoa_theta = add_angles(
            aoa_phi_rel, aoa_theta_rel, 
            los_aoa_phi[:,None], los_aoa_theta[:,None]
        )
        nlos_aod_phi, nlos_aod_theta = add_angles(
            aod_phi_rel, aod_theta_rel,
            los_aod_phi[:,None], los_aod_theta[:,None]
        )

        out = np.zeros((dvec.shape[0], nmp, AngleIndex.n_angles))
        out[...,AngleIndex.AOA_PHI] = nlos_aoa_phi
        out[...,AngleIndex.AOA_THETA] = nlos_aoa_theta
        out[...,AngleIndex.AOD_PHI] = nlos_aod_phi
        out[...,AngleIndex.AOD_THETA] = nlos_aod_theta

        return out

    # ======================================================================
    #       Delay Transformations
    # ======================================================================

    def _transform_delays(dvec: np.ndarray, nlos_delay: np.ndarray, fit: bool) -> np.ndarray:
        dist = np.linalg.norm(dvec, axis=1)
        los_delay = dist / LIGHT_SPEED
        rel = np.maximum(0.0, nlos_delay - los_delay[:,None])

        if fit: self.delay_scale = np.mean(rel) or 1.0
        return (rel / self.delay_scale).astype(np.float32)
    

    def _inverse_transform_delays(
        self, dvec: np.ndarray, delays: np.ndarray
    ) -> np.ndarray:
        # Compute LOS delays
        dist = np.linalg.norm(dvec,axis=1)
        los_delays = dist / LIGHT_SPEED
        return delays * self.delay_scale + los_delays[:,None]
