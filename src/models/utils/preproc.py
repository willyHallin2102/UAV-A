"""
src / models / utils / preproc.py
--------------------------------
Lightweight serialization / deserialization of selected sklearn.preprocessing
objects, storing only the fitted state required for inference.

Supported:
- StandardScaler
- MinMaxScaler
- OneHotEncoder
"""

from __future__ import annotations
import numpy as np

from functools import singledispatch
from typing import Any, Dict, Union, Type
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

Preproc = Union[StandardScaler, OneHotEncoder, MinMaxScaler]

# ======================================================================
#   Serialization
# ======================================================================

@singledispatch
def _serialize(proc: Any) -> Dict[str, Any]:
    raise TypeError(f"Unsupported preprocessing type: {type(proc).__name__}")


@_serialize.register(StandardScaler)
def _(proc: StandardScaler) -> Dict[str, Any]:
    return {
        "mean_": proc.mean_.tolist(), "scale_": proc.scale_.tolist(),
        "var_": proc.var_.tolist(), "n_samples_seen_": int(proc.n_samples_seen_),
    }

@_serialize.register(MinMaxScaler)
def _(proc: MinMaxScaler) -> Dict[str, Any]:
    return {
        "data_min_": proc.data_min_.tolist(), "data_max_": proc.data_max_.tolist(),
        "data_range_": proc.data_range_.tolist(), "scale_": proc.scale_.tolist(),
        "min_": proc.min_.tolist(), "feature_range": tuple(proc.feature_range),
    }

@_serialize.register(OneHotEncoder)
def _(proc: OneHotEncoder) -> Dict[str, Any]:
    return {
        "categories_": [cat.tolist() for cat in proc.categories_],

        "drop": proc.drop, "handle_unknown": proc.handle_unknown,
        "dtype": str(proc.dtype),

        "sparse_output": getattr(proc, "sparse_output", None),
        "sparse": getattr(proc, "sparse", None), 
        "min_frequency": getattr(proc, "min_frequency", None),
        "max_categories": getattr(proc, "max_categories", None),
    }

# ======================================================================
#   Deserialization
# ======================================================================

@singledispatch
def _deserialize(cls: Any, params: Dict[str, Any]) -> Preproc:
    raise TypeError(f"Unsupported preprocessor class: {cls}")


@_deserialize.register(type(StandardScaler()))
def _(cls: Type[StandardScaler], p: Dict[str, Any]) -> StandardScaler:
    proc = StandardScaler()
    proc.mean_ = np.asarray(p["mean_"], dtype=float)
    proc.scale_ = np.asarray(p["scale_"], dtype=float)
    proc.var_ = np.asarray(p["var_"], dtype=float)
    proc.n_samples_seen_ = int(p["n_samples_seen_"])
    proc.n_features_in_ = proc.mean_.shape[0]
    return proc

@_deserialize.register(type(MinMaxScaler()))
def _(cls: Type[MinMaxScaler], p: Dict[str, Any]) -> MinMaxScaler:
    proc = MinMaxScaler(feature_range=tuple(p["feature_range"]))
    proc.data_min_ = np.asarray(p["data_min_"], dtype=float)
    proc.data_max_ = np.asarray(p["data_max_"], dtype=float)
    proc.data_range_ = np.asarray(p["data_range_"], dtype=float)
    proc.scale_ = np.asarray(p["scale_"], dtype=float)
    proc.min_ = np.asarray(p["min_"], dtype=float)
    proc.n_features_in_ = proc.data_min_.shape[0]
    return proc

# @_deserialize.register(type(OneHotEncoder()))
# def _(cls: Type[OneHotEncoder], p: Dict[str, Any]) -> OneHotEncoder:
#     kwargs = {
#         "categories": [np.asarray(c) for c in p["categories_"]],
#         "drop": p["drop"], "handle_unknown": p["handle_unknown"], 
#         # "dtype": np.dtype(p["dtype"]), <- crash with this 
#     }
#     if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
#         kwargs["sparse_output"] = bool(p.get("sparse_output", False))
#     else: 
#         kwargs["sparse"] = bool(p.get("sparse", False))

#     if p.get("min_frequency") is not None: kwargs["min_frequency"] = p["min_frequency"]
#     if p.get("max_categories") is not None: kwargs["max_categories"] = p["max_categories"]

#     proc = OneHotEncoder(**kwargs)
#     proc.categories_ = [np.asarray(c) for c in p["categories_"]]
#     proc.n_features_in_ = len(proc.categories_)
#     proc._infrequent_enabled = False

#     return proc

# chatgpt, ... need review later...
@_deserialize.register(type(OneHotEncoder()))
def _(cls: Type[OneHotEncoder], p: Dict[str, Any]) -> OneHotEncoder:
    kwargs = {
        "categories": [np.asarray(c) for c in p["categories_"]],
        "drop": p["drop"], "handle_unknown": p["handle_unknown"],
    }

    if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
        kwargs["sparse_output"] = bool(p.get("sparse_output", False))
    else:
        kwargs["sparse"] = bool(p.get("sparse", False))

    if p.get("min_frequency") is not None:kwargs["min_frequency"] = p["min_frequency"]
    if p.get("max_categories") is not None:kwargs["max_categories"] = p["max_categories"]

    proc = OneHotEncoder(**kwargs)

    # === public fitted attrs ===
    proc.categories_ = [np.asarray(c) for c in p["categories_"]]
    proc.n_features_in_ = len(proc.categories_)

    # === private fitted attrs REQUIRED by transform() ===
    proc._infrequent_enabled, proc._drop_idx_after_grouping = False, None
    proc._n_features_outs = [len(c) for c in proc.categories_]
    proc._feature_indices = np.cumsum([0] + proc._n_features_outs)

    return proc


# ======================================================================
#   Function Calls or API for preproc <-> params
# ======================================================================

def serialize_preproc(proc: Preproc) -> Dict[str, Any]:
    """ Serialize a fitted sklearn preprocessor into a JSON-safe dict. """
    return {"type": type(proc).__name__, "params": _serialize(proc),}

def deserialize_preproc(data: Dict[str, Any]) -> Preproc:
    """ Reconstruct a fitted sklearn preprocessor from serialized state. """
    # Try to get the class by its string name first
    cls_name = data["type"]
    cls_map: Dict[str, Type[Preproc]] = {
        "StandardScaler": StandardScaler, 
        "MinMaxScaler": MinMaxScaler,
        "OneHotEncoder": OneHotEncoder,
    }
    internal_cls_map: Dict[str, Type[Preproc]] = {
        "_data.StandardScaler": StandardScaler,
        "_encoders.OneHotEncoder": OneHotEncoder,
        "_data.MinMaxScaler": MinMaxScaler,
    }
    cls = cls_map.get(cls_name)
    if cls is None:
        for key, value in internal_cls_map.items():
            if key in cls_name:
                cls = value
                break
    
    if cls is None: 
        raise ValueError(f"Unknown preprocessor type: {cls_name}")
    
    return _deserialize.dispatch(type(cls()))(cls, data["params"])
