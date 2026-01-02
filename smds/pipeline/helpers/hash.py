import hashlib
import json
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray


def hash_data(X: NDArray[np.float64], y: NDArray[np.float64]) -> str:
    X_cont = X if X.flags["C_CONTIGUOUS"] else np.ascontiguousarray(X)
    y_cont = y if y.flags["C_CONTIGUOUS"] else np.ascontiguousarray(y)

    hasher = hashlib.md5()
    hasher.update(X_cont.view(np.uint8))
    hasher.update(y_cont.view(np.uint8))
    return hasher.hexdigest()


def compute_shape_hash(
    shape_name: str,
    params: Dict[str, Any],
    data_hash: str,
    n_folds: int,
) -> str:
    safe_params = {k: str(v) for k, v in params.items()}
    params_str = json.dumps(safe_params, sort_keys=True)
    config_str = f"{shape_name}_{params_str}_{data_hash}_{n_folds}"
    return hashlib.md5(config_str.encode()).hexdigest()


def load_cached_shape_result(cache_file: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, "rb") as f:
            result: Dict[str, Any] = pickle.load(f)
            return result
    except (pickle.UnpicklingError, OSError, ValueError, TypeError):
        return None


def save_shape_result(cache_file: str, result_data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(result_data, f)
