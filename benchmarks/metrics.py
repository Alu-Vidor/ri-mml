from __future__ import annotations

import numpy as np


def interpolate_reference(
    t_source: np.ndarray,
    y_source: np.ndarray,
    t_target: np.ndarray,
) -> np.ndarray:
    return np.interp(t_target, t_source, y_source)


def curve_metrics(y_num: np.ndarray, y_ref: np.ndarray, delta: float = 1e-12) -> dict[str, float]:
    err = np.asarray(y_num) - np.asarray(y_ref)
    abs_err = np.abs(err)
    denom = np.maximum(np.abs(y_ref), delta)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(abs_err)),
        "mre": float(np.mean(abs_err / denom)),
        "e_inf": float(np.max(abs_err)),
    }
