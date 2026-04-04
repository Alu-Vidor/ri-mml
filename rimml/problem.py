from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray
ScalarFunction = Callable[[Array], Array | float]


def _as_1d_array(x: float | Array) -> Array:
    return np.atleast_1d(np.asarray(x, dtype=float))


def _wrap_function(function: ScalarFunction) -> Callable[[float | Array], Array]:
    def wrapped(x: float | Array) -> Array:
        points = _as_1d_array(x)
        values = function(points)
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            array = np.full(points.shape, float(array), dtype=float)
        return array.reshape(points.shape)

    return wrapped


@dataclass(frozen=True)
class FractionalProblem:
    """Model of the singularly perturbed fractional problem from the article."""

    alpha: float
    epsilon: float
    interval_end: float
    u0: float
    coefficient: ScalarFunction
    forcing: ScalarFunction
    reference_coefficient: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must satisfy 0 < alpha < 1")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.interval_end <= 0.0:
            raise ValueError("interval_end must be positive")
        object.__setattr__(self, "_coefficient", _wrap_function(self.coefficient))
        object.__setattr__(self, "_forcing", _wrap_function(self.forcing))

        a0 = float(self._coefficient(np.array([0.0]))[0])
        if a0 <= 0.0:
            raise ValueError("a(0) must be positive")
        object.__setattr__(self, "_a0", a0)

        if self.reference_coefficient is None:
            a_c = a0
        else:
            a_c = float(self.reference_coefficient)
        if a_c <= 0.0:
            raise ValueError("reference_coefficient must be positive")
        object.__setattr__(self, "_reference_coefficient", a_c)

    @property
    def a0(self) -> float:
        return self._a0

    @property
    def a_c(self) -> float:
        return self._reference_coefficient

    def evaluate_coefficient(self, x: float | Array) -> Array:
        return self._coefficient(x)

    def evaluate_forcing(self, x: float | Array) -> Array:
        return self._forcing(x)
