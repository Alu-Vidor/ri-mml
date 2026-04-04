from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special


@dataclass(frozen=True)
class GradedCompositeGaussLegendre:
    """Composite Gauss-Legendre rule on a graded mesh clustered near x = 0."""

    interval_end: float
    panels: int = 32
    order: int = 12
    grading: float = 2.0

    def __post_init__(self) -> None:
        if self.interval_end <= 0.0:
            raise ValueError("interval_end must be positive")
        if self.panels <= 0 or self.order <= 0:
            raise ValueError("panels and order must be positive integers")
        if self.grading <= 0.0:
            raise ValueError("grading must be positive")

        roots, weights = special.roots_legendre(self.order)
        fractions = np.linspace(0.0, 1.0, self.panels + 1) ** self.grading
        boundaries = self.interval_end * fractions

        nodes = []
        quad_weights = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            midpoint = 0.5 * (left + right)
            half_width = 0.5 * (right - left)
            nodes.append(midpoint + half_width * roots)
            quad_weights.append(half_width * weights)

        object.__setattr__(self, "_nodes", np.concatenate(nodes))
        object.__setattr__(self, "_weights", np.concatenate(quad_weights))

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def integrate(self, values: np.ndarray) -> float:
        return float(np.dot(self.weights, np.asarray(values, dtype=float)))
