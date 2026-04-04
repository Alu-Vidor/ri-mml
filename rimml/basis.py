from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy import linalg, special

from .mittag_leffler import mittag_leffler


@dataclass(frozen=True)
class ExponentSemigroup:
    """Enumerator for the additive semigroup Lambda_alpha = N_0 + alpha N_0."""

    alpha: float
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must satisfy 0 < alpha < 1")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")

    def first(self, size: int) -> np.ndarray:
        if size <= 0:
            raise ValueError("size must be positive")

        exponents: list[float] = []
        heap = [0.0]
        seen = {0}
        scale = 1.0 / self.tolerance

        while len(exponents) < size:
            value = heapq.heappop(heap)
            if not exponents or abs(value - exponents[-1]) > self.tolerance:
                exponents.append(value)

            for step in (self.alpha, 1.0):
                candidate = value + step
                key = round(candidate * scale)
                if key not in seen:
                    seen.add(key)
                    heapq.heappush(heap, candidate)

        return np.asarray(exponents, dtype=float)


@dataclass(frozen=True)
class MuntzBasis:
    """L2-orthonormal Muntz basis obtained from the explicit Gram matrix."""

    alpha: float
    interval_end: float
    size: int
    exponent_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        semigroup = ExponentSemigroup(self.alpha, self.exponent_tolerance)
        exponents = semigroup.first(self.size)
        gram = self.interval_end ** (exponents[:, None] + exponents[None, :] + 1.0)
        gram = gram / (exponents[:, None] + exponents[None, :] + 1.0)
        try:
            chol = np.linalg.cholesky(gram)
            coefficients = linalg.solve_triangular(chol, np.eye(self.size), lower=True)
        except np.linalg.LinAlgError:
            # For larger irrational exponent sets the Gram matrix can lose
            # positive definiteness numerically while remaining SPD analytically.
            eigenvalues, eigenvectors = np.linalg.eigh(gram)
            min_eigenvalue = max(float(np.max(eigenvalues)) * 1e-14, 1e-15)
            stabilized = np.maximum(eigenvalues, min_eigenvalue)
            coefficients = np.diag(1.0 / np.sqrt(stabilized)) @ eigenvectors.T

        object.__setattr__(self, "_exponents", exponents)
        object.__setattr__(self, "_gram_matrix", gram)
        object.__setattr__(self, "_coefficients", coefficients)

    @property
    def exponents(self) -> np.ndarray:
        return self._exponents

    @property
    def gram_matrix(self) -> np.ndarray:
        return self._gram_matrix

    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients

    def evaluate_monomials(self, x: float | np.ndarray) -> np.ndarray:
        points = np.atleast_1d(np.asarray(x, dtype=float))
        return points[:, None] ** self.exponents[None, :]

    def evaluate(self, x: float | np.ndarray) -> np.ndarray:
        monomials = self.evaluate_monomials(x)
        return monomials @ self.coefficients.T

    def evaluate_function(self, spectral_coefficients: np.ndarray, x: float | np.ndarray) -> np.ndarray:
        values = self.evaluate(x)
        return values @ np.asarray(spectral_coefficients, dtype=float)


@dataclass(frozen=True)
class OperatorInducedTrialBasis:
    """Trial basis Phi_j = R_c M_j built from the exact monomial image formula."""

    basis: MuntzBasis
    alpha: float
    epsilon: float
    reference_coefficient: float
    mittag_leffler_tolerance: float = 1e-13
    mittag_leffler_max_terms: int = 2000

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.reference_coefficient <= 0.0:
            raise ValueError("reference_coefficient must be positive")

        gamma_factors = special.gamma(self.basis.exponents + 1.0) / self.epsilon
        object.__setattr__(self, "_gamma_factors", gamma_factors)

    def evaluate_monomial_images(self, x: float | np.ndarray) -> np.ndarray:
        points = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.zeros((points.size, self.basis.size), dtype=float)
        positive = points > 0.0
        if not np.any(positive):
            return result

        points_positive = points[positive]
        x_alpha = points_positive**self.alpha
        for column, exponent in enumerate(self.basis.exponents):
            beta = exponent + self.alpha + 1.0
            ml = mittag_leffler(
                self.alpha,
                beta,
                -(self.reference_coefficient / self.epsilon) * x_alpha,
                tol=self.mittag_leffler_tolerance,
                max_terms=self.mittag_leffler_max_terms,
            )
            result[positive, column] = (
                self._gamma_factors[column]
                * points_positive ** (exponent + self.alpha)
                * np.asarray(ml, dtype=float)
            )
        return result

    def evaluate(self, x: float | np.ndarray) -> np.ndarray:
        monomial_images = self.evaluate_monomial_images(x)
        return monomial_images @ self.basis.coefficients.T
