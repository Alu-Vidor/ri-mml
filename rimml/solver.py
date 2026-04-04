from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import MuntzBasis, OperatorInducedTrialBasis
from .mittag_leffler import mittag_leffler
from .problem import FractionalProblem
from .quadrature import GradedCompositeGaussLegendre


@dataclass(frozen=True)
class SolverConfig:
    basis_size: int
    quadrature_panels: int = 48
    quadrature_order: int = 10
    quadrature_grading: float = 2.5
    exponent_tolerance: float = 1e-12
    mittag_leffler_tolerance: float = 1e-13
    mittag_leffler_max_terms: int = 2000

    def __post_init__(self) -> None:
        if self.basis_size <= 0:
            raise ValueError("basis_size must be positive")


@dataclass(frozen=True)
class SpectralSolution:
    problem: FractionalProblem
    basis: MuntzBasis
    trial_basis: OperatorInducedTrialBasis
    density_coefficients: np.ndarray
    system_matrix: np.ndarray
    rhs: np.ndarray

    def evaluate_density(self, x: float | np.ndarray) -> np.ndarray:
        return self.basis.evaluate(x) @ self.density_coefficients

    def evaluate_remainder(self, x: float | np.ndarray) -> np.ndarray:
        return self.trial_basis.evaluate(x) @ self.density_coefficients

    def evaluate_layer(self, x: float | np.ndarray) -> np.ndarray:
        points = np.atleast_1d(np.asarray(x, dtype=float))
        values = mittag_leffler(
            self.problem.alpha,
            1.0,
            -(self.problem.a0 / self.problem.epsilon) * points**self.problem.alpha,
        )
        return np.asarray(values, dtype=float)

    def evaluate(self, x: float | np.ndarray) -> np.ndarray:
        return self.problem.u0 * self.evaluate_layer(x) + self.evaluate_remainder(x)


class SpectralPetrovGalerkinSolver:
    """Full OOP implementation of the method presented in article.tex."""

    def __init__(self, problem: FractionalProblem, config: SolverConfig) -> None:
        self.problem = problem
        self.config = config
        self.quadrature = GradedCompositeGaussLegendre(
            interval_end=problem.interval_end,
            panels=config.quadrature_panels,
            order=config.quadrature_order,
            grading=config.quadrature_grading,
        )
        self.basis = MuntzBasis(
            alpha=problem.alpha,
            interval_end=problem.interval_end,
            size=config.basis_size,
            exponent_tolerance=config.exponent_tolerance,
        )
        self.trial_basis = OperatorInducedTrialBasis(
            basis=self.basis,
            alpha=problem.alpha,
            epsilon=problem.epsilon,
            reference_coefficient=problem.a_c,
            mittag_leffler_tolerance=config.mittag_leffler_tolerance,
            mittag_leffler_max_terms=config.mittag_leffler_max_terms,
        )

    def evaluate_layer(self, x: float | np.ndarray) -> np.ndarray:
        points = np.atleast_1d(np.asarray(x, dtype=float))
        values = mittag_leffler(
            self.problem.alpha,
            1.0,
            -(self.problem.a0 / self.problem.epsilon) * points**self.problem.alpha,
            tol=self.config.mittag_leffler_tolerance,
            max_terms=self.config.mittag_leffler_max_terms,
        )
        return np.asarray(values, dtype=float)

    def evaluate_hidden_rhs(self, x: float | np.ndarray) -> np.ndarray:
        points = np.atleast_1d(np.asarray(x, dtype=float))
        forcing = self.problem.evaluate_forcing(points)
        coefficient = self.problem.evaluate_coefficient(points)
        layer = self.evaluate_layer(points)
        return forcing - self.problem.u0 * (coefficient - self.problem.a0) * layer

    def assemble_system(self) -> tuple[np.ndarray, np.ndarray]:
        nodes = self.quadrature.nodes
        weights = self.quadrature.weights
        test_values = self.basis.evaluate(nodes)
        trial_values = self.trial_basis.evaluate(nodes)
        weighted_test = test_values * weights[:, None]

        coefficient = self.problem.evaluate_coefficient(nodes)
        hidden_rhs = self.evaluate_hidden_rhs(nodes)

        perturbation = (coefficient - self.problem.a_c)[:, None] * trial_values
        matrix = np.eye(self.config.basis_size) + weighted_test.T @ perturbation
        rhs = weighted_test.T @ hidden_rhs
        return matrix, rhs

    def solve(self) -> SpectralSolution:
        matrix, rhs = self.assemble_system()
        density_coefficients = np.linalg.solve(matrix, rhs)
        return SpectralSolution(
            problem=self.problem,
            basis=self.basis,
            trial_basis=self.trial_basis,
            density_coefficients=density_coefficients,
            system_matrix=matrix,
            rhs=rhs,
        )
