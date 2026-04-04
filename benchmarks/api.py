from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from rimml import FractionalProblem, SolverConfig, SpectralPetrovGalerkinSolver


Array = np.ndarray
ScalarFunction = Callable[[Array], Array | float]


@dataclass(frozen=True)
class ProblemConfig:
    alpha: float
    epsilon: float
    interval_end: float
    u0: float
    coefficient: ScalarFunction
    forcing: ScalarFunction
    reference_coefficient: float | None = None
    evaluation_points: int = 500


@dataclass(frozen=True)
class SolverRunConfig:
    basis_size: int
    quadrature_panels: int = 48
    quadrature_order: int = 10
    quadrature_grading: float = 2.5
    exponent_tolerance: float = 1e-12
    mittag_leffler_tolerance: float = 1e-13
    mittag_leffler_max_terms: int = 2000


@dataclass
class SolutionObject:
    t: Array
    u: Array
    runtime_seconds: float
    residual_norm: float
    relative_residual_norm: float
    rhs_norm: float
    condition_number: float
    modes: int
    system_matrix: Array
    rhs: Array
    metadata: dict
    raw_solution: object


def solve_fractional_ivp(
    problem_config: ProblemConfig,
    solver_config: SolverRunConfig,
) -> SolutionObject:
    problem = FractionalProblem(
        alpha=problem_config.alpha,
        epsilon=problem_config.epsilon,
        interval_end=problem_config.interval_end,
        u0=problem_config.u0,
        coefficient=problem_config.coefficient,
        forcing=problem_config.forcing,
        reference_coefficient=problem_config.reference_coefficient,
    )
    config = SolverConfig(
        basis_size=solver_config.basis_size,
        quadrature_panels=solver_config.quadrature_panels,
        quadrature_order=solver_config.quadrature_order,
        quadrature_grading=solver_config.quadrature_grading,
        exponent_tolerance=solver_config.exponent_tolerance,
        mittag_leffler_tolerance=solver_config.mittag_leffler_tolerance,
        mittag_leffler_max_terms=solver_config.mittag_leffler_max_terms,
    )

    start = perf_counter()
    solver = SpectralPetrovGalerkinSolver(problem, config)
    spectral_solution = solver.solve()
    runtime_seconds = perf_counter() - start

    t = np.linspace(0.0, problem_config.interval_end, problem_config.evaluation_points)
    u = np.asarray(spectral_solution.evaluate(t), dtype=float)
    residual = spectral_solution.system_matrix @ spectral_solution.density_coefficients - spectral_solution.rhs
    rhs_norm = float(np.linalg.norm(spectral_solution.rhs))
    residual_norm = float(np.linalg.norm(residual))
    return SolutionObject(
        t=t,
        u=u,
        runtime_seconds=runtime_seconds,
        residual_norm=residual_norm,
        relative_residual_norm=residual_norm / max(rhs_norm, 1e-30),
        rhs_norm=rhs_norm,
        condition_number=float(np.linalg.cond(spectral_solution.system_matrix)),
        modes=solver_config.basis_size,
        system_matrix=spectral_solution.system_matrix,
        rhs=spectral_solution.rhs,
        metadata={
            "alpha": problem_config.alpha,
            "epsilon": problem_config.epsilon,
            "interval_end": problem_config.interval_end,
            "u0": problem_config.u0,
        },
        raw_solution=spectral_solution,
    )
