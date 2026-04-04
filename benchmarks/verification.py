from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gamma

from rimml import ExponentSemigroup, mittag_leffler

from .api import ProblemConfig, SolverRunConfig, solve_fractional_ivp
from .utils import CONFIGS_DIR, RESULTS_DIR, write_csv, write_json


SEED = 0
EVALUATION_POINTS = 2049
BASIS_SIZES = [4, 6, 8, 10, 12, 16, 20, 24]


@dataclass(frozen=True)
class ExactVerificationCase:
    case_id: str
    alpha: float
    epsilon: float
    coefficient_value: float
    equilibrium_value: float
    u0: float
    interval_end: float


@dataclass(frozen=True)
class ManufacturedCase:
    case_id: str
    family: str
    alpha: float
    epsilon: float
    u0: float
    interval_end: float
    coefficient_kind: str


def _constant(values: np.ndarray, scalar: float) -> np.ndarray:
    return np.full_like(values, scalar, dtype=float)


def _caputo_power(alpha: float, exponent: float, t: np.ndarray) -> np.ndarray:
    if exponent == 0.0:
        return np.zeros_like(t, dtype=float)
    prefactor = gamma(exponent + 1.0) / gamma(exponent + 1.0 - alpha)
    return prefactor * np.power(t, exponent - alpha)


def _power_series(t: np.ndarray, terms: list[tuple[float, float]], u0: float) -> np.ndarray:
    values = np.full_like(t, u0, dtype=float)
    for coefficient, exponent in terms:
        values = values + coefficient * np.power(t, exponent)
    return values


def _caputo_series(alpha: float, t: np.ndarray, terms: list[tuple[float, float]]) -> np.ndarray:
    values = np.zeros_like(t, dtype=float)
    for coefficient, exponent in terms:
        values = values + coefficient * _caputo_power(alpha, exponent, t)
    return values


def _manufactured_case_data(case: ManufacturedCase, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = case.alpha
    if case.family == "smooth_analytic":
        terms = [(0.4, 1.0), (-0.15, 2.0), (0.05, 3.0)]
        coefficient = 1.0 + 0.25 * t
    elif case.family == "weak_endpoint_singularity":
        terms = [(0.8, alpha), (-0.2, 2.0 * alpha), (0.15, 1.0 + alpha)]
        coefficient = 1.0 + 0.1 * np.power(t, alpha)
    elif case.family == "trial_space_aligned":
        semigroup = ExponentSemigroup(alpha)
        exponents = [float(value) for value in semigroup.first(6)[1:5]]
        terms = [(0.9, exponents[0]), (-0.4, exponents[1]), (0.2, exponents[2]), (-0.1, exponents[3])]
        coefficient = _constant(t, 1.0)
    else:
        raise ValueError(case.family)

    exact = _power_series(t, terms, case.u0)
    forcing = case.epsilon * _caputo_series(alpha, t, terms) + coefficient * exact
    return exact, coefficient, forcing


def _exact_solution(case: ExactVerificationCase, t: np.ndarray) -> np.ndarray:
    argument = -(case.coefficient_value / case.epsilon) * np.power(t, case.alpha)
    ml = np.asarray(mittag_leffler(case.alpha, 1.0, argument), dtype=float)
    return case.equilibrium_value + (case.u0 - case.equilibrium_value) * ml


def _error_metrics(u_num: np.ndarray, u_ref: np.ndarray) -> dict[str, float]:
    error = np.asarray(u_num, dtype=float) - np.asarray(u_ref, dtype=float)
    abs_error = np.abs(error)
    l2_abs = float(np.sqrt(np.mean(error**2)))
    l2_ref = float(np.sqrt(np.mean(np.asarray(u_ref, dtype=float) ** 2)))
    return {
        "l2_abs": l2_abs,
        "l2_rel": l2_abs / max(l2_ref, 1e-30),
        "l_inf": float(np.max(abs_error)),
    }


def _observed_rates(frame: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, sub_df in frame.groupby(group_cols, dropna=False):
        ordered = sub_df.sort_values("N")
        previous_n = None
        previous_error = None
        for row in ordered.itertuples(index=False):
            current_error = float(getattr(row, metric))
            rate = None
            if previous_n is not None and previous_error is not None and current_error > 0.0 and previous_error > 0.0:
                rate = float(np.log(previous_error / current_error) / np.log(float(row.N) / previous_n))
            row_dict = {column: value for column, value in zip(group_cols, keys if isinstance(keys, tuple) else (keys,), strict=False)}
            row_dict.update({"N": int(row.N), f"{metric}_observed_rate": rate})
            rows.append(row_dict)
            previous_n = float(row.N)
            previous_error = current_error
    return pd.DataFrame(rows)


def _save_verification_plots(exact: pd.DataFrame, manufactured: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    exact_best = exact.groupby(["case_id", "alpha", "epsilon"], as_index=False)[["l2_abs", "l_inf"]].min()
    for alpha, sub_df in exact_best.groupby("alpha"):
        axes[0].plot(sub_df["epsilon"], sub_df["l2_abs"], marker="o", label=f"alpha={alpha:g}")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epsilon")
    axes[0].set_ylabel("best absolute L2 error")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    manufactured_best = manufactured.groupby(["family", "alpha", "epsilon", "N"], as_index=False)["l2_abs"].mean()
    for family, sub_df in manufactured_best.groupby("family"):
        grouped = sub_df.groupby("N", as_index=False)["l2_abs"].mean()
        axes[1].plot(grouped["N"], grouped["l2_abs"], marker="o", label=family)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("mean absolute L2 error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.savefig(RESULTS_DIR / "plots" / "verification_overview.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, family in zip(axes, ["smooth_analytic", "weak_endpoint_singularity", "trial_space_aligned"], strict=True):
        family_df = manufactured[manufactured["family"] == family]
        for case_id, sub_df in family_df.groupby("case_id"):
            axis.plot(sub_df["N"], sub_df["l2_abs"], marker="o", label=case_id)
        axis.set_title(family.replace("_", " "))
        axis.set_yscale("log")
        axis.set_xlabel("N")
        axis.set_ylabel("absolute L2 error")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    fig.savefig(RESULTS_DIR / "plots" / "manufactured_convergence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    exact_residual = exact.groupby("N", as_index=False)["relative_residual_norm"].max()
    manufactured_residual = manufactured.groupby("N", as_index=False)["relative_residual_norm"].max()
    ax.plot(exact_residual["N"], exact_residual["relative_residual_norm"], marker="o", label="closed-form exact")
    ax.plot(manufactured_residual["N"], manufactured_residual["relative_residual_norm"], marker="s", label="manufactured")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("max relative linear residual")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(RESULTS_DIR / "plots" / "verification_residuals.png", dpi=180)
    plt.close(fig)


def run_verification_suite() -> dict:
    np.random.seed(SEED)

    exact_cases = [
        ExactVerificationCase("mlf_charge_a0p75", 0.35, 1e-1, 0.75, 1.0, 0.2, 1.0),
        ExactVerificationCase("mlf_charge_a1p50", 0.65, 1e-2, 1.50, 1.2, -0.1, 1.0),
        ExactVerificationCase("mlf_discharge_a1p00", 0.85, 5e-3, 1.00, 0.0, 1.0, 1.0),
        ExactVerificationCase("mlf_charge_small_eps", 0.92, 1e-4, 1.00, 0.8, 0.05, 0.5),
    ]
    manufactured_cases = [
        ManufacturedCase("smooth_a035_eps1e-2", "smooth_analytic", 0.35, 1e-2, 0.15, 1.0, "affine"),
        ManufacturedCase("smooth_a075_eps1e-3", "smooth_analytic", 0.75, 1e-3, -0.1, 1.0, "affine"),
        ManufacturedCase("singular_a035_eps1e-2", "weak_endpoint_singularity", 0.35, 1e-2, 0.4, 1.0, "fractional_power"),
        ManufacturedCase("singular_a075_eps5e-4", "weak_endpoint_singularity", 0.75, 5e-4, 0.1, 1.0, "fractional_power"),
        ManufacturedCase("aligned_a045_eps1e-3", "trial_space_aligned", 0.45, 1e-3, 0.2, 1.0, "constant"),
        ManufacturedCase("aligned_a080_eps1e-4", "trial_space_aligned", 0.80, 1e-4, -0.05, 1.0, "constant"),
    ]

    solver_config_payload = {
        "seed": SEED,
        "evaluation_points": EVALUATION_POINTS,
        "basis_sizes": BASIS_SIZES,
        "solver": {
            "quadrature_panels": 48,
            "quadrature_order": 10,
            "quadrature_grading": 2.5,
            "exponent_tolerance": 1e-12,
            "mittag_leffler_tolerance": 1e-13,
            "mittag_leffler_max_terms": 2000,
        },
        "exact_cases": [case.__dict__ for case in exact_cases],
        "manufactured_cases": [case.__dict__ for case in manufactured_cases],
    }
    write_json(CONFIGS_DIR / "verification_suite.json", solver_config_payload)

    exact_rows: list[dict] = []
    for case in exact_cases:
        for basis_size in BASIS_SIZES:
            solution = solve_fractional_ivp(
                ProblemConfig(
                    alpha=case.alpha,
                    epsilon=case.epsilon,
                    interval_end=case.interval_end,
                    u0=case.u0,
                    coefficient=lambda x, value=case.coefficient_value: _constant(x, value),
                    forcing=lambda x, value=case.coefficient_value * case.equilibrium_value: _constant(x, value),
                    reference_coefficient=case.coefficient_value,
                    evaluation_points=EVALUATION_POINTS,
                ),
                SolverRunConfig(basis_size=basis_size),
            )
            reference = _exact_solution(case, solution.t)
            metrics = _error_metrics(solution.u, reference)
            exact_rows.append(
                {
                    "validation_category": "verification_exact",
                    "suite": "closed_form_exact",
                    "case_id": case.case_id,
                    "alpha": case.alpha,
                    "epsilon": case.epsilon,
                    "coefficient_value": case.coefficient_value,
                    "u0": case.u0,
                    "equilibrium_value": case.equilibrium_value,
                    "N": basis_size,
                    "evaluation_points": EVALUATION_POINTS,
                    "seed": SEED,
                    **metrics,
                    "runtime_seconds": solution.runtime_seconds,
                    "residual_norm": solution.residual_norm,
                    "relative_residual_norm": solution.relative_residual_norm,
                    "rhs_norm": solution.rhs_norm,
                    "condition_number": solution.condition_number,
                }
            )

    manufactured_rows: list[dict] = []
    for case in manufactured_cases:
        for basis_size in BASIS_SIZES:
            reference_grid = np.linspace(0.0, case.interval_end, EVALUATION_POINTS)
            reference, coefficient, forcing = _manufactured_case_data(case, reference_grid)
            solution = solve_fractional_ivp(
                ProblemConfig(
                    alpha=case.alpha,
                    epsilon=case.epsilon,
                    interval_end=case.interval_end,
                    u0=case.u0,
                    coefficient=lambda x, grid=reference_grid, values=coefficient: np.interp(x, grid, values),
                    forcing=lambda x, grid=reference_grid, values=forcing: np.interp(x, grid, values),
                    reference_coefficient=float(coefficient[0]),
                    evaluation_points=EVALUATION_POINTS,
                ),
                SolverRunConfig(basis_size=basis_size),
            )
            metrics = _error_metrics(solution.u, reference)
            manufactured_rows.append(
                {
                    "validation_category": "verification_convergence",
                    "suite": "manufactured_solution",
                    "case_id": case.case_id,
                    "family": case.family,
                    "alpha": case.alpha,
                    "epsilon": case.epsilon,
                    "u0": case.u0,
                    "coefficient_kind": case.coefficient_kind,
                    "N": basis_size,
                    "evaluation_points": EVALUATION_POINTS,
                    "seed": SEED,
                    **metrics,
                    "runtime_seconds": solution.runtime_seconds,
                    "residual_norm": solution.residual_norm,
                    "relative_residual_norm": solution.relative_residual_norm,
                    "rhs_norm": solution.rhs_norm,
                    "condition_number": solution.condition_number,
                }
            )

    exact_frame = pd.DataFrame(exact_rows)
    manufactured_frame = pd.DataFrame(manufactured_rows)
    exact_rates = _observed_rates(exact_frame, ["case_id", "alpha", "epsilon"], "l2_abs")
    manufactured_rates = _observed_rates(manufactured_frame, ["case_id", "family", "alpha", "epsilon"], "l2_abs")
    convergence_summary = manufactured_frame.groupby(["family", "case_id"], as_index=False).agg(
        best_l2_abs=("l2_abs", "min"),
        best_l2_rel=("l2_rel", "min"),
        best_l_inf=("l_inf", "min"),
        max_condition_number=("condition_number", "max"),
        max_relative_residual_norm=("relative_residual_norm", "max"),
    )

    write_csv(RESULTS_DIR / "verification_exact.csv", exact_frame)
    write_csv(RESULTS_DIR / "verification_convergence.csv", manufactured_frame)
    write_csv(RESULTS_DIR / "tables" / "verification_exact_best.csv", exact_frame.groupby(["case_id", "alpha", "epsilon"], as_index=False)[["l2_abs", "l2_rel", "l_inf"]].min())
    write_csv(RESULTS_DIR / "tables" / "verification_exact_rates.csv", exact_rates)
    write_csv(RESULTS_DIR / "tables" / "verification_convergence_best.csv", convergence_summary)
    write_csv(RESULTS_DIR / "tables" / "verification_convergence_rates.csv", manufactured_rates)
    _save_verification_plots(exact_frame, manufactured_frame)

    best_exact = exact_frame["l2_abs"].min()
    worst_exact = exact_frame["l2_abs"].max()
    best_manufactured = manufactured_frame["l2_abs"].min()
    worst_manufactured = manufactured_frame["l2_abs"].max()

    return {
        "seed": SEED,
        "basis_sizes": BASIS_SIZES,
        "evaluation_points": EVALUATION_POINTS,
        "verification_exact_cases": len(exact_cases),
        "verification_convergence_cases": len(manufactured_cases),
        "best_exact_l2_abs": float(best_exact),
        "worst_exact_l2_abs": float(worst_exact),
        "best_manufactured_l2_abs": float(best_manufactured),
        "worst_manufactured_l2_abs": float(worst_manufactured),
    }
