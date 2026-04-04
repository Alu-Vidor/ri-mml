from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rimml import mittag_leffler

from .api import ProblemConfig, SolverRunConfig, solve_fractional_ivp
from .utils import RESULTS_DIR, write_csv


@dataclass(frozen=True)
class ExactCase:
    name: str
    alpha: float
    epsilon: float
    v_star: float
    u0: float
    interval_end: float = 1.0


def exact_solution(case: ExactCase, t: np.ndarray) -> np.ndarray:
    ml = np.asarray(mittag_leffler(case.alpha, 1.0, -(t**case.alpha) / case.epsilon), dtype=float)
    return case.v_star + (case.u0 - case.v_star) * ml


def run_exact_benchmarks() -> dict:
    alphas = [0.45, 0.75, 0.95]
    epsilons = [1e-1, 1e-3, 1e-5]
    basis_sizes = [4, 6, 8, 10, 12, 16, 20]
    cases = []
    for alpha in alphas:
        for epsilon in epsilons:
            cases.append(ExactCase("charging", alpha, epsilon, v_star=1.0, u0=0.2))
            cases.append(ExactCase("discharging", alpha, epsilon, v_star=0.0, u0=1.0))

    rows: list[dict] = []
    for case in cases:
        for basis_size in basis_sizes:
            solution = solve_fractional_ivp(
                ProblemConfig(
                    alpha=case.alpha,
                    epsilon=case.epsilon,
                    interval_end=case.interval_end,
                    u0=case.u0,
                    coefficient=lambda x: np.ones_like(x),
                    forcing=lambda x, value=case.v_star: value * np.ones_like(x),
                    reference_coefficient=1.0,
                    evaluation_points=1500,
                ),
                SolverRunConfig(basis_size=basis_size),
            )
            exact = exact_solution(case, solution.t)
            error = solution.u - exact
            rows.append(
                {
                    "case": case.name,
                    "alpha": case.alpha,
                    "epsilon": case.epsilon,
                    "N": basis_size,
                    "l_inf": float(np.max(np.abs(error))),
                    "l2": float(np.sqrt(np.mean(error**2))),
                    "runtime_seconds": solution.runtime_seconds,
                    "residual_norm": solution.residual_norm,
                    "condition_number": solution.condition_number,
                }
            )

    frame = pd.DataFrame(rows)
    write_csv(RESULTS_DIR / "metrics_exact.csv", frame)
    write_csv(RESULTS_DIR / "tables" / "exact_convergence.csv", frame)
    _plot_exact(frame)
    return {
        "rows": len(frame),
        "alphas": alphas,
        "epsilons": epsilons,
        "basis_sizes": basis_sizes,
    }


def _plot_exact(frame: pd.DataFrame) -> None:
    metrics = [
        ("l_inf", "L∞ error", "exact_convergence_linf.png"),
        ("l2", "L2 error", "exact_convergence_l2.png"),
        ("runtime_seconds", "runtime (s)", "exact_convergence_runtime.png"),
        ("residual_norm", "residual norm", "exact_convergence_residual.png"),
    ]
    for metric, ylabel, filename in metrics:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        grouped = list(frame.groupby("case"))
        for idx, ax in enumerate(axes.flat):
            case_name, case_df = grouped[idx % len(grouped)]
            for (alpha, epsilon), sub_df in case_df.groupby(["alpha", "epsilon"]):
                label = f"alpha={alpha}, eps={epsilon:g}"
                ax.plot(sub_df["N"], sub_df[metric], marker="o", label=label)
            ax.set_title(case_name)
            ax.set_xlabel("N")
            ax.set_ylabel(ylabel)
            if metric in {"l_inf", "l2", "residual_norm"} and (case_df[metric] > 0).any():
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
        fig.savefig(RESULTS_DIR / "plots" / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    summary = frame.groupby(["case", "alpha", "epsilon"], as_index=False)["l_inf"].min()
    for case_name, case_df in summary.groupby("case"):
        ax.plot(case_df["epsilon"], case_df["l_inf"], marker="o", label=case_name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("best L∞ error over N")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(RESULTS_DIR / "plots" / "exact_error_vs_epsilon.png", dpi=180)
    plt.close(fig)
