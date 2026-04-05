from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erfcx, gamma

from .api import ProblemConfig, SolverRunConfig, solve_fractional_ivp
from .utils import REPORT_DIR, RESULTS_DIR, write_csv, write_json


ALPHA = 0.5
DELTA = 0.55
ALPHA_TABLE = 39.3
BETA = ALPHA_TABLE + DELTA
EPSILON = 1.0 / BETA
U0 = 1.0
COEFFICIENT_VALUE = 1.0
FORCING_VALUE = 0.0
MATERIAL = "Silver"
ARTICLE_CITATION = "M. Caputo, F. Mainardi, 'A New Dissipation Model Based on Memory Mechanism', Pure and Applied Geophysics, 1971."

SPECTRAL_BASIS_SIZES = [4, 6, 8, 10, 12, 16, 20, 24, 32]
L1_GRID_SIZES = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
CONTROL_GRID_POINTS = 40001


@dataclass(frozen=True)
class SilverBenchmarkCase:
    case_id: str
    interval_end: float


@dataclass(frozen=True)
class GridData:
    t: np.ndarray
    exact: np.ndarray


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| empty |\n|---|\n| no rows |"
    safe = frame.copy()
    formatted_rows: list[list[str]] = []
    for row in safe.itertuples(index=False):
        current: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                current.append(f"{float(value):.6e}")
            else:
                current.append(str(value))
        formatted_rows.append(current)
    header = "| " + " | ".join(str(column) for column in safe.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(safe.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in formatted_rows]
    return "\n".join([header, separator, *body])


def _control_grid(interval_end: float) -> np.ndarray:
    s = np.linspace(0.0, np.sqrt(interval_end), CONTROL_GRID_POINTS)
    return s**2


def silver_exact_solution(t: np.ndarray) -> np.ndarray:
    z = BETA * np.sqrt(t)
    return erfcx(z)


def _error_metrics(t: np.ndarray, u_num: np.ndarray, u_ref: np.ndarray) -> tuple[float, float]:
    error = np.asarray(u_num, dtype=float) - np.asarray(u_ref, dtype=float)
    err_inf = float(np.max(np.abs(error)))
    err_l2 = float(np.sqrt(np.trapezoid(error**2, t)))
    return err_inf, err_l2


def _fft_convolve(a: np.ndarray, b: np.ndarray, size: int) -> np.ndarray:
    fft_size = 1 << (size - 1).bit_length()
    a_hat = np.fft.rfft(a, fft_size)
    b_hat = np.fft.rfft(b, fft_size)
    return np.fft.irfft(a_hat * b_hat, fft_size)[:size]


def _series_inverse(coefficients: np.ndarray, size: int) -> np.ndarray:
    inverse = np.array([1.0 / coefficients[0]], dtype=float)
    current = 1
    while current < size:
        next_size = min(2 * current, size)
        coeff_trunc = coefficients[:next_size]
        product = _fft_convolve(coeff_trunc, inverse, next_size)
        correction = -product
        correction[0] += 2.0
        inverse = _fft_convolve(inverse, correction, next_size)
        current = next_size
    return inverse[:size]


def solve_l1_uniform(case: SilverBenchmarkCase, steps: int) -> tuple[np.ndarray, np.ndarray, float]:
    dt = case.interval_end / steps
    prefactor = EPSILON / (gamma(2.0 - ALPHA) * dt**ALPHA)

    start = perf_counter()
    indices = np.arange(steps, dtype=float)
    b = np.power(indices + 1.0, 1.0 - ALPHA) - np.power(indices, 1.0 - ALPHA)
    a = b[:-1] - b[1:]

    toeplitz_column = np.empty(steps, dtype=float)
    toeplitz_column[0] = 1.0 + prefactor
    if steps > 1:
        toeplitz_column[1:] = -prefactor * a

    rhs = prefactor * b * U0
    inverse_series = _series_inverse(toeplitz_column, steps)
    interior = _fft_convolve(inverse_series, rhs, steps)
    runtime_seconds = perf_counter() - start

    t = np.linspace(0.0, case.interval_end, steps + 1)
    u = np.empty(steps + 1, dtype=float)
    u[0] = U0
    u[1:] = interior
    return t, u, runtime_seconds


def _run_spectral(case: SilverBenchmarkCase, basis_size: int, control: GridData) -> dict:
    solution = solve_fractional_ivp(
        ProblemConfig(
            alpha=ALPHA,
            epsilon=EPSILON,
            interval_end=case.interval_end,
            u0=U0,
            coefficient=lambda x: np.full_like(x, COEFFICIENT_VALUE, dtype=float),
            forcing=lambda x: np.full_like(x, FORCING_VALUE, dtype=float),
            reference_coefficient=COEFFICIENT_VALUE,
            evaluation_points=control.t.size,
        ),
        SolverRunConfig(basis_size=basis_size),
    )
    u_control = np.asarray(solution.raw_solution.evaluate(control.t), dtype=float)
    err_inf, err_l2 = _error_metrics(control.t, u_control, control.exact)
    return {
        "case_id": case.case_id,
        "T": case.interval_end,
        "method": "Spectral method",
        "DoF": basis_size,
        "runtime_seconds": solution.runtime_seconds,
        "err_inf": err_inf,
        "err_L2": err_l2,
        "condition_number": solution.condition_number,
        "relative_residual_norm": solution.relative_residual_norm,
    }


def _run_l1(case: SilverBenchmarkCase, steps: int, control: GridData) -> dict:
    t_grid, u_grid, runtime_seconds = solve_l1_uniform(case, steps)
    u_control = np.interp(control.t, t_grid, u_grid)
    err_inf, err_l2 = _error_metrics(control.t, u_control, control.exact)
    return {
        "case_id": case.case_id,
        "T": case.interval_end,
        "method": "L1 finite difference",
        "DoF": steps,
        "runtime_seconds": runtime_seconds,
        "err_inf": err_inf,
        "err_L2": err_l2,
        "condition_number": np.nan,
        "relative_residual_norm": np.nan,
    }


def _best_row(frame: pd.DataFrame, method: str, metric: str) -> pd.Series:
    subset = frame[frame["method"] == method].sort_values(metric, kind="stable")
    return subset.iloc[0]


def _closest_pair(frame: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    spectral = frame[frame["method"] == "Spectral method"].reset_index(drop=True)
    l1 = frame[frame["method"] == "L1 finite difference"].reset_index(drop=True)
    if column == "runtime_seconds":
        distances = np.abs(
            np.log10(spectral[column].to_numpy()[:, None]) - np.log10(l1[column].to_numpy()[None, :])
        )
    else:
        distances = np.abs(spectral[column].to_numpy()[:, None] - l1[column].to_numpy()[None, :])
    i, j = np.unravel_index(np.argmin(distances), distances.shape)
    return spectral.iloc[i], l1.iloc[j]


def _plot_solutions(
    case: SilverBenchmarkCase,
    control: GridData,
    spectral_best: pd.Series,
    l1_best: pd.Series,
) -> None:
    spectral_solution = solve_fractional_ivp(
        ProblemConfig(
            alpha=ALPHA,
            epsilon=EPSILON,
            interval_end=case.interval_end,
            u0=U0,
            coefficient=lambda x: np.full_like(x, COEFFICIENT_VALUE, dtype=float),
            forcing=lambda x: np.full_like(x, FORCING_VALUE, dtype=float),
            reference_coefficient=COEFFICIENT_VALUE,
            evaluation_points=control.t.size,
        ),
        SolverRunConfig(basis_size=int(spectral_best["DoF"])),
    )
    u_spectral = np.asarray(spectral_solution.raw_solution.evaluate(control.t), dtype=float)
    l1_t, l1_u, _ = solve_l1_uniform(case, int(l1_best["DoF"]))
    u_l1 = np.interp(control.t, l1_t, l1_u)

    windows = [
        (case.interval_end, f"caputo_mainardi_{case.case_id}_solution_full.png", "t"),
        (0.01, f"caputo_mainardi_{case.case_id}_solution_zoom_0p01.png", "t"),
        (0.002, f"caputo_mainardi_{case.case_id}_solution_zoom_0p002.png", "t"),
    ]
    for x_max, filename, xlabel in windows:
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        mask = control.t <= x_max + 1e-15
        ax.plot(control.t[mask], control.exact[mask], linewidth=2.2, label="Exact Mittag-Leffler")
        ax.plot(control.t[mask], u_spectral[mask], "--", linewidth=1.8, label="Spectral method")
        ax.plot(control.t[mask], u_l1[mask], "-.", linewidth=1.8, label="L1 finite difference")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("u(t)")
        ax.set_title(
            f"{MATERIAL}, nu = {ALPHA}, beta = {BETA:.2f}, epsilon = 1 / {BETA:.2f}, T = {case.interval_end:g}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(RESULTS_DIR / "plots" / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    sqrt_t = np.sqrt(control.t)
    ax.plot(sqrt_t, control.exact, linewidth=2.2, label="Exact Mittag-Leffler")
    ax.plot(sqrt_t, u_spectral, "--", linewidth=1.8, label="Spectral method")
    ax.plot(sqrt_t, u_l1, "-.", linewidth=1.8, label="L1 finite difference")
    ax.set_xlabel("sqrt(t)")
    ax.set_ylabel("u(t)")
    ax.set_title(
        f"{MATERIAL}, nu = {ALPHA}, beta = {BETA:.2f}, epsilon = 1 / {BETA:.2f}, T = {case.interval_end:g}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(RESULTS_DIR / "plots" / f"caputo_mainardi_{case.case_id}_solution_sqrt_t.png", dpi=180)
    plt.close(fig)


def _plot_accuracy(frame: pd.DataFrame, case: SilverBenchmarkCase) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for method, sub_df in frame.groupby("method"):
        ordered = sub_df.sort_values("DoF")
        axes[0].loglog(ordered["DoF"], ordered["err_inf"], marker="o", label=method)
        axes[1].loglog(ordered["runtime_seconds"], ordered["err_inf"], marker="o", label=method)

    title = f"{MATERIAL}, nu = {ALPHA}, beta = {BETA:.2f}, epsilon = 1 / {BETA:.2f}, T = {case.interval_end:g}"
    axes[0].set_xlabel("DoF")
    axes[0].set_ylabel("Linf error")
    axes[0].set_title(f"Error vs DoF\n{title}")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("runtime (s)")
    axes[1].set_ylabel("Linf error")
    axes[1].set_title(f"Error vs runtime\n{title}")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.savefig(RESULTS_DIR / "plots" / f"caputo_mainardi_{case.case_id}_accuracy.png", dpi=180)
    plt.close(fig)


def _write_case_report(
    case: SilverBenchmarkCase,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    artifacts: list[str],
) -> str:
    report_path = REPORT_DIR / f"caputo_mainardi_{case.case_id}_report.md"
    spectral_best = _best_row(results, "Spectral method", "err_inf")
    l1_best = _best_row(results, "L1 finite difference", "err_inf")
    comparable_runtime_spectral, comparable_runtime_l1 = _closest_pair(results, "runtime_seconds")
    comparable_dof_spectral, comparable_dof_l1 = _closest_pair(results, "DoF")

    lines: list[str] = []
    lines.append(f"# Caputo-Mainardi Silver Relaxation Benchmark ({case.case_id})")
    lines.append("")
    lines.append(f"Source article: {ARTICLE_CITATION}")
    lines.append("")
    lines.append("## Physical setup")
    lines.append("")
    lines.append(
        "This benchmark models stress relaxation in silver after a unit step strain in the fractional standard linear solid of Caputo-Mainardi."
    )
    lines.append(
        "For Table 1 material parameters we use Silver with delta = 0.55, alpha = 39.3, nu = 0.50, hence beta = alpha + delta = 39.85 and epsilon = 1 / beta."
    )
    lines.append(
        "The normalized relaxation law is the Mittag-Leffler response u(t) = E_{0.5}(-39.85 t^{0.5}), which combines a sharp initial transition with a long memory tail."
    )
    lines.append("")
    lines.append("## Solver call")
    lines.append("")
    lines.append("The spectral solver is used as a black box through:")
    lines.append("")
    lines.append("```python")
    lines.append("solve_fractional_ivp(")
    lines.append("    ProblemConfig(")
    lines.append("        alpha=0.5,")
    lines.append(f"        epsilon={EPSILON:.16f},")
    lines.append(f"        interval_end={case.interval_end:g},")
    lines.append("        u0=1.0,")
    lines.append("        coefficient=lambda x: np.ones_like(x),")
    lines.append("        forcing=lambda x: np.zeros_like(x),")
    lines.append("        reference_coefficient=1.0,")
    lines.append("    ),")
    lines.append("    SolverRunConfig(basis_size=N),")
    lines.append(")")
    lines.append("```")
    lines.append("")
    lines.append("## Detailed results")
    lines.append("")
    lines.append(_markdown_table(results))
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append(_markdown_table(summary))
    lines.append("")
    lines.append("## Key observations")
    lines.append("")
    lines.append(
        f"- The characteristic initial-layer scale is epsilon^(1/nu) = epsilon^2 = {EPSILON**2:.6e}, so a uniform L1 grid must spend many steps resolving the very start of the trajectory."
    )
    lines.append(
        f"- Best Spectral method Linf error: {float(spectral_best['err_inf']):.6e} at DoF = {int(spectral_best['DoF'])}, runtime = {float(spectral_best['runtime_seconds']):.6e} s."
    )
    lines.append(
        f"- Best L1 finite difference Linf error: {float(l1_best['err_inf']):.6e} at DoF = {int(l1_best['DoF'])}, runtime = {float(l1_best['runtime_seconds']):.6e} s."
    )
    lines.append(
        f"- Closest-runtime comparison: Spectral method DoF = {int(comparable_runtime_spectral['DoF'])} gives Linf = {float(comparable_runtime_spectral['err_inf']):.6e}, while L1 finite difference DoF = {int(comparable_runtime_l1['DoF'])} gives Linf = {float(comparable_runtime_l1['err_inf']):.6e}."
    )
    lines.append(
        f"- Closest-DoF comparison: Spectral method DoF = {int(comparable_dof_spectral['DoF'])} gives Linf = {float(comparable_dof_spectral['err_inf']):.6e}, while L1 finite difference DoF = {int(comparable_dof_l1['DoF'])} gives Linf = {float(comparable_dof_l1['err_inf']):.6e}."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for artifact in artifacts:
        lines.append(f"- `{artifact}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def _case_summary(results: pd.DataFrame) -> pd.DataFrame:
    spectral_best = _best_row(results, "Spectral method", "err_inf")
    l1_best = _best_row(results, "L1 finite difference", "err_inf")
    runtime_spectral, runtime_l1 = _closest_pair(results, "runtime_seconds")
    dof_spectral, dof_l1 = _closest_pair(results, "DoF")
    rows = [
        {
            "comparison": "best Spectral method",
            "method": "Spectral method",
            "DoF": int(spectral_best["DoF"]),
            "runtime_seconds": float(spectral_best["runtime_seconds"]),
            "err_inf": float(spectral_best["err_inf"]),
            "err_L2": float(spectral_best["err_L2"]),
        },
        {
            "comparison": "best L1 finite difference",
            "method": "L1 finite difference",
            "DoF": int(l1_best["DoF"]),
            "runtime_seconds": float(l1_best["runtime_seconds"]),
            "err_inf": float(l1_best["err_inf"]),
            "err_L2": float(l1_best["err_L2"]),
        },
        {
            "comparison": "closest runtime",
            "method": "Spectral method",
            "DoF": int(runtime_spectral["DoF"]),
            "runtime_seconds": float(runtime_spectral["runtime_seconds"]),
            "err_inf": float(runtime_spectral["err_inf"]),
            "err_L2": float(runtime_spectral["err_L2"]),
        },
        {
            "comparison": "closest runtime",
            "method": "L1 finite difference",
            "DoF": int(runtime_l1["DoF"]),
            "runtime_seconds": float(runtime_l1["runtime_seconds"]),
            "err_inf": float(runtime_l1["err_inf"]),
            "err_L2": float(runtime_l1["err_L2"]),
        },
        {
            "comparison": "closest DoF",
            "method": "Spectral method",
            "DoF": int(dof_spectral["DoF"]),
            "runtime_seconds": float(dof_spectral["runtime_seconds"]),
            "err_inf": float(dof_spectral["err_inf"]),
            "err_L2": float(dof_spectral["err_L2"]),
        },
        {
            "comparison": "closest DoF",
            "method": "L1 finite difference",
            "DoF": int(dof_l1["DoF"]),
            "runtime_seconds": float(dof_l1["runtime_seconds"]),
            "err_inf": float(dof_l1["err_inf"]),
            "err_L2": float(dof_l1["err_L2"]),
        },
    ]
    return pd.DataFrame(rows)


def run_caputo_mainardi_silver_benchmark() -> dict:
    cases = [
        SilverBenchmarkCase(case_id="T10", interval_end=10.0),
        SilverBenchmarkCase(case_id="T1", interval_end=1.0),
    ]

    summaries: list[dict] = []
    for case in cases:
        control_t = _control_grid(case.interval_end)
        control = GridData(t=control_t, exact=silver_exact_solution(control_t))

        rows: list[dict] = []
        for basis_size in SPECTRAL_BASIS_SIZES:
            rows.append(_run_spectral(case, basis_size, control))
        for steps in L1_GRID_SIZES:
            rows.append(_run_l1(case, steps, control))

        results = pd.DataFrame(rows).sort_values(["method", "DoF"], kind="stable").reset_index(drop=True)
        summary_table = _case_summary(results)

        results_csv = RESULTS_DIR / f"caputo_mainardi_{case.case_id}_results.csv"
        summary_csv = RESULTS_DIR / "tables" / f"caputo_mainardi_{case.case_id}_summary.csv"
        write_csv(results_csv, results)
        write_csv(summary_csv, summary_table)

        spectral_best = _best_row(results, "Spectral method", "err_inf")
        l1_best = _best_row(results, "L1 finite difference", "err_inf")
        _plot_solutions(case, control, spectral_best, l1_best)
        _plot_accuracy(results, case)

        artifact_paths = [
            str(results_csv),
            str(summary_csv),
            str(RESULTS_DIR / "plots" / f"caputo_mainardi_{case.case_id}_solution_full.png"),
            str(RESULTS_DIR / "plots" / f"caputo_mainardi_{case.case_id}_solution_zoom_0p01.png"),
            str(RESULTS_DIR / "plots" / f"caputo_mainardi_{case.case_id}_solution_zoom_0p002.png"),
            str(RESULTS_DIR / "plots" / f"caputo_mainardi_{case.case_id}_solution_sqrt_t.png"),
            str(RESULTS_DIR / "plots" / f"caputo_mainardi_{case.case_id}_accuracy.png"),
        ]
        report_path = _write_case_report(case, results, summary_table, artifact_paths)

        summaries.append(
            {
                "case_id": case.case_id,
                "T": case.interval_end,
                "material": MATERIAL,
                "nu": ALPHA,
                "beta": BETA,
                "epsilon": EPSILON,
                "spectral_basis_sizes": SPECTRAL_BASIS_SIZES,
                "l1_grid_sizes": L1_GRID_SIZES,
                "best_spectral_err_inf": float(spectral_best["err_inf"]),
                "best_l1_err_inf": float(l1_best["err_inf"]),
                "report_markdown": report_path,
            }
        )

    summary_path = RESULTS_DIR / "caputo_mainardi_silver_summary.json"
    payload = {
        "article": ARTICLE_CITATION,
        "material": MATERIAL,
        "table_1_parameters": {
            "delta": DELTA,
            "alpha": ALPHA_TABLE,
            "nu": ALPHA,
            "beta": BETA,
            "epsilon": EPSILON,
        },
        "cases": summaries,
    }
    write_json(summary_path, payload)
    return payload
