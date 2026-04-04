from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .api import ProblemConfig, SolverRunConfig, solve_fractional_ivp
from .utils import RESULTS_DIR, write_csv


def run_gomez_benchmark() -> dict:
    gammas = [1.0, 0.98, 0.96]
    R = 1.0
    C = 0.1
    omega = 2.0 * np.pi * 60.0
    t_end = 0.08 * R * C
    rows = []
    curves = {}
    for gamma in gammas:
        alpha = gamma if gamma < 1.0 else 0.999999
        tau_gamma = (R * C) / (R * C) ** (1.0 - gamma)
        solution = solve_fractional_ivp(
            ProblemConfig(
                alpha=alpha,
                epsilon=tau_gamma,
                interval_end=t_end,
                u0=0.0,
                coefficient=lambda x: np.ones_like(x),
                forcing=lambda x, om=omega: np.sin(om * x),
                reference_coefficient=1.0,
                evaluation_points=1200,
            ),
            SolverRunConfig(basis_size=18),
        )
        voltage = solution.u
        charge = C * voltage
        current = (np.sin(omega * solution.t) - voltage) / R
        curves[gamma] = {"t": solution.t / (R * C), "charge": charge, "voltage": voltage, "current": current}
        rows.extend(
            {
                "gamma": gamma,
                "t_over_rc": float(tt),
                "charge": float(qt),
                "voltage": float(vt),
                "current": float(it),
            }
            for tt, qt, vt, it in zip(solution.t / (R * C), charge, voltage, current, strict=True)
        )

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
    for gamma, data in curves.items():
        axes[0].plot(data["t"], data["charge"], label=f"gamma={gamma:g}")
        axes[1].plot(data["t"], data["voltage"], label=f"gamma={gamma:g}")
        axes[2].plot(data["t"], data["current"], label=f"gamma={gamma:g}")
    axes[0].set_ylabel("charge")
    axes[1].set_ylabel("voltage")
    axes[2].set_ylabel("current")
    axes[2].set_xlabel("t / RC")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.savefig(RESULTS_DIR / "plots" / "gomez_rc_sanity.png", dpi=180)
    plt.close(fig)

    write_csv(RESULTS_DIR / "metrics_gomez.csv", pd.DataFrame(rows))
    return {
        "source": "additional_inf/ArticuloPublicado.PDF",
        "mode": "qualitative sanity-check",
        "parameters": {"R": R, "C": C, "omega": omega, "gammas": gammas},
    }
