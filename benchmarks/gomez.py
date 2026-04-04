from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .api import ProblemConfig, SolverRunConfig, solve_fractional_ivp
from .utils import RESULTS_DIR, write_csv, write_json


SOURCE_PAPER = "Gomez-Aguilar et al. fractional RC/CPE response paper"


def run_gomez_qualitative_reference() -> dict:
    gammas = [1.0, 0.98, 0.96]
    resistance = 1.0
    capacitance = 0.1
    omega = 2.0 * np.pi * 60.0
    t_end = 0.08 * resistance * capacitance
    rows = []
    curves = {}

    for gamma in gammas:
        alpha = min(gamma, 0.999999)
        tau_gamma = (resistance * capacitance) / (resistance * capacitance) ** (1.0 - gamma)
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
        charge = capacitance * voltage
        current = (np.sin(omega * solution.t) - voltage) / resistance
        curves[gamma] = {"t": solution.t / (resistance * capacitance), "charge": charge, "voltage": voltage, "current": current}
        rows.extend(
            {
                "benchmark_id": "gomez_qualitative_reference",
                "validation_category": "validation_literature_qualitative",
                "source_paper": SOURCE_PAPER,
                "model_match_level": "qualitative",
                "data_source": "digitized_pdf",
                "initialization_type": "pointwise",
                "claim_level": "qualitative_validation",
                "gamma": gamma,
                "t_over_rc": float(tt),
                "charge": float(qt),
                "voltage": float(vt),
                "current": float(it),
            }
            for tt, qt, vt, it in zip(solution.t / (resistance * capacitance), charge, voltage, current, strict=True)
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
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.savefig(RESULTS_DIR / "plots" / "gomez_qualitative_reference.png", dpi=180)
    plt.close(fig)

    frame = pd.DataFrame(rows)
    write_csv(RESULTS_DIR / "validation_gomez_qualitative.csv", frame)
    note = {
        "benchmark_id": "gomez_qualitative_reference",
        "validation_category": "validation_literature_qualitative",
        "source_paper": SOURCE_PAPER,
        "model_match_level": "qualitative",
        "data_source": "digitized_pdf",
        "initialization_type": "pointwise",
        "claim_level": "qualitative_validation",
        "message": (
            "Gomez is retained only as a structural reference for the qualitative trend with varying gamma. "
            "It is not used for strict quantitative validation of the present RI-MML solver."
        ),
    }
    write_json(RESULTS_DIR / "gomez_qualitative_reference.json", note)
    return note
