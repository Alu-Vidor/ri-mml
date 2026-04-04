from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rimml import mittag_leffler

from .utils import RESULTS_DIR, write_csv, write_json


SOURCE_PAPER = "Lopez-Villanueva et al. 2022 history-dependent initialization study"


def g1(alpha: float, t0: float, t: np.ndarray) -> np.ndarray:
    return (1.0 + t / t0) ** alpha - (t / t0) ** alpha


def g2(alpha: float, t0: float, t: np.ndarray) -> np.ndarray:
    z = t / t0
    return (1.0 + z) ** (alpha + 1.0) - (1.0 + alpha) * z**alpha


def g3(alpha: float, tau: float, t0: float, t: np.ndarray) -> np.ndarray:
    first = np.asarray(mittag_leffler(alpha, 1.0, -((t + t0) / tau) ** alpha), dtype=float)
    second = np.asarray(mittag_leffler(alpha, 1.0, -(t / tau) ** alpha), dtype=float)
    denominator = 1.0 - float(mittag_leffler(alpha, 1.0, -(t0 / tau) ** alpha))
    return (first - second) / denominator


def g4(t0: float, t: np.ndarray, resistances: np.ndarray, taus: np.ndarray) -> np.ndarray:
    numerator = np.zeros_like(t, dtype=float)
    denominator = 0.0
    for resistance, tau in zip(resistances, taus, strict=True):
        weight = resistance * (1.0 - np.exp(-t0 / tau))
        numerator = numerator + weight * np.exp(-t / tau)
        denominator += weight
    return numerator / denominator


def run_lopez_initialization_limit() -> dict:
    alpha = 0.5
    t = np.logspace(-3, 3, 1000)
    rows: list[dict] = []

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for t0 in [10.0, 100.0]:
        g1_values = g1(alpha, t0, t)
        g2_values = g2(alpha, t0, t)
        ax.plot(t, g1_values, label=f"g1, t0={t0:g}s")
        ax.plot(t, g2_values, "--", label=f"g2, t0={t0:g}s")
        rows.extend(
            {
                "benchmark_id": "lopez_initialization_limit",
                "validation_category": "limitations",
                "source_paper": SOURCE_PAPER,
                "model_match_level": "surrogate",
                "data_source": "manual_table_transcription",
                "initialization_type": "history-dependent",
                "claim_level": "limitation_only",
                "family": "g1_g2",
                "alpha": alpha,
                "t0": t0,
                "t": float(tt),
                "g1": float(v1),
                "g2": float(v2),
            }
            for tt, v1, v2 in zip(t, g1_values, g2_values, strict=True)
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("decay")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(RESULTS_DIR / "plots" / "lopez_limit_g1_g2.png", dpi=180)
    plt.close(fig)

    tau = 1.0
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for t0 in [tau, 2.0 * tau, 3.0 * tau]:
        g3_values = g3(alpha, tau, t0, t)
        exp_values = np.exp(-t / tau)
        ax.plot(t, g3_values, label=f"g3, t0={t0/tau:.0f}tau")
        ax.plot(t, exp_values, "--", label=f"exp, t0={t0/tau:.0f}tau")
        rows.extend(
            {
                "benchmark_id": "lopez_initialization_limit",
                "validation_category": "limitations",
                "source_paper": SOURCE_PAPER,
                "model_match_level": "surrogate",
                "data_source": "manual_table_transcription",
                "initialization_type": "history-dependent",
                "claim_level": "limitation_only",
                "family": "g3_vs_exp",
                "alpha": alpha,
                "t0": t0,
                "t": float(tt),
                "g3": float(v3),
                "exp": float(ve),
            }
            for tt, v3, ve in zip(t, g3_values, exp_values, strict=True)
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("decay")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(RESULTS_DIR / "plots" / "lopez_limit_g3.png", dpi=180)
    plt.close(fig)

    resistances = 1e-3 * np.array([1.0407, 1.9991, 4.5240, 10.5239, 71.3679], dtype=float)
    taus = np.array([0.1352, 1.8012, 12.3951, 70.7794, 686.9286], dtype=float)
    g4_values = g4(100.0, t, resistances, taus)
    rows.extend(
        {
            "benchmark_id": "lopez_initialization_limit",
            "validation_category": "limitations",
            "source_paper": SOURCE_PAPER,
            "model_match_level": "surrogate",
            "data_source": "manual_table_transcription",
            "initialization_type": "history-dependent",
            "claim_level": "limitation_only",
            "family": "g4_multiexponential_surrogate",
            "alpha": alpha,
            "t0": 100.0,
            "t": float(tt),
            "g4": float(v4),
        }
        for tt, v4 in zip(t, g4_values, strict=True)
    )

    frame = pd.DataFrame(rows)
    write_csv(RESULTS_DIR / "limitations_lopez_initialization.csv", frame)
    note = {
        "benchmark_id": "lopez_initialization_limit",
        "validation_category": "limitations",
        "source_paper": SOURCE_PAPER,
        "model_match_level": "surrogate",
        "data_source": "manual_table_transcription",
        "initialization_type": "history-dependent",
        "claim_level": "limitation_only",
        "message": (
            "Current RI-MML solver supports only a pointwise initial condition u(0)=u0. "
            "It does not implement history-dependent initialization, so Lopez is included only to document a boundary of applicability."
        ),
        "not_supported": [
            "history-dependent initialized operators",
            "prehistory-dependent CPE state reconstruction",
            "one-to-one replay of Lopez decay curves with the present solver",
        ],
        "supported_surrogate_only": [
            "analytic decay families g1/g2/g3 as external references",
            "multi-exponential surrogate g4 comparison",
        ],
    }
    write_json(RESULTS_DIR / "lopez_initialization_limit.json", note)
    return note
