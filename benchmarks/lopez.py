from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rimml import mittag_leffler

from .utils import RESULTS_DIR, write_csv, write_json


def g1(alpha: float, t0: float, t: np.ndarray) -> np.ndarray:
    return (1.0 + t / t0) ** alpha - (t / t0) ** alpha


def g2(alpha: float, t0: float, t: np.ndarray) -> np.ndarray:
    z = t / t0
    return (1.0 + z) ** (alpha + 1.0) - (1.0 + alpha) * z**alpha


def g3(alpha: float, tau: float, t0: float, t: np.ndarray) -> np.ndarray:
    num = np.asarray(mittag_leffler(alpha, 1.0, -((t + t0) / tau) ** alpha), dtype=float)
    num -= np.asarray(mittag_leffler(alpha, 1.0, -(t / tau) ** alpha), dtype=float)
    den = 1.0 - float(mittag_leffler(alpha, 1.0, -(t0 / tau) ** alpha))
    return num / den


def g4(t0: float, t: np.ndarray, resistances: np.ndarray, taus: np.ndarray) -> np.ndarray:
    numerator = np.zeros_like(t, dtype=float)
    denominator = 0.0
    for resistance, tau in zip(resistances, taus, strict=True):
        weight = resistance * (1.0 - np.exp(-t0 / tau))
        numerator += weight * np.exp(-t / tau)
        denominator += weight
    return numerator / denominator


def run_lopez_benchmarks() -> dict:
    alpha = 0.5
    t = np.logspace(-3, 3, 1000)
    t0_values = [10.0, 100.0]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    rows = []
    for t0 in t0_values:
        g1_vals = g1(alpha, t0, t)
        g2_vals = g2(alpha, t0, t)
        ax.plot(t, g1_vals, label=f"g1, t0={t0:g}s")
        ax.plot(t, g2_vals, "--", label=f"g2, t0={t0:g}s")
        rows.extend(
            {
                "benchmark": "L1",
                "alpha": alpha,
                "t0": t0,
                "t": float(tt),
                "g1": float(v1),
                "g2": float(v2),
            }
            for tt, v1, v2 in zip(t, g1_vals, g2_vals, strict=True)
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("t' (s)")
    ax.set_ylabel("decay")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(RESULTS_DIR / "plots" / "lopez_g1_g2.png", dpi=180)
    plt.close(fig)

    tau = 1.0
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for t0 in [tau, 2.0 * tau, 3.0 * tau]:
        g3_vals = g3(alpha, tau, t0, t)
        exp_vals = np.exp(-t / tau)
        ax.plot(t, g3_vals, label=f"g3, t0={t0/tau:.0f}tau")
        ax.plot(t, exp_vals, "--", label=f"exp, t0={t0/tau:.0f}tau")
        rows.extend(
            {
                "benchmark": "L2",
                "alpha": alpha,
                "t0": t0,
                "t": float(tt),
                "g3": float(v3),
                "exp": float(ve),
            }
            for tt, v3, ve in zip(t, g3_vals, exp_vals, strict=True)
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("t' (s)")
    ax.set_ylabel("decay")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(RESULTS_DIR / "plots" / "lopez_g3_vs_exp.png", dpi=180)
    plt.close(fig)

    resistances = 1e-3 * np.array([1.0407, 1.9991, 4.5240, 10.5239, 71.3679], dtype=float)
    taus = np.array([0.1352, 1.8012, 12.3951, 70.7794, 686.9286], dtype=float)
    g4_vals = g4(100.0, t, resistances, taus)
    rows.extend(
        {
            "benchmark": "g4",
            "alpha": alpha,
            "t0": 100.0,
            "t": float(tt),
            "g4": float(v4),
        }
        for tt, v4 in zip(t, g4_vals, strict=True)
    )

    frame = pd.DataFrame(rows)
    write_csv(RESULTS_DIR / "metrics_lopez.csv", frame)
    note = {
        "source": "additional_inf/energies-15-00792-v2.pdf",
        "message": (
            "Current RI-MML solver supports only a pointwise IVP state u(0)=u0. "
            "It does not represent history-dependent initialized operators, so Lopez tests are analytic limitation checks."
        ),
    }
    write_json(RESULTS_DIR / "lopez_note.json", note)
    return note

