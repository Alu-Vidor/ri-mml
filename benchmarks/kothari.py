from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rimml import mittag_leffler

from .api import ProblemConfig, SolverRunConfig, solve_fractional_ivp
from .metrics import curve_metrics, interpolate_reference
from .utils import DATA_DIR, DIGITIZED_DIR, RESULTS_DIR, ROOT, write_csv, write_json


KOTHARI_TABLE3 = [
    {"brand": "AVX", "capacitance_F": 0.47, "fractional": {"C_lambda": 0.476, "Rs_ohm": 0.0, "lambda": 0.902, "E_t": 6.18e-06}, "integer": {"C": 0.648, "Rs_ohm": 1.484, "E_t": 4.48e-05}},
    {"brand": "AVX", "capacitance_F": 1.0, "fractional": {"C_lambda": 0.765, "Rs_ohm": 0.0, "lambda": 0.898, "E_t": 1.34e-06}, "integer": {"C": 1.060, "Rs_ohm": 1.056, "E_t": 1.03e-05}},
    {"brand": "AVX", "capacitance_F": 1.5, "fractional": {"C_lambda": 1.147, "Rs_ohm": 0.0, "lambda": 0.909, "E_t": 3.37e-07}, "integer": {"C": 1.533, "Rs_ohm": 0.643, "E_t": 4.11e-06}},
    {"brand": "Kemet", "capacitance_F": 0.47, "fractional": {"C_lambda": 0.079, "Rs_ohm": 8.918, "lambda": 0.594, "E_t": 1.33e-04}, "integer": {"C": 0.325, "Rs_ohm": 29.82, "E_t": 1.20e-03}},
    {"brand": "Kemet", "capacitance_F": 1.0, "fractional": {"C_lambda": 0.626, "Rs_ohm": 2.742, "lambda": 0.873, "E_t": 1.01e-05}, "integer": {"C": 0.942, "Rs_ohm": 4.273, "E_t": 2.95e-05}},
    {"brand": "Kemet", "capacitance_F": 1.5, "fractional": {"C_lambda": 0.914, "Rs_ohm": 0.069, "lambda": 0.922, "E_t": 5.21e-07}, "integer": {"C": 1.172, "Rs_ohm": 0.789, "E_t": 5.16e-06}},
    {"brand": "Eaton", "capacitance_F": 0.47, "fractional": {"C_lambda": 0.037, "Rs_ohm": 0.0, "lambda": 0.471, "E_t": 5.39e-05}, "integer": {"C": 0.259, "Rs_ohm": 43.15, "E_t": 4.40e-03}},
    {"brand": "Eaton", "capacitance_F": 1.0, "fractional": {"C_lambda": 0.081, "Rs_ohm": 0.0, "lambda": 0.422, "E_t": 1.26e-05}, "integer": {"C": 0.701, "Rs_ohm": 19.52, "E_t": 6.50e-04}},
    {"brand": "Eaton", "capacitance_F": 1.5, "fractional": {"C_lambda": 0.103, "Rs_ohm": 0.0, "lambda": 0.443, "E_t": 1.27e-05}, "integer": {"C": 0.801, "Rs_ohm": 15.49, "E_t": 7.31e-04}},
]

KOTHARI_TABLE4 = [
    {"brand": "AVX", "capacitance_F": 0.47, "fractional": {"C_lambda": 0.457, "Rs_ohm": 0.0, "lambda": 0.933, "E_t": 2.60e-06}, "integer": {"C": 0.562, "Rs_ohm": 1.125, "E_t": 1.85e-05}},
    {"brand": "AVX", "capacitance_F": 1.0, "fractional": {"C_lambda": 0.744, "Rs_ohm": 0.0, "lambda": 0.935, "E_t": 1.32e-06}, "integer": {"C": 0.914, "Rs_ohm": 0.733, "E_t": 5.60e-06}},
    {"brand": "AVX", "capacitance_F": 1.5, "fractional": {"C_lambda": 1.056, "Rs_ohm": 0.0, "lambda": 0.935, "E_t": 3.77e-07}, "integer": {"C": 1.295, "Rs_ohm": 0.514, "E_t": 2.16e-06}},
    {"brand": "Kemet", "capacitance_F": 0.47, "fractional": {"C_lambda": 0.092, "Rs_ohm": 7.911, "lambda": 0.619, "E_t": 1.34e-04}, "integer": {"C": 0.344, "Rs_ohm": 25.70, "E_t": 1.14e-03}},
    {"brand": "Kemet", "capacitance_F": 1.0, "fractional": {"C_lambda": 0.507, "Rs_ohm": 3.481, "lambda": 0.839, "E_t": 9.11e-06}, "integer": {"C": 0.855, "Rs_ohm": 5.713, "E_t": 2.76e-05}},
    {"brand": "Kemet", "capacitance_F": 1.5, "fractional": {"C_lambda": 0.804, "Rs_ohm": 0.156, "lambda": 0.927, "E_t": 3.81e-07}, "integer": {"C": 1.014, "Rs_ohm": 0.929, "E_t": 3.59e-06}},
    {"brand": "Eaton", "capacitance_F": 0.47, "fractional": {"C_lambda": 0.038, "Rs_ohm": 0.0, "lambda": 0.473, "E_t": 5.98e-05}, "integer": {"C": 0.259, "Rs_ohm": 42.75, "E_t": 4.49e-03}},
    {"brand": "Eaton", "capacitance_F": 1.0, "fractional": {"C_lambda": 0.095, "Rs_ohm": 0.205, "lambda": 0.448, "E_t": 1.50e-05}, "integer": {"C": 0.727, "Rs_ohm": 17.09, "E_t": 6.45e-04}},
    {"brand": "Eaton", "capacitance_F": 1.5, "fractional": {"C_lambda": 0.093, "Rs_ohm": 0.0, "lambda": 0.439, "E_t": 1.71e-05}, "integer": {"C": 0.737, "Rs_ohm": 17.10, "E_t": 7.31e-04}},
]

KOTHARI_TABLE5 = [
    {"time_s": 10, "zero_initial": {"C_lambda": 0.771, "Rs_ohm": 158.93, "lambda": 0.922, "E_t": 2.92e-07}, "nonzero_initial": {"C_lambda": 1.275, "Rs_ohm": 0.0, "lambda": 0.93, "E_t": 3.17e-07}},
    {"time_s": 20, "zero_initial": {"C_lambda": 0.737, "Rs_ohm": 174.44, "lambda": 0.918, "E_t": 3.05e-07}, "nonzero_initial": {"C_lambda": 1.219, "Rs_ohm": 0.0, "lambda": 0.919, "E_t": 3.06e-07}},
    {"time_s": 30, "zero_initial": {"C_lambda": 0.669, "Rs_ohm": 189.71, "lambda": 0.907, "E_t": 3.35e-07}, "nonzero_initial": {"C_lambda": 1.147, "Rs_ohm": 0.0, "lambda": 0.909, "E_t": 3.37e-07}},
    {"time_s": 40, "zero_initial": {"C_lambda": 0.603, "Rs_ohm": 212.74, "lambda": 0.894, "E_t": 3.81e-07}, "nonzero_initial": {"C_lambda": 1.101, "Rs_ohm": 0.0, "lambda": 0.899, "E_t": 4.10e-07}},
    {"time_s": 50, "zero_initial": {"C_lambda": 0.541, "Rs_ohm": 242.01, "lambda": 0.885, "E_t": 5.40e-07}, "nonzero_initial": {"C_lambda": 1.138, "Rs_ohm": 0.0, "lambda": 0.911, "E_t": 1.42e-06}},
]


@dataclass(frozen=True)
class FigureDigitizationConfig:
    name: str
    image_path: str
    rect: tuple[int, int, int, int]
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    labels: dict[str, str]


FIGURE_CONFIGS = [
    FigureDigitizationConfig("fig3a", "tmp_pages/kajal_p7.png", (156, 377, 276, 221), (0.0, 30.0), (2.0, 3.5), {"experimental": "red", "proposed": "blue", "integer": "black"}),
    FigureDigitizationConfig("fig3b", "tmp_pages/kajal_p7.png", (489, 377, 276, 221), (0.0, 30.0), (2.1, 2.6), {"experimental": "red", "proposed": "blue", "integer": "black"}),
    FigureDigitizationConfig("fig4", "tmp_pages/kajal_p7.png", (105, 720, 311, 145), (0.0, 250.0), (2.2, 2.9), {"experimental": "red", "proposed": "blue", "integer": "black"}),
    FigureDigitizationConfig("fig5a", "tmp_pages/kajal_p8.png", (181, 88, 263, 207), (0.0, 50.0), (1.8, 2.8), {"experimental": "red", "proposed": "blue"}),
    FigureDigitizationConfig("fig5b", "tmp_pages/kajal_p8.png", (495, 90, 262, 205), (0.0, 50.0), (1.4, 2.8), {"experimental": "red", "proposed": "blue"}),
    FigureDigitizationConfig("fig6", "tmp_pages/kajal_p8.png", (135, 418, 286, 212), (0.0, 50.0), (1.8, 2.8), {"experimental": "red", "zero_initial_model": "blue"}),
    FigureDigitizationConfig("fig7a", "tmp_pages/kajal_p8.png", (512, 414, 301, 169), (0.0, 250.0), (1.0, 3.5), {"experimental": "red", "proposed": "blue", "mlf": "gray"}),
    FigureDigitizationConfig("fig7b", "tmp_pages/kajal_p8.png", (513, 642, 300, 170), (0.0, 50.0), (1.4, 2.2), {"experimental": "red", "proposed": "blue", "mlf": "gray"}),
]


def export_kothari_tables() -> None:
    write_json(
        DATA_DIR / "kothari_table3_nonzero_charge.json",
        {
            "source": "additional_inf/Kajal.pdf Table 3",
            "provenance": "Manually transcribed from scanned PDF page after visual verification.",
            "units": {"C_lambda": "F/s^(1-lambda)", "Rs_ohm": "ohm", "C": "F"},
            "rows": KOTHARI_TABLE3,
        },
    )
    write_json(
        DATA_DIR / "kothari_table4_nonzero_discharge.json",
        {
            "source": "additional_inf/Kajal.pdf Table 4",
            "provenance": "Manually transcribed from scanned PDF page after visual verification.",
            "units": {"C_lambda": "F/s^(1-lambda)", "Rs_ohm": "ohm", "C": "F"},
            "rows": KOTHARI_TABLE4,
        },
    )
    write_json(
        DATA_DIR / "kothari_table5_timeframe_transfer.json",
        {
            "source": "additional_inf/Kajal.pdf Table 5",
            "provenance": "Manually transcribed from scanned PDF page after visual verification.",
            "units": {"C_lambda": "F/s^(1-lambda)", "Rs_ohm": "ohm"},
            "rows": KOTHARI_TABLE5,
        },
    )


def ensure_kothari_page_images() -> None:
    for config in FIGURE_CONFIGS:
        path = ROOT / config.image_path
        if path.exists():
            continue
        doc = fitz.open(ROOT / "additional_inf" / "Kajal.pdf")
        page_number = int(Path(config.image_path).stem.split("_p")[1]) - 1
        pix = doc.load_page(page_number).get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(path)


def _mask_from_color(roi: np.ndarray, color_name: str) -> np.ndarray:
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    if color_name == "red":
        return (r > 150) & (r > g + 20) & (r > b + 20)
    if color_name == "blue":
        return (b > 140) & (b > r + 20) & (b > g + 20)
    if color_name == "black":
        return (r < 90) & (g < 90) & (b < 90)
    if color_name == "gray":
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        return (maxc - minc < 25) & (maxc > 105) & (maxc < 220)
    raise ValueError(color_name)


def _extract_curve(config: FigureDigitizationConfig, label: str, color_name: str) -> pd.DataFrame:
    image = cv2.imread(str(ROOT / config.image_path))
    x0, y0, width, height = config.rect
    roi = image[y0 : y0 + height, x0 : x0 + width]
    mask = _mask_from_color(roi, color_name)
    xs, ys = [], []
    for col in range(mask.shape[1]):
        rows = np.where(mask[:, col])[0]
        if rows.size == 0:
            continue
        row = float(np.median(rows))
        t = config.xlim[0] + col / max(mask.shape[1] - 1, 1) * (config.xlim[1] - config.xlim[0])
        v = config.ylim[1] - row / max(mask.shape[0] - 1, 1) * (config.ylim[1] - config.ylim[0])
        xs.append(t)
        ys.append(v)
    frame = pd.DataFrame({"time": xs, "voltage": ys, "label": label, "figure": config.name})
    if not frame.empty:
        frame = frame.groupby("time", as_index=False).median(numeric_only=True).assign(label=label, figure=config.name)
    return frame


def digitize_kothari_figures() -> dict[str, pd.DataFrame]:
    ensure_kothari_page_images()
    digitized = {}
    for config in FIGURE_CONFIGS:
        frames = []
        for label, color_name in config.labels.items():
            frame = _extract_curve(config, label, color_name)
            if not frame.empty:
                frames.append(frame)
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["time", "voltage", "label", "figure"])
        write_csv(DIGITIZED_DIR / f"kothari_{config.name}.csv", merged)
        digitized[config.name] = merged
    return digitized


def _find_row(rows: list[dict], brand: str, capacitance: float) -> dict:
    for row in rows:
        if row["brand"] == brand and abs(row["capacitance_F"] - capacitance) < 1e-12:
            return row
    raise KeyError((brand, capacitance))


def _terminal_from_hidden(hidden: np.ndarray, vin: float, r_ext: float, rs: float) -> np.ndarray:
    return (rs * vin + r_ext * hidden) / (r_ext + rs)


def _solve_hidden(alpha: float, c_lambda: float, rs: float, vin: float, x0: float, t_end: float, basis_size: int = 18) -> dict:
    epsilon = c_lambda * (270.0 + rs)
    sol = solve_fractional_ivp(
        ProblemConfig(
            alpha=alpha if alpha < 1.0 else 0.999999,
            epsilon=epsilon,
            interval_end=t_end,
            u0=x0,
            coefficient=lambda x: np.ones_like(x),
            forcing=lambda x, value=vin: value * np.ones_like(x),
            reference_coefficient=1.0,
            evaluation_points=1200,
        ),
        SolverRunConfig(basis_size=basis_size),
    )
    return {"t": sol.t, "u": _terminal_from_hidden(sol.u, vin, 270.0, rs), "solver": sol}


def _mlf_hidden(alpha: float, c_lambda: float, rs: float, vin: float, x0: float, t: np.ndarray) -> np.ndarray:
    epsilon = c_lambda * (270.0 + rs)
    ml = np.asarray(mittag_leffler(alpha, 1.0, -(t**alpha) / epsilon), dtype=float)
    return _terminal_from_hidden(vin + (x0 - vin) * ml, vin, 270.0, rs)


def _integer_hidden(capacitance: float, rs: float, vin: float, x0: float, t: np.ndarray) -> np.ndarray:
    epsilon = capacitance * (270.0 + rs)
    return _terminal_from_hidden(vin + (x0 - vin) * np.exp(-t / epsilon), vin, 270.0, rs)


def _scenario_from_digitized_y0(digitized: pd.DataFrame, fallback: float) -> float:
    exp = digitized[digitized["label"] == "experimental"].sort_values("time")
    if exp.empty:
        return fallback
    return float(exp["voltage"].iloc[0])


def _run_single_scenario(
    scenario: str,
    figure_name: str,
    brand: str,
    capacitance: float,
    mode: str,
    vin: float,
    duration: float,
    table_rows: list[dict],
    digitized: dict[str, pd.DataFrame],
) -> dict:
    row = _find_row(table_rows, brand, capacitance)
    digi = digitized[figure_name]
    y0 = _scenario_from_digitized_y0(digi, 0.0)
    frac = row["fractional"]
    rs = float(frac["Rs_ohm"])
    x0 = ((270.0 + rs) * y0 - rs * vin) / 270.0
    numerical = _solve_hidden(frac["lambda"], frac["C_lambda"], rs, vin, x0, duration)
    mlf = _mlf_hidden(frac["lambda"], frac["C_lambda"], rs, vin, x0, numerical["t"])
    integer_row = row["integer"]
    integer = _integer_hidden(integer_row["C"], float(integer_row["Rs_ohm"]), vin, x0, numerical["t"])
    exp = digi[digi["label"] == "experimental"].sort_values("time")
    metrics = {}
    integer_metrics = {}
    mlf_metrics = {}
    if not exp.empty:
        y_ref = interpolate_reference(exp["time"].to_numpy(), exp["voltage"].to_numpy(), numerical["t"])
        metrics = curve_metrics(numerical["u"], y_ref)
        integer_metrics = curve_metrics(integer, y_ref)
        mlf_metrics = curve_metrics(mlf, y_ref)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label in ["experimental", "proposed", "integer", "mlf", "zero_initial_model"]:
        sub = digi[digi["label"] == label]
        if not sub.empty:
            ax.plot(sub["time"], sub["voltage"], linestyle="--", linewidth=1.0, label=f"digitized {label}")
    ax.plot(numerical["t"], numerical["u"], label="our spectral solver", linewidth=2)
    ax.plot(numerical["t"], integer, label="integer-order baseline", linewidth=1.5)
    ax.plot(numerical["t"], mlf, label="direct MLF baseline", linewidth=1.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("voltage (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(RESULTS_DIR / "plots" / f"kothari_{figure_name}_overlay.png", dpi=180)
    plt.close(fig)

    return {
        "scenario": scenario,
        "figure": figure_name,
        "source_file": "additional_inf/Kajal.pdf",
        "source_refs": f"{figure_name} + {'Table 3' if table_rows is KOTHARI_TABLE3 else 'Table 4'}",
        "brand": brand,
        "capacitance_F": capacitance,
        "mode": mode,
        "mapping_status": "exact hidden-state reduction with output reconstruction",
        "digitized_used": True,
        "parameters_origin": "published table values; no refit",
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "mre": metrics.get("mre"),
        "e_inf": metrics.get("e_inf"),
        "integer_rmse": integer_metrics.get("rmse"),
        "mlf_rmse": mlf_metrics.get("rmse"),
        "runtime_seconds": numerical["solver"].runtime_seconds,
        "residual_norm": numerical["solver"].residual_norm,
        "condition_number": numerical["solver"].condition_number,
        "N": numerical["solver"].modes,
        "comments": "Parameters from published tables; initial voltage from digitized figure intercept when not tabulated.",
    }


def _run_table5_scenario(digitized: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    fig5 = digitized["fig5a"]
    fig6 = digitized["fig6"]
    initial_levels = np.linspace(1.85, 2.35, 5)
    for table_row, y0 in zip(KOTHARI_TABLE5, initial_levels, strict=True):
        frac = table_row["nonzero_initial"]
        rs = float(frac["Rs_ohm"])
        duration = float(table_row["time_s"])
        numerical = _solve_hidden(frac["lambda"], frac["C_lambda"], rs, 5.0, y0, duration)
        mlf = _mlf_hidden(frac["lambda"], frac["C_lambda"], rs, 5.0, y0, numerical["t"])
        metrics = {}
        if not fig5.empty:
            y_ref = interpolate_reference(fig5["time"].to_numpy(), fig5["voltage"].to_numpy(), numerical["t"])
            metrics = curve_metrics(numerical["u"], y_ref)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        if not fig5.empty:
            ax.plot(fig5["time"], fig5["voltage"], "--", label="digitized figure 5(a)")
        ax.plot(numerical["t"], numerical["u"], label=f"our solver, T={duration:g}s")
        ax.plot(numerical["t"], mlf, label="direct MLF baseline")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("voltage (V)")
        fig.savefig(RESULTS_DIR / "plots" / f"kothari_fig5a_T{int(duration)}.png", dpi=180)
        plt.close(fig)
        rows.append(
            {
                "scenario": "K4",
                "figure": "fig5a",
                "source_file": "additional_inf/Kajal.pdf",
                "source_refs": "Fig. 5(a) + Table 5 (non-zero initial condition)",
                "brand": "AVX",
                "capacitance_F": 1.5,
                "mode": "approximate-replay",
                "mapping_status": "exact hidden-state reduction; initial voltages approximated from figure",
                "digitized_used": True,
                "parameters_origin": "published Table 5 values; no refit",
                "duration_s": duration,
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "mre": metrics.get("mre"),
                "e_inf": metrics.get("e_inf"),
                "runtime_seconds": numerical["solver"].runtime_seconds,
                "residual_norm": numerical["solver"].residual_norm,
                "condition_number": numerical["solver"].condition_number,
                "N": numerical["solver"].modes,
                "comments": "Table 5 parameters direct; initial voltage levels approximated from Fig. 5(a).",
            }
        )
        zero = table_row["zero_initial"]
        zero_num = _solve_hidden(zero["lambda"], zero["C_lambda"], float(zero["Rs_ohm"]), 5.0, 0.0, duration)
        zero_metrics = {}
        if not fig6.empty:
            y_ref = interpolate_reference(fig6["time"].to_numpy(), fig6["voltage"].to_numpy(), zero_num["t"])
            zero_metrics = curve_metrics(zero_num["u"], y_ref)
        rows.append(
            {
                "scenario": "K4",
                "figure": "fig6",
                "source_file": "additional_inf/Kajal.pdf",
                "source_refs": "Fig. 6 + Table 5 (zero initial condition)",
                "brand": "AVX",
                "capacitance_F": 1.5,
                "mode": "approximate-replay",
                "mapping_status": "exact hidden-state reduction; zero-initial branch",
                "digitized_used": True,
                "parameters_origin": "published Table 5 values; no refit",
                "duration_s": duration,
                "rmse": zero_metrics.get("rmse"),
                "mae": zero_metrics.get("mae"),
                "mre": zero_metrics.get("mre"),
                "e_inf": zero_metrics.get("e_inf"),
                "runtime_seconds": zero_num["solver"].runtime_seconds,
                "residual_norm": zero_num["solver"].residual_norm,
                "condition_number": zero_num["solver"].condition_number,
                "N": zero_num["solver"].modes,
                "comments": "Zero-initial branch from Table 5 compared against Fig. 6 digitized curves.",
            }
        )
    return rows


def _run_long_cycle_scenario(digitized: dict[str, pd.DataFrame]) -> dict:
    row_charge = _find_row(KOTHARI_TABLE3, "AVX", 1.0)
    row_discharge = _find_row(KOTHARI_TABLE4, "AVX", 1.0)
    dt = 25.0
    segments = [(5.0, row_charge["fractional"], True), (0.0, row_discharge["fractional"], False)] * 3
    t_all, y_all = [], []
    runtimes, residuals, conds = [], [], []
    hidden0 = 2.4
    offset = 0.0
    for vin, params, charge_phase in segments:
        if charge_phase and offset == 0.0:
            hidden0 = 2.4
        numerical = _solve_hidden(params["lambda"], params["C_lambda"], params["Rs_ohm"], vin, hidden0, dt)
        t_all.append(offset + numerical["t"])
        y_all.append(numerical["u"])
        runtimes.append(numerical["solver"].runtime_seconds)
        residuals.append(numerical["solver"].residual_norm)
        conds.append(numerical["solver"].condition_number)
        rs = float(params["Rs_ohm"])
        hidden0 = ((270.0 + rs) * float(numerical["u"][-1]) - rs * vin) / 270.0
        offset += dt
    t_num = np.concatenate(t_all)
    y_num = np.concatenate(y_all)
    digi = digitized["fig4"]
    metrics = {}
    exp = digi[digi["label"] == "experimental"].sort_values("time")
    if not exp.empty:
        y_ref = interpolate_reference(exp["time"].to_numpy(), exp["voltage"].to_numpy(), t_num)
        metrics = curve_metrics(y_num, y_ref)
        cycle_rmses = []
        for start in [0.0, 50.0, 100.0]:
            mask = (t_num >= start) & (t_num < start + 50.0)
            if mask.any():
                cycle_rmses.append(float(np.sqrt(np.mean((y_num[mask] - y_ref[mask]) ** 2))))
    else:
        cycle_rmses = []
    peaks = [float(np.max(y_num[(t_num >= start) & (t_num < start + 50.0)])) for start in [0.0, 50.0, 100.0]]
    troughs = [float(np.min(y_num[(t_num >= start) & (t_num < start + 50.0)])) for start in [0.0, 50.0, 100.0]]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if not digi.empty:
        for label in ["experimental", "proposed", "integer"]:
            sub = digi[digi["label"] == label]
            if not sub.empty:
                ax.plot(sub["time"], sub["voltage"], "--", label=f"digitized {label}")
    ax.plot(t_num, y_num, label="our spectral replay")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("voltage (V)")
    fig.savefig(RESULTS_DIR / "plots" / "kothari_fig4_cycles.png", dpi=180)
    plt.close(fig)
    return {
        "scenario": "K3",
        "figure": "fig4",
        "source_file": "additional_inf/Kajal.pdf",
        "source_refs": "Fig. 4 + Tables 3/4",
        "brand": "AVX",
        "capacitance_F": 1.0,
        "mode": "approximate-replay",
        "mapping_status": "piecewise compatible but switching times inferred",
        "digitized_used": True,
        "parameters_origin": "published Table 3/4 values; no refit",
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "mre": metrics.get("mre"),
        "e_inf": metrics.get("e_inf"),
        "cycle_rmse_mean": float(np.mean(cycle_rmses)) if cycle_rmses else None,
        "peak_voltage_drift": peaks[-1] - peaks[0],
        "trough_voltage_drift": troughs[-1] - troughs[0],
        "runtime_seconds": float(np.sum(runtimes)),
        "residual_norm": float(np.max(residuals)),
        "condition_number": float(np.max(conds)),
        "N": 18,
        "comments": "Segment lengths inferred from digitized cycle geometry; structure compatible but timing is approximate.",
    }


def run_kothari_benchmarks() -> dict:
    export_kothari_tables()
    digitized = digitize_kothari_figures()
    write_json(
        RESULTS_DIR / "kothari_mapping_note.json",
        {
            "source": "additional_inf/Kajal.pdf Eq. (13), (14), (28)",
            "mapping": (
                "Use hidden state x(t)=CPE voltage satisfying C_lambda*(R+Rs) D_t^lambda x + x = v_in, "
                "then reconstruct terminal voltage as v_out=(Rs*v_in + R*x)/(R+Rs) with external R=270 ohm."
            ),
            "status": "exact for one-term series RC-CPE under piecewise-constant step input, with output reconstruction.",
        },
    )
    rows = [
        _run_single_scenario("K1", "fig3a", "Eaton", 0.47, "approximate-replay", 5.0, 30.0, KOTHARI_TABLE3, digitized),
        _run_single_scenario("K2", "fig3b", "Eaton", 1.5, "approximate-replay", 0.0, 30.0, KOTHARI_TABLE4, digitized),
        _run_single_scenario("K5", "fig5b", "Kemet", 1.5, "approximate-replay", 0.0, 50.0, KOTHARI_TABLE4, digitized),
        _run_single_scenario("K6", "fig7b", "AVX", 1.0, "approximate-replay", 0.0, 50.0, KOTHARI_TABLE4, digitized),
        _run_long_cycle_scenario(digitized),
    ]
    rows.extend(_run_table5_scenario(digitized))
    frame = pd.DataFrame(rows)
    write_csv(RESULTS_DIR / "metrics_kothari.csv", frame)
    return {"scenarios": sorted(frame["scenario"].unique().tolist()), "rows": len(frame)}
