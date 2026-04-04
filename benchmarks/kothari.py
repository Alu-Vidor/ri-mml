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


SOURCE_PAPER = "Kothari et al. 2021 supercapacitor RC-CPE study"

KOTHARI_TABLE3 = [
    {"brand": "AVX", "capacitance_F": 0.47, "fractional": {"C_lambda": 0.476, "Rs_ohm": 0.0, "lambda": 0.902}, "integer": {"C": 0.648, "Rs_ohm": 1.484}},
    {"brand": "AVX", "capacitance_F": 1.0, "fractional": {"C_lambda": 0.765, "Rs_ohm": 0.0, "lambda": 0.898}, "integer": {"C": 1.060, "Rs_ohm": 1.056}},
    {"brand": "AVX", "capacitance_F": 1.5, "fractional": {"C_lambda": 1.147, "Rs_ohm": 0.0, "lambda": 0.909}, "integer": {"C": 1.533, "Rs_ohm": 0.643}},
    {"brand": "Kemet", "capacitance_F": 0.47, "fractional": {"C_lambda": 0.079, "Rs_ohm": 8.918, "lambda": 0.594}, "integer": {"C": 0.325, "Rs_ohm": 29.82}},
    {"brand": "Kemet", "capacitance_F": 1.0, "fractional": {"C_lambda": 0.626, "Rs_ohm": 2.742, "lambda": 0.873}, "integer": {"C": 0.942, "Rs_ohm": 4.273}},
    {"brand": "Kemet", "capacitance_F": 1.5, "fractional": {"C_lambda": 0.914, "Rs_ohm": 0.069, "lambda": 0.922}, "integer": {"C": 1.172, "Rs_ohm": 0.789}},
    {"brand": "Eaton", "capacitance_F": 0.47, "fractional": {"C_lambda": 0.037, "Rs_ohm": 0.0, "lambda": 0.471}, "integer": {"C": 0.259, "Rs_ohm": 43.15}},
    {"brand": "Eaton", "capacitance_F": 1.0, "fractional": {"C_lambda": 0.081, "Rs_ohm": 0.0, "lambda": 0.422}, "integer": {"C": 0.701, "Rs_ohm": 19.52}},
    {"brand": "Eaton", "capacitance_F": 1.5, "fractional": {"C_lambda": 0.103, "Rs_ohm": 0.0, "lambda": 0.443}, "integer": {"C": 0.801, "Rs_ohm": 15.49}},
]

KOTHARI_TABLE4 = [
    {"brand": "AVX", "capacitance_F": 0.47, "fractional": {"C_lambda": 0.457, "Rs_ohm": 0.0, "lambda": 0.933}, "integer": {"C": 0.562, "Rs_ohm": 1.125}},
    {"brand": "AVX", "capacitance_F": 1.0, "fractional": {"C_lambda": 0.744, "Rs_ohm": 0.0, "lambda": 0.935}, "integer": {"C": 0.914, "Rs_ohm": 0.733}},
    {"brand": "AVX", "capacitance_F": 1.5, "fractional": {"C_lambda": 1.056, "Rs_ohm": 0.0, "lambda": 0.935}, "integer": {"C": 1.295, "Rs_ohm": 0.514}},
    {"brand": "Kemet", "capacitance_F": 0.47, "fractional": {"C_lambda": 0.092, "Rs_ohm": 7.911, "lambda": 0.619}, "integer": {"C": 0.344, "Rs_ohm": 25.70}},
    {"brand": "Kemet", "capacitance_F": 1.0, "fractional": {"C_lambda": 0.507, "Rs_ohm": 3.481, "lambda": 0.839}, "integer": {"C": 0.855, "Rs_ohm": 5.713}},
    {"brand": "Kemet", "capacitance_F": 1.5, "fractional": {"C_lambda": 0.804, "Rs_ohm": 0.156, "lambda": 0.927}, "integer": {"C": 1.014, "Rs_ohm": 0.929}},
    {"brand": "Eaton", "capacitance_F": 0.47, "fractional": {"C_lambda": 0.038, "Rs_ohm": 0.0, "lambda": 0.473}, "integer": {"C": 0.259, "Rs_ohm": 42.75}},
    {"brand": "Eaton", "capacitance_F": 1.0, "fractional": {"C_lambda": 0.095, "Rs_ohm": 0.205, "lambda": 0.448}, "integer": {"C": 0.727, "Rs_ohm": 17.09}},
    {"brand": "Eaton", "capacitance_F": 1.5, "fractional": {"C_lambda": 0.093, "Rs_ohm": 0.0, "lambda": 0.439}, "integer": {"C": 0.737, "Rs_ohm": 17.10}},
]

KOTHARI_TABLE5 = [
    {"time_s": 10, "zero_initial": {"C_lambda": 0.771, "Rs_ohm": 158.93, "lambda": 0.922}, "nonzero_initial": {"C_lambda": 1.275, "Rs_ohm": 0.0, "lambda": 0.93}},
    {"time_s": 20, "zero_initial": {"C_lambda": 0.737, "Rs_ohm": 174.44, "lambda": 0.918}, "nonzero_initial": {"C_lambda": 1.219, "Rs_ohm": 0.0, "lambda": 0.919}},
    {"time_s": 30, "zero_initial": {"C_lambda": 0.669, "Rs_ohm": 189.71, "lambda": 0.907}, "nonzero_initial": {"C_lambda": 1.147, "Rs_ohm": 0.0, "lambda": 0.909}},
    {"time_s": 40, "zero_initial": {"C_lambda": 0.603, "Rs_ohm": 212.74, "lambda": 0.894}, "nonzero_initial": {"C_lambda": 1.101, "Rs_ohm": 0.0, "lambda": 0.899}},
    {"time_s": 50, "zero_initial": {"C_lambda": 0.541, "Rs_ohm": 242.01, "lambda": 0.885}, "nonzero_initial": {"C_lambda": 1.138, "Rs_ohm": 0.0, "lambda": 0.911}},
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
    FigureDigitizationConfig("fig7b", "tmp_pages/kajal_p8.png", (513, 642, 300, 170), (0.0, 50.0), (1.4, 2.2), {"experimental": "red", "proposed": "blue", "mlf": "gray"}),
]


def export_kothari_tables() -> None:
    write_json(
        DATA_DIR / "kothari_table3_nonzero_charge.json",
        {
            "source_paper": SOURCE_PAPER,
            "source": "additional_inf/Kajal.pdf Table 3",
            "data_source": "manual_table_transcription",
            "rows": KOTHARI_TABLE3,
        },
    )
    write_json(
        DATA_DIR / "kothari_table4_nonzero_discharge.json",
        {
            "source_paper": SOURCE_PAPER,
            "source": "additional_inf/Kajal.pdf Table 4",
            "data_source": "manual_table_transcription",
            "rows": KOTHARI_TABLE4,
        },
    )
    write_json(
        DATA_DIR / "kothari_table5_timeframe_transfer.json",
        {
            "source_paper": SOURCE_PAPER,
            "source": "additional_inf/Kajal.pdf Table 5",
            "data_source": "manual_table_transcription",
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
    solution = solve_fractional_ivp(
        ProblemConfig(
            alpha=min(alpha, 0.999999),
            epsilon=epsilon,
            interval_end=t_end,
            u0=x0,
            coefficient=lambda x: np.ones_like(x),
            forcing=lambda x, value=vin: value * np.ones_like(x),
            reference_coefficient=1.0,
            evaluation_points=1400,
        ),
        SolverRunConfig(basis_size=basis_size),
    )
    return {"t": solution.t, "u": _terminal_from_hidden(solution.u, vin, 270.0, rs), "solver": solution}


def _mlf_hidden(alpha: float, c_lambda: float, rs: float, vin: float, x0: float, t: np.ndarray) -> np.ndarray:
    epsilon = c_lambda * (270.0 + rs)
    ml = np.asarray(mittag_leffler(alpha, 1.0, -(np.power(t, alpha) / epsilon)), dtype=float)
    return _terminal_from_hidden(vin + (x0 - vin) * ml, vin, 270.0, rs)


def _integer_hidden(capacitance: float, rs: float, vin: float, x0: float, t: np.ndarray) -> np.ndarray:
    epsilon = capacitance * (270.0 + rs)
    return _terminal_from_hidden(vin + (x0 - vin) * np.exp(-t / epsilon), vin, 270.0, rs)


def _scenario_from_digitized_y0(digitized: pd.DataFrame, fallback: float) -> float:
    experimental = digitized[digitized["label"] == "experimental"].sort_values("time")
    if experimental.empty:
        return fallback
    return float(experimental["voltage"].iloc[0])


def _base_metadata(benchmark_id: str, figure_name: str, mode: str, initialization_type: str, claim_level: str, model_match_level: str) -> dict:
    return {
        "benchmark_id": benchmark_id,
        "validation_category": "validation_literature_quantitative" if claim_level == "quantitative_validation" else "validation_literature_qualitative",
        "source_paper": SOURCE_PAPER,
        "source_file": "additional_inf/Kajal.pdf",
        "figure": figure_name,
        "mode": mode,
        "model_match_level": model_match_level,
        "data_source": "digitized_pdf",
        "initialization_type": initialization_type,
        "claim_level": claim_level,
        "supports_pointwise_only": True,
    }


def _attach_metrics(payload: dict, numerical: dict, metrics: dict | None, integer_metrics: dict | None = None, mlf_metrics: dict | None = None) -> dict:
    output = dict(payload)
    output.update(
        {
            "rmse": None if metrics is None else metrics.get("rmse"),
            "mae": None if metrics is None else metrics.get("mae"),
            "mre": None if metrics is None else metrics.get("mre"),
            "e_inf": None if metrics is None else metrics.get("e_inf"),
            "integer_rmse": None if integer_metrics is None else integer_metrics.get("rmse"),
            "mlf_rmse": None if mlf_metrics is None else mlf_metrics.get("rmse"),
            "runtime_seconds": numerical["solver"].runtime_seconds,
            "residual_norm": numerical["solver"].residual_norm,
            "relative_residual_norm": numerical["solver"].relative_residual_norm,
            "condition_number": numerical["solver"].condition_number,
            "N": numerical["solver"].modes,
        }
    )
    return output


def _save_overlay(filename: str, digitized: pd.DataFrame, numerical_t: np.ndarray, numerical_u: np.ndarray, integer: np.ndarray | None = None, mlf: np.ndarray | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label in sorted(digitized["label"].unique().tolist()):
        sub = digitized[digitized["label"] == label]
        ax.plot(sub["time"], sub["voltage"], "--", linewidth=1.0, label=f"digitized {label}")
    ax.plot(numerical_t, numerical_u, label="RI-MML replay", linewidth=2.0)
    if integer is not None:
        ax.plot(numerical_t, integer, label="integer baseline", linewidth=1.2)
    if mlf is not None:
        ax.plot(numerical_t, mlf, label="direct MLF baseline", linewidth=1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("voltage (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(RESULTS_DIR / "plots" / filename, dpi=180)
    plt.close(fig)


def _run_single_replay_scenario(
    benchmark_id: str,
    figure_name: str,
    brand: str,
    capacitance: float,
    vin: float,
    duration: float,
    table_rows: list[dict],
    digitized: dict[str, pd.DataFrame],
) -> dict:
    row = _find_row(table_rows, brand, capacitance)
    digitized_frame = digitized[figure_name]
    y0 = _scenario_from_digitized_y0(digitized_frame, 0.0)
    fractional = row["fractional"]
    rs = float(fractional["Rs_ohm"])
    x0 = ((270.0 + rs) * y0 - rs * vin) / 270.0
    numerical = _solve_hidden(fractional["lambda"], fractional["C_lambda"], rs, vin, x0, duration)
    mlf = _mlf_hidden(fractional["lambda"], fractional["C_lambda"], rs, vin, x0, numerical["t"])
    integer_row = row["integer"]
    integer = _integer_hidden(integer_row["C"], float(integer_row["Rs_ohm"]), vin, x0, numerical["t"])

    experimental = digitized_frame[digitized_frame["label"] == "experimental"].sort_values("time")
    metrics = integer_metrics = mlf_metrics = None
    if not experimental.empty:
        reference = interpolate_reference(experimental["time"].to_numpy(), experimental["voltage"].to_numpy(), numerical["t"])
        metrics = curve_metrics(numerical["u"], reference)
        integer_metrics = curve_metrics(integer, reference)
        mlf_metrics = curve_metrics(mlf, reference)

    _save_overlay(f"kothari_{figure_name}_replay.png", digitized_frame, numerical["t"], numerical["u"], integer=integer, mlf=mlf)
    payload = _base_metadata(benchmark_id, figure_name, "replay", "inferred", "qualitative_validation", "reduced")
    payload.update(
        {
            "brand": brand,
            "capacitance_F": capacitance,
            "vin": vin,
            "duration_s": duration,
            "parameters_origin": "published table values; no refit",
            "comparison_note": "curve replay under inferred initial state; not a one-to-one replication of the published model",
            "uncertainty_note": "Digitization uncertainty, inferred initial condition from the same curve, and figure-based time alignment ambiguity are all material.",
        }
    )
    return _attach_metrics(payload, numerical, metrics, integer_metrics, mlf_metrics)


def _run_strict_zero_initial_scenarios(digitized: dict[str, pd.DataFrame]) -> list[dict]:
    rows: list[dict] = []
    digitized_frame = digitized["fig6"]
    experimental = digitized_frame[digitized_frame["label"] == "experimental"].sort_values("time")
    for table_row in KOTHARI_TABLE5:
        fractional = table_row["zero_initial"]
        duration = float(table_row["time_s"])
        numerical = _solve_hidden(fractional["lambda"], fractional["C_lambda"], float(fractional["Rs_ohm"]), 5.0, 0.0, duration)
        metrics = None
        if not experimental.empty:
            reference = interpolate_reference(experimental["time"].to_numpy(), experimental["voltage"].to_numpy(), numerical["t"])
            metrics = curve_metrics(numerical["u"], reference)
        benchmark_id = f"kothari_strict_zero_initial_T{int(duration)}"
        payload = _base_metadata(benchmark_id, "fig6", "strict", "pointwise", "quantitative_validation", "reduced")
        payload.update(
            {
                "brand": "AVX",
                "capacitance_F": 1.5,
                "vin": 5.0,
                "duration_s": duration,
                "parameters_origin": "published Table 5 zero-initial branch",
                "comparison_note": "approximately reproduces selected literature curves under model-reduction and digitization assumptions",
                "uncertainty_note": "Digitization uncertainty remains, but initial condition and duration come directly from the published zero-initial branch.",
            }
        )
        rows.append(_attach_metrics(payload, numerical, metrics))
    _save_overlay("kothari_fig6_strict.png", digitized_frame, numerical["t"], numerical["u"])
    return rows


def _run_nonzero_table5_replay(digitized: dict[str, pd.DataFrame]) -> list[dict]:
    rows: list[dict] = []
    digitized_frame = digitized["fig5a"]
    inferred_initial_levels = np.linspace(1.85, 2.35, 5)
    for table_row, y0 in zip(KOTHARI_TABLE5, inferred_initial_levels, strict=True):
        fractional = table_row["nonzero_initial"]
        duration = float(table_row["time_s"])
        numerical = _solve_hidden(fractional["lambda"], fractional["C_lambda"], float(fractional["Rs_ohm"]), 5.0, float(y0), duration)
        experimental = digitized_frame[digitized_frame["label"] == "experimental"].sort_values("time")
        metrics = None
        if not experimental.empty:
            reference = interpolate_reference(experimental["time"].to_numpy(), experimental["voltage"].to_numpy(), numerical["t"])
            metrics = curve_metrics(numerical["u"], reference)
        payload = _base_metadata(f"kothari_replay_nonzero_initial_T{int(duration)}", "fig5a", "replay", "inferred", "qualitative_validation", "reduced")
        payload.update(
            {
                "brand": "AVX",
                "capacitance_F": 1.5,
                "vin": 5.0,
                "duration_s": duration,
                "parameters_origin": "published Table 5 nonzero branch with inferred initial voltage ladder",
                "comparison_note": "curve replay under inferred initial state; not validation accuracy",
                "uncertainty_note": "Initial state is inferred from figure geometry, so this branch is replay-only.",
            }
        )
        rows.append(_attach_metrics(payload, numerical, metrics))
    return rows


def _run_long_cycle_replay(digitized: dict[str, pd.DataFrame]) -> dict:
    row_charge = _find_row(KOTHARI_TABLE3, "AVX", 1.0)
    row_discharge = _find_row(KOTHARI_TABLE4, "AVX", 1.0)
    segments = [(5.0, row_charge["fractional"], True), (0.0, row_discharge["fractional"], False)] * 3
    duration = 25.0
    t_all, y_all = [], []
    hidden0 = 2.4
    offset = 0.0
    runtime = 0.0
    relative_residual = 0.0
    condition_number = 0.0
    residual_norm = 0.0
    for vin, params, reset_initial in segments:
        if reset_initial and offset == 0.0:
            hidden0 = 2.4
        numerical = _solve_hidden(params["lambda"], params["C_lambda"], float(params["Rs_ohm"]), vin, hidden0, duration)
        t_all.append(offset + numerical["t"])
        y_all.append(numerical["u"])
        runtime += numerical["solver"].runtime_seconds
        residual_norm = max(residual_norm, numerical["solver"].residual_norm)
        relative_residual = max(relative_residual, numerical["solver"].relative_residual_norm)
        condition_number = max(condition_number, numerical["solver"].condition_number)
        rs = float(params["Rs_ohm"])
        hidden0 = ((270.0 + rs) * float(numerical["u"][-1]) - rs * vin) / 270.0
        offset += duration
    t_num = np.concatenate(t_all)
    y_num = np.concatenate(y_all)
    digitized_frame = digitized["fig4"]
    experimental = digitized_frame[digitized_frame["label"] == "experimental"].sort_values("time")
    metrics = None
    if not experimental.empty:
        reference = interpolate_reference(experimental["time"].to_numpy(), experimental["voltage"].to_numpy(), t_num)
        metrics = curve_metrics(y_num, reference)
    _save_overlay("kothari_fig4_replay.png", digitized_frame, t_num, y_num)
    payload = _base_metadata("kothari_replay_long_cycle", "fig4", "replay", "inferred", "qualitative_validation", "reduced")
    payload.update(
        {
            "brand": "AVX",
            "capacitance_F": 1.0,
            "vin": "piecewise 5 V / 0 V",
            "duration_s": float(t_num[-1]),
            "parameters_origin": "published Tables 3 and 4 with inferred cycle timing",
            "comparison_note": "piecewise replay with switching times inferred from figure geometry",
            "uncertainty_note": "Switching schedule and initial state are inferred, so this is a replay study only.",
            "rmse": None if metrics is None else metrics.get("rmse"),
            "mae": None if metrics is None else metrics.get("mae"),
            "mre": None if metrics is None else metrics.get("mre"),
            "e_inf": None if metrics is None else metrics.get("e_inf"),
            "integer_rmse": None,
            "mlf_rmse": None,
            "runtime_seconds": runtime,
            "residual_norm": residual_norm,
            "relative_residual_norm": relative_residual,
            "condition_number": condition_number,
            "N": 18,
        }
    )
    return payload


def run_kothari_benchmarks() -> dict:
    export_kothari_tables()
    digitized = digitize_kothari_figures()

    strict_rows = _run_strict_zero_initial_scenarios(digitized)
    replay_rows = [
        _run_single_replay_scenario("kothari_replay_fig3a", "fig3a", "Eaton", 0.47, 5.0, 30.0, KOTHARI_TABLE3, digitized),
        _run_single_replay_scenario("kothari_replay_fig3b", "fig3b", "Eaton", 1.5, 0.0, 30.0, KOTHARI_TABLE4, digitized),
        _run_single_replay_scenario("kothari_replay_fig5b", "fig5b", "Kemet", 1.5, 0.0, 50.0, KOTHARI_TABLE4, digitized),
        _run_single_replay_scenario("kothari_replay_fig7b", "fig7b", "AVX", 1.0, 0.0, 50.0, KOTHARI_TABLE4, digitized),
        _run_long_cycle_replay(digitized),
        *_run_nonzero_table5_replay(digitized),
    ]

    strict_frame = pd.DataFrame(strict_rows)
    replay_frame = pd.DataFrame(replay_rows)
    combined_columns = sorted(set(strict_frame.columns).union(replay_frame.columns))

    write_csv(RESULTS_DIR / "validation_kothari_strict.csv", strict_frame)
    write_csv(RESULTS_DIR / "validation_kothari_replay.csv", replay_frame)
    write_csv(RESULTS_DIR / "tables" / "validation_literature_quantitative.csv", strict_frame)
    write_csv(RESULTS_DIR / "tables" / "validation_literature_qualitative_kothari.csv", replay_frame)
    write_json(
        RESULTS_DIR / "kothari_summary.json",
        {
            "source_paper": SOURCE_PAPER,
            "model_match_level": "reduced",
            "strict_mode_description": "Published table parameters and pointwise zero initial state only; no inferred intercept fitting.",
            "replay_mode_description": "Allows digitized curves, inferred initial conditions, and figure-based alignment for exploratory replay.",
            "strict_rows": len(strict_frame),
            "replay_rows": len(replay_frame),
        },
    )

    return {
        "strict_rows": int(len(strict_frame)),
        "replay_rows": int(len(replay_frame)),
        "best_strict_rmse": None if strict_frame["rmse"].dropna().empty else float(strict_frame["rmse"].dropna().min()),
        "best_replay_rmse": None if replay_frame["rmse"].dropna().empty else float(replay_frame["rmse"].dropna().min()),
        "output_files": {
            "strict": str(RESULTS_DIR / "validation_kothari_strict.csv"),
            "replay": str(RESULTS_DIR / "validation_kothari_replay.csv"),
        },
        "combined_columns": combined_columns,
    }
