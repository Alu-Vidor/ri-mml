from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import REPORT_DIR, RESULTS_DIR, read_json


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| empty |\n|---|\n| no rows |"
    safe = frame.copy()
    safe.columns = [str(c) for c in safe.columns]
    header = "| " + " | ".join(safe.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(safe.columns)) + " |"
    body = []
    for _, row in safe.iterrows():
        body.append("| " + " | ".join(str(row[col]) for col in safe.columns) + " |")
    return "\n".join([header, sep, *body])


def build_report_markdown() -> str:
    exact = pd.read_csv(RESULTS_DIR / "metrics_exact.csv")
    kothari = pd.read_csv(RESULTS_DIR / "metrics_kothari.csv")
    lopez = pd.read_csv(RESULTS_DIR / "metrics_lopez.csv")
    gomez = pd.read_csv(RESULTS_DIR / "metrics_gomez.csv")
    lopez_note = read_json(RESULTS_DIR / "lopez_note.json")
    kothari_mapping = read_json(RESULTS_DIR / "kothari_mapping_note.json")

    lines: list[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("This report was generated from a script-based benchmark harness around the existing RI-MML solver.")
    lines.append("Used sources: `additional_inf/Kajal.pdf`, `additional_inf/energies-15-00792-v2.pdf`, `additional_inf/ArticuloPublicado.PDF`, `additional_inf/j.apenergy.2020.115736.pdf`.")
    lines.append("")
    lines.append("## 2. Exact-solution convergence")
    lines.append("")
    lines.append("Exact mapping: `epsilon D_t^alpha u + u = V*` with the closed-form Mittag-Leffler solution.")
    lines.append("")
    best_exact = exact.groupby(["case", "alpha", "epsilon"], as_index=False)[["l_inf", "l2"]].min()
    lines.append("### Table A. Exact benchmark convergence")
    lines.append("")
    lines.append(_markdown_table(best_exact))
    lines.append("")
    lines.append("## 3. Kothari main benchmark")
    lines.append("")
    lines.append(f"Mapping note: {kothari_mapping['mapping']}")
    lines.append("Digitized curves from PDF figures were used wherever raw experimental traces were unavailable.")
    lines.append("")
    lines.append("### Table B. Kothari scenario-by-scenario metrics")
    lines.append("")
    cols_b = [
        c
        for c in [
            "scenario",
            "source_refs",
            "brand",
            "capacitance_F",
            "mode",
            "mapping_status",
            "digitized_used",
            "parameters_origin",
            "rmse",
            "mae",
            "mre",
            "e_inf",
        ]
        if c in kothari.columns
    ]
    lines.append(_markdown_table(kothari[cols_b]))
    lines.append("")
    lines.append("### Table C. Kothari comparison against integer-order baseline and MLF baseline")
    lines.append("")
    cols_c = [
        c
        for c in [
            "scenario",
            "source_refs",
            "rmse",
            "integer_rmse",
            "mlf_rmse",
            "N",
            "runtime_seconds",
            "residual_norm",
            "condition_number",
        ]
        if c in kothari.columns
    ]
    lines.append(_markdown_table(kothari[cols_c]))
    lines.append("")
    lines.append("## 4. Initialization stress-test (Lopez)")
    lines.append("")
    lines.append("Paper/file source: `additional_inf/energies-15-00792-v2.pdf`.")
    lines.append("Direct mapping to the current solver is unsupported because the solver uses pointwise initialization only.")
    lines.append("")
    lines.append("### Table D. Lopez initialization notes")
    lines.append("")
    lines.append(_markdown_table(pd.DataFrame([lopez_note])))
    lines.append("")
    lines.append(f"Generated {len(lopez)} analytic samples for `g1`, `g2`, `g3`, and `g4`.")
    lines.append("")
    lines.append("## 5. Simple fractional RC sanity-check (Gomez)")
    lines.append("")
    lines.append("Paper/file source: `additional_inf/ArticuloPublicado.PDF`.")
    lines.append("Nearest compatible formulation: `tau_gamma D_t^gamma v + v = sin(omega t)`. This section is qualitative only.")
    lines.append("")
    lines.append(f"Generated {len(gomez)} samples for gamma in `{{1, 0.98, 0.96}}`.")
    lines.append("")
    lines.append("## 6. Optional Wang external reference")
    lines.append("")
    lines.append("Wang 2020 was not force-fit onto the one-term solver. Published ultra-capacitor targets are included only as external references.")
    lines.append("")
    lines.append("| temperature | published MAE | published MRE | published RMSE |")
    lines.append("|---|---:|---:|---:|")
    lines.append("| 0 C | 26.4 mV | 1.45% | 32.8 mV |")
    lines.append("| 25 C | 37.3 mV | 2.10% | 45.1 mV |")
    lines.append("| 45 C | 28.2 mV | 1.52% | 35.4 mV |")
    lines.append("")
    lines.append("## 7. Conclusions")
    lines.append("")
    lines.append("- Exact comparisons are strict and reproducible.")
    lines.append("- Kothari scenarios use published table parameters first; no parameter identification was added in this first pass.")
    lines.append("- Lopez is handled as a limitation study, not a fake solver capability claim.")
    lines.append("- Gomez is qualitative and checks that the trend with decreasing gamma is recovered.")
    lines.append("")
    lines.append("## 8. Limitations / unsupported comparisons")
    lines.append("")
    lines.append("- `Kajal.pdf` has no text layer, so table values were transcribed manually into machine-readable JSON.")
    lines.append("- Kothari metrics are versus digitized figure curves from PDF pages when raw data are unavailable.")
    lines.append("- Some Kothari initial voltages and switching times are inferred from figure geometry, so those scenarios are marked approximate.")
    lines.append("- Wang 2020 was kept as a target metric box only because its model structure is richer than the present solver.")
    lines.append("")
    return "\n".join(lines)


def write_report_files() -> dict[str, str]:
    markdown = build_report_markdown()
    md_path = REPORT_DIR / "benchmark_report.md"
    html_path = REPORT_DIR / "benchmark_report.html"
    md_path.write_text(markdown, encoding="utf-8")
    try:
        import markdown as md

        html = md.markdown(markdown, extensions=["tables", "fenced_code"])
    except Exception:
        escaped = markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = f"<html><body><pre>{escaped}</pre></body></html>"
    html_path.write_text(html, encoding="utf-8")
    return {"markdown": str(md_path), "html": str(html_path)}
