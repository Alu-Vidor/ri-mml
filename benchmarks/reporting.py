from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import REPORT_DIR, RESULTS_DIR, read_json


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| empty |\n|---|\n| no rows |"
    safe = frame.copy()
    safe.columns = [str(column) for column in safe.columns]
    header = "| " + " | ".join(safe.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(safe.columns)) + " |"
    body = ["| " + " | ".join(str(row[column]) for column in safe.columns) + " |" for _, row in safe.iterrows()]
    return "\n".join([header, separator, *body])


def _safe_min(frame: pd.DataFrame, column: str) -> float | None:
    series = frame[column].dropna()
    if series.empty:
        return None
    return float(series.min())


def build_report_markdown() -> str:
    verification_exact = pd.read_csv(RESULTS_DIR / "verification_exact.csv")
    verification_convergence = pd.read_csv(RESULTS_DIR / "verification_convergence.csv")
    kothari_strict = pd.read_csv(RESULTS_DIR / "validation_kothari_strict.csv")
    kothari_replay = pd.read_csv(RESULTS_DIR / "validation_kothari_replay.csv")
    lopez_note = read_json(RESULTS_DIR / "lopez_initialization_limit.json")
    gomez_note = read_json(RESULTS_DIR / "gomez_qualitative_reference.json")
    wang_note = read_json(RESULTS_DIR / "wang_external_reference.json")
    wang_table = pd.read_csv(RESULTS_DIR / "validation_wang_external_reference.csv")

    exact_best = verification_exact.groupby(["case_id", "alpha", "epsilon"], as_index=False)[["l2_abs", "l2_rel", "l_inf"]].min()
    convergence_best = pd.read_csv(RESULTS_DIR / "tables" / "verification_convergence_best.csv")
    convergence_rates = pd.read_csv(RESULTS_DIR / "tables" / "verification_convergence_rates.csv")
    quantitative_best_rmse = _safe_min(kothari_strict, "rmse")
    replay_best_rmse = _safe_min(kothari_replay, "rmse")

    lines: list[str] = []
    lines.append("# Validation Report")
    lines.append("")
    lines.append("## What this report validates")
    lines.append("")
    lines.append("- Verification of the RI-MML implementation on closed-form Mittag-Leffler test problems in the same mathematical setting as the solver.")
    lines.append("- Verification of convergence behavior on manufactured solutions, including smooth, weakly singular, and trial-space-aligned families.")
    lines.append("- A narrow quantitative literature comparison for Kothari zero-initial scenarios that can be replayed without inferred initial state fitting.")
    lines.append("")
    lines.append("## What this report does not validate")
    lines.append("")
    lines.append("- It does not validate history-dependent initialization. The current solver supports pointwise initial condition only: `u(0)=u0`.")
    lines.append("- It does not claim one-to-one replication of Lopez, Wang, or Gomez models when their published setup differs structurally from the present solver.")
    lines.append("- It does not claim validation against raw experimental data when curves were digitized from PDF figures.")
    lines.append("")
    lines.append("## 1. Verification: Closed-Form Exact Problems")
    lines.append("")
    lines.append("These cases reproduce analytic Mittag-Leffler solutions with fixed evaluation grids and deterministic solver settings.")
    lines.append("")
    lines.append(_markdown_table(exact_best))
    lines.append("")
    lines.append(
        f"Best absolute L2 error across exact cases: `{verification_exact['l2_abs'].min():.3e}`. "
        f"Worst absolute L2 error across exact cases: `{verification_exact['l2_abs'].max():.3e}`."
    )
    lines.append("")
    lines.append("## 2. Verification: Manufactured Convergence")
    lines.append("")
    lines.append("Manufactured solutions are grouped into smooth analytic, weak endpoint singularity, and trial-space-aligned families.")
    lines.append("")
    lines.append(_markdown_table(convergence_best))
    lines.append("")
    best_family_rates = convergence_rates.groupby("family", dropna=False)["l2_abs_observed_rate"].max().dropna()
    if not best_family_rates.empty:
        rate_text = ", ".join(f"{family}: {rate:.2f}" for family, rate in best_family_rates.items())
        lines.append(f"Maximum observed local convergence-rate indicators by family: {rate_text}.")
        lines.append("")
    lines.append("## 3. Quantitatively Compared Literature Cases")
    lines.append("")
    if kothari_strict.empty:
        lines.append("No literature case survived the strict quantitative-validation filter.")
    else:
        lines.append("Only Kothari zero-initial scenarios remain in this section. They use published Table 5 parameters and pointwise zero initial state, but comparison still relies on digitized curves from PDF figures.")
        lines.append("")
        lines.append(_markdown_table(kothari_strict[[
            "benchmark_id",
            "figure",
            "mode",
            "model_match_level",
            "data_source",
            "initialization_type",
            "claim_level",
            "duration_s",
            "rmse",
            "mae",
            "e_inf",
        ]]))
        lines.append("")
        if quantitative_best_rmse is not None:
            lines.append(f"Best strict Kothari RMSE: `{quantitative_best_rmse:.3e}`.")
            lines.append("Observed strict-mode errors remain moderate rather than negligible, so this section supports only a cautious quantitative comparison.")
    lines.append("")
    lines.append("## 4. Qualitatively Compared Literature Cases")
    lines.append("")
    lines.append("These cases are retained as external references only. They may still report numerical distances, but those distances are not promoted to strict validation claims.")
    lines.append("")
    qualitative_rows = []
    if not kothari_replay.empty:
        for row in kothari_replay.itertuples(index=False):
            qualitative_rows.append(
                {
                    "benchmark_id": row.benchmark_id,
                    "mode": row.mode,
                    "model_match_level": row.model_match_level,
                    "data_source": row.data_source,
                    "initialization_type": row.initialization_type,
                    "claim_level": row.claim_level,
                    "rmse": row.rmse,
                    "note": row.comparison_note,
                }
            )
    for note in [gomez_note, wang_note]:
        qualitative_rows.append(
            {
                "benchmark_id": note["benchmark_id"],
                "mode": "external_reference",
                "model_match_level": note["model_match_level"],
                "data_source": note["data_source"],
                "initialization_type": note["initialization_type"],
                "claim_level": note["claim_level"],
                "rmse": None,
                "note": note["message"],
            }
        )
    lines.append(_markdown_table(pd.DataFrame(qualitative_rows)))
    lines.append("")
    if replay_best_rmse is not None:
        lines.append(f"Best replay-only Kothari RMSE: `{replay_best_rmse:.3e}`. This number is descriptive only and should not be read as validation accuracy.")
        lines.append("")
    lines.append("Published Wang metrics are preserved below as external context, not as a scorecard for the present solver.")
    lines.append("")
    lines.append(_markdown_table(wang_table))
    lines.append("")
    lines.append("## 5. Limitations")
    lines.append("")
    lines.append(_markdown_table(pd.DataFrame([lopez_note])))
    lines.append("")
    lines.append("## 6. Reproducibility")
    lines.append("")
    lines.append("- Single-command entrypoint: `python -m benchmarks`.")
    lines.append("- Deterministic settings are stored in `benchmarks/configs/verification_suite.json`.")
    lines.append("- Machine-readable artifacts are written to `benchmarks/results/` and paper-ready tables to `benchmarks/results/tables/`.")
    lines.append("- Benchmark claims are separated into verification, quantitative literature comparison, qualitative external reference, and limitations.")
    lines.append("- For constant-coefficient reductions the discrete matrix can collapse to the identity, so `condition_number = 1` and zero linear residual are expected but not the primary diagnostics.")
    lines.append("")
    return "\n".join(lines)


def build_paper_style_summary() -> str:
    verification_exact = pd.read_csv(RESULTS_DIR / "verification_exact.csv")
    verification_convergence = pd.read_csv(RESULTS_DIR / "verification_convergence.csv")
    kothari_strict = pd.read_csv(RESULTS_DIR / "validation_kothari_strict.csv")
    kothari_replay = pd.read_csv(RESULTS_DIR / "validation_kothari_replay.csv")

    lines: list[str] = []
    lines.append("# Paper-Style Validation Summary")
    lines.append("")
    lines.append("## Verified")
    lines.append("")
    lines.append(
        f"- The solver reproduces closed-form Mittag-Leffler test problems on a fixed grid, with best/worst absolute L2 errors of "
        f"`{verification_exact['l2_abs'].min():.3e}` / `{verification_exact['l2_abs'].max():.3e}` over the reported sweep."
    )
    lines.append(
        f"- The solver demonstrates convergence on manufactured solutions across smooth, weakly singular, and trial-space-aligned families; "
        f"best/worst absolute L2 errors in that suite are `{verification_convergence['l2_abs'].min():.3e}` / `{verification_convergence['l2_abs'].max():.3e}`."
    )
    lines.append("")
    lines.append("## Quantitatively Compared")
    lines.append("")
    if kothari_strict.empty:
        lines.append("- No external literature case currently supports a strict quantitative validation claim.")
    else:
        lines.append(
            "- A limited Kothari subset with published zero-initial conditions is compared quantitatively under model-reduction and PDF-digitization assumptions only, with moderate RMSE rather than strong one-to-one agreement."
        )
    lines.append("")
    lines.append("## Qualitatively Compared")
    lines.append("")
    lines.append(
        f"- Kothari replay scenarios with inferred initial state or inferred switching remain descriptive only ({len(kothari_replay)} replay rows)."
    )
    lines.append("- Gomez and Wang are retained as external structural references and do not support strict quantitative validation claims.")
    lines.append("")
    lines.append("## Not Supported / Limitations")
    lines.append("")
    lines.append("- History-dependent initialization is not implemented; the current solver supports pointwise initial condition only.")
    lines.append("- Lopez is therefore a limitation study, not an accuracy benchmark.")
    lines.append("- Benchmarks based only on digitized curves remain approximate and should not be described as raw experimental validation.")
    return "\n".join(lines)


def write_report_files() -> dict[str, str]:
    markdown = build_report_markdown()
    paper_summary = build_paper_style_summary()
    md_path = REPORT_DIR / "validation_report.md"
    html_path = REPORT_DIR / "validation_report.html"
    paper_summary_path = REPORT_DIR / "paper_validation_summary.md"
    md_path.write_text(markdown, encoding="utf-8")
    paper_summary_path.write_text(paper_summary, encoding="utf-8")
    try:
        import markdown as md

        html = md.markdown(markdown, extensions=["tables", "fenced_code"])
    except Exception:
        escaped = markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = f"<html><body><pre>{escaped}</pre></body></html>"
    html_path.write_text(html, encoding="utf-8")
    return {
        "markdown": str(md_path),
        "html": str(html_path),
        "paper_summary": str(paper_summary_path),
    }
