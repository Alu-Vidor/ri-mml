from __future__ import annotations

from .exact import run_exact_benchmarks
from .gomez import run_gomez_benchmark
from .kothari import run_kothari_benchmarks
from .lopez import run_lopez_benchmarks
from .reporting import write_report_files
from .utils import RESULTS_DIR, ensure_directories, write_json
import pandas as pd


def main() -> None:
    ensure_directories()
    exact_summary = run_exact_benchmarks()
    kothari_summary = run_kothari_benchmarks()
    lopez_summary = run_lopez_benchmarks()
    gomez_summary = run_gomez_benchmark()
    report_summary = write_report_files()

    exact_metrics = pd.read_csv(RESULTS_DIR / "metrics_exact.csv")
    kothari_metrics = pd.read_csv(RESULTS_DIR / "metrics_kothari.csv")
    summary = {
        "exact": exact_summary,
        "kothari": kothari_summary,
        "lopez": lopez_summary,
        "gomez": gomez_summary,
        "report": report_summary,
        "key_metrics": {
            "exact_best_linf": float(exact_metrics["l_inf"].min()),
            "exact_best_l2": float(exact_metrics["l2"].min()),
            "kothari_best_rmse": float(kothari_metrics["rmse"].dropna().min()),
            "kothari_worst_rmse": float(kothari_metrics["rmse"].dropna().max()),
            "kothari_rows": int(len(kothari_metrics)),
        },
    }
    write_json(RESULTS_DIR / "summary.json", summary)


if __name__ == "__main__":
    main()
