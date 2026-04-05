from __future__ import annotations

import numpy as np

from .caputo_mainardi_silver import run_caputo_mainardi_silver_benchmark
from .gomez import run_gomez_qualitative_reference
from .kothari import run_kothari_benchmarks
from .lopez import run_lopez_initialization_limit
from .reporting import write_report_files
from .utils import RESULTS_DIR, ensure_directories, write_json
from .verification import run_verification_suite
from .wang import run_wang_external_reference


def main() -> None:
    np.random.seed(0)
    ensure_directories()

    verification_summary = run_verification_suite()
    caputo_mainardi_summary = run_caputo_mainardi_silver_benchmark()
    kothari_summary = run_kothari_benchmarks()
    lopez_summary = run_lopez_initialization_limit()
    gomez_summary = run_gomez_qualitative_reference()
    wang_summary = run_wang_external_reference()
    report_summary = write_report_files()

    summary = {
        "entrypoint": "python -m benchmarks",
        "deterministic_seed": 0,
        "solver_initialization_support": "pointwise_only",
        "verified": verification_summary,
        "real_material_benchmarks": {
            "caputo_mainardi_silver": caputo_mainardi_summary,
        },
        "validation_literature_quantitative": {
            "kothari_strict": kothari_summary["strict_rows"],
        },
        "validation_literature_qualitative": {
            "kothari_replay": kothari_summary["replay_rows"],
            "gomez": gomez_summary,
            "wang": wang_summary,
        },
        "limitations": {
            "lopez": lopez_summary,
        },
        "report": report_summary,
    }
    write_json(RESULTS_DIR / "summary.json", summary)


if __name__ == "__main__":
    main()
