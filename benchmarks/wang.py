from __future__ import annotations

import pandas as pd

from .utils import RESULTS_DIR, write_csv, write_json


SOURCE_PAPER = "Wang et al. 2020 lithium-ion battery and ultra-capacitor study"


def run_wang_external_reference() -> dict:
    frame = pd.DataFrame(
        [
            {"temperature_C": 0, "published_mae_mV": 26.4, "published_mre_percent": 1.45, "published_rmse_mV": 32.8},
            {"temperature_C": 25, "published_mae_mV": 37.3, "published_mre_percent": 2.10, "published_rmse_mV": 45.1},
            {"temperature_C": 45, "published_mae_mV": 28.2, "published_mre_percent": 1.52, "published_rmse_mV": 35.4},
        ]
    )
    write_csv(RESULTS_DIR / "validation_wang_external_reference.csv", frame)
    note = {
        "benchmark_id": "wang_external_reference",
        "validation_category": "validation_literature_qualitative",
        "source_paper": SOURCE_PAPER,
        "model_match_level": "qualitative",
        "data_source": "manual_table_transcription",
        "initialization_type": "history-dependent",
        "claim_level": "qualitative_validation",
        "message": (
            "Wang is treated as an external reference only. The published model structure is richer than the present one-term RI-MML solver, "
            "so no strict quantitative validation claim is made."
        ),
    }
    write_json(RESULTS_DIR / "wang_external_reference.json", note)
    return note
