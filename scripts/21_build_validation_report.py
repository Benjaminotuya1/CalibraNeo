"""Build the reproducible candidate-level validation report and experiment queue."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibraneo.evidence import attach_iedb_evidence
from calibraneo.inference import load_predictor
from calibraneo.validation import (
    build_experiment_plan,
    validation_report,
)


def main() -> int:
    root = ROOT
    master = pd.read_csv(root / "data_processed" / "MASTER_CALIBRANEO_DATASET.csv", low_memory=False)
    pool = pd.read_csv(root / "data_processed" / "nature_2317_9tools_benchmark.csv")
    source = master[master["dataset_source"].eq("Nature_Clinical_NeoVaccine_2317")].reset_index(drop=True)
    if len(source) != len(pool):
        raise ValueError("Clinical benchmark and feature rows have different lengths")
    predictor = load_predictor(root / "models" / "calibraneo_stage1_bundle.pkl")
    pool["CalibraNeo_Prob"] = predictor.predict_proba(source)
    iedb = pd.read_csv(root / "data_external" / "iedb" / "iedb_evidence_normalized.csv")
    matched = attach_iedb_evidence(pool, iedb)
    plan = build_experiment_plan(matched)
    results = root / "results"
    results.mkdir(exist_ok=True)
    plan.to_csv(results / "candidate_experiment_queue.csv", index=False)
    benchmark_path = results / "table1_clinical_benchmark_vs_9tools.csv"
    benchmark = pd.read_csv(benchmark_path) if benchmark_path.is_file() else None
    report = validation_report(
        model_version=predictor.model_version,
        candidate_count=len(pool),
        external_evidence_rows=len(iedb),
        exact_tcell_matches=int(matched["iedb_tcell_support"].eq(True).sum()),
        benchmark_metrics=benchmark,
    )
    (results / "computational_validation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "ok",
        "queue_rows": len(plan),
        "report": str(results / "computational_validation_report.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
