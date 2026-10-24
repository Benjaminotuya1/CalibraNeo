"""Freeze a hash-pinned prediction table before experimental validation."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibraneo.inference import load_predictor
from calibraneo.task_models import predict_endpoint


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    processed = ROOT / "data_processed"
    models = ROOT / "models"
    results = ROOT / "results"
    master = pd.read_csv(processed / "MASTER_CALIBRANEO_DATASET.csv", low_memory=False)
    pool = master[master["dataset_source"].eq("Nature_Clinical_NeoVaccine_2317")].copy()
    predictor = load_predictor(models / "calibraneo_stage1_bundle.pkl")
    pool["presentation_score_frozen"] = predictor.predict_proba(pool)
    immunogenicity_model = models / "calibraneo_immunogenicity_ott.pkl"
    if immunogenicity_model.is_file():
        pool["immunogenicity_score_frozen"] = predict_endpoint(
            immunogenicity_model, pool, peptide_column="peptide"
        )
    else:
        pool["immunogenicity_score_frozen"] = pd.NA
    output = results / "prospective_frozen_predictions.csv"
    pool.to_csv(output, index=False)
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": sha256(processed / "MASTER_CALIBRANEO_DATASET.csv"),
        "presentation_model_sha256": sha256(models / "calibraneo_stage1_bundle.pkl"),
        "immunogenicity_model_sha256": sha256(immunogenicity_model)
        if immunogenicity_model.is_file() else None,
        "output_sha256": sha256(output),
        "rows": int(len(pool)),
        "status": "frozen_before_wet_lab",
        "claim_boundary": "Frozen computational predictions are not biological validation.",
    }
    (results / "prospective_freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
