"""Train the RNADecayCafe gene-level RNA stability context model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibraneo.rna_model import train_rna_context_model


def main() -> int:
    source = ROOT / "data_external" / "rna" / "RNADecayCafe_AvgKdegs_genes_v1.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Download the RNADecayCafe table first: {source}")
    table = pd.read_csv(source)
    bundle = train_rna_context_model(
        table,
        ROOT / "models" / "calibraneo_rna_context.pkl",
    )
    manifest_path = ROOT / "models" / "endpoint_model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("trained", {})["rna_context_rnadecaycafe"] = {
        "path": "models/calibraneo_rna_context.pkl",
        "source": "RNADecayCafe AvgKdegs genes table",
        "metrics": bundle["metrics"],
        "limitations": [
            "gene-level context, not custom ORF stability",
            "cell-line aggregation",
            "not a delivery or vaccine formulation model",
        ],
    }
    manifest["not_trained"].pop("rna_stability_ttdb_rnadecaycafe", None)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(bundle["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
