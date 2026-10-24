"""Combine available T-cell sources and run leave-one-study-out evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibraneo.task_models import harmonize_immunogenicity_sources, sequence_features


def main() -> int:
    raw = ROOT / "data_raw"
    external = ROOT / "data_external" / "iedb" / "iedb_tcell_normalized.csv"
    ott = pd.read_csv(
        raw / "big_benchmarks" / "ott_clinical_vaccine" / "immunizing-peptides-with-tcell.csv"
    )
    iedb = pd.read_csv(external)
    table = harmonize_immunogenicity_sources(ott=ott, iedb=iedb)
    table["study_group"] = (
        table["study"].astype(str)
        + "|"
        + table["publication"].astype(str)
        + "|"
        + table["peptide"].astype(str)
    )
    table.to_csv(ROOT / "data_external" / "immunogenicity_combined_normalized.csv", index=False)
    features = sequence_features(table)
    y = table["label"].astype(int).to_numpy()
    rows = []
    for study in sorted(table["study"].unique()):
        test = table["study"].eq(study).to_numpy()
        train = ~test
        if y[train].sum() == 0 or y[train].sum() == train.sum():
            rows.append({"held_out_study": study, "status": "not_estimable", "reason": "training labels have one class"})
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features.loc[train])
        x_test = scaler.transform(features.loc[test])
        model = HistGradientBoostingClassifier(
            max_iter=120, max_depth=3, learning_rate=0.05, random_state=42
        )
        model.fit(x_train, y[train])
        probabilities = model.predict_proba(x_test)[:, 1]
        rows.append(
            {
                "held_out_study": study,
                "status": "evaluated",
                "rows": int(test.sum()),
                "positive_rate": float(y[test].mean()),
                "auc_roc": float(roc_auc_score(y[test], probabilities))
                if len(np.unique(y[test])) == 2 else np.nan,
                "auprc": float(average_precision_score(y[test], probabilities)),
                "group_definition": "study held out; peptide features only",
            }
        )
    report = {
        "sources": {
            "Ott": int((table["study"] == "Ott").sum()),
            "IEDB": int((table["study"] == "IEDB").sum()),
        },
        "rows": int(len(table)),
        "unique_peptides": int(table["peptide"].nunique()),
        "loso": rows,
        "limitations": [
            "IEDB sample is bounded and heterogeneous",
            "Ott and IEDB endpoints are not identical clinical assays",
            "two-source LOSO is a feasibility audit, not definitive generalization",
        ],
    }
    (ROOT / "results" / "combined_immunogenicity_loso.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(
        ROOT / "results" / "combined_immunogenicity_loso.csv", index=False
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
