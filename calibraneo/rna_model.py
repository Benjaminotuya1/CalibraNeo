"""RNA stability context model trained from RNADecayCafe gene-level labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


RNA_FEATURES = [
    "avg_log_RPKM_total",
    "avg_log_RPKM_3pend",
]


def train_rna_context_model(
    table: pd.DataFrame,
    output_path: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a grouped gene holdout model for gene-level half-life context."""
    required = {"feature_ID", "avg_halflife", *RNA_FEATURES}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"RNA table missing columns: {', '.join(sorted(missing))}")
    clean = table.dropna(subset=["feature_ID", "avg_halflife", *RNA_FEATURES]).copy()
    groups = clean["feature_ID"].astype(str)
    y = pd.to_numeric(clean["avg_halflife"], errors="coerce").to_numpy(float)
    x = clean[RNA_FEATURES].to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    clean = clean.loc[valid].reset_index(drop=True)
    y = y[valid]
    x = x[valid]
    groups = groups.loc[valid].reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    fit_idx, test_idx = next(splitter.split(x, y, groups))
    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x[fit_idx], y[fit_idx])
    predictions = model.predict(x[test_idx])
    metrics = {
        "mae": float(mean_absolute_error(y[test_idx], predictions)),
        "r2": float(r2_score(y[test_idx], predictions)),
        "fit_rows": int(len(fit_idx)),
        "test_rows": int(len(test_idx)),
        "test_genes": int(groups.iloc[test_idx].nunique()),
    }
    bundle = {
        "endpoint": "gene_rna_half_life_context",
        "model": model,
        "feature_names": RNA_FEATURES,
        "model_version": "rnadecaycafe-context-grouped-v1",
        "metrics": metrics,
        "source": "RNADecayCafe_v1 AvgKdegs_genes_v1.csv",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    return bundle


def annotate_gene_stability(
    candidates: pd.DataFrame,
    stability_table: pd.DataFrame,
    *,
    gene_column: str = "gene",
) -> pd.DataFrame:
    """Attach observed gene-level half-life context without inventing ORF labels."""
    if gene_column not in candidates:
        raise ValueError(f"Missing candidate gene column: {gene_column}")
    required = {"feature_ID", "avg_halflife"}
    missing = required.difference(stability_table.columns)
    if missing:
        raise ValueError(f"Stability table missing columns: {', '.join(sorted(missing))}")
    lookup = (
        stability_table.assign(
            feature_ID=stability_table["feature_ID"].astype(str).str.upper()
        )
        .groupby("feature_ID")["avg_halflife"]
        .median()
    )
    result = candidates.copy()
    result["gene_half_life_context"] = (
        result[gene_column].astype(str).str.upper().map(lookup)
    )
    result["rna_stability_evidence_status"] = np.where(
        result["gene_half_life_context"].notna(),
        "gene_context_observed",
        "gene_context_unknown",
    )
    return result
