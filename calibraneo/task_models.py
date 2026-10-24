"""Endpoint-specific models and sequence feature extraction.

Models in this module are deliberately separate: T-cell response and
immunopeptidomics are different labels. Missing RNA or translation labels are
reported as unavailable rather than learned from a proxy endpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


AA_HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
AA_CHARGE = {"R": 1.0, "K": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}
FEATURE_COLUMNS = [
    "sequence_length", "hydrophobicity", "charge", "aromatic_fraction",
    "anchor_p2_hydrophobicity", "anchor_omega_hydrophobicity",
    "anchor_p2_charge", "anchor_omega_charge",
]


def sequence_features(
    frame: pd.DataFrame,
    *,
    peptide_column: str = "peptide",
) -> pd.DataFrame:
    """Calculate endpoint-neutral peptide features from amino-acid sequences."""
    if peptide_column not in frame:
        raise ValueError(f"Missing peptide column: {peptide_column}")
    peptides = frame[peptide_column].astype(str).str.upper().str.strip()

    def mean_property(sequence: str, mapping: dict[str, float]) -> float:
        values = [mapping.get(char, 0.0) for char in sequence]
        return float(np.mean(values)) if values else 0.0

    rows = []
    for peptide in peptides:
        rows.append(
            {
                "sequence_length": len(peptide),
                "hydrophobicity": mean_property(peptide, AA_HYDRO),
                "charge": sum(AA_CHARGE.get(char, 0.0) for char in peptide),
                "aromatic_fraction": sum(char in "FWYH" for char in peptide)
                / max(len(peptide), 1),
                "anchor_p2_hydrophobicity": AA_HYDRO.get(peptide[1], 0.0)
                if len(peptide) > 1 else 0.0,
                "anchor_omega_hydrophobicity": AA_HYDRO.get(peptide[-1], 0.0)
                if peptide else 0.0,
                "anchor_p2_charge": AA_CHARGE.get(peptide[1], 0.0)
                if len(peptide) > 1 else 0.0,
                "anchor_omega_charge": AA_CHARGE.get(peptide[-1], 0.0)
                if peptide else 0.0,
            }
        )
    return pd.DataFrame(rows, index=frame.index, columns=FEATURE_COLUMNS)


def train_endpoint_model(
    frame: pd.DataFrame,
    *,
    peptide_column: str,
    label_column: str,
    group_columns: list[str],
    endpoint: str,
    output_path: str | Path,
    extra_features: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Train one endpoint model with a grouped untouched holdout."""
    required = {peptide_column, label_column, *group_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing {endpoint} columns: {', '.join(sorted(missing))}")
    clean = frame.dropna(subset=[label_column]).copy()
    features = sequence_features(clean, peptide_column=peptide_column)
    feature_names = list(FEATURE_COLUMNS)
    if extra_features:
        for name in extra_features:
            if name not in clean:
                raise ValueError(f"Missing extra feature: {name}")
            features[name] = pd.to_numeric(clean[name], errors="coerce").fillna(0.0)
            feature_names.append(name)
    y = clean[label_column].astype(int).to_numpy()
    groups = clean[group_columns].astype(str).agg("|".join, axis=1)
    if len(np.unique(y)) < 2:
        raise ValueError(f"{endpoint} requires both positive and negative labels")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    fit_idx, test_idx = next(splitter.split(features, y, groups))
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(features.iloc[fit_idx][feature_names])
    x_test = scaler.transform(features.iloc[test_idx][feature_names])
    model = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.06, max_depth=4, random_state=seed
    )
    model.fit(x_fit, y[fit_idx])
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "auc_roc": float(roc_auc_score(y[test_idx], probabilities)),
        "auprc": float(average_precision_score(y[test_idx], probabilities)),
        "brier": float(brier_score_loss(y[test_idx], probabilities)),
        "fit_rows": int(len(fit_idx)),
        "test_rows": int(len(test_idx)),
        "test_groups": int(groups.iloc[test_idx].nunique()),
    }
    bundle = {
        "endpoint": endpoint,
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_version": f"{endpoint}-grouped-v1",
        "metrics": metrics,
        "label_column": label_column,
        "group_columns": group_columns,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    return bundle


def predict_endpoint(
    bundle_path: str | Path,
    frame: pd.DataFrame,
    *,
    peptide_column: str,
) -> np.ndarray:
    """Apply a saved endpoint model to a candidate table."""
    bundle = joblib.load(bundle_path)
    features = sequence_features(frame, peptide_column=peptide_column)
    for name in bundle["feature_names"]:
        if name not in features:
            if name not in frame:
                raise ValueError(f"Missing endpoint feature: {name}")
            features[name] = pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
    values = bundle["scaler"].transform(features[bundle["feature_names"]])
    return bundle["model"].predict_proba(values)[:, 1]
