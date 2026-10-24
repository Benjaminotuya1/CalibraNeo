"""Inference against the versioned CalibraNeo model bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class CalibraNeoPredictor:
    """Loads and applies the trained Stage 1 model without redefining it."""

    bundle: dict[str, Any]

    @property
    def feature_names(self) -> list[str]:
        return list(self.bundle["bio_features"])

    @property
    def model_version(self) -> str:
        return str(self.bundle.get("model_version", "stage1-unknown"))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing model features: {', '.join(missing)}")
        values = frame[self.feature_names].to_numpy(dtype=np.float32)
        scaled = self.bundle["scaler"].transform(values)
        raw = (
            self.bundle["gbdt"].predict_proba(scaled)[:, 1]
            + self.bundle["mlp"].predict_proba(scaled)[:, 1]
        ) / 2.0
        calibrator = self.bundle.get("calibrator", self.bundle.get("iso_cal"))
        if calibrator is None:
            raise ValueError("Model bundle does not contain a calibrator")
        return np.asarray(calibrator.transform(raw), dtype=float)


def load_predictor(model_path: str | Path) -> CalibraNeoPredictor:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    bundle = joblib.load(path)
    required = {"scaler", "gbdt", "mlp", "bio_features"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Invalid model bundle; missing keys: {sorted(missing)}")
    return CalibraNeoPredictor(bundle)
