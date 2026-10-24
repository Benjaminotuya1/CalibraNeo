"""Transparent LOH-aware cassette selection.

This is deliberately named an optimizer rather than RL: it is a deterministic
greedy approximation with an explicit objective and reproducible tie-breaking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_loh_aware_cassette(
    candidates: pd.DataFrame,
    capacity: int,
    dominant_loh_risk: float,
    *,
    diversity_decay: float = 0.45,
) -> pd.DataFrame:
    required = {"hla_clean", "CalibraNeo_Prob"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Missing selection columns: {', '.join(sorted(missing))}")
    if capacity < 1:
        raise ValueError("capacity must be positive")
    if not 0 <= dominant_loh_risk <= 1:
        raise ValueError("dominant_loh_risk must be between 0 and 1")

    frame = candidates.copy().reset_index(drop=True)
    frame["_hla_risk"] = 0.10
    dominant = frame["hla_clean"].value_counts().index[0] if len(frame) else None
    if dominant is not None:
        frame.loc[frame["hla_clean"].eq(dominant), "_hla_risk"] = dominant_loh_risk

    selected: list[int] = []
    counts: dict[str, int] = {}
    for _ in range(min(capacity, len(frame))):
        best_index: int | None = None
        best_key: tuple[float, str] | None = None
        for index, row in frame.iterrows():
            if index in selected:
                continue
            allele = str(row["hla_clean"])
            reward = (
                float(row["CalibraNeo_Prob"])
                * (1.0 - float(row["_hla_risk"]))
                * np.exp(-diversity_decay * counts.get(allele, 0))
            )
            key = (reward, f"{row.get('Peptide ID', '')}|{index}")
            if best_key is None or key > best_key:
                best_index, best_key = index, key
        if best_index is None:
            break
        selected.append(best_index)
        allele = str(frame.loc[best_index, "hla_clean"])
        counts[allele] = counts.get(allele, 0) + 1

    result = frame.loc[selected].drop(columns="_hla_risk")
    result = result.reset_index(drop=True)
    result.index = np.arange(1, len(result) + 1)
    result.index.name = "slot"
    return result
