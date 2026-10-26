"""Transparent LOH-aware cassette selection.

This is deliberately named an optimizer rather than RL: it is a deterministic
greedy approximation with an explicit objective and reproducible tie-breaking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations


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


def exact_small_pool_hierarchical_selection(
    candidates: pd.DataFrame,
    capacity: int,
    *,
    threshold: int = 3,
    peptide_probability_column: str = "presentation_probability",
    loh_probability_column: str = "loh_probability",
    max_candidates: int = 30,
) -> tuple[pd.DataFrame, float]:
    """Exhaustively optimize P(S >= threshold) for a bounded candidate pool.

    This exact reference implementation is intentionally limited to small
    pools. It is a correctness baseline for the future scalable DP/MILP
    implementation, not a large-pool optimizer.
    """
    required = {
        "hla_clean",
        peptide_probability_column,
        loh_probability_column,
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Missing exact-selection columns: {', '.join(sorted(missing))}")
    if len(candidates) > max_candidates:
        raise ValueError(f"Exact selection is limited to {max_candidates} candidates")
    if capacity < 1 or capacity > len(candidates):
        raise ValueError("capacity must be within candidate count")
    if threshold < 1:
        raise ValueError("threshold must be positive")

    frame = candidates.reset_index(drop=True).copy()

    def objective(indices: tuple[int, ...]) -> float:
        grouped: dict[str, list[float]] = {}
        loh: dict[str, float] = {}
        for index in indices:
            row = frame.iloc[index]
            allele = str(row["hla_clean"])
            grouped.setdefault(allele, []).append(float(row[peptide_probability_column]))
            loh[allele] = float(row[loh_probability_column])
        polynomial = np.array([1.0])
        for allele, probabilities in grouped.items():
            conditional = np.array([1.0])
            for probability in probabilities:
                next_values = np.zeros(len(conditional) + 1)
                next_values[:-1] += conditional * (1.0 - probability)
                next_values[1:] += conditional * probability
                conditional = next_values
            group_polynomial = np.zeros(len(conditional))
            group_polynomial[0] = loh[allele]
            group_polynomial += (1.0 - loh[allele]) * conditional
            combined = np.zeros(len(polynomial) + len(group_polynomial) - 1)
            for left, left_value in enumerate(polynomial):
                combined[left:left + len(group_polynomial)] += (
                    left_value * group_polynomial
                )
            polynomial = combined
        return float(1.0 - polynomial[:threshold].sum())

    best_indices: tuple[int, ...] | None = None
    best_value = -1.0
    for indices in combinations(range(len(frame)), capacity):
        value = objective(indices)
        if value > best_value:
            best_indices, best_value = indices, value
    if best_indices is None:
        raise RuntimeError("No exact candidate subset was evaluated")
    return frame.iloc[list(best_indices)].reset_index(drop=True), best_value
