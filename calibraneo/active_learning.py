"""Active-learning experiment selection and transparent optimizer comparisons."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def experiment_information_gain(
    candidates: pd.DataFrame,
    *,
    score_column: str = "computational_priority_score",
    cost_column: str | None = None,
) -> pd.DataFrame:
    """Rank assays by uncertainty times evidence gap, divided by cost."""
    result = candidates.copy()
    score = pd.to_numeric(result.get(score_column, 0.5), errors="coerce").fillna(0.5)
    uncertainty = 1.0 - (score - 0.5).abs() * 2.0
    completeness = pd.to_numeric(
        result.get("evidence_completeness", 0.0), errors="coerce"
    ).fillna(0.0)
    cost = (
        pd.to_numeric(result[cost_column], errors="coerce").fillna(1.0)
        if cost_column and cost_column in result
        else pd.Series(1.0, index=result.index)
    )
    result["expected_information_gain"] = (
        uncertainty.clip(0.0, 1.0) * (1.0 - completeness) / cost.clip(lower=0.01)
    )
    return result.sort_values(
        "expected_information_gain", ascending=False, kind="mergesort"
    )


def compare_selection_methods(
    candidates: pd.DataFrame,
    capacity: int,
    *,
    score_column: str = "CalibraNeo_Prob",
) -> pd.DataFrame:
    """Compare greedy, exact small-pool optimum, and a diversity-aware baseline."""
    if capacity < 1 or capacity > len(candidates):
        raise ValueError("capacity must be within candidate count")
    frame = candidates.reset_index(drop=True)
    scores = pd.to_numeric(frame[score_column], errors="coerce").fillna(0.0).to_numpy()
    greedy = list(np.argsort(-scores, kind="stable")[:capacity])
    rows = [{"method": "score_greedy", "objective": float(scores[greedy].sum()), "selected_indices": greedy}]
    if len(frame) <= 30:
        best = max(combinations(range(len(frame)), capacity), key=lambda x: float(scores[list(x)].sum()))
        rows.append({"method": "exact_small_pool", "objective": float(scores[list(best)].sum()), "selected_indices": list(best)})
    if "hla_clean" in frame:
        selected: list[int] = []
        counts: dict[str, int] = {}
        for _ in range(capacity):
            available = [i for i in range(len(frame)) if i not in selected]
            pick = max(
                available,
                key=lambda i: (
                    scores[i] / (1.0 + counts.get(str(frame.loc[i, "hla_clean"]), 0)),
                    -i,
                ),
            )
            selected.append(pick)
            allele = str(frame.loc[pick, "hla_clean"])
            counts[allele] = counts.get(allele, 0) + 1
        rows.append({"method": "hla_diversity_greedy", "objective": float(scores[selected].sum()), "selected_indices": selected})
    return pd.DataFrame(rows)
