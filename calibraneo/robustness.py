"""Robustness and consensus metrics for candidate shortlists."""

from __future__ import annotations

import numpy as np
import pandas as pd


def shortlist_consensus(
    rankings: pd.DataFrame,
    *,
    candidate_column: str = "candidate_id",
    condition_column: str = "condition_id",
    rank_column: str = "rank",
    top_k: int = 20,
) -> pd.DataFrame:
    """Calculate appearance and rank stability across analytical conditions."""
    required = {candidate_column, condition_column, rank_column}
    missing = required.difference(rankings.columns)
    if missing:
        raise ValueError(f"Missing ranking columns: {', '.join(sorted(missing))}")
    subset = rankings[rankings[rank_column] <= top_k].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[candidate_column, "appearance_fraction", "mean_rank", "rank_std"]
        )
    total_conditions = rankings[condition_column].nunique()
    grouped = subset.groupby(candidate_column)[rank_column].agg(
        appearance_count="count",
        mean_rank="mean",
        rank_std="std",
    )
    grouped["rank_std"] = grouped["rank_std"].fillna(0.0)
    grouped["appearance_fraction"] = grouped["appearance_count"] / total_conditions
    return grouped.reset_index()[
        [candidate_column, "appearance_fraction", "mean_rank", "rank_std"]
    ].sort_values(
        ["appearance_fraction", "mean_rank", candidate_column],
        ascending=[False, True, True],
    )


def rank_uncertainty_score(
    appearance_fraction: float,
    rank_std: float,
    *,
    max_rank_std: float = 10.0,
) -> float:
    """Return a bounded stability score; this is not biological probability."""
    if not 0 <= appearance_fraction <= 1:
        raise ValueError("appearance_fraction must be in [0, 1]")
    if rank_std < 0 or max_rank_std <= 0:
        raise ValueError("rank standard deviations must be non-negative")
    return float(appearance_fraction * np.exp(-rank_std / max_rank_std))
