"""Computational validation and experiment-priority utilities.

This module combines model outputs only as an evidence-aware decision layer.
It does not train a synthetic immunogenicity model from missing labels and does
not convert absence of a database match into negative biological evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SCORE_COLUMNS = (
    "presentation_score",
    "immunogenicity_score",
    "rna_stability_score",
    "translation_support",
)


def _bounded(value: pd.Series) -> pd.Series:
    return pd.to_numeric(value, errors="coerce").clip(0.0, 1.0)


def add_validation_scores(
    candidates: pd.DataFrame,
    *,
    presentation_column: str = "CalibraNeo_Prob",
) -> pd.DataFrame:
    """Add transparent ranking, uncertainty, and evidence-completeness columns.

    The composite is a weighted mean over observed component scores. Therefore,
    a missing downstream assay does not silently become a zero score.
    """
    if presentation_column not in candidates:
        raise ValueError(f"Missing presentation column: {presentation_column}")
    result = candidates.copy()
    result["presentation_score"] = _bounded(result[presentation_column])

    if "immunogenicity_model_score" in result:
        result["immunogenicity_score"] = _bounded(
            result["immunogenicity_model_score"]
        )
    elif "iedb_tcell_support" in result:
        result["immunogenicity_score"] = result["iedb_tcell_support"].map(
            lambda value: float(value) if pd.notna(value) else np.nan
        )
    else:
        result["immunogenicity_score"] = np.nan

    for column in ("rna_stability_score", "translation_support"):
        if column not in result:
            result[column] = np.nan
        result[column] = _bounded(result[column])

    weights = {
        "presentation_score": 0.55,
        "immunogenicity_score": 0.25,
        "rna_stability_score": 0.10,
        "translation_support": 0.10,
    }
    observed_weight = pd.Series(0.0, index=result.index)
    weighted_total = pd.Series(0.0, index=result.index)
    for column, weight in weights.items():
        observed = result[column].notna()
        observed_weight = observed_weight + observed.astype(float) * weight
        weighted_total = weighted_total + result[column].fillna(0.0) * weight
    result["evidence_completeness"] = observed_weight / sum(weights.values())
    result["computational_priority_score"] = (
        weighted_total / observed_weight.replace(0.0, np.nan)
    ).fillna(0.0)

    if "rank_std" in result:
        result["ranking_uncertainty"] = _bounded(
            1.0 - pd.to_numeric(result["rank_std"], errors="coerce").fillna(10.0) / 10.0
        )
    else:
        result["ranking_uncertainty"] = np.nan
    result["validation_confidence"] = (
        result["computational_priority_score"]
        * result["evidence_completeness"]
        * result["ranking_uncertainty"].fillna(1.0)
    ).clip(0.0, 1.0)
    result["decision_status"] = np.select(
        [
            result["evidence_completeness"] < 0.50,
            result["validation_confidence"] < 0.50,
        ],
        ["needs_experimental_evidence", "computational_review"],
        default="prioritize_for_validation",
    )
    return result.sort_values(
        ["computational_priority_score", "validation_confidence"],
        ascending=False,
        kind="mergesort",
    )


def recommend_experiment(row: pd.Series) -> dict[str, Any]:
    """Return the highest-value missing validation step for one candidate."""
    checks = [
        (
            "tumor_expression_assay",
            "Confirm mutant allele and transcript expression from matched tumor DNA/RNA.",
            row.get("tumor_expression_support"),
        ),
        (
            "immunopeptidomics_assay",
            "Test peptide-HLA display by targeted or discovery immunopeptidomics.",
            row.get("immunopeptidomics_support"),
        ),
        (
            "t_cell_assay",
            "Test antigen-specific T-cell recognition using an orthogonal functional assay.",
            row.get("iedb_tcell_support", row.get("t_cell_support")),
        ),
        (
            "translation_assay",
            "Measure antigen production from the proposed ORF in the intended cell context.",
            row.get("translation_support"),
        ),
        (
            "rna_stability_assay",
            "Measure RNA persistence and integrity for the proposed construct.",
            row.get("rna_stability_score"),
        ),
    ]
    for assay, reason, value in checks:
        if pd.isna(value):
            return {
                "recommended_assay": assay,
                "reason": reason,
                "priority": float(row.get("validation_confidence", 0.0)),
            }
    return {
        "recommended_assay": "replication_or_independent_validation",
        "reason": "Repeat the strongest available result in an independent model or cohort.",
        "priority": float(row.get("validation_confidence", 0.0)),
    }


def build_experiment_plan(candidates: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Create an auditable experiment queue ordered by computational priority."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    scored = add_validation_scores(candidates).head(top_n).copy()
    recommendations = scored.apply(
        recommend_experiment, axis=1, result_type="expand"
    ).reset_index(drop=True)
    return pd.concat([scored.reset_index(drop=True), recommendations], axis=1)


def validation_report(
    *,
    model_version: str,
    candidate_count: int,
    external_evidence_rows: int,
    exact_tcell_matches: int,
    benchmark_metrics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable validation status report for the app/CLI."""
    metrics: dict[str, Any] = {}
    if benchmark_metrics is not None and not benchmark_metrics.empty:
        for column in benchmark_metrics.columns:
            values = benchmark_metrics[column].dropna()
            if len(values) and pd.api.types.is_numeric_dtype(values):
                metrics[column] = {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                }
    return {
        "model_version": model_version,
        "candidate_count": int(candidate_count),
        "external_evidence_rows": int(external_evidence_rows),
        "exact_tcell_matches": int(exact_tcell_matches),
        "external_match_fraction": (
            float(exact_tcell_matches / candidate_count) if candidate_count else 0.0
        ),
        "benchmark_numeric_summary": metrics,
        "validated_claims": [
            "model inference uses the versioned serialized bundle",
            "calibration and untouched-test metrics are reported separately",
            "external evidence is provenance-tracked and endpoint-specific",
            "missing evidence remains unknown",
        ],
        "not_proven": [
            "tumor-specific mutation expression",
            "translation or RNA stability of a proposed construct",
            "peptide processing and in-tumor presentation",
            "T-cell activation in the intended patient",
            "safety or therapeutic benefit",
        ],
    }
