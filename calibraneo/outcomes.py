"""Endpoint-specific model registry and conservative prediction combination.

Different biological outcomes are not interchangeable labels. This module
provides a typed status registry and combines only already-generated,
endpoint-specific predictions; it does not manufacture missing outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EndpointStatus:
    name: str
    target: str
    trainable: bool
    internally_validated: bool
    externally_validated: bool
    validation_scope: str
    evidence_boundary: str


ENDPOINT_STATUSES = (
    EndpointStatus(
        "presentation",
        "peptide-HLA presentation-related label",
        True, True, True,
        "grouped internal test; Zenodo record-level transfer; Nature direct failure",
        "does not prove tumour presentation or immunogenicity",
    ),
    EndpointStatus(
        "immunogenicity",
        "T-cell response",
        True, True, False,
        "Ott/Zaghla descriptive endpoint audits; weak cross-study LOSO",
        "does not prove patient-specific T-cell activation or tumour killing",
    ),
    EndpointStatus(
        "rna_context",
        "gene-level RNA half-life context",
        True, True, False,
        "grouped RNADecayCafe holdout; R2 approximately 0.0544",
        "does not predict custom circRNA half-life",
    ),
    EndpointStatus(
        "circRNA_translation",
        "circRNA translation evidence or measured yield",
        False, False, False,
        "sequence proxies and benchmark runner only",
        "requires construct-level assays and permitted labeled data",
    ),
    EndpointStatus(
        "lnp_delivery",
        "formulation-to-organ delivery outcome",
        False, False, False,
        "interface contract only; no local compatible labeled table",
        "spleen delivery does not imply tumour-cell delivery",
    ),
    EndpointStatus(
        "tumour_killing",
        "tumour-cell killing assay outcome",
        False, False, False,
        "no harmonized local endpoint dataset",
        "requires matched HLA, effector, tumour, and assay context",
    ),
    EndpointStatus(
        "cytokine_safety",
        "dose/route/species-specific toxicity outcome",
        False, False, False,
        "literature-curated interaction rules only",
        "not a generic safety predictor",
    ),
)


def endpoint_validation_status() -> pd.DataFrame:
    """Return the auditable status of every proposed outcome endpoint."""
    return pd.DataFrame([asdict(status) for status in ENDPOINT_STATUSES])


def combine_endpoint_predictions(
    predictions: pd.DataFrame,
    *,
    id_column: str,
    endpoint_columns: Iterable[str],
    weights: Mapping[str, float] | None = None,
    uncertainty_column: str | None = None,
) -> pd.DataFrame:
    """Combine compatible endpoint scores with explicit missingness.

    This is a decision-layer ensemble, not a new biological endpoint model.
    A missing component is excluded from the observed weighted mean and lowers
    completeness. It is never converted to a negative biological label.
    """
    endpoint_columns = list(endpoint_columns)
    if not endpoint_columns:
        raise ValueError("At least one endpoint column is required")
    if id_column not in predictions:
        raise ValueError(f"Missing identifier column: {id_column}")
    missing = set(endpoint_columns).difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing endpoint columns: {', '.join(sorted(missing))}")
    if predictions[id_column].duplicated().any():
        raise ValueError("Prediction identifiers must be unique before combination")
    weights = dict(weights or {name: 1.0 for name in endpoint_columns})
    if set(weights) != set(endpoint_columns):
        raise ValueError("weights must contain exactly the endpoint columns")
    if any(float(value) <= 0 for value in weights.values()):
        raise ValueError("endpoint weights must be positive")
    result = predictions.copy()
    observed_weight = pd.Series(0.0, index=result.index)
    total = pd.Series(0.0, index=result.index)
    for name in endpoint_columns:
        values = pd.to_numeric(result[name], errors="coerce").clip(0.0, 1.0)
        observed = values.notna()
        weight = float(weights[name])
        observed_weight = observed_weight + observed.astype(float) * weight
        total = total + values.fillna(0.0) * weight
    result["combined_score"] = (total / observed_weight.replace(0.0, np.nan)).fillna(0.0)
    result["endpoint_completeness"] = observed_weight / sum(weights.values())
    if uncertainty_column is None:
        result["combined_uncertainty"] = 1.0 - result["endpoint_completeness"]
    else:
        if uncertainty_column not in result:
            raise ValueError(f"Missing uncertainty column: {uncertainty_column}")
        result["combined_uncertainty"] = pd.to_numeric(
            result[uncertainty_column], errors="coerce"
        ).clip(0.0, 1.0).fillna(1.0)
    result["uncertainty_adjusted_score"] = (
        result["combined_score"] * (1.0 - result["combined_uncertainty"])
    )
    result["combination_status"] = np.select(
        [
            result["endpoint_completeness"].eq(0.0),
            result["endpoint_completeness"] < 1.0,
        ],
        ["no_observed_endpoint", "partial_endpoint_evidence"],
        default="all_requested_endpoints_observed",
    )
    return result.sort_values(
        "uncertainty_adjusted_score", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
