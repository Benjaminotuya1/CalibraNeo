"""Evidence-aware candidate scoring and nomination gates.

Scores from different biological assays are intentionally kept separate. An
unknown measurement is not converted to zero evidence, and no candidate can
become biologically ready from computational scores alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


EVIDENCE_COLUMNS = [
    "presentation_score",
    "immunogenicity_score",
    "immunopeptidomics_support",
    "normal_tissue_support",
    "translation_support",
    "rna_stability_score",
    "delivery_support",
    "therapeutic_evidence",
]


@dataclass(frozen=True)
class EvidencePolicy:
    """Explicit policy for converting evidence into a status label."""

    min_presentation: float = 0.50
    min_immunogenicity: float = 0.50
    require_ms_support: bool = True
    require_t_cell_support: bool = True


def evidence_level(row: pd.Series, policy: EvidencePolicy | None = None) -> str:
    policy = policy or EvidencePolicy()
    presentation = row.get("presentation_score")
    immunogenicity = row.get("immunogenicity_score")
    ms = row.get("immunopeptidomics_support")
    t_cell = row.get("t_cell_support")

    if pd.isna(presentation) or presentation < policy.min_presentation:
        return "computational_insufficient"
    if pd.isna(immunogenicity) or immunogenicity < policy.min_immunogenicity:
        return "presentation_only"
    if policy.require_ms_support and ms is not True:
        return "putative_immunogenicity"
    if policy.require_t_cell_support and t_cell is not True:
        return "presentation_supported"
    return "biologically_supported"


def annotate_evidence(
    frame: pd.DataFrame,
    policy: EvidencePolicy | None = None,
) -> pd.DataFrame:
    """Add explicit evidence status without inventing missing measurements."""
    result = frame.copy()
    for column in EVIDENCE_COLUMNS:
        if column not in result:
            result[column] = np.nan
    result["evidence_level"] = result.apply(evidence_level, axis=1, policy=policy)
    result["biological_candidate_ready"] = False
    result["nomination_status"] = "NO_GO_without_independent_validation"
    return result


def build_evidence_table(
    candidates: pd.DataFrame,
    *,
    source: str,
    source_version: str,
) -> pd.DataFrame:
    """Create the normalized multi-track evidence table used by all adapters."""
    result = annotate_evidence(candidates)
    result["evidence_source"] = source
    result["evidence_source_version"] = source_version
    result["evidence_schema_version"] = "1.0"
    return result


def evidence_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize what is measured versus unknown in an evidence table."""
    measured = {
        column: int(frame[column].notna().sum())
        for column in EVIDENCE_COLUMNS
        if column in frame
    }
    return {
        "rows": int(len(frame)),
        "measured_counts": measured,
        "biologically_ready_rows": int(
            frame.get("biological_candidate_ready", pd.Series(dtype=bool)).sum()
        ),
        "claim_boundary": "No computational row is biologically candidate-ready.",
    }


def attach_iedb_evidence(
    candidates: pd.DataFrame,
    iedb: pd.DataFrame,
    *,
    peptide_column: str = "peptide(mut)",
    hla_column: str = "hla_clean",
) -> pd.DataFrame:
    """Attach exact-sequence/HLA IEDB support without treating no-match as negative."""
    required = {"peptide", "hla_allele", "is_positive"}
    missing = required.difference(iedb.columns)
    if missing:
        raise ValueError(f"Missing IEDB columns: {', '.join(sorted(missing))}")
    result = candidates.copy()
    lookup = iedb.copy()
    lookup["_peptide_key"] = lookup["peptide"].astype(str).str.upper().str.strip()
    lookup["_hla_key"] = lookup["hla_allele"].astype(str).str.upper().str.replace(" ", "", regex=False)
    lookup = lookup.groupby(["_peptide_key", "_hla_key"])["is_positive"].max()
    peptide_key = result[peptide_column].astype(str).str.upper().str.strip()
    hla_key = result[hla_column].astype(str).str.upper().str.replace(" ", "", regex=False)
    keys = pd.MultiIndex.from_arrays([peptide_key, hla_key])
    values = lookup.reindex(keys)
    result["iedb_tcell_support"] = values.to_numpy()
    result["iedb_tcell_support"] = result["iedb_tcell_support"].where(
        result["iedb_tcell_support"].notna(), np.nan
    )
    return result
