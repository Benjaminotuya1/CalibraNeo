"""Auditable extensions for shared antigens, circRNA, cytokines, and hand-offs.

These utilities are computational triage tools. They do not predict validated
translation, tissue delivery, cytokine safety, or therapeutic response.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import re
import shutil
import subprocess
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SharedAntigen:
    antigen_id: str
    peptide: str
    hla_allele: str
    tumour_prevalence: float = 0.0
    clonality: float = 0.0
    normal_tissue_risk: float = 0.0
    evidence_level: str = "unknown"


def normalize_shared_antigens(
    records: pd.DataFrame,
    *,
    source: str,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize a permitted shared-antigen export without copying its payload.

    ``column_map`` maps canonical names to source columns. Required canonical
    fields are ``antigen_id``, ``peptide``, and ``hla_allele``. The source and
    evidence fields are retained for auditability.
    """
    column_map = dict(column_map or {})
    canonical = {
        name: column_map.get(name, name)
        for name in (
            "antigen_id",
            "peptide",
            "hla_allele",
            "tumour_prevalence",
            "clonality",
            "normal_tissue_risk",
            "evidence_level",
        )
    }
    required = {"antigen_id", "peptide", "hla_allele"}
    missing = {name for name in required if canonical[name] not in records.columns}
    if missing:
        raise ValueError(f"Missing shared-antigen source columns: {', '.join(sorted(missing))}")
    result = pd.DataFrame(
        {
            name: records[column]
            for name, column in canonical.items()
            if column in records.columns
        }
    )
    for name, default in (
        ("tumour_prevalence", 0.0),
        ("clonality", 0.0),
        ("normal_tissue_risk", 0.0),
        ("evidence_level", "unknown"),
    ):
        if name not in result:
            result[name] = default
    result["source"] = str(source)
    result["raw_sequence_redistributed"] = False
    return result[
        [
            "antigen_id",
            "peptide",
            "hla_allele",
            "tumour_prevalence",
            "clonality",
            "normal_tissue_risk",
            "evidence_level",
            "source",
            "raw_sequence_redistributed",
        ]
    ]


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def population_hla_coverage(
    candidates: pd.DataFrame,
    allele_frequencies: Mapping[str, float],
    *,
    allele_column: str = "hla_allele",
    score_column: str = "presentation_probability",
    antigen_column: str = "antigen_id",
) -> pd.DataFrame:
    """Estimate allele-frequency-weighted shared-antigen coverage.

    The result is a prioritisation estimate under supplied allele frequencies,
    not a population prevalence or clinical response probability.
    """
    required = {allele_column, score_column, antigen_column}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Missing shared-antigen columns: {', '.join(sorted(missing))}")
    if not allele_frequencies:
        raise ValueError("allele_frequencies must not be empty")
    frequencies = {
        str(allele): _validate_probability(value, f"frequency[{allele}]")
        for allele, value in allele_frequencies.items()
    }
    frame = candidates.copy()
    frame["_allele_frequency"] = frame[allele_column].astype(str).map(frequencies).fillna(0.0)
    frame["_presentation_probability"] = pd.to_numeric(
        frame[score_column], errors="coerce"
    ).clip(0.0, 1.0).fillna(0.0)
    frame["_coverage_contribution"] = (
        frame["_allele_frequency"] * frame["_presentation_probability"]
    )
    result = (
        frame.groupby(antigen_column, as_index=False)
        .agg(
            weighted_coverage=("_coverage_contribution", "sum"),
            supported_alleles=(allele_column, "nunique"),
            mean_presentation=("_presentation_probability", "mean"),
        )
        .sort_values(
            ["weighted_coverage", "supported_alleles"],
            ascending=[False, False],
            kind="mergesort",
        )
    )
    result["coverage_interpretation"] = "allele-frequency-weighted computational estimate"
    return result.reset_index(drop=True)


def score_shared_antigens(
    candidates: pd.DataFrame,
    *,
    prevalence_column: str = "tumour_prevalence",
    clonality_column: str = "clonality",
    normal_risk_column: str = "normal_tissue_risk",
    presentation_column: str = "presentation_probability",
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Rank shared antigens with explicit, inspectable evidence weights."""
    required = {
        prevalence_column,
        clonality_column,
        normal_risk_column,
        presentation_column,
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Missing shared-antigen score columns: {', '.join(sorted(missing))}")
    weights = dict(
        weights
        or {
            "prevalence": 0.25,
            "clonality": 0.30,
            "presentation": 0.35,
            "normal_tissue_safety": 0.10,
        }
    )
    if abs(sum(weights.values()) - 1.0) > 1e-8:
        raise ValueError("shared-antigen weights must sum to 1")
    frame = candidates.copy()
    values = {}
    for name, column in (
        ("prevalence", prevalence_column),
        ("clonality", clonality_column),
        ("presentation", presentation_column),
    ):
        values[name] = pd.to_numeric(frame[column], errors="coerce").clip(0.0, 1.0).fillna(0.0)
    values["normal_tissue_safety"] = 1.0 - pd.to_numeric(
        frame[normal_risk_column], errors="coerce"
    ).clip(0.0, 1.0).fillna(1.0)
    frame["shared_antigen_priority"] = sum(
        weights[name] * values[name] for name in weights
    )
    frame["shared_antigen_score_status"] = "computational triage; not safety validation"
    return frame.sort_values(
        "shared_antigen_priority", ascending=False, kind="mergesort"
    ).reset_index(drop=True)


def _longest_homopolymer(sequence: str) -> int:
    return max((len(run) for run in re.findall(r"(A+|C+|G+|U+)", sequence)), default=0)


def circ_rna_constraints(
    rna: str,
    *,
    bsj_index: int | None = None,
    orf_start_index: int = 0,
    ires_sequence: str | None = None,
    ires_max_cassette_nt: int | None = None,
    target_gc: float = 0.55,
    rnafold_executable: str = "RNAfold",
) -> dict[str, float | int | bool | str]:
    """Calculate auditable circRNA constraints and sequence proxies.

    If ViennaRNA's ``RNAfold`` is installed, its minimum free energy (MFE) is
    reported. The self-complementarity value remains a fallback/secondary
    descriptor and must not be described as secondary-structure prediction.
    Rolling translation is defined as a frame-completeness and in-frame-stop
    proxy over one traversal of the circle; it is not a translation assay.
    """
    sequence = re.sub(r"\s+", "", rna.upper().replace("T", "U"))
    if not sequence or set(sequence) - set("ACGU"):
        raise ValueError("rna must contain only A, C, G, U, or T")
    if len(sequence) < 30:
        raise ValueError("rna must contain at least 30 nucleotides")
    if bsj_index is None:
        bsj_index = len(sequence) // 2
    if not 0 <= bsj_index < len(sequence):
        raise ValueError("bsj_index must be within the RNA sequence")
    if not 0 <= orf_start_index < len(sequence):
        raise ValueError("orf_start_index must be within the RNA sequence")
    bsj_in_frame = (bsj_index - orf_start_index) % 3 == 0
    junction = sequence[max(0, bsj_index - 30):bsj_index] + sequence[
        bsj_index:min(len(sequence), bsj_index + 30)
    ]
    gc_fraction = (sequence.count("G") + sequence.count("C")) / len(sequence)
    junction_gc = (junction.count("G") + junction.count("C")) / len(junction)
    reverse_complement = str.maketrans("ACGU", "UGCA")
    rc = sequence.translate(reverse_complement)[::-1]
    self_complement = max(
        (
            sum(left == right for left, right in zip(sequence[i:i + 12], rc[j:j + 12]))
            for i in range(max(1, len(sequence) - 11))
            for j in range(max(1, len(sequence) - 11))
        ),
        default=0,
    )
    ires = (ires_sequence or "").upper().replace("T", "U")
    if ires and set(ires) - set("ACGU"):
        raise ValueError("ires_sequence must contain only A, C, G, U, or T")
    ires_overlap = bool(ires) and ires in sequence
    overlap_fraction = (
        sequence.count(ires) * len(ires) / len(sequence) if ires else 0.0
    )
    ires_compatibility = max(0.0, 1.0 - min(1.0, overlap_fraction))
    if ires_max_cassette_nt is not None and ires_max_cassette_nt < 1:
        raise ValueError("ires_max_cassette_nt must be positive")
    ires_length_compatible = (
        ires_max_cassette_nt is None or len(sequence) <= ires_max_cassette_nt
    )
    gc_proxy = max(0.0, 1.0 - abs(gc_fraction - target_gc) / 0.45)
    structure_proxy = max(0.0, 1.0 - self_complement / 12.0)
    bsj_proxy = max(0.0, 1.0 - abs(junction_gc - target_gc) / 0.45)
    codons = [
        sequence[(orf_start_index + offset):(orf_start_index + offset + 3)]
        for offset in range(0, len(sequence) - 2, 3)
    ]
    in_frame_stop_count = sum(codon in {"UAA", "UAG", "UGA"} for codon in codons)
    frame_complete = (len(sequence) - orf_start_index) % 3 == 0
    rolling_translation_proxy = (
        0.45 * float(bsj_in_frame)
        + 0.25 * float(frame_complete)
        + 0.20 * max(0.0, 1.0 - in_frame_stop_count / max(len(codons), 1))
        + 0.10 * gc_proxy
    )
    mfe, structure_method = _rnafold_mfe(sequence, rnafold_executable)
    return {
        "length_nt": len(sequence),
        "gc_fraction": gc_fraction,
        "bsj_index": bsj_index,
        "orf_start_index": orf_start_index,
        "bsj_in_frame": bsj_in_frame,
        "bsj_window_gc_fraction": junction_gc,
        "rnafold_mfe_kcal_mol": mfe,
        "structure_method": structure_method,
        "bsj_structure_proxy": bsj_proxy,
        "self_complementarity_12nt_proxy": self_complement,
        "internal_structure_proxy": structure_proxy,
        "homopolymer_max": _longest_homopolymer(sequence),
        "ires_sequence_present": bool(ires),
        "ires_overlap_detected": ires_overlap,
        "ires_overlap_fraction": overlap_fraction,
        "ires_compatibility": ires_compatibility,
        "ires_length_compatible": ires_length_compatible,
        "rolling_frame_complete": frame_complete,
        "rolling_in_frame_stop_count": in_frame_stop_count,
        "rolling_translation_proxy": rolling_translation_proxy,
        "validated_translation_model": False,
        "interpretation": "sequence constraints/proxies; requires circularisation and translation assays",
    }


def _rnafold_mfe(sequence: str, executable: str) -> tuple[float | None, str]:
    """Return RNAfold MFE when available, otherwise an explicit fallback status."""
    if shutil.which(executable) is None:
        return None, "self_complementarity_fallback_rnafold_unavailable"
    try:
        completed = subprocess.run(
            [executable, "--noPS"],
            input=sequence + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None, "self_complementarity_fallback_rnafold_failed"
    match = re.search(r"\(([-+]?\d+(?:\.\d+)?)\)", completed.stdout)
    if not match:
        return None, "self_complementarity_fallback_rnafold_unparseable"
    return float(match.group(1)), "ViennaRNA_RNAfold_MFE"


def circ_rna_candidate_score(
    features: pd.DataFrame,
    *,
    bsj_column: str = "bsj_structure_proxy",
    ires_column: str = "ires_compatibility",
    rolling_column: str = "rolling_translation_proxy",
    length_column: str = "length_nt",
    target_length_nt: int | None = None,
    require_bsj_in_frame: bool = True,
    require_ires_length_compatible: bool = True,
) -> pd.DataFrame:
    """Rank circRNA candidates with transparent proxy weights."""
    required = {bsj_column, ires_column, rolling_column, length_column}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Missing circRNA feature columns: {', '.join(sorted(missing))}")
    frame = features.copy()
    if require_bsj_in_frame and "bsj_in_frame" in frame:
        frame = frame.loc[frame["bsj_in_frame"].astype(bool)].copy()
    if require_ires_length_compatible and "ires_length_compatible" in frame:
        frame = frame.loc[frame["ires_length_compatible"].astype(bool)].copy()
    length = pd.to_numeric(frame[length_column], errors="coerce").fillna(0.0)
    length_score = (
        pd.Series(1.0, index=frame.index)
        if target_length_nt is None
        else (1.0 - (length - target_length_nt).abs() / max(target_length_nt, 1))
    )
    frame["circRNA_constraint_score"] = (
        0.40 * pd.to_numeric(frame[bsj_column], errors="coerce").clip(0.0, 1.0).fillna(0.0)
        + 0.30 * pd.to_numeric(frame[ires_column], errors="coerce").clip(0.0, 1.0).fillna(0.0)
        + 0.20 * pd.to_numeric(frame[rolling_column], errors="coerce").clip(0.0, 1.0).fillna(0.0)
        + 0.10 * length_score.clip(0.0, 1.0)
    )
    frame["circRNA_score_status"] = "heuristic; not validated translation"
    return frame.sort_values(
        "circRNA_constraint_score", ascending=False, kind="mergesort"
    ).reset_index(drop=True)


def joint_weight_sensitivity(
    components: pd.DataFrame,
    *,
    id_column: str,
    base_weights: Mapping[str, float] | None = None,
    top_k: int = 10,
    perturbation: float = 0.5,
) -> pd.DataFrame:
    """Measure top-k Jaccard stability under one-at-a-time weight changes."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if id_column not in components.columns:
        raise ValueError(f"Missing identifier column: {id_column}")
    base = dict(base_weights or {
        "antigen": 0.30, "coverage": 0.20, "loh_diversity": 0.15,
        "circRNA": 0.15, "ires": 0.10, "lnp": 0.10,
    })
    if not 0 <= perturbation < 1:
        raise ValueError("perturbation must be in [0, 1)")
    baseline = joint_platform_score(
        components.assign(uncertainty=components.get("uncertainty", 0.0)),
        weights=base,
    )
    baseline_ids = set(baseline.head(top_k)[id_column])
    rows = []
    for name in base:
        for direction in ("down", "up"):
            altered = dict(base)
            altered[name] *= 1.0 - perturbation if direction == "down" else 1.0 + perturbation
            total = sum(altered.values())
            altered = {key: value / total for key, value in altered.items()}
            ranked = joint_platform_score(
                components.assign(uncertainty=components.get("uncertainty", 0.0)),
                weights=altered,
            )
            ids = set(ranked.head(top_k)[id_column])
            union = baseline_ids | ids
            rows.append(
                {
                    "component": name,
                    "direction": direction,
                    "top_k": top_k,
                    "baseline_top_k_count": len(baseline_ids),
                    "perturbed_top_k_count": len(ids),
                    "top_k_jaccard": len(baseline_ids & ids) / len(union) if union else 1.0,
                    "weights": altered,
                }
            )
    return pd.DataFrame(rows)


CYTOKINE_RULES = {
    "IL12": {"secreted": True, "risk": 0.85},
    "IL18": {"secreted": True, "risk": 0.75},
    "IL10": {"secreted": True, "risk": 0.65},
    "IL15": {"secreted": True, "risk": 0.70},
    "GMCSF": {"secreted": True, "risk": 0.55},
    "IFNG": {"secreted": True, "risk": 0.80},
}


def rank_cytokine_triplets(
    cytokines: Iterable[str],
    *,
    max_risk: float = 2.2,
) -> pd.DataFrame:
    """Rank cytokine triplets using literature-curated directional rules.

    The output is a hypothesis-ranking aid, not a toxicity or efficacy model.
    """
    names = [str(name).upper().replace("-", "") for name in cytokines]
    unknown = sorted(set(names).difference(CYTOKINE_RULES))
    if unknown:
        raise ValueError(f"Unknown cytokines require explicit curation: {', '.join(unknown)}")
    rows = []
    for triplet in combinations(sorted(set(names)), 3):
        risk = sum(float(CYTOKINE_RULES[name]["risk"]) for name in triplet)
        synergy = 0.0
        antagonism = 0.0
        if {"IL12", "IL18"}.issubset(triplet):
            synergy += 1.0
        if {"IL10", "IL12"}.issubset(triplet):
            antagonism += 1.0
        if {"IL10", "IFNG"}.issubset(triplet):
            antagonism += 0.5
        score = synergy - antagonism - max(0.0, risk - max_risk)
        rows.append(
            {
                "cytokine_triplet": "+".join(triplet),
                "directional_synergy_score": score,
                "risk_proxy": risk,
                "synergy_flags": "IL12+IL18" if {"IL12", "IL18"}.issubset(triplet) else "",
                "antagonism_flags": "+".join(
                    flag for flag, condition in (
                        ("IL10+IL12", {"IL10", "IL12"}.issubset(triplet)),
                        ("IL10+IFNG", {"IL10", "IFNG"}.issubset(triplet)),
                    ) if condition
                ),
                "status": "literature-curated hypothesis; not a safety prediction",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["directional_synergy_score", "risk_proxy"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_platform_interface(
    *,
    cassette_length_nt: int,
    gc_fraction: float,
    antigen_mode: str,
    ires_id: str | None = None,
    lnp_model_id: str | None = None,
) -> dict[str, object]:
    """Create a versioned hand-off contract for downstream IRES/LNP models."""
    if antigen_mode not in {"personalised", "shared", "dual"}:
        raise ValueError("antigen_mode must be personalised, shared, or dual")
    if cassette_length_nt < 1:
        raise ValueError("cassette_length_nt must be positive")
    _validate_probability(gc_fraction, "gc_fraction")
    return {
        "schema_version": "calibraneo-platform-interface-1.0",
        "antigen_mode": antigen_mode,
        "cassette": {
            "length_nt": int(cassette_length_nt),
            "gc_fraction": float(gc_fraction),
        },
        "ires": {
            "model_id": ires_id,
            "required_inputs": ["cassette_length_nt", "gc_fraction", "bsj_sequence"],
            "status": "contract_only" if ires_id is None else "model_reference",
        },
        "lnp_qsar": {
            "model_id": lnp_model_id,
            "required_inputs": [
                "cargo_length_nt",
                "cargo_gc_fraction",
                "sort_component",
                "molar_ratios",
            ],
            "status": "contract_only" if lnp_model_id is None else "model_reference",
        },
        "joint_objective_status": "not optimized without validated downstream models",
    }


def joint_platform_score(
    components: pd.DataFrame,
    *,
    weights: Mapping[str, float] | None = None,
    uncertainty_column: str = "uncertainty",
) -> pd.DataFrame:
    """Combine validated/available components without hiding missing evidence.

    Every component must be a [0, 1] score. Uncertainty is a [0, 1] penalty;
    it is reported separately so users can inspect sensitivity to the penalty.
    """
    default_weights = {
        "antigen": 0.30,
        "coverage": 0.20,
        "loh_diversity": 0.15,
        "circRNA": 0.15,
        "ires": 0.10,
        "lnp": 0.10,
    }
    weights = dict(weights or default_weights)
    if not weights or abs(sum(weights.values()) - 1.0) > 1e-8:
        raise ValueError("joint objective weights must be non-empty and sum to 1")
    missing = set(weights).difference(components.columns)
    if missing:
        raise ValueError(f"Missing joint-objective components: {', '.join(sorted(missing))}")
    if uncertainty_column not in components.columns:
        raise ValueError(f"Missing uncertainty column: {uncertainty_column}")
    frame = components.copy()
    score = 0.0
    for name, weight in weights.items():
        score = score + float(weight) * pd.to_numeric(
            frame[name], errors="coerce"
        ).clip(0.0, 1.0).fillna(0.0)
    uncertainty = pd.to_numeric(
        frame[uncertainty_column], errors="coerce"
    ).clip(0.0, 1.0).fillna(1.0)
    frame["joint_platform_score"] = score
    frame["joint_platform_score_uncertainty_adjusted"] = score * (1.0 - uncertainty)
    frame["joint_objective_status"] = (
        "configuration-level objective; downstream biological validation required"
    )
    return frame.sort_values(
        "joint_platform_score_uncertainty_adjusted",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
