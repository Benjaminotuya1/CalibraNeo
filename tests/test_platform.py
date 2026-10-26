import numpy as np
import pandas as pd
import pytest

from calibraneo.platform import (
    build_platform_interface,
    circ_rna_candidate_score,
    circ_rna_constraints,
    joint_platform_score,
    joint_weight_sensitivity,
    normalize_shared_antigens,
    population_hla_coverage,
    rank_cytokine_triplets,
    score_shared_antigens,
)
from calibraneo.outcomes import endpoint_validation_status
from calibraneo.outcomes import combine_endpoint_predictions


def test_shared_antigen_coverage_and_scoring():
    candidates = pd.DataFrame(
        [
            {"antigen_id": "A", "peptide": "SIINFEKL", "hla_allele": "HLA-A*02:01", "presentation_probability": 0.8, "tumour_prevalence": 0.9, "clonality": 0.8, "normal_tissue_risk": 0.1},
            {"antigen_id": "A", "peptide": "SIINFEKL", "hla_allele": "HLA-A*03:01", "presentation_probability": 0.6, "tumour_prevalence": 0.9, "clonality": 0.8, "normal_tissue_risk": 0.1},
        ]
    )
    coverage = population_hla_coverage(
        candidates, {"HLA-A*02:01": 0.4, "HLA-A*03:01": 0.2}
    )
    assert coverage.loc[0, "weighted_coverage"] == pytest.approx(0.44)
    assert score_shared_antigens(candidates).loc[0, "antigen_id"] == "A"
    normalized = normalize_shared_antigens(candidates, source="test")
    assert normalized.loc[0, "source"] == "test"
    assert bool(normalized.loc[0, "raw_sequence_redistributed"]) is False


def test_circ_rna_constraints_are_explicit_proxies():
    result = circ_rna_constraints("AUG" + "GCA" * 20, bsj_index=30, ires_sequence="GCA")
    assert result["length_nt"] == 63
    assert result["validated_translation_model"] is False
    assert result["ires_overlap_detected"] is True


def test_cytokine_and_interface_contracts():
    ranked = rank_cytokine_triplets(["IL12", "IL18", "IL15", "IL10"])
    assert "IL12+IL15+IL18" in ranked["cytokine_triplet"].tolist()
    contract = build_platform_interface(
        cassette_length_nt=900, gc_fraction=0.55, antigen_mode="dual"
    )
    assert contract["schema_version"].endswith("1.0")
    assert contract["lnp_qsar"]["status"] == "contract_only"
    components = pd.DataFrame(
        [{"antigen": 1.0, "coverage": 0.8, "loh_diversity": 0.5,
          "circRNA": 0.7, "ires": 0.0, "lnp": 0.0, "uncertainty": 0.2}]
    )
    assert joint_platform_score(components).loc[0, "joint_platform_score"] > 0
    sensitivity = joint_weight_sensitivity(
        components.assign(candidate_id=["one"]),
        id_column="candidate_id",
        top_k=1,
    )
    assert len(sensitivity) == 12


@pytest.mark.parametrize("bad_index", [-1, 63, 100])
def test_circ_rna_rejects_malformed_bsj(bad_index):
    with pytest.raises(ValueError):
        circ_rna_constraints("AUG" + "GCA" * 20, bsj_index=bad_index)


def test_circ_rna_enforces_frame_and_length_filters():
    result = circ_rna_constraints(
        "AUG" + "GCA" * 20,
        bsj_index=31,
        ires_max_cassette_nt=30,
    )
    assert result["bsj_in_frame"] is False
    assert result["ires_length_compatible"] is False


def test_circ_rna_reports_explicit_rnafold_fallback():
    result = circ_rna_constraints(
        "AUG" + "GCA" * 20,
        rnafold_executable="definitely-not-installed-rnafold",
    )
    assert result["rnafold_mfe_kcal_mol"] is None
    assert "fallback" in result["structure_method"]


def test_circ_rna_rejects_invalid_sequence_and_ires():
    with pytest.raises(ValueError):
        circ_rna_constraints("AUG" + "GCA" * 20 + "N")
    with pytest.raises(ValueError):
        circ_rna_constraints("AUG" + "GCA" * 20, ires_sequence="NNN")


def test_empty_candidate_score_is_supported():
    empty = pd.DataFrame(
        columns=["bsj_structure_proxy", "ires_compatibility",
                 "rolling_translation_proxy", "length_nt"]
    )
    ranked = circ_rna_candidate_score(empty)
    assert ranked.empty


def test_circ_candidate_score_filters_out_of_frame():
    features = pd.DataFrame(
        [
            {"id": "bad", "bsj_structure_proxy": 1, "ires_compatibility": 1,
             "rolling_translation_proxy": 1, "length_nt": 90,
             "bsj_in_frame": False, "ires_length_compatible": True},
            {"id": "good", "bsj_structure_proxy": 0.5, "ires_compatibility": 1,
             "rolling_translation_proxy": 0.5, "length_nt": 90,
             "bsj_in_frame": True, "ires_length_compatible": True},
        ]
    )
    assert circ_rna_candidate_score(features).loc[0, "id"] == "good"


def test_population_coverage_unknown_alleles_are_zero():
    candidates = pd.DataFrame(
        [{"antigen_id": "A", "hla_allele": "HLA-UNKNOWN",
          "presentation_probability": 1.0}]
    )
    result = population_hla_coverage(candidates, {"HLA-A*02:01": 0.4})
    assert result.loc[0, "weighted_coverage"] == 0


def test_population_coverage_rejects_empty_frequency_map():
    candidates = pd.DataFrame(
        [{"antigen_id": "A", "hla_allele": "HLA-A*02:01",
          "presentation_probability": 1.0}]
    )
    with pytest.raises(ValueError):
        population_hla_coverage(candidates, {})


def test_normalization_rejects_missing_peptide():
    with pytest.raises(ValueError):
        normalize_shared_antigens(
            pd.DataFrame([{"antigen_id": "A", "hla_allele": "HLA-A*02:01"}]),
            source="test",
        )


def test_normalization_preserves_duplicate_peptides():
    records = pd.DataFrame(
        [
            {"antigen_id": "A", "peptide": "PEPTIDE", "hla_allele": "HLA-A*02:01"},
            {"antigen_id": "A", "peptide": "PEPTIDE", "hla_allele": "HLA-A*03:01"},
        ]
    )
    assert len(normalize_shared_antigens(records, source="test")) == 2


def test_cytokine_unknown_rule_is_explicit_error():
    with pytest.raises(ValueError):
        rank_cytokine_triplets(["IL12", "IL18", "UNKNOWN"])


def test_joint_score_missing_components_and_nan_are_safe():
    components = pd.DataFrame(
        [{"antigen": 1.0, "coverage": float("nan"), "loh_diversity": 0.5,
          "circRNA": 0.7, "ires": 0.0, "lnp": 0.0, "uncertainty": 1.0}]
    )
    result = joint_platform_score(components)
    assert result.loc[0, "joint_platform_score"] >= 0
    assert result.loc[0, "joint_platform_score_uncertainty_adjusted"] == 0
    with pytest.raises(ValueError):
        joint_platform_score(components.drop(columns=["lnp"]))


def test_interface_rejects_invalid_modes_and_lengths():
    with pytest.raises(ValueError):
        build_platform_interface(cassette_length_nt=0, gc_fraction=0.5, antigen_mode="dual")
    with pytest.raises(ValueError):
        build_platform_interface(cassette_length_nt=90, gc_fraction=0.5, antigen_mode="unknown")


def test_endpoint_registry_exposes_validation_boundaries():
    status = endpoint_validation_status()
    assert set(["presentation", "circRNA_translation", "lnp_delivery"]).issubset(
        set(status["name"])
    )
    assert bool(status.loc[status["name"].eq("presentation"), "externally_validated"].iloc[0])
    assert not bool(status.loc[status["name"].eq("lnp_delivery"), "trainable"].iloc[0])


def test_endpoint_combination_preserves_missingness_and_rejects_duplicates():
    predictions = pd.DataFrame(
        [
            {"id": "a", "presentation": 0.8, "immunogenicity": np.nan},
            {"id": "b", "presentation": 0.2, "immunogenicity": 0.6},
            {"id": "c", "presentation": np.nan, "immunogenicity": np.nan},
        ]
    )
    combined = combine_endpoint_predictions(
        predictions,
        id_column="id",
        endpoint_columns=["presentation", "immunogenicity"],
    )
    assert combined.loc[combined["id"].eq("a"), "combination_status"].iloc[0] == "partial_endpoint_evidence"
    assert combined.loc[combined["id"].eq("c"), "combination_status"].iloc[0] == "no_observed_endpoint"
    with pytest.raises(ValueError):
        combine_endpoint_predictions(
            pd.concat([predictions, predictions.iloc[[0]]]),
            id_column="id",
            endpoint_columns=["presentation", "immunogenicity"],
        )
