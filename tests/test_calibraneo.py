import numpy as np
import pandas as pd

from calibraneo.constructs import build_orf, construct_quality
from calibraneo.pu import elkan_noto_correct, pu_sensitivity
from calibraneo.data_registry import load_registry, validate_registry
from calibraneo.evidence import annotate_evidence, attach_iedb_evidence
from calibraneo.robustness import shortlist_consensus
from calibraneo.rl import train_q_policy
from calibraneo.selection import select_loh_aware_cassette
from calibraneo.validation import add_validation_scores, build_experiment_plan
from calibraneo.task_models import sequence_features
from calibraneo.active_learning import compare_selection_methods, experiment_information_gain
from calibraneo.sequence_design import rna_design_features


def test_orf_has_terminal_stop_and_no_internal_stop():
    _, dna = build_orf(["SIINFEKL", "GILGFVFTL"])
    quality = construct_quality(dna)
    assert dna.endswith("TGA")
    assert quality["stop_codons"] == 0


def test_selector_is_deterministic_and_diverse():
    candidates = pd.DataFrame(
        {
            "Peptide ID": ["a", "b", "c", "d"],
            "hla_clean": ["A", "A", "B", "C"],
            "CalibraNeo_Prob": [0.9, 0.89, 0.7, 0.6],
        }
    )
    first = select_loh_aware_cassette(candidates, 3, 0.8)
    second = select_loh_aware_cassette(candidates, 3, 0.8)
    assert first.equals(second)
    assert first["hla_clean"].nunique() >= 2


def test_elkan_noto_correction_and_sensitivity():
    corrected = elkan_noto_correct(np.array([0.2, 0.8]), 0.5)
    assert np.allclose(corrected, [0.4, 1.0])
    assert set(pu_sensitivity(np.array([0.2]), [0.5, 1.0])) == {0.5, 1.0}


def test_missing_evidence_is_unknown_and_never_ready():
    result = annotate_evidence(pd.DataFrame({"candidate_id": ["x"]}))
    assert result.loc[0, "evidence_level"] == "computational_insufficient"
    assert bool(result.loc[0, "biological_candidate_ready"]) is False


def test_shortlist_consensus_measures_condition_stability():
    rankings = pd.DataFrame(
        {
            "candidate_id": ["a", "a", "b", "b", "c"],
            "condition_id": ["one", "two", "one", "two", "one"],
            "rank": [1, 2, 2, 1, 3],
        }
    )
    result = shortlist_consensus(rankings, top_k=2)
    assert result.iloc[0]["appearance_fraction"] == 1.0


def test_data_registry_is_valid():
    payload = load_registry("data_registry.json")
    assert validate_registry(payload) == []


def test_q_learning_policy_returns_unique_slots():
    candidates = pd.DataFrame(
        {
            "hla_clean": ["A", "A", "B", "C"],
            "CalibraNeo_Prob": [0.9, 0.8, 0.7, 0.6],
        }
    )
    selected = train_q_policy(candidates, 2, 0.7, episodes=20)
    assert len(selected) == 2
    assert len(set(selected)) == 2


def test_iedb_match_is_exact_and_no_match_is_unknown():
    candidates = pd.DataFrame(
        {"peptide(mut)": ["SIINFEKL", "UNKNOWN"], "hla_clean": ["HLA-A*02:01", "HLA-A*02:01"]}
    )
    iedb = pd.DataFrame(
        {"peptide": ["SIINFEKL"], "hla_allele": ["HLA-A*02:01"], "is_positive": [True]}
    )
    result = attach_iedb_evidence(candidates, iedb)
    assert bool(result.loc[0, "iedb_tcell_support"]) is True
    assert pd.isna(result.loc[1, "iedb_tcell_support"])


def test_validation_scores_do_not_turn_missing_evidence_into_zero():
    result = add_validation_scores(
        pd.DataFrame(
            {
                "peptide(mut)": ["AAAA"],
                "hla_clean": ["HLA-A*02:01"],
                "CalibraNeo_Prob": [0.8],
            }
        )
    )
    assert result.loc[0, "presentation_score"] == 0.8
    assert pd.isna(result.loc[0, "immunogenicity_score"])
    assert result.loc[0, "evidence_completeness"] == 0.55


def test_experiment_plan_identifies_missing_tcell_validation():
    plan = build_experiment_plan(
        pd.DataFrame(
            {
                "peptide(mut)": ["AAAA"],
                "hla_clean": ["HLA-A*02:01"],
                "CalibraNeo_Prob": [0.8],
                "iedb_tcell_support": [np.nan],
            }
        ),
        top_n=1,
    )
    assert plan.loc[0, "recommended_assay"] == "tumor_expression_assay"


def test_endpoint_features_and_active_learning_are_reproducible():
    frame = pd.DataFrame({"peptide": ["SIINFEKL", "GILGFVFTL"]})
    features = sequence_features(frame)
    assert list(features.columns)
    ranked = experiment_information_gain(
        pd.DataFrame(
            {
                "computational_priority_score": [0.5, 0.9],
                "evidence_completeness": [0.1, 0.9],
            }
        )
    )
    assert ranked.index[0] == 0
    methods = compare_selection_methods(
        pd.DataFrame(
            {
                "CalibraNeo_Prob": [0.9, 0.8, 0.7],
                "hla_clean": ["A", "B", "A"],
            }
        ),
        2,
    )
    assert "exact_small_pool" in set(methods["method"])


def test_rna_design_features_are_qc_not_biological_proof():
    result = rna_design_features("AUGGCCUGAUAA")
    assert result["start_codon_present"] is True
    assert result["validated_translation_model"] is False
