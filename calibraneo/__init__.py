"""Reusable CalibraNeo inference and design utilities."""

from .inference import CalibraNeoPredictor, load_predictor
from .selection import select_loh_aware_cassette
from .constructs import build_orf, calculate_cai, construct_quality
from .evidence import (
    EvidencePolicy,
    annotate_evidence,
    attach_iedb_evidence,
    build_evidence_table,
    evidence_summary,
)
from .robustness import rank_uncertainty_score, shortlist_consensus
from .rl import train_q_policy
from .validation import (
    add_validation_scores,
    build_experiment_plan,
    recommend_experiment,
    validation_report,
)
from .task_models import predict_endpoint, sequence_features, train_endpoint_model
from .patient import (
    assemble_patient_candidates,
    read_expression,
    read_hla,
    read_loh,
    read_vcf,
)
from .active_learning import compare_selection_methods, experiment_information_gain
from .sequence_design import rna_design_features
from .rna_model import annotate_gene_stability, train_rna_context_model

__all__ = [
    "CalibraNeoPredictor",
    "load_predictor",
    "select_loh_aware_cassette",
    "build_orf",
    "calculate_cai",
    "construct_quality",
    "EvidencePolicy",
    "annotate_evidence",
    "attach_iedb_evidence",
    "build_evidence_table",
    "evidence_summary",
    "rank_uncertainty_score",
    "shortlist_consensus",
    "train_q_policy",
    "add_validation_scores",
    "build_experiment_plan",
    "recommend_experiment",
    "validation_report",
    "predict_endpoint",
    "sequence_features",
    "train_endpoint_model",
    "assemble_patient_candidates",
    "read_expression",
    "read_hla",
    "read_loh",
    "read_vcf",
    "compare_selection_methods",
    "experiment_information_gain",
    "rna_design_features",
    "annotate_gene_stability",
    "train_rna_context_model",
]
