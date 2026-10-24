"""CalibraNeo exploratory interface.

The interface intentionally distinguishes analytical ranking from biological
validation. It does not nominate clinical vaccine candidates.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import json

from calibraneo import (
    build_orf,
    construct_quality,
    build_evidence_table,
    attach_iedb_evidence,
    evidence_summary,
    load_predictor,
    select_loh_aware_cassette,
    train_q_policy,
    add_validation_scores,
    build_experiment_plan,
    validation_report,
    predict_endpoint,
    experiment_information_gain,
    assemble_patient_candidates,
)

st.set_page_config(page_title="CalibraNeo: Shortlist Audit Studio", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
PROC_DIR = BASE_DIR / "data_processed"
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = BASE_DIR / "models" / "calibraneo_stage1_bundle.pkl"
IMMUNOGENICITY_MODEL_PATH = BASE_DIR / "models" / "calibraneo_immunogenicity_ott.pkl"
MS_MODEL_PATH = BASE_DIR / "models" / "calibraneo_immunopeptidomics_zaghla.pkl"

st.title("CalibraNeo: Provenance-Aware Neoantigen Shortlist Audit")
st.caption(
    "Analytical ranking and LOH-sensitivity exploration only. "
    "No output is biologically validated or ready for clinical use."
)

@st.cache_resource
def get_predictor():
    return load_predictor(MODEL_PATH)


@st.cache_data
def load_clinical_pool() -> pd.DataFrame:
    path = PROC_DIR / "nature_2317_9tools_benchmark.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Clinical pool not found: {path}")
    frame = pd.read_csv(path)
    master_path = PROC_DIR / "MASTER_CALIBRANEO_DATASET.csv"
    master = pd.read_csv(master_path, low_memory=False)
    source = master[master["dataset_source"].eq("Nature_Clinical_NeoVaccine_2317")].reset_index(drop=True)
    if len(source) != len(frame):
        raise ValueError("Clinical benchmark and feature rows have different lengths")
    predictor = get_predictor()
    frame["CalibraNeo_Prob"] = predictor.predict_proba(source)
    return frame


try:
    pool_df = load_clinical_pool()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. LOH-aware shortlist optimizer",
    "2. ORF construct inspection",
    "3. Benchmark and evidence boundaries",
    "4. Computational validation",
    "5. Patient-specific inputs",
])

with tab1:
    st.subheader("LOH-aware cassette optimizer")
    st.info(
        "LOH means loss of an HLA allele by the tumor. It is not the vaccine ORF. "
        "The optimizer chooses peptide-HLA targets while trying not to depend on one vulnerable allele. "
        "The Q-learning option is a real research baseline, not biological proof."
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        cancer_types = ["All Solid Tumors"] + sorted(pool_df["Cancer type"].dropna().unique().tolist())
        selected_cancer = st.selectbox("Tumor cohort", cancer_types)
    with col_b:
        capacity = st.slider("Cassette capacity", min_value=5, max_value=34, value=20)
    with col_c:
        dominant_loh_risk = st.slider("Dominant-HLA LOH risk", 0.0, 0.90, 0.65, 0.05)

    candidates = pool_df if selected_cancer == "All Solid Tumors" else pool_df[pool_df["Cancer type"].eq(selected_cancer)]
    if len(candidates) < capacity:
        st.warning("The selected cohort has fewer candidates than the requested capacity; using the full cohort.")
    method = st.radio(
        "Selection method",
        ["Transparent greedy optimizer", "Experimental Q-learning policy"],
        help="Q-learning is a computational research baseline; it does not prove biological superiority.",
    )
    if method == "Experimental Q-learning policy":
        if len(candidates) < capacity:
            st.error("Q-learning requires at least as many candidates as slots.")
            st.stop()
        selected_indices = train_q_policy(
            candidates,
            capacity,
            dominant_loh_risk,
            episodes=300,
        )
        selected = candidates.reset_index(drop=True).loc[selected_indices]
        selected.index = np.arange(1, len(selected) + 1)
        selected.index.name = "slot"
    else:
        selected = select_loh_aware_cassette(candidates, capacity, dominant_loh_risk)
    iedb_path = BASE_DIR / "data_external" / "iedb" / "iedb_evidence_normalized.csv"
    if iedb_path.is_file():
        selected = attach_iedb_evidence(
            selected,
            pd.read_csv(iedb_path),
            peptide_column="peptide(mut)",
            hla_column="hla_clean",
        )
    if IMMUNOGENICITY_MODEL_PATH.is_file():
        selected["immunogenicity_model_score"] = predict_endpoint(
            IMMUNOGENICITY_MODEL_PATH,
            selected,
            peptide_column="peptide(mut)",
        )
    else:
        selected["immunogenicity_model_score"] = np.nan
    selected = add_validation_scores(selected)
    st.session_state["cassette_df"] = selected

    survival = selected["CalibraNeo_Prob"] * (
        1.0 - np.where(
            selected["hla_clean"].eq(selected["hla_clean"].value_counts().index[0]),
            dominant_loh_risk,
            0.10,
        )
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Selected slots", len(selected))
    m2.metric("Distinct HLA alleles", selected["hla_clean"].nunique())
    m3.metric("Expected ≥1 surviving target", f"{(1 - np.prod(1 - survival)) * 100:.2f}%")
    st.caption(
        "Computational priority combines the presentation score with only the "
        "external evidence that is actually observed. It is not an immunogenicity probability."
    )
    st.dataframe(selected, use_container_width=True)

with tab2:
    st.subheader("ORF construct inspection")
    st.info(
        "This tab constructs a test sequence from selected peptide strings. It is not a final vaccine "
        "design: translation, RNA stability, processing, delivery, safety, and T-cell response still need "
        "separate models and laboratory validation."
    )
    selected = st.session_state.get("cassette_df", pool_df.head(20))
    linker = st.selectbox("Peptide linker", ["AAY", "GPGPG", "KK"])
    peptides = selected["peptide(mut)"].astype(str).str.strip().tolist()
    try:
        protein, dna = build_orf(peptides, linker)
        quality = construct_quality(dna)
        rna = dna.replace("T", "U")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ORF length", f"{quality['length_nt']} nt")
        m2.metric("GC content", f"{quality['gc_percent']:.1f}%")
        m3.metric("Codon preference score", f"{quality['codon_preference_score']:.3f}")
        m4.metric("Internal stop codons", int(quality["stop_codons"]))
        st.caption(
            "The codon preference score is a transparent relative score, not a validated "
            "species-specific CAI. No folding free energy is inferred here."
        )
        st.markdown("**What CalibraNeo currently predicts:** a calibrated presentation-related score for a peptide-HLA record.")
        st.markdown(
            "**What it does not currently prove:** immunogenicity, RNA performance, safe delivery, or therapeutic benefit."
        )
        st.code(rna, language="text")
        st.download_button("Download RNA ORF", rna, "calibraneo_orf.txt", "text/plain")
    except ValueError as error:
        st.error(str(error))

with tab3:
    st.subheader("Benchmark results and claim boundary")
    evidence = build_evidence_table(pool_df, source="Nature clinical benchmark", source_version="local")
    summary = evidence_summary(evidence)
    st.metric("Rows with independent biological evidence", summary["biologically_ready_rows"])
    st.caption(
        "Unknown evidence remains unknown. The current local benchmark contains "
        "computational scores, not independent MS or T-cell validation."
    )
    if iedb_path.is_file():
        iedb = pd.read_csv(iedb_path)
        st.metric("Imported official IEDB evidence rows", len(iedb))
        evidence_with_iedb = attach_iedb_evidence(pool_df, iedb)
        st.metric(
            "Exact local peptide-HLA rows with IEDB T-cell support",
            int(evidence_with_iedb["iedb_tcell_support"].eq(True).sum()),
        )
        st.dataframe(iedb.head(20), use_container_width=True)
    else:
        st.info("No external evidence snapshot found. Run: python -m calibraneo.cli external-data")
    for filename, title in [
        ("table1_clinical_benchmark_vs_9tools.csv", "Clinical benchmark"),
        ("table1_stage2_leave_one_tool_out.csv", "Stage 2 leave-one-tool-out ablation"),
        ("table2_rl_cassette_optimization_under_hlaloh.csv", "LOH simulation"),
    ]:
        path = RESULTS_DIR / filename
        if path.is_file():
            st.markdown(f"**{title}**")
            st.dataframe(pd.read_csv(path), use_container_width=True)
    st.warning(
        "The current evidence supports shortlist sensitivity and provenance auditing. "
        "It does not establish peptide presentation, immunogenicity, vaccine efficacy, safety, "
        "or clinical readiness. The co-delivery siRNA design is intentionally not presented as a recommendation."
    )
    for filename in [
        "Fig1_Calibration_Benchmark.png",
        "Fig2_RL_Cassette_HLA_LOH_Resilience.png",
        "Fig3_Reliability_Diagram.png",
        "Fig4_Biophysical_Feature_Importance.png",
    ]:
        path = RESULTS_DIR / "figures" / filename
        if path.is_file():
            st.image(str(path), caption=filename)

with tab4:
    st.subheader("Computational validation and experiment queue")
    st.info(
        "This page validates the computational workflow, not a patient's response. "
        "It reports provenance, model outputs, uncertainty, and the next experiment "
        "needed to resolve each candidate's largest evidence gap."
    )
    iedb = (
        pd.read_csv(iedb_path)
        if iedb_path.is_file()
        else pd.DataFrame(columns=["peptide", "hla_allele", "is_positive"])
    )
    evidence_with_iedb = attach_iedb_evidence(pool_df, iedb)
    if IMMUNOGENICITY_MODEL_PATH.is_file():
        evidence_with_iedb["immunogenicity_model_score"] = predict_endpoint(
            IMMUNOGENICITY_MODEL_PATH,
            evidence_with_iedb,
            peptide_column="peptide(mut)",
        )
    scored_pool = add_validation_scores(evidence_with_iedb)
    plan = build_experiment_plan(scored_pool, top_n=min(20, len(scored_pool)))
    plan = experiment_information_gain(plan)
    benchmark_path = RESULTS_DIR / "table1_clinical_benchmark_vs_9tools.csv"
    benchmark = pd.read_csv(benchmark_path) if benchmark_path.is_file() else None
    report = validation_report(
        model_version=get_predictor().model_version,
        candidate_count=len(pool_df),
        external_evidence_rows=len(iedb),
        exact_tcell_matches=int(
            evidence_with_iedb["iedb_tcell_support"].eq(True).sum()
        ),
        benchmark_metrics=benchmark,
    )
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Model version", report["model_version"])
    metric_b.metric("Candidate records", report["candidate_count"])
    metric_c.metric("External evidence rows", report["external_evidence_rows"])
    metric_d.metric("Exact T-cell matches", report["exact_tcell_matches"])
    st.markdown("**Validation checks completed**")
    for check in report["validated_claims"]:
        st.success(check)
    audit_path = RESULTS_DIR / "validation_audit_report.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        st.markdown("**Independent-cohort audit**")
        audit_rows = []
        for cohort, payload in audit.get("external_cohorts", {}).items():
            metrics = payload["metrics"]
            audit_rows.append(
                {
                    "cohort": cohort,
                    "rows": payload["rows"],
                    "positive_rate": payload["positive_rate"],
                    "AUC": metrics["auc_roc"]["estimate"],
                    "AUC 95% low": metrics["auc_roc"]["lower_95"],
                    "AUC 95% high": metrics["auc_roc"]["upper_95"],
                    "group_overlap": payload["group_overlap_with_training"],
                    "independent": payload["fully_independent"],
                }
            )
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)
        zaghla_flags = audit.get("zaghla_quality_flags", {})
        if zaghla_flags.get("usable_as_independent_validation") is False:
            st.warning(
                "The Zaghla-derived table is not treated as independent validation: "
                "its filled binding-rank field is a label-derived proxy."
            )
        st.caption(
            "Bootstrap intervals quantify sampling uncertainty only. The Ott cohort "
            "is an important negative transfer result and prevents a claim of universal superiority."
        )
    st.markdown("**Evidence-gated experiment queue**")
    st.dataframe(
        plan[
            [
                "peptide(mut)",
                "hla_clean",
                "CalibraNeo_Prob",
                "computational_priority_score",
                "evidence_completeness",
                "validation_confidence",
                "expected_information_gain",
                "decision_status",
                "recommended_assay",
                "reason",
            ]
        ],
        use_container_width=True,
    )
    st.markdown("**Claims that remain unproven**")
    for claim in report["not_proven"]:
        st.warning(claim)
    st.download_button(
        "Download experiment queue",
        plan.to_csv(index=False),
        "calibraneo_experiment_queue.csv",
        "text/csv",
    )
    comparison_path = RESULTS_DIR / "method_comparison_status.csv"
    if comparison_path.is_file():
        st.markdown("**Comparator availability and results**")
        st.dataframe(pd.read_csv(comparison_path), use_container_width=True)
        st.caption(
            "A method is not treated as compared unless it was run on the same "
            "frozen candidates with compatible labels and provenance."
        )

with tab5:
    st.subheader("Patient-specific candidate filter")
    st.info(
        "This workspace accepts research inputs and filters candidates by typed HLA. "
        "It does not call variants, infer HLA from incomplete files, or make a clinical recommendation."
    )
    hla_text = st.text_area(
        "HLA alleles, one per line or comma-separated",
        value="HLA-A*02:01\nHLA-A*03:01",
    )
    hla_alleles = [
        allele.strip().upper().replace(" ", "")
        for allele in hla_text.replace(",", "\n").splitlines()
        if allele.strip()
    ]
    expression_upload = st.file_uploader(
        "Optional expression CSV (gene, expression_tpm)",
        type=["csv"],
        key="patient_expression",
    )
    loh_upload = st.file_uploader(
        "Optional HLA-LOH CSV (hla_allele, loh_probability)",
        type=["csv"],
        key="patient_loh",
    )
    if hla_alleles:
        patient_candidates = pool_df.rename(
            columns={"peptide(mut)": "peptide", "hla_clean": "hla_allele"}
        ).copy()
        expression = (
            pd.read_csv(expression_upload)
            if expression_upload is not None
            else None
        )
        loh = pd.read_csv(loh_upload) if loh_upload is not None else None
        try:
            patient_candidates = assemble_patient_candidates(
                patient_candidates,
                hla_alleles=hla_alleles,
                expression=expression,
                loh=loh,
            )
            if not patient_candidates.empty:
                patient_candidates["presentation_score"] = patient_candidates[
                    "CalibraNeo_Prob"
                ]
                st.metric("Patient-HLA candidate records", len(patient_candidates))
                st.metric("Unique candidate peptides", patient_candidates["peptide"].nunique())
                st.dataframe(
                    patient_candidates[
                        [
                            "peptide",
                            "hla_allele",
                            "CalibraNeo_Prob",
                            "expression_tpm",
                            "loh_probability",
                            "patient_input_status",
                        ]
                    ].sort_values("CalibraNeo_Prob", ascending=False),
                    use_container_width=True,
                )
                st.download_button(
                    "Download patient-filtered candidates",
                    patient_candidates.to_csv(index=False),
                    "calibraneo_patient_candidates.csv",
                    "text/csv",
                )
            else:
                st.warning("No benchmark candidates match the supplied HLA alleles.")
        except ValueError as error:
            st.error(str(error))
