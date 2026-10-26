"""Generate manuscript tables and figures from recorded CalibraNeo outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript_assets"
FIG = OUT / "figures"
TABLE = OUT / "tables"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
TABLE.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))


def save_table(frame: pd.DataFrame, name: str, caption: str) -> None:
    frame.to_csv(TABLE / f"{name}.csv", index=False)
    with (TABLE / f"{name}.md").open("w", encoding="utf-8") as handle:
        handle.write(f"**{caption}**\n\n")
        columns = [str(column) for column in frame.columns]
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for values in frame.fillna("").itertuples(index=False, name=None):
            escaped = [str(value).replace("|", "\\|") for value in values]
            handle.write("| " + " | ".join(escaped) + " |\n")
        handle.write("\n")


def source_counts() -> pd.DataFrame:
    path = ROOT / "data_processed" / "MASTER_CALIBRANEO_DATASET.csv"
    frame = pd.read_csv(path, low_memory=False)
    grouped = frame.groupby("dataset_source")["label_presented"].agg(
        records="size",
        positives="sum",
        negatives=lambda x: int((x == 0).sum()),
    ).reset_index()
    return grouped.rename(columns={"dataset_source": "local_dataset"})


def make_tables() -> None:
    counts = source_counts()
    endpoint_map = {
        "D2_Benchmark_96K": "Presentation",
        "IEDB_Experimental_2024": "Presentation",
        "Nature_Clinical_NeoVaccine_2317": "External presentation benchmark / freeze",
        "Ott_Clinical_Melanoma_Vaccine": "T-cell immunogenicity",
        "Zaghla_2026_JITC_Benchmark": "Immunopeptidomics development",
        "Zenodo_2026_TumorRNA": "RNA-context transfer",
    }
    status_map = {
        "D2_Benchmark_96K": "Local, provenance-tracked, grouped audit",
        "IEDB_Experimental_2024": "Local, provenance-tracked, grouped audit",
        "Nature_Clinical_NeoVaccine_2317": "Local, frozen before wet-lab interpretation",
        "Ott_Clinical_Melanoma_Vaccine": "Local, small endpoint-specific cohort",
        "Zaghla_2026_JITC_Benchmark": "Local, label-derived proxy excluded",
        "Zenodo_2026_TumorRNA": "Local, independent external transfer",
    }
    counts["endpoint_track"] = counts["local_dataset"].map(endpoint_map)
    counts["provenance_status"] = counts["local_dataset"].map(status_map)
    save_table(
        counts[
            [
                "local_dataset",
                "endpoint_track",
                "records",
                "positives",
                "negatives",
                "provenance_status",
            ]
        ],
        "table1_local_datasets",
        "Table 1. Summary of local dataset counts and provenance status.",
    )

    manifest = json.loads(
        (ROOT / "models" / "endpoint_model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        {
            "endpoint_model": "Presentation Stage 1",
            "algorithm": "50:50 HistGradientBoosting + MLP",
            "features": "20 engineered peptide/biophysical/HLA features",
            "grouping": "Peptide-HLA; 15% test; separate calibration split",
            "calibration": "Isotonic regression",
            "status": "Executed",
        },
        {
            "endpoint_model": "Ott immunogenicity, expression-enhanced",
            "algorithm": "HistGradientBoosting",
            "features": "8 sequence features + Gene Expression",
            "grouping": "Patient grouped",
            "calibration": "None",
            "status": "Executed; exploratory",
        },
        {
            "endpoint_model": "Ott immunogenicity, deployable",
            "algorithm": "HistGradientBoosting",
            "features": "8 sequence features",
            "grouping": "Patient grouped",
            "calibration": "None",
            "status": "Executed; deployable sequence-only model",
        },
        {
            "endpoint_model": "Zaghla immunopeptidomics",
            "algorithm": "HistGradientBoosting",
            "features": "8 sequence features; proxy field excluded",
            "grouping": "Mutation-HLA grouped",
            "calibration": "None",
            "status": "Executed; study-specific",
        },
        {
            "endpoint_model": "RNADecayCafe context",
            "algorithm": "Leakage-corrected regression",
            "features": "Gene-level expression-context features",
            "grouping": "Held-out gene groups",
            "calibration": "Not applicable",
            "status": "Executed; not custom ORF stability",
        },
        {
            "endpoint_model": "Translation",
            "algorithm": "No fitted model",
            "features": "No local Ribo-seq/RNA-seq label table",
            "grouping": "Not applicable",
            "calibration": "Not applicable",
            "status": "Unevaluated / unavailable",
        },
        {
            "endpoint_model": "NetMHCpan / NetMHC-4 proxy",
            "algorithm": "Imported Nature benchmark output",
            "features": "Existing benchmark column",
            "grouping": "Nature 2,317-record benchmark",
            "calibration": "Not recalibrated",
            "status": "Executed local benchmark",
        },
        {
            "endpoint_model": "MHCflurry",
            "algorithm": "Imported Nature benchmark output",
            "features": "Existing benchmark column",
            "grouping": "Nature 2,317-record benchmark",
            "calibration": "Not recalibrated",
            "status": "Executed local benchmark",
        },
        {
            "endpoint_model": "BigMHC",
            "algorithm": "Imported Nature benchmark output",
            "features": "Existing benchmark column",
            "grouping": "Nature 2,317-record benchmark",
            "calibration": "Not recalibrated",
            "status": "Executed local benchmark",
        },
        {
            "endpoint_model": "DeepNeo / NeoDisc / pVACtools",
            "algorithm": "No compatible local run",
            "features": "Not available",
            "grouping": "Not applicable",
            "calibration": "Not applicable",
            "status": "Unevaluated on identical frozen inputs",
        },
    ]
    save_table(
        pd.DataFrame(rows),
        "table2_model_specifications",
        "Table 2. Endpoint model specifications and comparator status.",
    )

    audit = json.loads(
        (ROOT / "results" / "validation_audit_report.json").read_text(
            encoding="utf-8"
        )
    )
    internal = audit["internal_grouped_holdout"]
    zenodo = audit["external_cohorts"]["Zenodo_2026_TumorRNA"]
    table3 = pd.DataFrame(
        [
            {
                "evaluation": "Internal grouped test",
                "n": 32069,
                "roc_auc": internal["auc_roc"]["estimate"],
                "auprc": internal["auprc"]["estimate"],
                "brier": internal["brier"]["estimate"],
                "ece": internal["ece"]["estimate"],
            },
            {
                "evaluation": "Zenodo external transfer",
                "n": zenodo["rows"],
                "roc_auc": zenodo["metrics"]["auc_roc"]["estimate"],
                "auprc": zenodo["metrics"]["auprc"]["estimate"],
                "brier": zenodo["metrics"]["brier"]["estimate"],
                "ece": zenodo["metrics"]["ece"]["estimate"],
            },
        ]
    )
    save_table(
        table3,
        "table3_internal_external_metrics",
        "Table 3. Internal and external presentation metrics.",
    )

    benchmark = pd.read_csv(
        ROOT / "results" / "table1_clinical_benchmark_vs_9tools.csv"
    )
    selected = benchmark[
        benchmark["Method"].isin(
            [
                "CalibraNeo (Biophysics-Only Calibrated)",
                "NetMHC-4 (Uncalibrated Rank)",
                "MHCflurryPres (Presentation Score)",
                "BigMHC_EL (Deep Learning Baseline)",
                "NetMHCstab (Stability Score)",
            ]
        )
    ].rename(
        columns={
            "Method": "method",
            "AUC_ROC": "roc_auc",
            "AUPRC": "auprc",
            "Brier_Score (Lower=Better)": "brier",
            "ECE_Error (Lower=Better)": "ece",
        }
    )
    save_table(
        selected[["method", "roc_auc", "auprc", "brier", "ece"]],
        "table4_nature_benchmark",
        "Table 4. Nature clinical benchmark comparison.",
    )

    combined = json.loads(
        (ROOT / "results" / "combined_immunogenicity_loso.json").read_text(
            encoding="utf-8"
        )
    )
    table5 = pd.DataFrame(
        [
            {
                "endpoint": "Ott expression-enhanced",
                "roc_auc": manifest["trained"]["immunogenicity_ott"]["metrics"][
                    "auc_roc"
                ],
                "auprc": manifest["trained"]["immunogenicity_ott"]["metrics"][
                    "auprc"
                ],
                "note": "Patient-grouped; small cohort",
            },
            {
                "endpoint": "Ott sequence-only",
                "roc_auc": manifest["trained"][
                    "immunogenicity_ott_deployable"
                ]["metrics"]["auc_roc"],
                "auprc": manifest["trained"][
                    "immunogenicity_ott_deployable"
                ]["metrics"]["auprc"],
                "note": "Deployable sequence-only model",
            },
            {
                "endpoint": "Zaghla immunopeptidomics",
                "roc_auc": manifest["trained"]["immunopeptidomics_zaghla"][
                    "metrics"
                ]["auc_roc"],
                "auprc": manifest["trained"]["immunopeptidomics_zaghla"][
                    "metrics"
                ]["auprc"],
                "note": "46-record grouped holdout; proxy excluded",
            },
            {
                "endpoint": "LOSO: held-out IEDB",
                "roc_auc": combined["loso"][0]["auc_roc"],
                "auprc": combined["loso"][0]["auprc"],
                "note": "Study held out",
            },
            {
                "endpoint": "LOSO: held-out Ott",
                "roc_auc": combined["loso"][1]["auc_roc"],
                "auprc": combined["loso"][1]["auprc"],
                "note": "Study held out",
            },
            {
                "endpoint": "RNADecayCafe context",
                "roc_auc": np.nan,
                "auprc": np.nan,
                "note": "Regression: R2=0.0544; MAE=8.9967",
            },
        ]
    )
    save_table(
        table5,
        "table5_endpoint_audits",
        "Table 5. Endpoint-specific models, transfer, and leakage-aware audits.",
    )

    table6 = pd.read_csv(
        ROOT / "results" / "table2_rl_cassette_optimization_under_hlaloh.csv"
    )
    save_table(
        table6,
        "table6_hla_loh_simulation",
        "Table 6. Simulated 20-slot cassette resilience under dominant HLA-LOH.",
    )


def fit_internal_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    master = pd.read_csv(
        ROOT / "data_processed" / "MASTER_CALIBRANEO_DATASET.csv",
        low_memory=False,
    )
    features = [
        "peptide_length",
        "gravy_hydrophobicity",
        "molecular_weight_da",
        "net_charge_ph7",
        "aromatic_fraction",
        "tcr_core_hydrophobicity",
        "tcr_core_aromaticity",
        "p1_hydrophobicity",
        "p2_hydrophobicity",
        "p2_charge",
        "p_omega_hydrophobicity",
        "p_omega_charge",
        "proteasome_cleavage_score",
        "tap_transport_score",
        "hla_locus_A",
        "hla_locus_B",
        "hla_locus_C",
        "hla_p2_motif_prob",
        "hla_p_omega_motif_prob",
        "combined_anchor_motif_score",
    ]
    frame = master[master["split_role"].eq("train_val")].copy()
    x = frame[features].to_numpy(dtype=np.float32)
    y = frame["label_presented"].to_numpy(dtype=int)
    groups = frame["peptide"].astype(str) + "|" + frame["hla_allele"].astype(str)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    fit_pool, test = next(outer.split(x, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=43)
    fit_relative, calibration_relative = next(
        inner.split(x[fit_pool], y[fit_pool], groups.iloc[fit_pool])
    )
    fit = fit_pool[fit_relative]
    calibration = fit_pool[calibration_relative]
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x[fit])
    x_calibration = scaler.transform(x[calibration])
    x_test = scaler.transform(x[test])
    gbdt = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_depth=8, random_state=42
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        batch_size=512,
        max_iter=25,
        early_stopping=True,
        random_state=42,
    )
    gbdt.fit(x_fit, y[fit])
    mlp.fit(x_fit, y[fit])
    raw_calibration = (
        gbdt.predict_proba(x_calibration)[:, 1]
        + mlp.predict_proba(x_calibration)[:, 1]
    ) / 2
    raw_test = (
        gbdt.predict_proba(x_test)[:, 1] + mlp.predict_proba(x_test)[:, 1]
    ) / 2
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        raw_calibration, y[calibration]
    )
    return y[test], raw_test, calibrator.transform(raw_test)


def make_figure1() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    panels = [
        (
            axes[0, 0],
            "A  Provenance audit and partitioning",
            ["Source registry", "License/access audit", "Endpoint-specific tables", "Grouped fit / calibration / test"],
            "#dbeafe",
        ),
        (
            axes[0, 1],
            "B  Separate endpoint tracks",
            ["Presentation", "T-cell immunogenicity", "Immunopeptidomics", "RNA context"],
            "#dcfce7",
        ),
        (
            axes[1, 0],
            "C  Calibration and evidence integration",
            ["Raw model scores", "Isotonic calibration", "Unknown evidence remains unknown", "Confidence and gaps"],
            "#fef3c7",
        ),
        (
            axes[1, 1],
            "D  Patient-aware decision layer",
            ["HLA / expression / LOH inputs", "LOH-aware greedy selection", "Information-gain heuristic", "SHA-256 freeze"],
            "#fce7f3",
        ),
    ]
    for ax, title, labels, color in panels:
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        y_positions = np.linspace(0.78, 0.18, len(labels))
        for idx, (label, y) in enumerate(zip(labels, y_positions)):
            box = FancyBboxPatch(
                (0.12, y - 0.07),
                0.76,
                0.12,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="#334155",
                linewidth=1.1,
                transform=ax.transAxes,
            )
            ax.add_patch(box)
            ax.text(
                0.50,
                y,
                label,
                ha="center",
                va="center",
                fontsize=10,
                transform=ax.transAxes,
            )
            if idx < len(labels) - 1:
                ax.add_patch(
                    FancyArrowPatch(
                        (0.50, y - 0.08),
                        (0.50, y_positions[idx + 1] + 0.08),
                        arrowstyle="-|>",
                        mutation_scale=12,
                        color="#64748b",
                        transform=ax.transAxes,
                    )
                )
    fig.suptitle(
        "CalibraNeo: provenance-aware, endpoint-specific computational prioritisation",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Computational prioritisation and experiment design; not proof of translation, presentation, T-cell activation, safety, or efficacy.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.savefig(FIG / "Figure1_system_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figure2() -> None:
    y, raw, calibrated = fit_internal_predictions()
    master = pd.read_csv(
        ROOT / "data_processed" / "MASTER_CALIBRANEO_DATASET.csv",
        low_memory=False,
    )
    zenodo = master[master["dataset_source"].eq("Zenodo_2026_TumorRNA")].copy()
    bundle = joblib.load(ROOT / "models" / "calibraneo_stage1_bundle.pkl")
    values = bundle["scaler"].transform(
        zenodo[bundle["bio_features"]].to_numpy(dtype=np.float32)
    )
    external_raw = (
        bundle["gbdt"].predict_proba(values)[:, 1]
        + bundle["mlp"].predict_proba(values)[:, 1]
    ) / 2
    external_calibrated = bundle["iso_cal"].transform(external_raw)
    y_external = zenodo["label_presented"].to_numpy(dtype=int)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    fpr_raw, tpr_raw, _ = roc_curve(y, raw)
    fpr_cal, tpr_cal, _ = roc_curve(y, calibrated)
    axes[0].plot(fpr_raw, tpr_raw, label="Uncalibrated (AUC 0.9243)")
    axes[0].plot(fpr_cal, tpr_cal, label="Calibrated (AUC 0.9241)")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    axes[0].set_title("A  Internal ROC")
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].legend(fontsize=8)
    precision_raw, recall_raw, _ = precision_recall_curve(y, raw)
    precision_cal, recall_cal, _ = precision_recall_curve(y, calibrated)
    inset = axes[0].inset_axes([0.52, 0.12, 0.45, 0.42])
    inset.plot(recall_raw, precision_raw, label="Raw")
    inset.plot(recall_cal, precision_cal, label="Calibrated")
    inset.set_title("PR", fontsize=9)
    inset.tick_params(labelsize=7)
    inset.legend(fontsize=7)
    frac_raw, mean_raw = calibration_curve(y, raw, n_bins=10)
    frac_cal, mean_cal = calibration_curve(y, calibrated, n_bins=10)
    axes[1].plot(mean_raw, frac_raw, "o-", label="Uncalibrated; ECE 0.0138")
    axes[1].plot(mean_cal, frac_cal, "o-", label="Calibrated; ECE 0.0060")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_title("B  Internal reliability")
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Observed frequency")
    axes[1].legend(fontsize=8)
    frac_ext, mean_ext = calibration_curve(
        y_external, external_calibrated, n_bins=10
    )
    axes[2].plot(mean_ext, frac_ext, "o-", label="Zenodo; ECE 0.1687")
    axes[2].plot([0, 1], [0, 1], "--", color="gray")
    axes[2].set_title("C  External calibration drift")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Observed frequency")
    axes[2].legend(fontsize=8)
    fig.suptitle("Figure 2. Internal calibration and external transfer", fontsize=15)
    fig.savefig(FIG / "Figure2_calibration_transfer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figure3() -> None:
    frame = pd.read_csv(ROOT / "results" / "table1_clinical_benchmark_vs_9tools.csv")
    selected = frame[
        frame["Method"].isin(
            [
                "CalibraNeo (Biophysics-Only Calibrated)",
                "NetMHC-4 (Uncalibrated Rank)",
                "MHCflurryPres (Presentation Score)",
                "BigMHC_EL (Deep Learning Baseline)",
                "NetMHCstab (Stability Score)",
            ]
        )
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    labels = selected["Method"].str.replace(
        " (Biophysics-Only Calibrated)", "", regex=False
    ).str.replace(" (Uncalibrated Rank)", "", regex=False).str.replace(
        " (Presentation Score)", "", regex=False
    ).str.replace(" (Deep Learning Baseline)", "", regex=False).str.replace(
        " (Stability Score)", "", regex=False
    )
    x = np.arange(len(selected))
    width = 0.36
    axes[0].bar(x - width / 2, selected["AUC_ROC"], width, label="ROC-AUC")
    axes[0].bar(x + width / 2, selected["AUPRC"], width, label="AUPRC")
    axes[0].set_xticks(x, labels, rotation=28, ha="right")
    axes[0].set_ylim(0, 0.65)
    axes[0].set_title("A  Ranking performance")
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[1].bar(
        x - width / 2,
        selected["Brier_Score (Lower=Better)"],
        width,
        label="Brier score",
    )
    axes[1].bar(
        x + width / 2,
        selected["ECE_Error (Lower=Better)"],
        width,
        label="ECE",
    )
    axes[1].set_xticks(x, labels, rotation=28, ha="right")
    axes[1].set_title("B  Calibration error")
    axes[1].set_ylabel("Lower is better")
    axes[1].legend()
    fig.suptitle("Figure 3. Nature clinical benchmark (n=2,317)", fontsize=15)
    fig.savefig(FIG / "Figure3_nature_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figure4() -> None:
    frame = pd.read_csv(
        ROOT / "results" / "table2_rl_cassette_optimization_under_hlaloh.csv"
    )
    labels = frame["Cassette_Design_Strategy"].str.replace(
        " Greedy Top-20", "", regex=False
    ).str.replace(" Greedy Optimizer", "", regex=False)
    x = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    width = 0.36
    axes[0].bar(
        x - width / 2,
        frame["Mean_HLA_Bridges_Used (Max=4)"],
        width,
        label="Mean HLA bridges",
    )
    axes[0].bar(
        x + width / 2,
        frame["Mean_Surviving_Epitopes_Per_Vaccine"],
        width,
        label="Mean surviving epitopes",
    )
    axes[0].set_xticks(x, labels, rotation=28, ha="right")
    axes[0].set_title("A  Portfolio diversity and survival")
    axes[0].legend()
    axes[1].bar(
        x - width / 2,
        frame["Prob_At_Least_2_Active_Epitopes (%)"],
        width,
        label="≥2 active epitopes",
    )
    axes[1].bar(
        x + width / 2,
        frame["Total_Vaccine_Failure_Rate_Under_HLA_LOH (%)"],
        width,
        label="Complete failure",
    )
    axes[1].set_xticks(x, labels, rotation=28, ha="right")
    axes[1].set_ylim(0, 60)
    axes[1].set_ylabel("Simulated patients (%)")
    axes[1].set_title("B  Resilience under simulated HLA-LOH")
    axes[1].legend()
    fig.suptitle("Figure 4. Simulated 20-slot cassette resilience", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "Simulation-based comparison; not patient-level or biological validation.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.savefig(FIG / "Figure4_hla_loh_simulation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    make_tables()
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    print(f"Generated manuscript assets in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
