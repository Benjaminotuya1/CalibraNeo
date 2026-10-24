"""Audit CalibraNeo with grouped holdouts, bootstrap intervals, and shift checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibraneo.inference import load_predictor


def ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.digitize(probabilities, edges) - 1
    result = 0.0
    for index in range(bins):
        mask = ids == index
        if np.any(mask):
            result += abs(y_true[mask].mean() - probabilities[mask].mean()) * (
                mask.sum() / len(y_true)
            )
    return float(result)


def metric_values(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    return {
        "auc_roc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "ece": ece(y_true, probabilities),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    seed: int = 42,
    repeats: int = 1000,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for _ in range(repeats):
        indices = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        rows.append(metric_values(y_true[indices], probabilities[indices]))
    frame = pd.DataFrame(rows)
    return {
        column: {
            "estimate": float(metric_values(y_true, probabilities)[column]),
            "lower_95": float(frame[column].quantile(0.025)),
            "upper_95": float(frame[column].quantile(0.975)),
        }
        for column in frame.columns
    }


def main() -> int:
    proc = ROOT / "data_processed"
    results = ROOT / "results"
    master = pd.read_csv(proc / "MASTER_CALIBRANEO_DATASET.csv", low_memory=False)
    predictor = load_predictor(ROOT / "models" / "calibraneo_stage1_bundle.pkl")
    features = predictor.feature_names
    train = master[master["split_role"].eq("train_val")].copy()
    values = train[features].to_numpy(dtype=np.float32)
    labels = train["label_presented"].to_numpy(dtype=int)
    groups = train["peptide"].astype(str) + "|" + train["hla_allele"].astype(str)
    split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    fit_index, test_index = next(split.split(values, labels, groups))
    probabilities = predictor.predict_proba(train.iloc[test_index])
    internal_metrics = bootstrap_metrics(labels[test_index], probabilities)

    group_audit = {
        "rows": int(len(train)),
        "unique_peptide_hla_groups": int(groups.nunique()),
        "fit_rows": int(len(fit_index)),
        "test_rows": int(len(test_index)),
        "fit_groups": int(groups.iloc[fit_index].nunique()),
        "test_groups": int(groups.iloc[test_index].nunique()),
        "group_overlap": int(
            len(set(groups.iloc[fit_index]).intersection(set(groups.iloc[test_index])))
        ),
        "duplicate_peptide_hla_rows": int(len(train) - groups.nunique()),
    }
    train_group_set = set(groups)

    def evaluate_cohort(source_name: str) -> dict[str, object]:
        cohort = master[master["dataset_source"].eq(source_name)].copy()
        cohort_probabilities = predictor.predict_proba(cohort)
        cohort_labels = cohort["label_presented"].to_numpy(dtype=int)
        cohort_groups = (
            cohort["peptide"].astype(str) + "|" + cohort["hla_allele"].astype(str)
        )
        overlap = int(len(set(cohort_groups).intersection(train_group_set)))
        return {
            "rows": int(len(cohort)),
            "positive_rate": float(cohort_labels.mean()),
            "metrics": bootstrap_metrics(cohort_labels, cohort_probabilities),
            "group_overlap_with_training": overlap,
            "fully_independent": overlap == 0,
        }

    zaghla = master[
        master["dataset_source"].eq("Zaghla_2026_JITC_Benchmark")
    ].copy()
    external_labels = zaghla["label_presented"].to_numpy(dtype=int)
    external_probabilities = predictor.predict_proba(zaghla)
    external_metrics = bootstrap_metrics(external_labels, external_probabilities)
    external_groups = (
        zaghla["peptide"].astype(str) + "|" + zaghla["hla_allele"].astype(str)
    )
    external_overlap = int(len(set(external_groups).intersection(train_group_set)))
    external_baselines = {}
    for name in ("proteasome_cleavage_score", "tap_transport_score", "combined_anchor_motif_score"):
        baseline = zaghla[name].to_numpy(dtype=float)
        scaled = (baseline - baseline.min()) / (baseline.max() - baseline.min() or 1.0)
        external_baselines[name] = metric_values(external_labels, scaled)
    zaghla_label_proxy = {
        "binding_rank_pct_filled_unique_values": sorted(
            zaghla["binding_rank_pct_filled"].unique().tolist()
        ),
        "binding_rank_pct_filled_perfectly_separates_label": bool(
            zaghla.groupby("label_presented")["binding_rank_pct_filled"].nunique().max() == 1
            and zaghla.groupby("binding_rank_pct_filled")["label_presented"].nunique().max() == 1
        ),
        "usable_as_independent_validation": False,
        "reason": "The imported filled binding rank is a label-derived proxy in this table.",
    }
    other_cohorts = {
        name: evaluate_cohort(name)
        for name in (
            "Ott_Clinical_Melanoma_Vaccine",
            "Zenodo_2026_TumorRNA",
        )
    }
    subgroup_rows: list[dict[str, object]] = []
    if "hla_allele" in zaghla:
        for allele, subset in zaghla.groupby("hla_allele"):
            if len(subset) >= 10 and subset["label_presented"].nunique() == 2:
                subgroup_rows.append(
                    {
                        "hla_allele": str(allele),
                        "rows": int(len(subset)),
                        **metric_values(
                            subset["label_presented"].to_numpy(dtype=int),
                            predictor.predict_proba(subset),
                        ),
                    }
                )

    nature = pd.read_csv(proc / "nature_2317_9tools_benchmark.csv")
    nature_source = master[
        master["dataset_source"].eq("Nature_Clinical_NeoVaccine_2317")
    ].reset_index(drop=True)
    nature_probabilities = predictor.predict_proba(nature_source)
    nature_labels = nature["label_presented"].to_numpy(dtype=int)
    shift = {
        "internal_positive_rate": float(labels.mean()),
        "zaghla_positive_rate": float(external_labels.mean()),
        "nature_positive_rate": float(nature_labels.mean()),
        "internal_feature_mean": {
            name: float(train[name].mean()) for name in features
        },
        "zaghla_feature_mean": {
            name: float(zaghla[name].mean()) for name in features
        },
    }
    report = {
        "model_version": predictor.model_version,
        "internal_grouped_holdout": internal_metrics,
        "external_zaghla": external_metrics,
        "external_cohorts": other_cohorts,
        "nature_benchmark": bootstrap_metrics(nature_labels, nature_probabilities),
        "group_leakage_audit": group_audit,
        "external_group_overlap_with_training": external_overlap,
        "external_baselines": external_baselines,
        "zaghla_quality_flags": zaghla_label_proxy,
        "external_baseline_note": (
            "Mechanistic feature baselines are min-max scaled within the external "
            "set for Brier/ECE; their AUC is ranking-only and not a calibrated probability."
        ),
        "hla_subgroups_external": subgroup_rows,
        "dataset_shift": shift,
        "interpretation": {
            "group_overlap_zero_required": True,
            "external_validation_is_not_patient_specific": True,
            "confidence_intervals_are_bootstrap_not_clinical": True,
            "zaghla_auc_is_not_claimed_as_independent": True,
        },
    }
    (results / "validation_audit_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(subgroup_rows).to_csv(
        results / "validation_hla_subgroups.csv", index=False
    )
    summary = pd.DataFrame(
        [
            {"dataset": "internal_grouped_holdout", **{k: v["estimate"] for k, v in internal_metrics.items()}},
            {"dataset": "external_zaghla", **{k: v["estimate"] for k, v in external_metrics.items()}},
            {"dataset": "nature_benchmark", **{k: v["estimate"] for k, v in report["nature_benchmark"].items()}},
        ]
    )
    summary.to_csv(results / "validation_audit_summary.csv", index=False)
    print(json.dumps({
        "status": "ok",
        "group_overlap": group_audit["group_overlap"],
        "internal_auc": internal_metrics["auc_roc"],
        "external_auc": external_metrics["auc_roc"],
        "report": str(results / "validation_audit_report.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
