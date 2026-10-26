"""Run the requested leakage, calibration, source-LOSO, and LOH hardening audits."""

from __future__ import annotations

import json
import argparse
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibraneo.inference import load_predictor
from calibraneo.selection import exact_small_pool_hierarchical_selection


def metric_row(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def bootstrap_rows(
    y: np.ndarray,
    p: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    repeats: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    if groups is None:
        units = np.arange(len(y))
        members = {int(i): np.array([i]) for i in units}
    else:
        units = np.unique(groups)
        members = {unit: np.flatnonzero(groups == unit) for unit in units}
    draws = []
    for _ in range(repeats):
        chosen = rng.choice(units, size=len(units), replace=True)
        indices = np.concatenate([members[unit] for unit in chosen])
        if len(np.unique(y[indices])) < 2:
            continue
        draws.append(metric_row(y[indices], p[indices]))
    point = metric_row(y, p)
    result = {}
    for key, value in point.items():
        values = np.array([row[key] for row in draws])
        result[key] = {
            "estimate": value,
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "bootstrap_unit": "group" if groups is not None else "record",
        }
    return result


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(logits, y)
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


def calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    observed, predicted = calibration_curve(y, p, n_bins=bins, strategy="uniform")
    return pd.DataFrame(
        {
            "bin": np.arange(1, len(observed) + 1),
            "mean_predicted": predicted,
            "observed_fraction": observed,
            "prevalence_baseline": float(y.mean()),
            "binning": "10 equal-width bins",
        }
    )


def decision_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    rows = []
    prevalence = float(y.mean())
    for threshold in np.linspace(0.01, 0.99, 99):
        predicted_positive = p >= threshold
        tp = np.sum(predicted_positive & (y == 1))
        fp = np.sum(predicted_positive & (y == 0))
        n = len(y)
        odds = threshold / (1.0 - threshold)
        net_benefit = tp / n - fp / n * odds
        rows.append(
            {
                "threshold": threshold,
                "model_net_benefit": net_benefit,
                "treat_all_net_benefit": prevalence - (1 - prevalence) * odds,
                "treat_none_net_benefit": 0.0,
            }
        )
    return pd.DataFrame(rows)


def source_loso(master: pd.DataFrame, predictor) -> pd.DataFrame:
    rows = []
    sources = ["IEDB_Experimental_2024", "D2_Benchmark_96K"]
    for held_out in sources:
        train = master[
            master["dataset_source"].isin([source for source in sources if source != held_out])
        ].copy()
        test = master[master["dataset_source"].eq(held_out)].copy()
        features = predictor.feature_names
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[features])
        x_test = scaler.transform(test[features])
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_depth=8, random_state=42
        )
        model.fit(x_train, train["label_presented"].astype(int))
        p = model.predict_proba(x_test)[:, 1]
        rows.append({"held_out_source": held_out, "train_rows": len(train), "test_rows": len(test), **metric_row(test["label_presented"].to_numpy(int), p)})
    return pd.DataFrame(rows)


def shared_score_loh_simulation(
    pool: pd.DataFrame,
    *,
    n_patients: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    top_hlas = pool["hla_clean"].value_counts().head(12).index.tolist()
    methods = ["score_greedy", "lambda_0", "lambda_045", "hla_diversity", "round_robin"]
    scenarios = ["dominant_single", "random_single", "two_allele"]
    rows = []
    for scenario in scenarios:
        for loh_rate in (0.05, 0.10, 0.20, 0.30):
            outcomes = {method: [] for method in methods}
            for _ in range(n_patients):
                patient_hlas = list(rng.choice(top_hlas, size=4, replace=False))
                dominant = patient_hlas[0]
                pieces = []
                for allele in patient_hlas:
                    count = 55 if allele == dominant else 25
                    pieces.append(pool[pool["hla_clean"].eq(allele)].sample(count, replace=True, random_state=int(rng.integers(1_000_000))))
                candidates = pd.concat(pieces, ignore_index=True)
                candidates["presentation_probability"] = candidates[
                    "presentation_score_frozen"
                ].clip(0, 1)
                candidates = candidates.reset_index(drop=True)
                allele_values = candidates["hla_clean"].astype(str).to_numpy()
                presentation = candidates["presentation_probability"].to_numpy(float)
                if scenario == "dominant_single":
                    loh = {a: (0.75 if a == dominant else loh_rate) for a in patient_hlas}
                elif scenario == "random_single":
                    lost = str(rng.choice(patient_hlas))
                    loh = {a: (1.0 if a == lost else 0.0) for a in patient_hlas}
                else:
                    lost = rng.choice(patient_hlas, size=2, replace=False)
                    loh = {a: (1.0 if a in set(lost) else 0.0) for a in patient_hlas}
                candidates["loh_probability"] = candidates["hla_clean"].map(loh)
                candidates["effective_probability"] = candidates["presentation_probability"] * (1 - candidates["loh_probability"])
                effective = candidates["effective_probability"].to_numpy(float)
                selections = {}
                selections["score_greedy"] = np.argsort(-presentation, kind="stable")[:20].tolist()
                for method, decay in [("lambda_0", 0.0), ("lambda_045", 0.45)]:
                    chosen, counts = [], {}
                    for _slot in range(20):
                        scores = effective.copy()
                        if chosen:
                            scores[chosen] = -np.inf
                        scores *= np.exp(
                            -decay
                            * np.array([counts.get(allele, 0) for allele in allele_values])
                        )
                        pick = int(np.argmax(scores))
                        chosen.append(pick)
                        allele = allele_values[pick]
                        counts[allele] = counts.get(allele, 0) + 1
                    selections[method] = chosen
                chosen, counts = [], {}
                for _slot in range(20):
                    scores = effective.copy()
                    if chosen:
                        scores[chosen] = -np.inf
                    scores /= 1 + np.array(
                        [counts.get(allele, 0) for allele in allele_values]
                    )
                    pick = int(np.argmax(scores))
                    chosen.append(pick)
                    allele = allele_values[pick]
                    counts[allele] = counts.get(allele, 0) + 1
                selections["hla_diversity"] = chosen
                chosen = []
                for allele in patient_hlas:
                    allele_indices = np.flatnonzero(allele_values == allele)
                    chosen.extend(allele_indices[np.argsort(-presentation[allele_indices])[:5]].tolist())
                selections["round_robin"] = chosen[:20]
                for method, indices in selections.items():
                    selected = candidates.loc[indices]
                    surviving = selected["effective_probability"].sum()
                    outcomes[method].append(
                        {
                            "bridges": selected["hla_clean"].nunique(),
                            "surviving_score": surviving,
                            "failure": int(surviving <= 0),
                        }
                    )
            for method, values in outcomes.items():
                frame = pd.DataFrame(values)
                for metric in ("bridges", "surviving_score", "failure"):
                    mean = float(frame[metric].mean())
                    lower, upper = np.quantile(
                        frame[metric].to_numpy(), [0.025, 0.975]
                    )
                    rows.append(
                        {
                            "scenario": scenario,
                            "secondary_loh_rate": loh_rate,
                            "method": method,
                            "metric": metric,
                            "estimate": mean,
                            "lower_95": float(lower),
                            "upper_95": float(upper),
                            "n_patients": n_patients,
                            "seed": seed,
                        }
                    )
    return pd.DataFrame(rows)


def stochastic_shared_score_loh_simulation(
    pool: pd.DataFrame,
    *,
    n_patients: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate stochastic surviving-epitope outcomes under shared scores.

    For each patient and policy, this draws HLA-loss indicators Z_h and
    presentation indicators X_i ~ Bernoulli(p_i), then computes
    S = sum_i X_i * (1 - Z_h(i)). The output is descriptive Monte Carlo
    estimation under hypothetical LOH assumptions, not patient validation.
    """
    rng = np.random.default_rng(seed)
    top_hlas = pool["hla_clean"].value_counts().head(12).index.tolist()
    methods = ["score_greedy", "lambda_0", "lambda_045", "hla_diversity", "round_robin"]
    scenarios = ["dominant_single", "random_single", "two_allele"]
    rows = []
    for scenario in scenarios:
        for loh_rate in (0.05, 0.10, 0.20, 0.30):
            outcomes = {method: [] for method in methods}
            for _ in range(n_patients):
                patient_hlas = list(rng.choice(top_hlas, size=4, replace=False))
                dominant = patient_hlas[0]
                pieces = []
                for allele in patient_hlas:
                    count = 55 if allele == dominant else 25
                    pieces.append(
                        pool[pool["hla_clean"].eq(allele)].sample(
                            count,
                            replace=True,
                            random_state=int(rng.integers(1_000_000)),
                        )
                    )
                candidates = pd.concat(pieces, ignore_index=True).reset_index(drop=True)
                presentation = candidates["presentation_score_frozen"].clip(0, 1).to_numpy(float)
                allele_values = candidates["hla_clean"].astype(str).to_numpy()
                presentation_draw = rng.random(len(candidates)) < presentation
                selections = {}
                selections["score_greedy"] = np.argsort(-presentation, kind="stable")[:20].tolist()
                for method, decay in [("lambda_0", 0.0), ("lambda_045", 0.45)]:
                    chosen, counts = [], {}
                    for _slot in range(20):
                        scores = presentation.copy()
                        if chosen:
                            scores[chosen] = -np.inf
                        scores *= np.exp(
                            -decay * np.array(
                                [counts.get(allele, 0) for allele in allele_values]
                            )
                        )
                        pick = int(np.argmax(scores))
                        chosen.append(pick)
                        allele = allele_values[pick]
                        counts[allele] = counts.get(allele, 0) + 1
                    selections[method] = chosen
                chosen, counts = [], {}
                for _slot in range(20):
                    scores = presentation.copy()
                    if chosen:
                        scores[chosen] = -np.inf
                    scores /= 1 + np.array(
                        [counts.get(allele, 0) for allele in allele_values]
                    )
                    pick = int(np.argmax(scores))
                    chosen.append(pick)
                    allele = allele_values[pick]
                    counts[allele] = counts.get(allele, 0) + 1
                selections["hla_diversity"] = chosen
                chosen = []
                for allele in patient_hlas:
                    allele_indices = np.flatnonzero(allele_values == allele)
                    chosen.extend(
                        allele_indices[np.argsort(-presentation[allele_indices])[:5]].tolist()
                    )
                selections["round_robin"] = chosen[:20]

                if scenario == "dominant_single":
                    loh = {
                        allele: (0.75 if allele == dominant else loh_rate)
                        for allele in patient_hlas
                    }
                    loss = {
                        allele: bool(rng.random() < probability)
                        for allele, probability in loh.items()
                    }
                elif scenario == "random_single":
                    lost = str(rng.choice(patient_hlas))
                    loss = {allele: allele == lost for allele in patient_hlas}
                else:
                    lost = set(rng.choice(patient_hlas, size=2, replace=False))
                    loss = {allele: allele in lost for allele in patient_hlas}
                loh_probability = np.array(
                    [float(loss[allele]) for allele in allele_values],
                    dtype=float,
                )
                for method, indices in selections.items():
                    selected = candidates.iloc[indices]
                    selected_alleles = selected["hla_clean"].astype(str).to_numpy()
                    surviving = int(
                        np.sum(
                            presentation_draw[indices]
                            & ~np.array([loss[allele] for allele in selected_alleles])
                        )
                    )
                    outcomes[method].append(
                        {
                            "bridges": selected["hla_clean"].nunique(),
                            "surviving_epitopes": surviving,
                            "at_least_2": int(surviving >= 2),
                            "complete_failure": int(surviving == 0),
                        }
                    )
            for method, values in outcomes.items():
                frame = pd.DataFrame(values)
                for metric in ("bridges", "surviving_epitopes", "at_least_2", "complete_failure"):
                    mean = float(frame[metric].mean())
                    if metric in {"at_least_2", "complete_failure"}:
                        z = 1.959963984540054
                        successes = float(frame[metric].sum())
                        denominator = float(len(frame))
                        center = (successes + z**2 / 2) / (denominator + z**2)
                        margin = (
                            z
                            * math.sqrt(
                                successes * (denominator - successes) / denominator
                                + z**2 / 4
                            )
                            / (denominator + z**2)
                        )
                        lower, upper = center - margin, center + margin
                    else:
                        lower, upper = np.quantile(
                            frame[metric].to_numpy(), [0.025, 0.975]
                        )
                    rows.append(
                        {
                            "scenario": scenario,
                            "secondary_loh_rate": loh_rate,
                            "method": method,
                            "metric": metric,
                            "estimate": mean,
                            "lower_95": float(lower),
                            "upper_95": float(upper),
                            "n_patients": n_patients,
                            "seed": seed,
                            "simulation": "stochastic_X_and_Z",
                        }
                    )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CalibraNeo validation and LOH audits.")
    parser.add_argument(
        "--stochastic-patients",
        type=int,
        default=2000,
        help="Number of patients for the stochastic LOH simulation (default: 2000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible LOH simulations (default: 42).",
    )
    args = parser.parse_args()
    if args.stochastic_patients < 1:
        parser.error("--stochastic-patients must be positive")
    proc = ROOT / "data_processed"
    results = ROOT / "results"
    master = pd.read_csv(proc / "MASTER_CALIBRANEO_DATASET.csv", low_memory=False)
    predictor = load_predictor(ROOT / "models" / "calibraneo_stage1_bundle.pkl")
    group = master["peptide"].astype(str) + "|" + master["hla_allele"].astype(str)
    split_manifest = master[
        ["dataset_source", "split_role", "peptide", "hla_allele", "label_presented"]
    ].copy()
    split_manifest["peptide_hla_group"] = group
    train_mask = master["split_role"].eq("train_val")
    train_indices = np.flatnonzero(train_mask.to_numpy())
    train_groups = group[train_mask]
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    fit_pool_relative, test_relative = next(
        outer.split(train_indices, master.loc[train_mask, "label_presented"], train_groups)
    )
    fit_pool_indices = train_indices[fit_pool_relative]
    test_indices = train_indices[test_relative]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=43)
    fit_relative, cal_relative = next(
        inner.split(
            fit_pool_indices,
            master.iloc[fit_pool_indices]["label_presented"],
            group.iloc[fit_pool_indices],
        )
    )
    fit_indices = fit_pool_indices[fit_relative]
    cal_indices = fit_pool_indices[cal_relative]
    split_manifest["model_partition"] = "not_used"
    split_manifest.loc[fit_indices, "model_partition"] = "fit"
    split_manifest.loc[cal_indices, "model_partition"] = "calibration"
    split_manifest.loc[test_indices, "model_partition"] = "internal_test"
    split_manifest["record_id"] = np.arange(len(split_manifest))
    split_manifest.to_csv(results / "split_manifest.csv", index=False)
    split_counts = (
        split_manifest.groupby(["dataset_source", "model_partition", "label_presented"])
        .size()
        .rename("records")
        .reset_index()
    )
    split_counts.to_csv(results / "split_source_label_counts.csv", index=False)
    train_groups = set(group[train_mask])
    overlap_rows = []
    for role in sorted(master["split_role"].unique()):
        role_groups = set(group[master["split_role"].eq(role)])
        overlap_rows.append(
            {
                "split_role": role,
                "records": int((master["split_role"] == role).sum()),
                "unique_groups": len(role_groups),
                "overlap_with_train_val": len(role_groups & train_groups)
                if role != "train_val"
                else 0,
            }
        )
    pd.DataFrame(overlap_rows).to_csv(results / "split_overlap_audit.csv", index=False)

    nature = master[master["dataset_source"].eq("Nature_Clinical_NeoVaccine_2317")].copy()
    nature_group = group[nature.index].to_numpy()
    independent_nature = nature[~np.isin(nature_group, list(train_groups))].copy()
    independent_nature["CalibraNeo_Prob"] = predictor.predict_proba(independent_nature)
    independent_nature.to_csv(results / "nature_overlap_excluded_predictions.csv", index=False)
    y_nature = nature["label_presented"].to_numpy(int)
    p_nature = predictor.predict_proba(nature)
    zenodo = master[master["dataset_source"].eq("Zenodo_2026_TumorRNA")].copy()
    y_zenodo = zenodo["label_presented"].to_numpy(int)
    p_zenodo = predictor.predict_proba(zenodo)
    calibration_summary = pd.DataFrame(
        [
            {"cohort": "Nature all", **calibration_slope_intercept(y_nature, p_nature)},
            {"cohort": "Nature overlap-excluded", **calibration_slope_intercept(independent_nature["label_presented"].to_numpy(int), independent_nature["CalibraNeo_Prob"])},
            {"cohort": "Zenodo", **calibration_slope_intercept(y_zenodo, p_zenodo)},
        ]
    )
    calibration_summary.to_csv(results / "calibration_slope_intercept.csv", index=False)
    calibration_bins(y_nature, p_nature).to_csv(results / "reliability_nature.csv", index=False)
    calibration_bins(y_zenodo, p_zenodo).to_csv(results / "reliability_zenodo.csv", index=False)
    decision_curve(y_nature, p_nature).to_csv(results / "decision_curve_nature_calibraneo.csv", index=False)
    source_loso(master, predictor).to_csv(results / "leave_one_source_out.csv", index=False)
    test = master[master["split_role"].eq("train_val")].copy()
    groups = test["peptide"].astype(str) + "|" + test["hla_allele"].astype(str)
    split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    _, test_idx = next(split.split(test, test["label_presented"], groups))
    y_test = test.iloc[test_idx]["label_presented"].to_numpy(int)
    p_test = predictor.predict_proba(test.iloc[test_idx])
    pd.DataFrame(
        [
            {"cohort": "Internal grouped test", **bootstrap_rows(y_test, p_test, groups=groups.iloc[test_idx].to_numpy())},
            {"cohort": "Nature", **bootstrap_rows(y_nature, p_nature)},
            {"cohort": "Zenodo", **bootstrap_rows(y_zenodo, p_zenodo)},
        ]
    ).to_json(results / "group_aware_bootstrap_metrics.json", orient="records", indent=2)
    pool = pd.read_csv(results / "prospective_frozen_predictions.csv", low_memory=False)
    pool["hla_clean"] = pool["hla_allele"].astype(str)
    loh = shared_score_loh_simulation(pool, n_patients=200)
    loh.to_csv(results / "shared_score_loh_sensitivity.csv", index=False)
    stochastic_loh = stochastic_shared_score_loh_simulation(
        pool,
        n_patients=args.stochastic_patients,
        seed=args.seed,
    )
    stochastic_loh.to_csv(
        results / "stochastic_shared_score_loh_sensitivity.csv", index=False
    )
    small = pool.head(10).copy()
    small["presentation_probability"] = small["presentation_score_frozen"]
    small["loh_probability"] = 0.10
    small["hla_clean"] = small["hla_clean"].astype(str)
    exact, objective = exact_small_pool_hierarchical_selection(
        small, 5, threshold=3
    )
    (results / "exact_dp_reference.json").write_text(
        json.dumps(
            {
                "status": "bounded_exact_reference",
                "candidate_count": len(small),
                "capacity": 5,
                "threshold": 3,
                "objective_probability_at_least_threshold": objective,
                "selected_rows": exact.index.tolist(),
                "not_large_pool_milp": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Generated hardened validation outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
