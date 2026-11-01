"""Train an exploratory IRES activity model.

These models are evidence-limited: the IRES benchmark contains selected
extremes. Metrics are reported as exploratory estimates, not validated
biological performance.
"""

from __future__ import annotations

import csv
import importlib
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict


ROOT = Path(__file__).resolve().parents[1]
IRES_PATH = ROOT / "circdesign_data" / "ires" / "ires_benchmark.csv"
FIGURE_PATH = ROOT / "manuscript_assets" / "figures" / "ires_model_performance.png"
KMER_NAMES = ["".join(kmer) for kmer in product("ACGT", repeat=3)]


def sequence_features(sequence: str) -> dict[str, float]:
    gc = (sequence.count("G") + sequence.count("C")) / len(sequence)
    features = {"gc_content": gc, "length": float(len(sequence))}
    for kmer in KMER_NAMES:
        features[f"kmer_{kmer}"] = sum(
            sequence[index : index + 3] == kmer
            for index in range(len(sequence) - 2)
        ) / len(sequence)
    return features


def mfe(sequence: str) -> float | None:
    """Return linear RNA MFE when ViennaRNA is installed, otherwise None."""
    try:
        RNA = importlib.import_module("RNA")
    except ImportError:
        return None
    structure, energy = RNA.fold(sequence.replace("T", "U"))
    del structure
    return float(energy)


def vienna_version() -> str:
    try:
        RNA = importlib.import_module("RNA")
    except ImportError:
        return "unavailable"
    return str(getattr(RNA, "__version__", "unknown"))


def make_feature_matrix(sequences: list[str]) -> tuple[np.ndarray, list[str]]:
    rows = []
    mfe_values = [mfe(sequence) for sequence in sequences]
    if any(value is None for value in mfe_values):
        raise RuntimeError("ViennaRNA is required for the MFE feature")
    for sequence, mfe_value in zip(sequences, mfe_values):
        features = sequence_features(sequence)
        features["mfe"] = float(mfe_value)
        rows.append(features)
    names = ["gc_content", "length", *[f"kmer_{kmer}" for kmer in KMER_NAMES], "mfe"]
    return np.array([[row[name] for name in names] for row in rows]), names


def train_ires_model() -> None:
    with IRES_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sequences = [row["sequence_dna"] for row in rows]
    raw_targets = np.array([float(row["activity_mean"]) for row in rows])
    print(
        "activity_mean summary: "
        f"min={raw_targets.min():.6f}, "
        f"max={raw_targets.max():.6f}, "
        f"median={np.median(raw_targets):.6f}, "
        f"mean={raw_targets.mean():.6f}"
    )
    if raw_targets.max() > 50:
        targets = np.log2(raw_targets + 1)
        target_name = "log2_activity"
        print("target_transform=log2(activity_mean + 1)")
    else:
        targets = raw_targets
        target_name = "activity_mean"
        print("target_transform=none")
    features, feature_names = make_feature_matrix(sequences)

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    predictions = cross_val_predict(model, features, targets, cv=folds, n_jobs=-1)
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(targets, predictions, alpha=0.75, edgecolor="none")
    limits = [min(targets.min(), predictions.min()), max(targets.max(), predictions.max())]
    axis.plot(limits, limits, linestyle="--", color="black", linewidth=1)
    axis.set_xlabel(f"Actual {target_name}")
    axis.set_ylabel(f"Cross-validated predicted {target_name}")
    axis.set_title("IRES activity model (5-fold cross-validation)")
    axis.text(
        0.05,
        0.95,
        f"R² = {r2:.3f}\nMAE = {mae:.2f}",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)

    model.fit(features, targets)
    importance_order = np.argsort(model.feature_importances_)[::-1][:10]

    print("IRES MODEL")
    print(f"  n={len(targets)}, feature_count={len(feature_names)}")
    print(f"  5-fold cross-validated R2={r2:.6f}")
    print(f"  5-fold cross-validated MAE={mae:.6f}")
    print("  top_10_feature_importances:")
    for index in importance_order:
        print(f"    {feature_names[index]}={model.feature_importances_[index]:.6f}")
    print(f"  plot_saved={FIGURE_PATH}")


def main() -> None:
    print(f"ViennaRNA version={vienna_version()}")
    train_ires_model()


if __name__ == "__main__":
    main()
