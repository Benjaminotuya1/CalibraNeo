"""Benchmark circRNA sequence constraints against a permitted construct table.

The input table must contain construct_id and rna_sequence. Optional columns
are bsj_index, orf_start_index, ires_sequence, circularisation_efficiency, and
translation_efficiency. No labels are inferred from publication status.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from calibraneo.platform import circ_rna_constraints


def _numeric_correlation(frame: pd.DataFrame, outcome: str) -> float | None:
    if outcome not in frame or frame[outcome].notna().sum() < 3:
        return None
    valid = frame[["circRNA_constraint_score", outcome]].dropna()
    if len(valid) < 3 or valid[outcome].nunique() < 2:
        return None
    return float(valid["circRNA_constraint_score"].corr(valid[outcome], method="spearman"))


def benchmark(input_path: Path, output_path: Path) -> dict[str, object]:
    records = pd.read_csv(input_path)
    required = {"construct_id", "rna_sequence"}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"Missing benchmark columns: {', '.join(sorted(missing))}")
    rows = []
    for row in records.to_dict("records"):
        features = circ_rna_constraints(
            str(row["rna_sequence"]),
            bsj_index=None if pd.isna(row.get("bsj_index")) else int(row["bsj_index"]),
            orf_start_index=(
                0 if pd.isna(row.get("orf_start_index"))
                else int(row["orf_start_index"])
            ),
            ires_sequence=(
                None if pd.isna(row.get("ires_sequence"))
                else str(row["ires_sequence"])
            ),
        )
        score = (
            0.40 * float(features["bsj_in_frame"])
            + 0.30 * float(features["ires_compatibility"])
            + 0.20 * float(features["rolling_translation_proxy"])
            + 0.10 * max(0.0, 1.0 - abs(float(features["gc_fraction"]) - 0.55) / 0.45)
        )
        rows.append(
            {
                "construct_id": row["construct_id"],
                **features,
                "circRNA_constraint_score": score,
                **{
                    name: row[name]
                    for name in ("circularisation_efficiency", "translation_efficiency")
                    if name in row
                },
            }
        )
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    correlations = {
        outcome: _numeric_correlation(result, outcome)
        for outcome in ("circularisation_efficiency", "translation_efficiency")
    }
    return {
        "construct_count": int(len(result)),
        "outcome_correlations": correlations,
        "output": str(output_path),
        "interpretation": (
            "Retrospective sequence benchmark; correlations are omitted when "
            "fewer than three measured outcomes are available."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = benchmark(args.input, args.output)
    print(summary)


if __name__ == "__main__":
    main()
