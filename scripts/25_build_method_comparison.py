"""Build an honest comparator availability/results table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    benchmark = pd.read_csv(results / "table1_clinical_benchmark_vs_9tools.csv")
    rows = []
    local_names = {
        "NetMHCpan": "NetMHC-4 (Uncalibrated Rank)",
        "MHCflurry": "MHCflurryPres (Presentation Score)",
        "BigMHC": "BigMHC_EL (Deep Learning Baseline)",
    }
    for display, local in local_names.items():
        match = benchmark[benchmark["Method"].eq(local)]
        if match.empty:
            rows.append({"method": display, "status": "not_evaluated", "source": "local benchmark"})
        else:
            row = match.iloc[0]
            rows.append(
                {
                    "method": display,
                    "status": "evaluated_local_benchmark",
                    "source": "Nature clinical benchmark",
                    "auc_roc": row["AUC_ROC"],
                    "auprc": row["AUPRC"],
                }
            )
    rows.extend(
        {
            "method": name,
            "status": "not_evaluated",
            "source": "no reproducible local run or compatible output",
            "limitation": limitation,
        }
        for name, limitation in {
            "DeepNeo": "Model/tool output is not present locally.",
            "NeoDisc": "Workflow was not run on the same frozen candidates.",
            "pVACtools": "Pipeline was not run; external dependencies and inputs are required.",
        }.items()
    )
    rows.append(
        {
            "method": "CalibraNeo",
            "status": "evaluated_local_benchmark",
            "source": "Nature clinical benchmark",
            "auc_roc": float(
                benchmark.loc[
                    benchmark["Method"].eq("CalibraNeo (Biophysics-Only Calibrated)"),
                    "AUC_ROC",
                ].iloc[0]
            ),
            "auprc": float(
                benchmark.loc[
                    benchmark["Method"].eq("CalibraNeo (Biophysics-Only Calibrated)"),
                    "AUPRC",
                ].iloc[0]
            ),
        }
    )
    pd.DataFrame(rows).to_csv(results / "method_comparison_status.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
