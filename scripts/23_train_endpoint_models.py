"""Train available endpoint models and write an explicit model/data status manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibraneo.task_models import train_endpoint_model


def main() -> int:
    models = ROOT / "models"
    raw = ROOT / "data_raw"
    processed = ROOT / "data_processed"
    manifest_path = models / "endpoint_model_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("trained", {})
        manifest.setdefault("not_trained", {})
    else:
        manifest = {"trained": {}, "not_trained": {}}

    ott = pd.read_csv(
        raw / "big_benchmarks" / "ott_clinical_vaccine" / "immunizing-peptides-with-tcell.csv"
    )
    ott["peptide"] = ott["Immunizing peptide sequence"].astype(str).str.replace(
        r"[^A-Z]", "", regex=True
    )
    ott["t_cell_response"] = pd.to_numeric(
        ott["Minimal epitope CD8+ response"], errors="coerce"
    )
    ott["patient_group"] = ott["Patient ID"].fillna(ott["Patient"]).astype(str)
    bundle = train_endpoint_model(
        ott,
        peptide_column="peptide",
        label_column="t_cell_response",
        group_columns=["patient_group"],
        endpoint="immunogenicity_ott",
        output_path=models / "calibraneo_immunogenicity_ott.pkl",
        extra_features=["Gene Expression"],
    )
    deployable_bundle = train_endpoint_model(
        ott,
        peptide_column="peptide",
        label_column="t_cell_response",
        group_columns=["patient_group"],
        endpoint="immunogenicity_ott_deployable",
        output_path=models / "calibraneo_immunogenicity_ott_deployable.pkl",
    )
    manifest["trained"]["immunogenicity_ott"] = {
        "path": "models/calibraneo_immunogenicity_ott.pkl",
        "source": "Ott clinical melanoma vaccine table",
        "endpoint": "minimal epitope CD8 response",
        "metrics": bundle["metrics"],
        "limitations": [
            "small cohort",
            "patient grouped holdout only",
            "not a pan-cancer immunogenicity model",
        ],
    }
    manifest["trained"]["immunogenicity_ott_deployable"] = {
        "path": "models/calibraneo_immunogenicity_ott_deployable.pkl",
        "source": "Ott clinical melanoma vaccine table",
        "endpoint": "minimal epitope CD8 response",
        "metrics": deployable_bundle["metrics"],
        "limitations": [
            "sequence-only deployment model",
            "small cohort",
            "not a pan-cancer immunogenicity model",
        ],
    }

    zaghla = pd.read_csv(processed / "MASTER_CALIBRANEO_DATASET.csv", low_memory=False)
    zaghla = zaghla[zaghla["dataset_source"].eq("Zaghla_2026_JITC_Benchmark")].copy()
    zaghla["study_group"] = zaghla["mutation"].astype(str) + "|" + zaghla["hla_allele"].astype(str)
    ms_bundle = train_endpoint_model(
        zaghla,
        peptide_column="peptide",
        label_column="label_presented",
        group_columns=["study_group"],
        endpoint="immunopeptidomics_zaghla",
        output_path=models / "calibraneo_immunopeptidomics_zaghla.pkl",
    )
    manifest["trained"]["immunopeptidomics_zaghla"] = {
        "path": "models/calibraneo_immunopeptidomics_zaghla.pkl",
        "source": "Zaghla-derived MS experiment table",
        "endpoint": "mutant peptide detected in MS experiment",
        "metrics": ms_bundle["metrics"],
        "limitations": [
            "small study",
            "not a public HLA Ligand Atlas/SysteMHC Atlas snapshot",
            "label-derived binding-rank field excluded from features",
        ],
    }

    unavailable = {
        "immunogenicity_tesla_neodb_nepdb": "No licensed/downloaded normalized files are present locally.",
        "immunopeptidomics_hla_ligand_atlas_systemhc": "No normalized downloadable snapshot is present locally.",
        "translation_rpfdb": "No normalized Ribo-seq/RNA-seq translation table is present locally.",
    }
    manifest["not_trained"] = unavailable
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
