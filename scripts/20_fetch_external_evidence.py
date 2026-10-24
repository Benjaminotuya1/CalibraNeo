"""Fetch official IEDB evidence snapshots into a separate, harmonized layer.

The data are not merged into MASTER_CALIBRANEO_DATASET.csv because IEDB assay
records describe different biological endpoints from the local presentation
labels. They are normalized into a stable evidence table and joined by exact
peptide/HLA keys only during downstream evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


API = "https://query-api.iedb.org"


def fetch(endpoint: str, limit: int, pages: int) -> list[dict]:
    rows: list[dict] = []
    for page in range(pages):
        # The public API currently accepts `limit` but rejects offset-based
        # pagination; use one bounded snapshot until its cursor API is stable.
        if page:
            break
        query = urlencode({"limit": limit})
        request = Request(
            f"{API}/{endpoint}?{query}",
            headers={"User-Agent": "CalibraNeo-research/1.0"},
        )
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
        if not payload:
            break
        rows.extend(payload)
        if len(payload) < limit:
            break
        time.sleep(0.25)
    return rows


def normalize(rows: list[dict], assay_type: str) -> pd.DataFrame:
    records = []
    for row in rows:
        peptide = str(row.get("epitope__name") or "").strip().upper()
        hla = str(row.get("mhc_restriction__name") or "").strip()
        if not peptide or not hla or any(base not in "ACDEFGHIKLMNPQRSTVWY" for base in peptide):
            continue
        qualitative = str(row.get("assay__qualitative_measurement") or "").lower()
        records.append(
            {
                "source": "IEDB",
                "assay_type": assay_type,
                "assay_id": row.get("assay_id"),
                "peptide": peptide,
                "hla_allele": hla,
                "mhc_class": row.get("mhc_restriction__class"),
                "qualitative_measurement": qualitative,
                "is_positive": qualitative in {"positive", "positive-high", "positive-low"},
                "response_measured": row.get("assay__response_measured"),
                "reference_pmid": row.get("reference__pmid"),
            }
        )
    return pd.DataFrame.from_records(records).drop_duplicates()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data_external/iedb")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--pages", type=int, default=1)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_tables = []
    manifest = {
        "source": API,
        "query_limits": {"limit": args.limit, "pages": args.pages},
        "tables": {},
        "note": "Separate evidence layer; not merged into the local binary-label master table.",
    }
    for endpoint, assay_type, filename in [
        ("tcell_export", "tcell", "iedb_tcell_normalized.csv"),
        ("mhc_export", "mhc_ligand", "iedb_mhc_normalized.csv"),
    ]:
        rows = fetch(endpoint, args.limit, args.pages)
        table = normalize(rows, assay_type)
        path = output / filename
        table.to_csv(path, index=False)
        manifest["tables"][filename] = {
            "rows_raw": len(rows),
            "rows_normalized": len(table),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        all_tables.append(table)
    combined = pd.concat(all_tables, ignore_index=True)
    combined_path = output / "iedb_evidence_normalized.csv"
    combined.to_csv(combined_path, index=False)
    manifest["tables"][combined_path.name] = {
        "rows": len(combined),
        "sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
