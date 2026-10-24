"""Patient-specific input validation and candidate assembly."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


def read_vcf(path: str | Path) -> pd.DataFrame:
    """Read simple SNV VCF records; complex variants are retained for review."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            raise ValueError("VCF row must contain at least 5 tab-separated fields")
        rows.append(
            {
                "chrom": parts[0],
                "position": int(parts[1]),
                "ref": parts[3].upper(),
                "alt": parts[4].split(",")[0].upper(),
                "variant_type": "SNV"
                if len(parts[3]) == len(parts[4].split(",")[0]) == 1
                else "non_snv_review_required",
            }
        )
    return pd.DataFrame(rows)


def read_hla(path: str | Path) -> list[str]:
    alleles = [
        re.sub(r"\s+", "", line.strip()).upper()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not alleles:
        raise ValueError("HLA file contains no alleles")
    return sorted(set(alleles))


def read_expression(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"gene", "expression_tpm"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Expression file missing columns: {', '.join(sorted(missing))}")
    frame["expression_tpm"] = pd.to_numeric(frame["expression_tpm"], errors="coerce")
    return frame.dropna(subset=["gene", "expression_tpm"])


def read_loh(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"hla_allele", "loh_probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"LOH file missing columns: {', '.join(sorted(missing))}")
    frame["loh_probability"] = pd.to_numeric(frame["loh_probability"], errors="coerce").clip(0, 1)
    return frame.dropna(subset=["hla_allele", "loh_probability"])


def assemble_patient_candidates(
    candidates: pd.DataFrame,
    *,
    hla_alleles: list[str],
    expression: pd.DataFrame | None = None,
    loh: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Filter candidates to patient HLA and annotate expression/LOH uncertainty."""
    required = {"peptide", "hla_allele"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table missing columns: {', '.join(sorted(missing))}")
    result = candidates.copy()
    result["hla_allele"] = result["hla_allele"].astype(str).str.replace(" ", "", regex=False).str.upper()
    result = result[result["hla_allele"].isin(set(hla_alleles))].copy()
    if expression is not None and "gene" in result:
        result = result.merge(expression[["gene", "expression_tpm"]], on="gene", how="left")
        result["expression_observed"] = result["expression_tpm"].notna()
    else:
        result["expression_tpm"] = pd.NA
        result["expression_observed"] = False
    if loh is not None:
        result = result.merge(loh[["hla_allele", "loh_probability"]], on="hla_allele", how="left")
    else:
        result["loh_probability"] = pd.NA
    result["patient_input_status"] = "patient_specific_inputs_present"
    result.loc[result["expression_observed"].eq(False), "patient_input_status"] = "expression_missing"
    return result.reset_index(drop=True)
