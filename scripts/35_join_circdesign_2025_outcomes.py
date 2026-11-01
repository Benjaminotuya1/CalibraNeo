"""Join exact 2025 circRNA sequence labels with construct-level outcomes.

The source workbook reports a generic ``circRNA`` group for the requested
panels, so those measurements are retained in an audit file but are not
assigned to named constructs without explicit source attribution.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "circdesign_data" / "raw_papers"
BENCHMARK = ROOT / "circdesign_data" / "benchmark"
PDF_TEXT = RAW / "41551_2025_1344_MOESM1_ESM.pdf.txt"
WORKBOOK = RAW / "41551_2025_1344_MOESM3_ESM.xlsx"
DOI = "10.1038/s41551-025-01344-5"


REQUESTED = [
    "circRNA-SIINFEKL",
    "circRNA-OVA",
    "circRNA-Adpgk",
    "circRNA-E7",
    "circRNA-Trp2/gp100",
    "circRNA-RBD",
]


def clean_sequence(value: str) -> str:
    return re.sub(r"[^ACGTU]", "", value.upper())


def extract_pdf_sequence(label: str) -> str | None:
    text = PDF_TEXT.read_text(encoding="utf-8", errors="ignore")
    start = text.rfind("Supplementary Table 1 | RNA sequences.")
    end = text.index("Supplementary Table 2 | Comparison", start)
    table = text[start:end]
    lines = table.splitlines()
    start_index = None
    label_line_count = 1
    sequence_parts: list[str] = []
    for index, raw_line in enumerate(lines):
        normalized = re.sub(r"\s+", "", raw_line)
        if normalized == label:
            start_index = index
            break
        if index + 1 < len(lines):
            next_normalized = re.sub(r"\s+", "", lines[index + 1])
            if normalized + next_normalized == label:
                start_index = index
                label_line_count = 2
                break
        if normalized.startswith(label) and normalized[len(label) :]:
            suffix = normalized[len(label) :]
            if re.fullmatch(r"[ACGTU]+", suffix):
                start_index = index
                sequence_parts.append(suffix)
                break
    if start_index is None:
        return None
    for raw_line in lines[start_index + label_line_count :]:
        normalized = re.sub(r"\s+", "", raw_line).upper()
        if not normalized:
            continue
        if normalized.startswith(("CIRCRNA", "LINE1", "RBM3", "CRTMV")):
            break
        if re.fullmatch(r"[ACGTU]+", normalized):
            sequence_parts.append(normalized)
        elif sequence_parts:
            break
    sequence = "".join(sequence_parts)
    return sequence or None


def write_exact_fasta(sequence: str, construct_id: str) -> str:
    filename = f"{construct_id}.fasta"
    path = BENCHMARK / "fasta" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">{construct_id}\n{sequence}\n", encoding="utf-8")
    return f"benchmark/fasta/{filename}"


def extract_outcome_audit() -> dict[str, object]:
    workbook = load_workbook(WORKBOOK, data_only=True, read_only=False)
    expression_sheet = workbook["Fig. 5"]
    stability_sheet = workbook["Fig. 2"]

    # Fig. 5c: row 28 is the 10 ug dose; B:F is the generic circRNA group.
    expression_values = [
        expression_sheet.cell(28, column).value for column in range(2, 7)
    ]
    # Fig. 2j: rows 155-158 are the 4 C block; I:K are circRNA replicates.
    stability_values = [
        stability_sheet.cell(158, column).value for column in range(9, 12)
    ]
    audit_path = BENCHMARK / "2025_outcome_extraction_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_sheet",
                "source_panel",
                "source_group",
                "condition",
                "raw_values",
                "mean",
                "assignment_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_sheet": "Fig. 5",
                "source_panel": "Fig5c",
                "source_group": "circRNA",
                "condition": "10 ug",
                "raw_values": "|".join(str(v) for v in expression_values),
                "mean": sum(expression_values) / len(expression_values),
                "assignment_status": "NOT_ASSIGNED_TO_NAMED_CONSTRUCT",
            }
        )
        writer.writerow(
            {
                "source_sheet": "Fig. 2",
                "source_panel": "Fig2j",
                "source_group": "circRNA",
                "condition": "Day 7, 4 C",
                "raw_values": "|".join(str(v) for v in stability_values),
                "mean": sum(stability_values) / len(stability_values),
                "assignment_status": "NOT_ASSIGNED_TO_NAMED_CONSTRUCT",
            }
        )
    return {
        "expression_values": expression_values,
        "stability_values": stability_values,
    }


def main() -> None:
    outcomes = extract_outcome_audit()
    rows = []
    for construct_id in REQUESTED:
        sequence = extract_pdf_sequence(construct_id)
        sequence_path = write_exact_fasta(sequence, construct_id) if sequence else "NOT_FOUND"
        rows.append(
            {
                "construct_id": construct_id,
                "sequence_fasta": sequence_path,
                "expression_mfi_10ug": "NOT_FOUND",
                "stability_day7_4C": "NOT_FOUND",
                "source_doi": DOI,
            }
        )
    output = BENCHMARK / "constructs.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "construct_id",
                "sequence_fasta",
                "expression_mfi_10ug",
                "stability_day7_4C",
                "source_doi",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("master_rows=", len(rows))
    print("exact_sequence_matches=", sum(row["sequence_fasta"] != "NOT_FOUND" for row in rows))
    print("expression_circRNA_replicates=", outcomes["expression_values"])
    print("stability_circRNA_replicates=", outcomes["stability_values"])
    print("paired_construct_outcomes=0")


if __name__ == "__main__":
    main()
