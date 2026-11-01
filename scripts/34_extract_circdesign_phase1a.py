"""Extract explicitly printed circRNA and IRES sequences from supplied papers.

This script is intentionally conservative: it does not infer sequences,
outcomes, ORF boundaries, or frame status when the source does not provide
them unambiguously.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "circdesign_data" / "raw_papers"
DATA = ROOT / "circdesign_data"
DOI_2025 = "10.1038/s41551-025-01344-5"
DOI_2026 = "10.1038/s41551-026-01725-4"
DOI_2023 = "10.1101/2023.07.09.548293"


def clean_sequence(text: str) -> str:
    return re.sub(r"[^ACGTU]", "", text.upper())


def alphabet(sequence: str) -> str:
    symbols = set(sequence)
    if symbols <= {"A", "C", "G", "U"}:
        return "RNA"
    if symbols <= {"A", "C", "G", "T"}:
        return "DNA"
    if symbols <= {"A", "C", "G", "T", "U"}:
        return "MIXED"
    return "INVALID"


def fasta_text(records: list[tuple[str, str]]) -> str:
    chunks = []
    for identifier, sequence in records:
        chunks.append(f">{identifier}\n{sequence}\n")
    return "".join(chunks)


def extract_2025() -> tuple[list[dict[str, str]], list[tuple[str, str]]]:
    text = (RAW / "41551_2025_1344_MOESM1_ESM.pdf.txt").read_text(
        encoding="utf-8", errors="ignore"
    )
    start = text.rfind("Supplementary Table 1 | RNA sequences.")
    end = text.index("Supplementary Table 2 | Comparison", start)
    table = text[start:end]
    names = [
        "LINE1 (ORF2-138-86)",
        "Rbm3-22NT",
        "crTMV (IRESmp75)",
        "LINE1-SIINFEKL",
        "crTMV-SIINFEKL",
        "Rbm3-SIINFEKL",
        "circRNA-ISQ",
        "circRNA-FLAG",
        "circRNA-Adpgk",
        "circRNA-E7",
        "circRNA-trp2177-190/gp10023-33",
        "circRNA-trp1103-132/trp1455-463",
        "circRNA-RBD440-459",
        "circRNA-OVA",
    ]
    records = []
    fasta = []
    def flexible_marker(name: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9]+", name)
        return r"[\s\-_/()]*".join(re.escape(token) for token in tokens)

    matches = []
    for order, name in enumerate(names):
        match = re.search(rf"(?m)^\s*{flexible_marker(name)}", table)
        if match:
            matches.append((match.start(), match.end(), order, name))
    matches.sort()
    for index, (_, end, _, name) in enumerate(matches):
        next_start = matches[index + 1][0] if index + 1 < len(matches) else len(table)
        sequence = clean_sequence(table[end:next_start])
        status = "verified_from_supplementary_table" if sequence else "NOT_FOUND_SEQUENCE"
        records.append(
            {
                "construct_id": name,
                "source_doi": DOI_2025,
                "sequence_fasta": (
                    f"benchmark/fasta/{name.replace('/', '_')}.fasta"
                    if sequence
                    else "NOT_FOUND"
                ),
                "outcome_type": (
                    "NOT_FOUND"
                ),
                "outcome_value": "NOT_FOUND",
                "outcome_units": "NOT_FOUND",
                "notes": (
                    f"{status}; sequence_alphabet={alphabet(sequence)}; "
                    "ORF frame status NOT_DETERMINED because BSJ/ORF boundaries "
                    "are not unambiguously specified in the extracted table; "
                    "the paper reports a 32% paper-level circularisation rate, "
                    "but it is not assigned to this row without construct-level attribution"
                ),
                "sequence_alphabet": alphabet(sequence) if sequence else "NOT_FOUND",
                "sequence_symbol_validation": (
                    "PASS" if sequence and re.fullmatch(r"[ACGTU]+", sequence) else "NOT_FOUND"
                ),
                "orf_frame_status": "NOT_DETERMINED",
                "source_file": "41551_2025_1344_MOESM1_ESM.pdf",
                "source_location": "Supplementary Table 1",
                "extraction_status": status,
            }
        )
        if sequence:
            fasta.append((name, sequence))
    return records, fasta


def extract_table_sequences(
    text: str, table_title: str, next_title: str
) -> list[tuple[str, str]]:
    start = text.rfind(table_title)
    end = text.find(next_title, start + len(table_title))
    if end == -1:
        end = len(text)
    lines = text[start:end].splitlines()
    records: list[tuple[str, str]] = []
    current: str | None = None
    sequence_parts: list[str] = []
    pending_label: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith(table_title) or line in {
            "IRES name Sequence (5’-3’)",
            "Feature Schematic diagram or/and sequence (5’-3’)",
            "Name Sequence (5’-3’)",
        }:
            pending_label = []
            continue
        compact = clean_sequence(line)
        inline = re.match(r"^([A-Za-z0-9()αβ\-]+)\s+([ACGTU]{20,})$", line.upper())
        if inline:
            if current and sequence_parts:
                records.append((current, "".join(sequence_parts)))
            current = inline.group(1)
            sequence_parts = [inline.group(2)]
            pending_label = []
            continue
        if re.fullmatch(r"[ACGTU\s]+", line.upper()) and (
            len(compact) >= 20 or current is not None
        ):
            if current is None:
                current = "_".join(pending_label)
                pending_label = []
            sequence_parts.append(compact)
            continue
        if current and sequence_parts:
            records.append((current, "".join(sequence_parts)))
            sequence_parts = []
            current = None
        pending_label.append(line.replace(" ", "_"))
    if current and sequence_parts:
        records.append((current, "".join(sequence_parts)))
    normalized_ids = {
        "264": "MNSV-264",
        "J": "Human_echovirus_11",
        "SwineVesic_ular": "SwineVesicular",
    }
    normalized = []
    pte_count = 0
    for identifier, sequence in records:
        if identifier in {"(PTE)", "PTE"}:
            pte_count += 1
            identifier = "PEMV2_(PTE)" if pte_count == 1 else "SCV_(PTE-TSS)"
        elif identifier.startswith("EV-D94_"):
            identifier = "EV-D94"
        else:
            identifier = normalized_ids.get(identifier, identifier)
        identifier = identifier.replace("MNSV-_264", "MNSV-264")
        normalized.append((identifier, sequence))
    return normalized


def write_sequence_table(
    records: list[tuple[str, str]], csv_path: Path, fasta_path: Path, source: str
) -> None:
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text(fasta_text(records), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence_id",
                "source_doi",
                "sequence_fasta",
                "sequence_alphabet",
                "length_nt",
                "known_efficiency",
                "notes",
            ],
        )
        writer.writeheader()
        for identifier, sequence in records:
            writer.writerow(
                {
                    "sequence_id": identifier,
                    "source_doi": source,
                    "sequence_fasta": fasta_path.name,
                    "sequence_alphabet": alphabet(sequence),
                    "length_nt": len(sequence),
                    "known_efficiency": "NOT_FOUND",
                    "notes": "Explicit sequence text extracted; measured efficiency not assigned.",
                }
            )


def main() -> None:
    records, fasta = extract_2025()
    fasta_dir = DATA / "benchmark" / "fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    for identifier, sequence in fasta:
        filename = identifier.replace("/", "_") + ".fasta"
        (fasta_dir / filename).write_text(
            fasta_text([(identifier, sequence)]), encoding="utf-8"
        )

    benchmark_path = DATA / "benchmark" / "constructs.csv"
    fields = list(records[0])
    with benchmark_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    ires_text = (RAW / "41551_2026_1725_MOESM1_ESM.pdf.txt").read_text(
        encoding="utf-8", errors="ignore"
    )
    ires = extract_table_sequences(
        ires_text, "Supplementary Table 1 | IRES sequences.", "Supplementary Table 2"
    )
    cite = extract_table_sequences(
        ires_text, "Supplementary Table 2 | CITE and H2SL sequences.", "Supplementary Table 3"
    )
    write_sequence_table(
        ires,
        DATA / "ires" / "ires_reference.csv",
        DATA / "ires" / "All_IRES.fa",
        DOI_2026,
    )
    write_sequence_table(
        cite,
        DATA / "ires" / "cite_reference.csv",
        DATA / "ires" / "CITE_H2SL.fa",
        DOI_2026,
    )

    preprint_rows = []
    for identifier in ("CR1", "CR2", "CR3", "CR4", "CR5", "CR6"):
        preprint_rows.append(
            {
                "construct_id": identifier,
                "source_doi": DOI_2023,
                "sequence_fasta": "NOT_FOUND_SEQUENCE",
                "outcome_type": "NOT_FOUND",
                "outcome_value": "NOT_FOUND",
                "outcome_units": "NOT_FOUND",
                "notes": "Construct name appears in the supplied PDF context, but exact nucleotide sequence was not found in text or supplied tables.",
                "sequence_alphabet": "NOT_FOUND",
                "sequence_symbol_validation": "NOT_FOUND",
                "orf_frame_status": "NOT_DETERMINED",
                "source_file": "2023.07.09.548293v2.full.pdf",
                "source_location": "CR1-CR6 search",
                "extraction_status": "NOT_FOUND_SEQUENCE",
            }
        )
    with (DATA / "benchmark" / "preprint_2023_construct_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(preprint_rows[0]))
        writer.writeheader()
        writer.writerows(preprint_rows)

    print(
        {
            "construct_rows": len(records),
            "construct_sequences": sum(bool(row["sequence_fasta"] != "NOT_FOUND") for row in records),
            "ires_sequences": len(ires),
            "cite_sequences": len(cite),
            "cr1_cr6_sequence_status": "NOT_FOUND_SEQUENCE",
        }
    )


if __name__ == "__main__":
    main()
