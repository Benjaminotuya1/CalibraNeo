"""Extract conservative 2023 half-life outcomes and an IRES activity benchmark."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "circdesign_data" / "raw_papers"
BENCHMARK = ROOT / "circdesign_data" / "benchmark"
IRES = ROOT / "circdesign_data" / "ires"
FASTA = BENCHMARK / "fasta"
DOI = "10.1101/2023.07.09.548293"
SEQUENCE_SOURCE = "anie74170-sup-0002-suppmat.xlsx"


def load_existing_constructs() -> list[dict[str, str]]:
    path = BENCHMARK / "constructs.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def extract_rabies_sequences() -> dict[str, str]:
    workbook = RAW / SEQUENCE_SOURCE
    if not workbook.exists():
        raise FileNotFoundError(f"Required supplement not found: {workbook}")

    sheet = load_workbook(workbook, data_only=True, read_only=True)["Rabies seq"]
    sequences: dict[str, str] = {}
    expected = {"CR1", "CR2", "CR3", "CR4", "CR5", "CR6", "CR-Ther", "CR-Novo"}
    for row in sheet.iter_rows(values_only=True):
        construct_id = row[0]
        if construct_id in expected:
            components = row[1:7]
            if any(not isinstance(component, str) for component in components):
                raise ValueError(f"{construct_id} has a missing B-G sequence component")
            sequence = re.sub(r"\s+", "", "".join(components)).upper()
            if not sequence or set(sequence) - set("ACGT"):
                raise ValueError(f"{construct_id} has an invalid DNA alphabet")
            if len(sequence) != 2504:
                raise ValueError(
                    f"{construct_id} full-length sequence has length {len(sequence)}, expected 2504"
                )
            if construct_id in sequences and sequences[construct_id] != sequence:
                raise ValueError(f"Duplicate conflicting sequence for {construct_id}")
            sequences[construct_id] = sequence

    if set(sequences) != expected:
        raise ValueError(f"Missing rabies sequences: {sorted(expected - set(sequences))}")

    FASTA.mkdir(parents=True, exist_ok=True)
    for construct_id, sequence in sequences.items():
        path = FASTA / f"{construct_id}.fasta"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(f">{construct_id}|source={SEQUENCE_SOURCE}|alphabet=DNA\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")
    return sequences


def add_2023_rows(
    rows: list[dict[str, str]], rabies_sequences: dict[str, str]
) -> list[dict[str, str]]:
    fields = [
        "construct_id",
        "sequence_fasta",
        "expression_mfi_10ug",
        "stability_day7_4C",
        "in_solution_half_life_h",
        "in_cell_half_life_h",
        "source_doi",
    ]
    normalized = []
    existing_2023_ids = {
        "CR1",
        "CR2",
        "CR3",
        "CR4",
        "CR5",
        "CR6",
        "CR-Novo",
        "CR-Ther",
    }
    for row in rows:
        if row["construct_id"] in existing_2023_ids:
            continue
        normalized.append(
            {
                "construct_id": row["construct_id"],
                "sequence_fasta": row["sequence_fasta"],
                "expression_mfi_10ug": row.get("expression_mfi_10ug", "NOT_FOUND"),
                "stability_day7_4C": row.get("stability_day7_4C", "NOT_FOUND"),
                "in_solution_half_life_h": "NOT_FOUND",
                "in_cell_half_life_h": "NOT_FOUND",
                "source_doi": row["source_doi"],
            }
        )

    outcomes = {
        "CR1": ("NOT_FOUND", "24.34"),
        "CR2": ("NOT_FOUND", "NOT_FOUND"),
        "CR3": ("NOT_FOUND", "19.31"),
        "CR4": ("NOT_FOUND", "NOT_FOUND"),
        "CR5": ("NOT_FOUND", "9.18"),
        "CR6": ("NOT_FOUND", "NOT_FOUND"),
        "CR-Novo": ("9.14", "10.21"),
        "CR-Ther": ("9.68", "NOT_FOUND"),
    }
    for construct_id, (solution, cell) in outcomes.items():
        normalized.append(
            {
                "construct_id": construct_id,
                "sequence_fasta": (
                    f"benchmark/fasta/{construct_id}.fasta"
                    if construct_id in rabies_sequences
                    else "NOT_FOUND_SEQUENCE"
                ),
                "expression_mfi_10ug": "NOT_FOUND",
                "stability_day7_4C": "NOT_FOUND",
                "in_solution_half_life_h": solution,
                "in_cell_half_life_h": cell,
                "source_doi": DOI,
            }
        )

    output = BENCHMARK / "constructs.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)
    return normalized


def write_2023_outcome_audit() -> None:
    rows = [
        {
            "construct_group": "CR1-CR3",
            "outcome_type": "in_solution_half_life",
            "reported_value": "15.56-20.62",
            "units": "hours",
            "assignment_status": "GROUP_RANGE_NOT_MAPPED_TO_INDIVIDUALS",
            "source_location": "2023 preprint, Results, Fig. 2d",
        },
        {
            "construct_group": "CR-Novo",
            "outcome_type": "in_solution_half_life",
            "reported_value": "9.14",
            "units": "hours",
            "assignment_status": "EXACT",
            "source_location": "2023 preprint, Results, Fig. 2d",
        },
        {
            "construct_group": "CR-Ther",
            "outcome_type": "in_solution_half_life",
            "reported_value": "9.68",
            "units": "hours",
            "assignment_status": "EXACT",
            "source_location": "2023 preprint, Results, Fig. 2d",
        },
        {
            "construct_group": "CR1",
            "outcome_type": "in_cell_half_life",
            "reported_value": "24.34",
            "units": "hours",
            "assignment_status": "EXACT",
            "source_location": "2023 preprint, Results, Fig. 2e",
        },
        {
            "construct_group": "CR3",
            "outcome_type": "in_cell_half_life",
            "reported_value": "19.31",
            "units": "hours",
            "assignment_status": "EXACT",
            "source_location": "2023 preprint, Results, Fig. 2e",
        },
        {
            "construct_group": "CR5",
            "outcome_type": "in_cell_half_life",
            "reported_value": "9.18",
            "units": "hours",
            "assignment_status": "EXACT",
            "source_location": "2023 preprint, Results, Fig. 2e",
        },
        {
            "construct_group": "CR-Novo",
            "outcome_type": "in_cell_half_life",
            "reported_value": "10.21",
            "units": "hours",
            "assignment_status": "EXACT",
            "source_location": "2023 preprint, Results, Fig. 2e",
        },
    ]
    with (BENCHMARK / "2023_outcome_extraction_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_ires_benchmark() -> list[dict[str, str]]:
    source = RAW / "pcbi.1005734.s013.tab"
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        records = []
        for row in reader:
            try:
                first = float(row["Cap-independent activity(eGFP expression,a.u) Replicate_1"])
                second = float(row["Cap-independent activity(eGFP expression,a.u) Replicate_2"])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(first) and math.isfinite(second)):
                continue
            records.append(
                {
                    "oligo_index": row["Oligo_Index"],
                    "set": row["Set"],
                    "ires_element": row["IRES_element"],
                    "activity_replicate_1": f"{first:g}",
                    "activity_replicate_2": f"{second:g}",
                    "activity_mean": f"{(first + second) / 2:g}",
                    "promoter_activity": row["Promoter_activity(eGFP(+)/total reads)"],
                    "splicing_score": row["Splicing_score(cDNA/gDNA)"],
                    "sequence_dna": row["Oligo_sequence"],
                    "source_file": source.name,
                }
            )
    records.sort(key=lambda record: float(record["activity_mean"]))
    selected = records[:50] + records[-50:]
    for rank, record in enumerate(selected[:50], 1):
        record["activity_bin"] = "lowest_50"
        record["activity_rank_within_bin"] = str(rank)
    for rank, record in enumerate(selected[50:], 1):
        record["activity_bin"] = "highest_50"
        record["activity_rank_within_bin"] = str(rank)

    fields = [
        "activity_bin",
        "activity_rank_within_bin",
        "oligo_index",
        "set",
        "ires_element",
        "activity_replicate_1",
        "activity_replicate_2",
        "activity_mean",
        "promoter_activity",
        "splicing_score",
        "sequence_dna",
        "source_file",
    ]
    output = IRES / "ires_benchmark.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    return selected


def main() -> None:
    rabies_sequences = extract_rabies_sequences()
    constructs = add_2023_rows(load_existing_constructs(), rabies_sequences)
    write_2023_outcome_audit()
    ires = extract_ires_benchmark()
    complete_constructs = sum(
        row["sequence_fasta"] not in {"NOT_FOUND", "NOT_FOUND_SEQUENCE"}
        and (
            row["in_solution_half_life_h"] != "NOT_FOUND"
            or row["in_cell_half_life_h"] != "NOT_FOUND"
        )
        for row in constructs
    )
    complete_ires = sum(bool(row["sequence_dna"]) and row["activity_mean"] != "" for row in ires)
    print("main_construct_rows=", len(constructs))
    print("main_complete_sequence_plus_outcome_pairs=", complete_constructs)
    print("ires_rows=", len(ires))
    print("ires_complete_sequence_plus_activity_pairs=", complete_ires)
    print("ires_lowest_rows=", sum(r["activity_bin"] == "lowest_50" for r in ires))
    print("ires_highest_rows=", sum(r["activity_bin"] == "highest_50" for r in ires))


if __name__ == "__main__":
    main()
