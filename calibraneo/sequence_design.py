"""Transparent RNA/translation features; no claim of validated biological prediction."""

from __future__ import annotations

import math
import re


def rna_design_features(rna: str) -> dict[str, float | int | bool]:
    sequence = re.sub(r"\s+", "", rna.upper())
    if not sequence or set(sequence) - set("ACGU"):
        raise ValueError("RNA sequence must contain only A, C, G, and U")
    gc = (sequence.count("G") + sequence.count("C")) / len(sequence)
    au = (sequence.count("A") + sequence.count("U")) / len(sequence)
    homopolymer = max((len(run) for run in re.findall(r"(A+|C+|G+|U+)", sequence)), default=0)
    start_ok = sequence.startswith("AUG")
    stop_codons = sum(sequence[index:index + 3] in {"UAA", "UAG", "UGA"} for index in range(0, len(sequence) - 2, 3))
    return {
        "length_nt": len(sequence),
        "gc_fraction": gc,
        "au_fraction": au,
        "homopolymer_max": homopolymer,
        "start_codon_present": start_ok,
        "in_frame_stop_count": stop_codons,
        "sequence_qc_score": max(0.0, 1.0 - abs(gc - 0.55) - max(0, homopolymer - 6) / 20),
        "validated_rna_stability_model": False,
        "validated_translation_model": False,
    }
