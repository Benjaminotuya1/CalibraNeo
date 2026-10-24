"""Conservative codon and construct calculations used by the UI."""

from __future__ import annotations

from collections import Counter

OPTIMAL_CODONS = {
    "A": "GCC", "R": "CGC", "N": "AAC", "D": "GAC", "C": "TGC",
    "Q": "CAG", "E": "GAG", "G": "GGC", "H": "CAC", "I": "ATC",
    "L": "CTG", "K": "AAG", "M": "ATG", "F": "TTC", "P": "CCC",
    "S": "AGC", "T": "ACC", "W": "TGG", "Y": "TAC", "V": "GTG",
}


def build_orf(peptides: list[str], linker: str = "AAY") -> tuple[str, str]:
    if not peptides or any(not peptide.isalpha() for peptide in peptides):
        raise ValueError("At least one peptide containing amino-acid letters is required")
    protein = "M" + linker.join(peptides).upper()
    dna = "".join(OPTIMAL_CODONS.get(aa, "") for aa in protein)
    if len(dna) != len(protein) * 3:
        raise ValueError("Unsupported amino acid in peptide sequence")
    return protein, dna + "TGA"


def calculate_cai(dna: str) -> float:
    """Calculate a transparent relative codon score, not a species CAI."""
    codons = [dna[index:index + 3] for index in range(0, len(dna) - 2, 3)]
    valid = [codon for codon in codons if codon != "TGA"]
    if not valid:
        return 0.0
    preferred = set(OPTIMAL_CODONS.values())
    return sum(codon in preferred for codon in valid) / len(valid)


def construct_quality(dna: str) -> dict[str, float | int]:
    coding = dna[:-3] if dna.endswith("TGA") else dna
    gc = sum(base in "GC" for base in coding)
    return {
        "length_nt": len(dna),
        "gc_percent": 100.0 * gc / len(coding) if coding else 0.0,
        "codon_preference_score": calculate_cai(dna),
        "homopolymer_max": max((len(run) for run in coding.replace("A", " ").replace("C", " ").replace("G", " ").split()), default=0),
        "stop_codons": Counter(coding[index:index + 3] for index in range(0, len(coding) - 2, 3)).get("TGA", 0),
    }
