"""Build the final computational stopping-gate report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    manifest = json.loads(
        (ROOT / "models" / "endpoint_model_manifest.json").read_text(encoding="utf-8")
    )
    blockers = [
        {
            "item": "TESLA/NEPdb/NeoDB combined immunogenicity",
            "status": "blocked_by_data_access_or_normalization",
            "next_action": "Obtain licensed files and add source/publication-grouped adapter.",
        },
        {
            "item": "HLA Ligand Atlas/SysteMHC training",
            "status": "blocked_by_download/access in this environment",
            "next_action": "Download a permitted release and add immunopeptidomics adapter.",
        },
        {
            "item": "RPFdb translation model",
            "status": "blocked_by_missing_local_labeled_table",
            "next_action": "Obtain matched Ribo-seq/RNA-seq translation-efficiency table.",
        },
        {
            "item": "Full VCF-to-peptide generation",
            "status": "blocked_by_reference_and_consequence_dependencies",
            "next_action": "Provide transcript/protein reference and consequence caller.",
        },
        {
            "item": "Biological validation",
            "status": "requires_wet_lab",
            "next_action": "Test frozen candidates with expression, MS, and T-cell assays.",
        },
    ]
    gate = {
        "status": "computational_release_ready_with_explicit_blockers",
        "completed": [
            "leakage-audited presentation model",
            "endpoint-specific immunogenicity and immunopeptidomics models",
            "RNA stability context model without target leakage",
            "patient HLA/expression/ASE/LOH/copy-number parsers",
            "protein-context peptide generation",
            "leave-one-study-out combined immunogenicity audit",
            "prospective prediction freezing",
            "available comparator status",
            "automated tests and app smoke test",
        ],
        "blocked": blockers,
        "stop_rule": "Stop computational expansion until blocked data/access or wet-lab outcomes arrive.",
        "model_manifest": manifest,
    }
    (ROOT / "results" / "publication_readiness_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
