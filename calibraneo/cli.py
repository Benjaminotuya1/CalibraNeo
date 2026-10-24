"""Command-line entry point for reproducible CalibraNeo workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .data_registry import load_registry, validate_registry

def main() -> int:
    parser = argparse.ArgumentParser(description="Run CalibraNeo workflows")
    parser.add_argument(
        "command",
        choices=("train", "optimize", "figures", "registry", "external-data", "validation", "audit", "endpoint-models", "rna-model", "combined-immunogenicity", "freeze", "comparators", "gate", "all"),
        help="Workflow to execute",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    commands = {
        "train": root / "scripts" / "17_train_and_benchmark_calibraneo.py",
        "optimize": root / "scripts" / "18_rl_cassette_optimizer_and_figures.py",
        "figures": root / "scripts" / "19_build_clean_ipynb_and_app.py",
        "validation": root / "scripts" / "21_build_validation_report.py",
        "audit": root / "scripts" / "22_audit_validation.py",
        "endpoint-models": root / "scripts" / "23_train_endpoint_models.py",
        "rna-model": root / "scripts" / "26_train_rna_context_model.py",
        "combined-immunogenicity": root / "scripts" / "27_combined_immunogenicity_loso.py",
        "freeze": root / "scripts" / "24_freeze_prospective_predictions.py",
        "comparators": root / "scripts" / "25_build_method_comparison.py",
        "gate": root / "scripts" / "28_build_publication_gate.py",
    }
    if args.command == "registry":
        payload = load_registry(root / "data_registry.json")
        errors = validate_registry(payload)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print(json.dumps({"sources": len(payload["sources"]), "status": "valid"}, indent=2))
        return 0
    if args.command == "external-data":
        script = root / "scripts" / "20_fetch_external_evidence.py"
        return subprocess.run([sys.executable, str(script)], cwd=root).returncode
    if args.command == "validation":
        script = root / "scripts" / "21_build_validation_report.py"
        return subprocess.run([sys.executable, str(script)], cwd=root).returncode
    if args.command == "audit":
        script = root / "scripts" / "22_audit_validation.py"
        return subprocess.run([sys.executable, str(script)], cwd=root).returncode
    if args.command == "endpoint-models":
        script = root / "scripts" / "23_train_endpoint_models.py"
        return subprocess.run([sys.executable, str(script)], cwd=root).returncode
    if args.command == "rna-model":
        script = root / "scripts" / "26_train_rna_context_model.py"
        return subprocess.run([sys.executable, str(script)], cwd=root).returncode
    if args.command == "combined-immunogenicity":
        script = root / "scripts" / "27_combined_immunogenicity_loso.py"
        return subprocess.run([sys.executable, str(script)], cwd=root).returncode
    if args.command == "gate":
        script = root / "scripts" / "28_build_publication_gate.py"
        return subprocess.run([sys.executable, str(script)], cwd=root).returncode
    if args.command == "freeze":
        script = root / "scripts" / "24_freeze_prospective_predictions.py"
        return subprocess.run([sys.executable, str(script)], cwd=root).returncode
    if args.command == "comparators":
        script = root / "scripts" / "25_build_method_comparison.py"
        return subprocess.run([sys.executable, str(script)], cwd=root).returncode
    selected = ("train", "optimize", "figures", "validation", "audit", "endpoint-models", "rna-model", "combined-immunogenicity", "freeze", "comparators", "gate") if args.command == "all" else (args.command,)
    for name in selected:
        result = subprocess.run([sys.executable, str(commands[name])], cwd=root)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
