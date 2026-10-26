"""Write the endpoint validation registry used by the app and manuscript audit."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibraneo.outcomes import endpoint_validation_status


ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "results" / "endpoint_validation_status.json"
table = endpoint_validation_status()
payload = {
    "schema_version": "1.0",
    "interpretation": (
        "Endpoint-specific validation status. Combining scores is a decision "
        "layer and does not establish a validated therapeutic outcome model."
    ),
    "endpoints": table.to_dict(orient="records"),
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(output)
