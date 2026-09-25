"""Verify the public next-paper repository without private input data."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    manifest = json.loads((root / "release_manifest.json").read_text())
    assert manifest["raw_private_inputs_excluded"] is True
    required = [
        "README.md",
        "DATA_LICENSE.md",
        "CITATION.cff",
        "REPRODUCIBILITY.md",
        "manuscript/main.md",
        "manuscript/supplement.md",
        "manuscript/main.pdf",
        "manuscript/supplement.pdf",
        "data/processed/np025_dataset.npz",
        "data/processed/np028_dataset.npz",
        "figures/nature_v1/nature_v1_manifest.json",
    ]
    assert all((root / p).exists() for p in required)
    figures = json.loads((root / "figures/nature_v1/nature_v1_manifest.json").read_text())
    assert len(figures["figures"]) == 8
    for tag in [
        "np038_capacity_matched_information",
        "np039_route_direct_joint_bootstrap",
        "np040_route_global_summary",
        "np041_all_lead_decision_sensitivity",
    ]:
        verification = root / "results" / tag / "verification.json"
        assert json.loads(verification.read_text())["status"] == "PASS"
    for rel, expected in manifest["files"].items():
        p = root / rel
        if not p.exists():
            # Large processed arrays are release assets and are verified after
            # the asset is downloaded and extracted into the same root.
            continue
        assert sha256(p) == expected, rel
    print({"status": "PASS", "file_count": manifest["file_count"], "figures": 8})


if __name__ == "__main__":
    main()
