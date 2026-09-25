"""Assemble the public processed-data/code release for the next paper."""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = Path("/mnt/c/Users/admin/Desktop/next-paper")
OUT = ROOT / "public_release_next_paper"


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ["README.md", "LICENSE", "DATA_LICENSE.md", "CITATION.cff", "requirements.txt"]:
        p = OUT / name
        if p.exists():
            p.unlink()

    copy_file(DOCS / "next_paper_nature_submission_draft.md", OUT / "manuscript/main.md")
    copy_file(DOCS / "next_paper_current_supplement.md", OUT / "manuscript/supplement.md")
    copy_file(DOCS / "references.bib", OUT / "manuscript/references.bib")
    copy_file(DOCS / "renewable_energy_submission/renewable_energy_submission.pdf", OUT / "manuscript/main.pdf")
    copy_file(DOCS / "renewable_energy_submission/current_supplementary/supplementary_current.pdf", OUT / "manuscript/supplement.pdf")

    script_dst = OUT / "code/script"
    for p in (ROOT / "script").glob("*.py"):
        if p.name.startswith(("run_np", "verify_np", "build_", "refresh_", "next_paper_", "check_public_release")):
            copy_file(p, script_dst / p.name)
    for name in ["pyproject.toml", "uv.lock"]:
        copy_file(ROOT / name, OUT / "code" / name)
    if (ROOT / "src").exists():
        shutil.copytree(ROOT / "src", OUT / "code/src", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))

    fig_dst = OUT / "figures/nature_v1"
    for p in (ROOT / "outputs/next_paper/manuscript_figures/nature_v1").glob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".svg", ".png", ".json", ".md"}:
            copy_file(p, fig_dst / p.name)

    tags = [
        "np023", "np025", "np026", "np027", "np028", "np029", "np030",
        "np031_calendar_control", "np032_np033", "np034_weather_ablation",
        "np035_calibration", "np036_joint_timing_baseline",
        "np037_threshold_sensitivity", "np038_capacity_matched_information",
        "np039_route_direct_joint_bootstrap", "np040_route_global_summary",
        "np041_all_lead_decision_sensitivity",
    ]
    allowed = {".csv", ".json", ".npz", ".md", ".png", ".pdf", ".svg"}
    for tag in tags:
        src = ROOT / "outputs/next_paper" / tag
        if not src.exists():
            continue
        for p in src.rglob("*"):
            if p.is_file() and p.suffix.lower() in allowed:
                copy_file(p, OUT / "results" / tag / p.relative_to(src))

    copy_file(ROOT / "outputs/next_paper/site_metadata.json", OUT / "data/metadata/site_metadata.json")
    copy_file(ROOT / "outputs/next_paper/atmospheric_materials.json", OUT / "data/metadata/atmospheric_materials.json")
    for tag in ["np025", "np028"]:
        copy_file(ROOT / f"outputs/next_paper/{tag}/dataset.npz", OUT / f"data/processed/{tag}_dataset.npz")
        copy_file(ROOT / f"outputs/next_paper/{tag}/test_predictions.npz", OUT / f"data/processed/{tag}_test_predictions.npz")

    # Keep files larger than GitHub's normal repository-file limit as release
    # assets. The Git-tracked manifest and verification receipts remain small.
    asset_root = OUT / "release_assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    large_files = [
        p for p in OUT.rglob("*")
        if p.is_file() and ".git" not in p.parts and p.stat().st_size > 90_000_000
    ]
    asset_path = asset_root / "processed_arrays.tar.gz"
    if asset_path.exists():
        asset_path.unlink()
    if large_files:
        with tarfile.open(asset_path, "w:gz") as tar:
            for p in large_files:
                tar.add(p, arcname=p.relative_to(OUT))
        for p in large_files:
            p.unlink()

    (OUT / "README.md").write_text(
        """# Wind-power process predictability

Public processed-data and code release for *Different information horizons
shape wind-power event predictability*. This repository contains analysis
code, issue-time feature/label arrays, processed aggregate data, frozen
predictions, protocols, scores, verification receipts and figure manifests.

Raw SCADA, turbine-level identifiers, coordinates, operational-status records
and credentials are excluded. Processed arrays are published so the reported
score arithmetic and NP038-NP041 verification checks can be reproduced.

Install Python >=3.11 from requirements.txt. Run the verification scripts in
code/script. SHA-256 values are recorded in release_manifest.json.
""",
        encoding="utf-8",
    )
    (OUT / "LICENSE").write_text(
        "MIT License\n\nCopyright (c) 2026 ljpy\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction.\n\nTHE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND.\n",
        encoding="utf-8",
    )
    (OUT / "DATA_LICENSE.md").write_text(
        """# Data license and privacy boundary

Processed aggregate power, normalized features, labels, masks, predictions,
scores and figures are CC BY 4.0 unless a source-specific condition is stated.
Open-Meteo fields retain CC BY 4.0 attribution and source URLs. Raw SCADA,
turbine-level operational records, exact farm coordinates and credentials are
not redistributed.
""",
        encoding="utf-8",
    )
    (OUT / "CITATION.cff").write_text(
        """cff-version: 1.2.0
title: Wind-power process predictability
message: Cite this repository when using its processed arrays, protocols, or code.
type: software
authors:
  - family-names: Gong
    given-names: Peiyan
  - family-names: Yuan
    given-names: Chaoxia
  - family-names: Tian
    given-names: Jie
  - family-names: Shi
    given-names: Yanxi
license: MIT
version: 0.1.0
date-released: 2026-09-25
repository-code: https://github.com/longjingpy/wind-power-process-predictability
""",
        encoding="utf-8",
    )
    (OUT / "requirements.txt").write_text(
        "numpy>=2.1,<3\npandas>=2.2,<3\nscipy>=1.12,<2\nscikit-learn>=1.7,<1.8\njoblib>=1.4\nmatplotlib>=3.9\npyarrow>=15\n",
        encoding="utf-8",
    )
    (OUT / ".gitignore").write_text(
        "release_assets/processed_arrays.tar.gz\n__pycache__/\n.pytest_cache/\n",
        encoding="utf-8",
    )
    (OUT / "RELEASE_ASSETS.md").write_text(
        "The large processed prediction arrays are distributed as the GitHub "
        "Release asset processed_arrays.tar.gz. Download it into the repository "
        "root and verify its SHA-256 against release_manifest.json before "
        "extracting. No raw SCADA or turbine-level private records are included.\n",
        encoding="utf-8",
    )
    (OUT / "REPRODUCIBILITY.md").write_text(
        """# Reproducibility index

1. Install the pinned Python environment from requirements.txt.
2. Run the NP038, NP039, NP040 and NP041 verification scripts under code/script.
3. Recompute the score tables from the processed arrays in data/processed and results.
4. Check release_manifest.json before using any result.

The package contains processed aggregate/feature/label arrays and frozen
predictions. It does not contain raw turbine-level SCADA, coordinates,
operational-status records, API credentials or private source responses.
Anonymous readers can clone this repository and run the verification scripts
without an account.
""",
        encoding="utf-8",
    )
    files = [p for p in OUT.rglob("*") if p.is_file() and ".git" not in p.parts]
    manifest = {
        "release": "wind-power-process-predictability",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "raw_private_inputs_excluded": True,
        "file_count": len(files),
        "files": {str(p.relative_to(OUT)): digest(p) for p in sorted(files)},
    }
    (OUT / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT), "files": len(files)}))


if __name__ == "__main__":
    main()
