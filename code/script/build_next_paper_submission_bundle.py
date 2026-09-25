"""Build a curated, hash-manifested handoff bundle for the next paper."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

RESEARCH = Path("/mnt/d/projects/WindPowerForcast")
DOCS = Path("/mnt/c/Users/admin/Desktop/next-paper")
STAGING = DOCS / "temp/submission_bundle_staging"
ZIP_PATH = DOCS / "next_paper_submission_bundle.zip"
MANIFEST_NAME = "submission_bundle_manifest.json"

UPLOAD_FILES = [
    "next_paper_submission_draft.md",
    "next_paper_submission_review.pdf",
    "next_paper_submission_tables.md",
    "Renewable_Energy_Highlights.txt",
    "Renewable_Energy_Abstract_250w.txt",
    "Renewable_Energy_Keywords.txt",
    "renewable_energy_declarations_template.md",
    "renewable_energy_submission_checklist.md",
    "Renewable_Energy_Title_Page.md",
    "next_paper_submission_package.md",
]
AUDIT_FILES = [
    "next_paper_methods_reproducibility.md",
    "next_paper_figure_captions.md",
    "next_paper_figure_table_plan.md",
    "next_paper_claim_ledger.md",
    "next_paper_citation_audit.md",
    "next_paper_reviewer_audit.md",
    "PROJECT_PROGRESS.md",
    "next_paper_author_metadata.md",
]
FIGURE_FILES = [
    "fig1_process_objects.svg",
    "fig1_process_objects.png",
    "fig1_process_objects.pdf",
    "fig2_lead_time_process_skill.png",
    "fig2_lead_time_process_skill.pdf",
    "fig3_occurrence_vs_arrival_time.png",
    "fig3_occurrence_vs_arrival_time.pdf",
    "fig4_range_target_alignment.png",
    "fig4_range_target_alignment.pdf",
    "fig5_target_aligned_information.png",
    "fig5_target_aligned_information.pdf",
    "graphical_abstract.svg",
    "graphical_abstract.png",
    "graphical_abstract.pdf",
]
EVIDENCE_FILES = [
    ("np028_protocol.json", RESEARCH / "outputs/next_paper/np028/protocol.json"),
    ("np028_test_summary.csv", RESEARCH / "outputs/next_paper/np028/test_summary.csv"),
    ("np028_verification.json", RESEARCH / "outputs/next_paper/np028/verification.json"),
    ("np028_independent_verification.json", RESEARCH / "outputs/next_paper/np028/independent_verification.json"),
    ("np029_routing_summary.csv", RESEARCH / "outputs/next_paper/np029/routing_summary.csv"),
    ("np029_state_identity.csv", RESEARCH / "outputs/next_paper/np029/state_identity.csv"),
    ("np029_verification.json", RESEARCH / "outputs/next_paper/np029/verification.json"),
    ("np029_independent_verification.json", RESEARCH / "outputs/next_paper/np029/independent_verification.json"),
    ("np030_monthly_summary.csv", RESEARCH / "outputs/next_paper/np030/monthly_summary.csv"),
    ("np030_verification.json", RESEARCH / "outputs/next_paper/np030/verification.json"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def add_file(source: Path, destination: Path, records: list[dict]) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    records.append({
        "path": destination.relative_to(STAGING).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    })


def main() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    STAGING.mkdir(parents=True)
    records: list[dict] = []
    for name in UPLOAD_FILES:
        add_file(DOCS / name, STAGING / "upload" / name, records)
    for name in AUDIT_FILES:
        add_file(DOCS / name, STAGING / "audit" / name, records)
    figure_root = RESEARCH / "outputs/next_paper/manuscript_figures"
    for name in FIGURE_FILES:
        add_file(figure_root / name, STAGING / "figures" / name, records)
    for name, source in EVIDENCE_FILES:
        add_file(source, STAGING / "audit/evidence" / name, records)
    manifest = {
        "bundle": "next-paper-submission",
        "created": "2026-09-24",
        "purpose": "Curated handoff package; no raw input data included",
        "author_metadata": "TO_CONFIRM",
        "primary_target": "Renewable Energy",
        "files": sorted(records, key=lambda r: r["path"]),
    }
    (STAGING / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(STAGING.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(STAGING).as_posix())
    print(json.dumps({"zip": str(ZIP_PATH), "files": len(records), "bytes": ZIP_PATH.stat().st_size}))


if __name__ == "__main__":
    main()
