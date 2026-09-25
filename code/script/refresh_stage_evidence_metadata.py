"""
Backfill publication-facing metadata fields for stage summaries.

This utility upgrades existing stage summaries to the newer schema used by the
manuscript generator:
  - claim_ids
  - evidence_scope
  - comparator_set_id
  - common_intersection_id
  - pass_gate
  - figure_roles
  - citation_keys
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paper_materials import normalize_stage_summary


def _claim_ids_for_summary(summary: dict[str, Any]) -> list[str]:
    stage_type = str(summary.get("stage_type", "")).strip().lower()
    task_name = str(summary.get("task_name", "")).strip().lower()
    if stage_type == "same_contract_seven_model":
        return ["C1"]
    if stage_type == "same_contract_family":
        return ["C1", "C3"]
    if stage_type == "dataset_contract":
        return ["C5"]
    if stage_type in {"gnss_prior", "usable_gnss_direction_prior"}:
        return ["C2"]
    if stage_type == "ablation_substage":
        if "prior" in task_name or "gating" in task_name:
            return ["C2"]
        if "graph" in task_name or "complexity" in task_name:
            return ["C3"]
        if "training" in task_name or "recipe" in task_name:
            return ["C4"]
        if "data" in task_name or "gapfill" in task_name:
            return ["C5"]
    if stage_type == "ablation_results":
        return ["C2", "C3", "C4", "C5"]
    return []


def _citation_keys_for_summary(summary: dict[str, Any]) -> list[str]:
    stage_type = str(summary.get("stage_type", "")).strip().lower()
    if stage_type in {"same_contract_family", "ablation_substage"}:
        return ["hochreiter1997_lstm", "nielsen1996_arx", "chevillon2007_direct_multistep"]
    if stage_type in {"dataset_contract", "gnss_prior", "usable_gnss_direction_prior"}:
        return ["kalman1960_filtering", "hersbach2020_era5", "bevis1994_gps_meteorology"]
    if stage_type == "same_contract_seven_model":
        return ["nielsen1996_arx", "chevillon2007_direct_multistep"]
    return []


def _merge_unique(existing: Any, incoming: list[str]) -> list[str]:
    merged: list[str] = []
    for item in list(existing or []) + list(incoming):
        text = str(item).strip()
        if not text or text in merged:
            continue
        merged.append(text)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill claim/evidence metadata for all stage summaries.")
    parser.add_argument("--materials-root", type=Path, default=ROOT / "outputs/paper_materials")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-claims", action="store_true")
    args = parser.parse_args()

    summaries = sorted(args.materials_root.glob("stage_*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No summary.json found under {args.materials_root}")

    changed = 0
    for path in summaries:
        raw = json.loads(path.read_text(encoding="utf-8"))
        claim_ids = _claim_ids_for_summary(raw)
        citation_keys = _citation_keys_for_summary(raw)
        if args.force_claims:
            raw["claim_ids"] = claim_ids
        else:
            raw["claim_ids"] = _merge_unique(raw.get("claim_ids"), claim_ids)
        raw["citation_keys"] = _merge_unique(raw.get("citation_keys"), citation_keys)
        normalized = normalize_stage_summary(raw)
        before = json.dumps(raw, ensure_ascii=False, sort_keys=True)
        after = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        if before == after:
            continue
        changed += 1
        if not args.dry_run:
            path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

    mode = "DRY-RUN" if args.dry_run else "WRITE"
    print(f"[{mode}] touched={changed} summaries under {args.materials_root}")


if __name__ == "__main__":
    main()

