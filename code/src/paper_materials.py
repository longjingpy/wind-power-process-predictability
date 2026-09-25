"""
Helpers for stage-driven paper materials and draft generation.

The paper draft is generated from versioned stage folders under
``outputs/paper_materials`` so that completed engineering tasks can be
summarized reproducibly without hand-editing the `.docx`.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any


LABEL_FUSION = "Fusion Model"
LABEL_TEMPORAL_ONLY = "Temporal-Only Model"
LABEL_LSTM_GNSS_GNN = "LSTM+GNSS-GNN"
LABEL_VANILLA_LSTM = "Lightweight LSTM Baseline"
LABEL_VANILLA_TRANSFORMER = "Lightweight Transformer Baseline"
LABEL_RAINFORMER_SPATIAL = "Pure Spatial Rainformer Control"
LABEL_PERSISTENCE = "Persistence"
LABEL_AR1 = "ARX Wind-Speed Baseline"
PAPER_8_MODEL_ORDER = [
    LABEL_PERSISTENCE,
    LABEL_AR1,
    LABEL_VANILLA_LSTM,
    LABEL_VANILLA_TRANSFORMER,
    LABEL_RAINFORMER_SPATIAL,
    LABEL_TEMPORAL_ONLY,
    LABEL_LSTM_GNSS_GNN,
    LABEL_FUSION,
]
FAMILY_3_MODEL_ORDER = [LABEL_TEMPORAL_ONLY, LABEL_LSTM_GNSS_GNN, LABEL_FUSION]


def normalize_stage_id(value: str | int) -> str:
    if isinstance(value, int):
        return f"stage_{value:02d}"
    text = str(value).strip()
    if re.fullmatch(r"stage_\d{2}", text):
        return text
    if re.fullmatch(r"\d+", text):
        return f"stage_{int(text):02d}"
    return text


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip()).strip("_").lower()
    return text or "stage"


def build_stage_dir(materials_root: Path, stage_id: str | int, task_name: str) -> Path:
    stage_key = normalize_stage_id(stage_id)
    return materials_root / f"{stage_key}_{slugify(task_name)}"


def reset_stage_dir(stage_dir: Path) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    (stage_dir / "figures").mkdir(parents=True, exist_ok=True)
    (stage_dir / "tables").mkdir(parents=True, exist_ok=True)


def copy_asset(src: Path, dst_dir: Path) -> str:
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Missing asset: {src}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst.name


def ensure_jsonable(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(k): ensure_jsonable(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [ensure_jsonable(v) for v in payload]
    if isinstance(payload, Path):
        return str(payload)
    return payload


def _normalize_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        # keep order while de-duplicating
        seen: set[str] = set()
        uniq: list[str] = []
        for item in out:
            if item in seen:
                continue
            seen.add(item)
            uniq.append(item)
        return uniq
    text = str(value).strip()
    return [text] if text else []


def _infer_evidence_scope(summary: dict[str, Any]) -> str:
    stage_type = str(summary.get("stage_type", ""))
    if stage_type in {"teacher_student_prior", "gnss_tree_prior", "usable_gnss_direction_prior"}:
        return "train_fit"
    contract = summary.get("dataset_contract", {}) or {}
    split_mode = str(contract.get("split_mode", "")).strip().lower()
    if split_mode == "paper":
        return "held_out"
    if split_mode:
        return "same_contract"
    return "stage_specific"


def _infer_comparator_set_id(summary: dict[str, Any]) -> str:
    model_contract = summary.get("model_contract", {}) or {}
    labels = _normalize_list_of_str(model_contract.get("labels", []))
    label_set = set(labels)
    if label_set == set(PAPER_8_MODEL_ORDER):
        return "paper_facing_8model_v1"
    if label_set == set(FAMILY_3_MODEL_ORDER):
        return "family_3model_v1"
    stage_type = str(summary.get("stage_type", "")).strip()
    task_name = slugify(str(summary.get("task_name", "")).strip())
    if stage_type == "ablation_substage":
        return f"ablation_{task_name or 'substage'}"
    if stage_type:
        return f"{stage_type}_set"
    return "unspecified_set"


def _infer_common_intersection_id(summary: dict[str, Any]) -> str:
    contract = summary.get("dataset_contract", {}) or {}
    split_mode = str(contract.get("split_mode", "")).strip().lower() or "unknown"
    common_orig = contract.get("common_origin_count")
    common_rows = contract.get("common_row_count")
    if common_orig is not None and common_rows is not None:
        return f"{split_mode}_orig{int(common_orig)}_rows{int(common_rows)}"
    if "origin_count" in contract and "eligible_count" in contract:
        return f"{split_mode}_origin{int(contract['origin_count'])}_eligible{int(contract['eligible_count'])}"
    return f"{split_mode}_unspecified"


def _infer_pass_gate(summary: dict[str, Any]) -> bool:
    status = str(summary.get("status", "")).strip().lower()
    evidence_level = str(summary.get("evidence_level", "")).strip().lower()
    role = str(summary.get("manuscript_role", "")).strip().lower()
    if status != "completed":
        return False
    if evidence_level != "formal":
        return False
    if role in {"internal_only"}:
        return False
    return True


def _infer_figure_roles(summary: dict[str, Any]) -> dict[str, str]:
    figures = list(summary.get("figures", []) or [])
    existing = summary.get("figure_roles", {}) or {}
    role_map: dict[str, str] = {str(k): str(v) for k, v in dict(existing).items()}
    stage_type = str(summary.get("stage_type", "")).strip().lower()
    manuscript_role = str(summary.get("manuscript_role", "")).strip().lower()
    evidence_level = str(summary.get("evidence_level", "")).strip().lower()
    use_as_main = bool(summary.get("use_as_main_result", False))
    for idx, fig in enumerate(figures):
        path = str(fig.get("path", "")).strip()
        if not path:
            continue
        if path in role_map and role_map[path] in {"main", "appendix", "internal"}:
            fig["role"] = role_map[path]
            continue
        role = str(fig.get("role", "")).strip().lower()
        if role not in {"main", "appendix", "internal"}:
            kind = str(fig.get("kind", "")).strip().lower()
            if manuscript_role == "internal_only" or evidence_level in {"internal", "internal_audit"}:
                role = "internal"
            elif kind == "manuscript_primary":
                role = "main"
            elif use_as_main and idx == 0:
                role = "main"
            elif stage_type in {"same_contract_family", "same_contract_seven_model"} and idx == 0:
                role = "main"
            else:
                role = "appendix"
        role_map[path] = role
        fig["role"] = role
    return role_map


def normalize_stage_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out = dict(summary or {})
    out.setdefault("claim_ids", [])
    out["claim_ids"] = _normalize_list_of_str(out.get("claim_ids"))
    out.setdefault("citation_keys", [])
    out["citation_keys"] = _normalize_list_of_str(out.get("citation_keys"))
    out.setdefault("evidence_scope", _infer_evidence_scope(out))
    out["evidence_scope"] = str(out.get("evidence_scope", _infer_evidence_scope(out))).strip().lower()
    out.setdefault("comparator_set_id", _infer_comparator_set_id(out))
    out["comparator_set_id"] = str(out.get("comparator_set_id", _infer_comparator_set_id(out))).strip() or "unspecified_set"
    out.setdefault("common_intersection_id", _infer_common_intersection_id(out))
    out["common_intersection_id"] = str(out.get("common_intersection_id", _infer_common_intersection_id(out))).strip() or "unknown_unspecified"
    out.setdefault("pass_gate", _infer_pass_gate(out))
    out["pass_gate"] = bool(out.get("pass_gate"))
    role_map = _infer_figure_roles(out)
    out["figure_roles"] = role_map
    return out


def write_stage_summary(stage_dir: Path, payload: dict[str, Any]) -> Path:
    payload = normalize_stage_summary(payload)
    summary_path = stage_dir / "summary.json"
    summary_path.write_text(
        json.dumps(ensure_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_path


def resolve_stage_asset(stage_dir: Path, asset_path: str | Path) -> Path:
    asset = Path(asset_path)
    if asset.is_absolute():
        return asset
    return stage_dir / asset


def load_stage_summaries(materials_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not materials_root.exists():
        raise FileNotFoundError(f"Materials root not found: {materials_root}")
    results: list[tuple[Path, dict[str, Any]]] = []
    for stage_dir in sorted(p for p in materials_root.iterdir() if p.is_dir() and p.name.startswith("stage_")):
        summary_path = stage_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = normalize_stage_summary(summary)
        results.append((stage_dir, summary))
    if not results:
        raise FileNotFoundError(f"No stage summaries found under {materials_root}")
    return results
