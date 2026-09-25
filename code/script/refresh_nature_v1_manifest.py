"""Refresh the source manifests after the final figure exports are checked."""
from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

OUT = Path("/mnt/d/projects/WindPowerForcast/outputs/next_paper/manuscript_figures/nature_v1")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    figures = []
    for stem in [
        "fig1_process_objects",
        "fig2_process_capability",
        "fig3_pizhou_core_skill",
        "fig4_information_decomposition_enhanced",
        "fig5_weather_variable_ablation",
        "fig6_target_aligned_routing_landscape",
        "fig7_support_and_stability",
        "fig8_decision_value",
    ]:
        files = {}
        for ext in ["pdf", "svg", "png"]:
            path = OUT / f"{stem}.{ext}"
            files[ext] = {"path": str(path.relative_to(OUT.parent.parent.parent)), "sha256": digest(path)}
        figures.append({"figure": stem, "files": files, "source_manifest": f"{stem}_source_manifest.json"})
        manifest_path = OUT / f"{stem}_source_manifest.json"
        existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        # Preserve the builder's source paths and hashes while refreshing only
        # the exported-file hashes. This keeps CSV provenance attached to each
        # figure after a manifest refresh.
        manifest = dict(existing)
        manifest.update({"created_utc": datetime.now(timezone.utc).isoformat(),
                         "script": "script/build_nature_figures_enhanced.py / script/build_nature_fig1.py",
                         "captions": "outputs/next_paper/manuscript_figures/nature_v1/captions.md",
                         "figure": stem, "files": files})
        (OUT / f"{stem}_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "nature_v1_manifest.json").write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "figures": figures}, indent=2) + "\n", encoding="utf-8")
    print(OUT / "nature_v1_manifest.json")


if __name__ == "__main__":
    main()
