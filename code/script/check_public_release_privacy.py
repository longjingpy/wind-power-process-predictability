"""Fail closed if a public-release staging tree contains private inputs."""
from __future__ import annotations

import re
import sys
from pathlib import Path

FORBIDDEN_PARTS = {
    "raw",
    "private",
    "scada",
    "turbine",
    "coordinates",
    "operational_status",
    "whereTurbins.csv",
}
FORBIDDEN_TEXT = re.compile(r"(?i)(api[_ -]?key|authorization:|bearer\\s+[A-Za-z0-9._-]{12,})")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "public_release_next_paper").resolve()
    bad_paths = []
    bad_text = []
    for p in root.rglob("*"):
        if ".git" in p.parts or not p.is_file():
            continue
        lower_parts = {part.lower() for part in p.relative_to(root).parts}
        if lower_parts & FORBIDDEN_PARTS:
            bad_paths.append(str(p.relative_to(root)))
            continue
        if p.stat().st_size <= 5_000_000 and p.suffix.lower() in {".md", ".json", ".txt", ".yml", ".yaml", ".cff"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if FORBIDDEN_TEXT.search(text):
                bad_text.append(str(p.relative_to(root)))
    assert not bad_paths, bad_paths
    assert not bad_text, bad_text
    print({"status": "PASS", "files_scanned": sum(1 for p in root.rglob("*") if p.is_file() and ".git" not in p.parts)})


if __name__ == "__main__":
    main()
