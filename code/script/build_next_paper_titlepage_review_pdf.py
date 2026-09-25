"""Build a title-page-inclusive review PDF for the next-paper handoff."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path("/mnt/c/Users/admin/Desktop/next-paper")
TMP = ROOT / "temp/next_paper_titlepage_review.md"
OUTPUT = ROOT / "next_paper_submission_review_titlepage.pdf"


def main() -> None:
    title = (ROOT / "Renewable_Energy_Title_Page.md").read_text(encoding="utf-8")
    body = (ROOT / "next_paper_submission_draft.md").read_text(encoding="utf-8")
    TMP.write_text(title + "\n\n\\newpage\n\n" + body, encoding="utf-8")
    command = [
        "pandoc", str(TMP), "--from", "markdown", "--standalone", "--pdf-engine=xelatex",
        "-V", "geometry:margin=1in", "-V", "mainfont=TeX Gyre Termes", "-V", "fontsize=10.5pt",
        "-o", str(OUTPUT),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
