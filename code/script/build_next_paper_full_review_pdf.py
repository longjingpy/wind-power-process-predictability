"""Build a full review PDF with title page, text, tables, and main figures."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path("/mnt/c/Users/admin/Desktop/next-paper")
FIG = Path("/mnt/d/projects/WindPowerForcast/outputs/next_paper/manuscript_figures")
TMP = ROOT / "temp/next_paper_full_review.md"
OUTPUT = ROOT / "next_paper_full_review.pdf"


def main() -> None:
    title = (ROOT / "Renewable_Energy_Title_Page.md").read_text(encoding="utf-8")
    body = (ROOT / "next_paper_submission_draft.md").read_text(encoding="utf-8")
    tables = (ROOT / "next_paper_submission_tables.md").read_text(encoding="utf-8")
    data_marker = "\n## Data and code availability\n"
    refs_marker = "\n## References\n"
    body_start = body.index(data_marker)
    refs_start = body.index(refs_marker)
    main_text = body[:body_start]
    refs = body[refs_start:]
    table_start = tables.index("## Table 1")
    table_text = tables[table_start:]
    figure_text = [
        "## Figure plates",
        "",
        "The following plates are review exports. Final journal dimensions, numbering, and captions remain subject to the target Guide for Authors.",
        "",
        f"### Graphical abstract\n\n![Graphical abstract]({FIG / 'graphical_abstract.pdf'})",
        f"### Fig. 1. Process objects\n\n![Fig. 1]({FIG / 'fig1_process_objects.pdf'})",
        f"### Fig. 2. Lead-time process skill\n\n![Fig. 2]({FIG / 'fig2_lead_time_process_skill.pdf'})",
        f"### Fig. 3. Occurrence and arrival timing\n\n![Fig. 3]({FIG / 'fig3_occurrence_vs_arrival_time.pdf'})",
        f"### Fig. 4. Range target alignment\n\n![Fig. 4]({FIG / 'fig4_range_target_alignment.pdf'})",
        f"### Fig. 5. Target-aligned information\n\n![Fig. 5]({FIG / 'fig5_target_aligned_information.pdf'})",
    ]
    text = "\n\n".join([title, "\\newpage", main_text, "## Data and code availability", body[body_start + len(data_marker):refs_start], "\\newpage", "\n".join(figure_text), "\\newpage", table_text, refs])
    TMP.write_text(text, encoding="utf-8")
    command = [
        "pandoc", str(TMP), "--from", "markdown", "--standalone", "--pdf-engine=xelatex",
        "-V", "geometry:margin=0.75in", "-V", "mainfont=TeX Gyre Termes", "-V", "fontsize=10pt",
        "-V", "graphics:true", "-o", str(OUTPUT),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
