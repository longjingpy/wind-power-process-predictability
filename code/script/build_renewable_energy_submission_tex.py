"""Build an Elsevier elsarticle submission source with ordered figures/tables."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

DOCS = Path("/mnt/c/Users/admin/Desktop/next-paper")
ROOT = Path("/mnt/d/projects/WindPowerForcast")
WORK = DOCS / "renewable_energy_submission"
SOURCE_MD = DOCS / "next_paper_nature_submission_draft.md"
TABLE_MD = DOCS / "next_paper_submission_tables.md"
ABSTRACT = DOCS / "Renewable_Energy_Abstract_250w.txt"
BIB = DOCS / "references.bib"
BST = ROOT / "manuscript/applied_energy/submission_v26/elsarticle-num.bst"
FIG = ROOT / "outputs/next_paper/manuscript_figures/nature_v1"

AFF1 = "State Key Laboratory of Climate System Prediction and Risk Management/Key Laboratory of Meteorological Disaster, Ministry of Education/Collaborative Innovation Center on Forecast and Evaluation of Meteorological Disasters, Nanjing University of Information Science and Technology, Nanjing 210044, China"
AFF2 = "Jiangsu Key Laboratory of Intelligent Weather Forecasting and Applications Based on Big Data/School of Artificial Intelligence, Nanjing University of Information Science and Technology, Nanjing 210044, China"


def table_section(text: str, title_prefix: str) -> str:
    start = text.index(title_prefix)
    rest = text[start:]
    end = rest.find("\n## ", len(title_prefix))
    section = rest if end < 0 else rest[:end]
    lines = section.splitlines()
    table_lines = [line for line in lines if line.startswith("|")]
    if len(table_lines) < 3:
        raise ValueError(title_prefix)
    # The final column is an internal source path. Keep journal-facing columns only.
    header_cells = table_lines[0].strip().strip("|").split("|")
    remove_source_column = bool(header_cells and header_cells[-1].strip().lower() == "source")
    cleaned = []
    for line in table_lines:
        cells = line.strip().strip("|").split("|")
        if remove_source_column and cells:
            cells = cells[:-1]
        cleaned.append("| " + " | ".join(c.strip() for c in cells) + " |")
    caption = lines[0].replace("## ", "", 1)
    return f"**{caption}**\n\n" + "\n".join(cleaned)


def figure_block(number: int, caption: str, filename: str) -> str:
    # Use raw LaTeX so every journal figure is constrained to the text width.
    # Pandoc's bundled Markdown reader does not support the attributes
    # extension in this environment, and an unconstrained PDF figure can be
    # clipped at the right edge of the preprint page.
    path = str(FIG / filename)
    return (
        "\n\n\\begin{figure}[H]\n"
        "\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{{path}}}\n"
        f"\\caption{{{caption}}}\n"
        "\\end{figure}\n\n"
    )


def make_source() -> tuple[str, str, str]:
    text = SOURCE_MD.read_text(encoding="utf-8")
    abstract_start = text.index("## Abstract\n") + len("## Abstract\n")
    abstract_end = text.index("\n## 1. Introduction", abstract_start)
    abstract = text[abstract_start:abstract_end].strip()
    data_marker = "\n## Data and code availability\n"
    refs_marker = "\n## References\n"
    text = text[text.index("## 1. Introduction"):]
    refs_start = text.index(refs_marker)
    data_start = text.index(data_marker)
    main = text[:data_start]
    # References are generated from the verified BibTeX matrix rather than a
    # hand-numbered Markdown list.
    refs = ""
    # Pandoc's Markdown reader handles inline math reliably; flatten display
    # equations for this preprint source while retaining the full equations in
    # the manuscript candidate and supplementary methods.
    main = re.sub(
        r"\\\((.*?)\\\)",
        lambda m: "$" + " ".join(m.group(1).split()) + "$",
        main,
        flags=re.DOTALL,
    )
    main = re.sub(
        r"\\\[(.*?)\\\]",
        lambda m: "$$\n" + m.group(1).strip() + "\n$$",
        main,
        flags=re.DOTALL,
    )
    # Normalize manuscript headings in one pass so subsection levels are not
    # accidentally collapsed when the section numbers are removed.
    def heading(match: re.Match[str]) -> str:
        level = match.group(1)
        title = match.group(2)
        title = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title)
        return ("# " if level == "##" else "## ") + title

    main = re.sub(r"^(#{2,3})\s+(.+)$", heading, main, flags=re.MULTILINE)
    # Ordered first citations: Fig.1, Table 1, Fig.2/Table 2, Fig.3, Fig.4,
    # Fig.5, Fig.6, Fig.7 and Fig.8. Tables 3 and 4 remain in the Supplementary material.
    # Table 1 is already cited in 2.1 and appears immediately after that subsection.
    table1 = table_section(TABLE_MD.read_text(encoding="utf-8"), "## Table 1.")
    table2 = table_section(TABLE_MD.read_text(encoding="utf-8"), "## Table 2.")
    marker1 = "This boundary separates information that can be used at issue time from information that can only be used to diagnose an information gap after the event.\n"
    main = main.replace(marker1, marker1 + "\n" + table1 + "\n", 1)
    marker2 = "The timing increment becomes smaller as the issue lead increases, whereas broad state-risk information remains available at longer leads. The two objects therefore provide complementary forecast outputs: a state-risk vector over all windows and a conditional arrival distribution on downward-event windows.\n"
    fig2 = figure_block(2, "Process-state and first-passage skill retain different issue-time horizons. Points show frozen test scores; vertical ranges show the paired seven-day block bootstrap intervals (2,000 resamples).", "fig2_process_capability.pdf")
    main = main.replace(marker2, marker2 + "\n" + table2 + fig2, 1)
    marker_core = "These results define two complementary forecast objects for the information decomposition: event-state risk and first-passage timing.\n"
    fig3 = figure_block(3, "Pizhou core state-risk and first-passage skill across issue lead (NP005/NP008/NP009; 6,496 target windows). Points show absolute scores and intervals show paired seven-day block bootstrap uncertainty.", "fig3_pizhou_core_skill.pdf")
    main = main.replace(marker_core, marker_core + fig3, 1)
    fig4 = figure_block(4, "Absolute state scores and target-conditional information increments for NP025 and NP028. Lines show absolute test scores; ribbons show paired seven-day block bootstrap intervals on the matched Suining windows.", "fig4_information_decomposition_enhanced.pdf")
    fig5 = figure_block(5, "Weather-variable ablation for state and timing information. Bars compare calendar, thermodynamic, wind-vector and full-weather arms at 15, 240 and 720 min under the NP025/NP028 tree budgets.", "fig5_weather_variable_ablation.pdf")
    main = main.replace("\n## The target-specific pattern persists under a fixed pre-issue weather clock", fig4 + fig5 + "\n## The target-specific pattern persists under a fixed pre-issue weather clock", 1)
    marker4 = "The route therefore supports a lead- and protocol-conditioned timing allocation rule while leaving the joint state-risk probabilities unchanged.\n"
    fig6 = figure_block(6, "Target-aligned routing across all issue leads. Ribbons are paired seven-day block bootstrap intervals; archived NP026 uses 6,996 windows and strict NP029 uses 3,772 windows. The state identity panel reports the maximum absolute state-probability difference.", "fig6_target_aligned_routing_landscape.pdf")
    main = main.replace(marker4, marker4 + "\n" + fig6, 1)
    marker5 = "The complete lead-by-lead and monthly matrices, Pizhou core protocols, missingness audit and decision-cost sensitivity are retained in the Supplementary materials.\n"
    fig7 = figure_block(7, "Support windows, issue-time clocks and within-period stability. The timeline marks the data and information contracts; the right panel reports frozen monthly point estimates from NP027/NP030.", "fig7_support_and_stability.pdf")
    fig8 = figure_block(8, "Forecast-to-decision sensitivity using frozen probabilities. Panel a shows all-lead full-rolling cost reduction versus no alert; panel b retains the NP023 fusion comparison at its available leads; panel c gives the all-lead cost-ratio surface.", "fig8_decision_value.pdf")
    main = main.replace(marker5, marker5 + "\n" + fig7 + fig8, 1)
    # Add Fig.1 after the three research questions and before the contribution paragraph.
    marker0 = "Third, can this information decomposition be converted into a target-aligned timing product using issue-time inputs?\n"
    fig1 = figure_block(1, "Process objects and information boundary.", "fig1_process_objects.pdf")
    main = main.replace(marker0, marker0 + fig1, 1)
    # Main manuscript should not include the source map; preserve only references.
    return main, refs, abstract


def escape_latex(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def main() -> None:
    if WORK.exists():
        # Keep the separately compiled Supplementary directory.  The main
        # manuscript is rebuilt from scratch, while the supplement is a
        # durable review artifact linked from the project documents.
        for child in WORK.iterdir():
            if child.name == "current_supplementary":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        WORK.mkdir(parents=True)
    main, refs, abstract = make_source()
    fragment = WORK / "body.md"
    fragment.write_text(main + "\n" + refs, encoding="utf-8")
    body_tex = WORK / "body.tex"
    subprocess.run(["pandoc", str(fragment), "--from", "markdown", "--to", "latex", "--standalone", "-o", str(body_tex)], check=True)
    converted = body_tex.read_text(encoding="utf-8")
    if "\\begin{document}" in converted:
        converted = converted.split("\\begin{document}", 1)[1].rsplit("\\end{document}", 1)[0]
    body_tex.write_text(converted, encoding="utf-8")
    shutil.copy2(ROOT / "manuscript/applied_energy/submission_v26/elsarticle.cls", WORK / "elsarticle.cls")
    shutil.copy2(BIB, WORK / "references.bib")
    shutil.copy2(BST, WORK / "elsarticle-num.bst")
    wrapper = WORK / "renewable_energy_submission.tex"
    title = "Different information horizons shape wind-power event predictability"
    tex = f'''\\documentclass[preprint,12pt]{{elsarticle}}
\\usepackage{{fontspec}}
\\setmainfont{{TeX Gyre Termes}}
\\usepackage{{amsmath,amssymb,booktabs,longtable,array,graphicx,adjustbox,caption,float,placeins,xurl,calc}}
\\usepackage[unicode,colorlinks=true,allcolors=blue]{{hyperref}}
\\providecommand{{\\tightlist}}{{\\setlength{{\\itemsep}}{{0pt}}\\setlength{{\\parskip}}{{0pt}}}}
\\providecommand{{\\pandocbounded}}[1]{{\\begin{{adjustbox}}{{max width=\\linewidth,max totalheight=.65\\textheight,keepaspectratio}}#1\\end{{adjustbox}}}}
\\providecommand{{\\real}}[1]{{#1}}
\\setlength{{\\emergencystretch}}{{3em}}
\\journal{{Renewable Energy}}\n\\biboptions{{super,sort&compress}}
\\begin{{document}}
\\begin{{frontmatter}}
\\title{{{title}}}
\\author[aff1,aff2]{{Peiyan Gong}}
\\ead{{202483300332@nuist.edu.cn}}
\\author[aff1,aff2]{{Chaoxia Yuan\\corref{{cor1}}}}
\\ead{{chaoxia.yuan@nuist.edu.cn}}
\\author[aff1,aff2]{{Jie Tian}}
\\author[aff1,aff2]{{Yanxi Shi}}
\\cortext[cor1]{{Corresponding author.}}
\\fntext[orcid]{{ORCID: Peiyan Gong 0009-0000-8793-2932; Chaoxia Yuan 0000-0001-5121-869X; Jie Tian 0009-0004-4231-315X; Yanxi Shi 0009-0000-1095-4017.}}
\\address[aff1]{{{AFF1}}}
\\address[aff2]{{{AFF2}}}
\\begin{{abstract}}
{escape_latex(abstract)}
\\end{{abstract}}
\\begin{{keyword}}
wind power forecasting \\sep event predictability \\sep first-passage timing \\sep numerical weather prediction \\sep power history \\sep target-aligned information
\\end{{keyword}}
\\end{{frontmatter}}
\\input{{body.tex}}
\\bibliographystyle{{elsarticle-num}}
\\bibliography{{references}}
\\section*{{Funding}}
This work was financially supported by the State Key Laboratory of Climate System Prediction and Risk Management (CPRM) initiative project (CPRM-2025-NUIST-012) and the National Natural Science Foundation of China (42088101 and 41875099).
\\section*{{CRediT authorship contribution statement}}
Peiyan Gong: Conceptualization, methodology, software, formal analysis, visualization, writing -- original draft. Chaoxia Yuan: Supervision, project administration, funding acquisition, writing -- review and editing. Jie Tian: Validation, data curation, writing -- review and editing. Yanxi Shi: Validation, data curation, writing -- review and editing. All authors approved the final manuscript.
\\section*{{Declaration of competing interest}}
The authors declare no competing interests.
\\section*{{Data and code availability}}
Code will be released publicly. Public-source wind-farm datasets will be released or linked with their source records and licenses. Private wind-farm SCADA, coordinates and operational-status records will not be released or redistributed.
\\end{{document}}
'''
    wrapper.write_text(tex, encoding="utf-8")
    subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", wrapper.name], cwd=WORK, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["bibtex", wrapper.stem], cwd=WORK, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", wrapper.name], cwd=WORK, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", wrapper.name], cwd=WORK, check=True, stdout=subprocess.DEVNULL)
    print(wrapper)


if __name__ == "__main__":
    main()
