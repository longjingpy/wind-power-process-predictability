"""Build the complete Applied Energy supplementary material, including S13-S18."""
from pathlib import Path
import json
import re
import shutil
import subprocess

import pypandoc
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "manuscript" / "applied_energy" / "supplementary_complete.md"
BUILD = ROOT / "temp" / "v19_supplementary_build"
OUT = ROOT / "manuscript" / "applied_energy" / "supplementary.pdf"
TEXLIVE = Path("D:/texlive/2026/bin/windows")


def normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def write_utf8_lf(path: Path, value: str) -> None:
    path.write_bytes(normalize_newlines(value).encode("utf-8"))


def bibliography() -> tuple[str, set[str]]:
    primary = normalize_newlines(
        (ROOT / "manuscript" / "applied_energy" / "references.bib").read_text(encoding="utf-8")
    )
    additions = normalize_newlines(
        (ROOT / "manuscript" / "references_v18_additions.bib").read_text(encoding="utf-8")
    )
    key_pattern = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")
    keys = set(key_pattern.findall(primary))
    blocks = re.split(r"(?m)(?=^@\w+\s*\{)", additions.strip())
    extra = [block for block in blocks if block and (match := key_pattern.search(block))
             and match.group(1) not in keys]
    merged = primary.rstrip() + ("\n\n" + "\n\n".join(extra) if extra else "") + "\n"
    return merged, set(key_pattern.findall(merged))


def executable(name: str) -> str:
    candidate = TEXLIVE / f"{name}.exe"
    return str(candidate) if candidate.exists() else name


def run(command: list[str], index: int) -> None:
    result = subprocess.run(command, cwd=BUILD, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    write_utf8_lf(BUILD / f"compile_{index}.txt", result.stdout + "\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Compile step failed: {command}; see {BUILD / f'compile_{index}.txt'}")


def remove_longtable_preamble_blank_lines(value: str) -> str:
    """Keep Pandoc's longtable column specification free of paragraph breaks."""
    pattern = re.compile(r"(\\begin\{longtable\}.*?)(@\{\}\})", flags=re.DOTALL)

    def clean(match: re.Match[str]) -> str:
        preamble = re.sub(r"\n[ \t]*(?=\n)", "\n", match.group(1))
        return preamble + match.group(2)

    return pattern.sub(clean, value)


def make_long_paths_breakable(value: str) -> str:
    """Use url's breakable path handling for literal repository paths."""
    return re.sub(r"\\texttt\{([^{}]*?/[^{}]*)\}", r"\\path{\1}", value)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    markdown = normalize_newlines(SRC.read_text(encoding="utf-8"))
    figures=re.findall(r'!\[[^\]]*\]\((figures_v\d+/[^)]+)\)',markdown)
    for figure in figures:
        source=ROOT/'manuscript'/figure
        shutil.copy2(source,BUILD/source.name)
        markdown=markdown.replace(figure,source.name)
    conversion = normalize_newlines(
        pypandoc.convert_text(
            markdown, "latex", format="markdown+tex_math_single_backslash",
            extra_args=["--natbib", "--wrap=none"],
        )
    )
    conversion = remove_longtable_preamble_blank_lines(conversion)
    conversion = make_long_paths_breakable(conversion)
    conversion = re.sub(r'The complete thirteen arms.*?(?=\n\n)',
                        lambda m: m.group(0).replace(r'\_', r'\_\allowbreak{}'),
                        conversion, flags=re.DOTALL)
    conversion = re.sub(r'The data interface is prepared by.*?(?=\n\n)',
                        lambda m: m.group(0).replace(r'\_', r'\_\allowbreak{}'),
                        conversion, flags=re.DOTALL)
    conversion = re.sub(
        r'(\\subsubsection\{Table S(?:75|76|77|78|79|80|81|82|83)\..*?)(\\begin\{longtable\}.*?\\end\{longtable\})',
        lambda m: m.group(1) + r'\begingroup\AtBeginEnvironment{longtable}{\footnotesize}'
        + r'\setlength{\tabcolsep}{3pt}' + '\n' + m.group(2) + '\n' + r'\endgroup',
        conversion, flags=re.DOTALL)
    for number in [59, 62, 75, 76, 77, 78, 79, 80]:
        heading = r'\subsubsection{Table S' + str(number) + '.'
        conversion = conversion.replace(heading, r'\Needspace{17\baselineskip}' + heading)
    conversion = conversion.replace(r'\begin{figure}', r'\begin{figure}[H]')
    bib, keys = bibliography()
    cited = set(re.findall(r"@([A-Za-z0-9_-]+)", markdown))
    if missing := cited - keys:
        raise ValueError(f"Unresolved citation keys: {sorted(missing)}")
    write_utf8_lf(BUILD / "references.bib", bib)
    template = next((ROOT / "manuscript" / "applied_energy" / "official_template").rglob("elsarticle-num.bst"))
    shutil.copy2(template, BUILD / "elsarticle-num.bst")
    tex = r"""\documentclass[a4paper,11pt]{article}
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\usepackage{geometry,amsmath,amssymb,booktabs,hyperref,longtable,array,calc,caption,graphicx,adjustbox,xurl,float,needspace}
\usepackage[numbers,sort&compress]{natbib}
\geometry{margin=25mm}
\setcounter{secnumdepth}{0}
\setlength{\tabcolsep}{1.5pt}
\AtBeginEnvironment{longtable}{\scriptsize}
\setlength{\emergencystretch}{3em}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\newcounter{none}
\renewcommand{\thefigure}{S\arabic{figure}}
\providecommand{\pandocbounded}[1]{\begin{adjustbox}{max width=\linewidth}#1\end{adjustbox}}
\begin{document}
""" + conversion + r"""
\bibliographystyle{elsarticle-num}
\bibliography{references}
\end{document}
"""
    write_utf8_lf(BUILD / "supplementary.tex", tex)
    commands = [
        [executable("xelatex"), "-interaction=nonstopmode", "-halt-on-error", "supplementary.tex"],
        [executable("bibtex"), "supplementary"],
        [executable("xelatex"), "-interaction=nonstopmode", "-halt-on-error", "supplementary.tex"],
        [executable("xelatex"), "-interaction=nonstopmode", "-halt-on-error", "supplementary.tex"],
    ]
    for index, command in enumerate(commands):
        run(command, index)
    log = (BUILD / "supplementary.log").read_text(encoding="utf-8", errors="replace")
    if "There were undefined references" in log or re.search(r"Citation .* undefined", log):
        raise ValueError("Unresolved citations remain in the final LaTeX log")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUILD / "supplementary.pdf", OUT)
    reader = PdfReader(str(OUT))
    report = {
        "status": "COMPILED_LAYOUT_REVIEW_PENDING",
        "pages": len(reader.pages),
        "figures": len(figures),
        "cited_references": len(cited),
        "bibliography_entries": len(keys),
        "table_heading_count": len(re.findall(r"^### Table S\d+\.", markdown, flags=re.MULTILINE)),
        "longtable_count": conversion.count(r"\begin{longtable}"),
        "pdf": str(OUT),
        "overfull_box_count": log.count("Overfull \\hbox"),
    }
    write_utf8_lf(BUILD / "build_report.json", json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
