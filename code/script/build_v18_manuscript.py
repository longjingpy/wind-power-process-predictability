"""Build the current Applied Energy manuscript from the unified body source."""
from pathlib import Path
import json
import re
import shutil
import subprocess

import pypandoc
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "manuscript" / "applied_energy" / "manuscript_body.md"
BUILD = ROOT / "temp" / "v19_manuscript_build"
OUT = ROOT / "manuscript" / "applied_energy" / "main.pdf"
TEXLIVE = Path("D:/texlive/2026/bin/windows")


def normalize_newlines(value: str) -> str:
    """Use one UTF-8/LF representation for source and generated text."""
    return value.replace("\r\n", "\n").replace("\r", "\n")


def write_utf8_lf(path: Path, value: str) -> None:
    path.write_bytes(normalize_newlines(value).encode("utf-8"))


def read_body() -> tuple[str, str, str]:
    text = normalize_newlines(SRC.read_text(encoding="utf-8"))
    lines = text.split("\n")
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"Expected a title heading in {SRC}")
    title = lines[0][2:].strip()
    try:
        abstract_heading = lines.index("## Abstract")
        first_section = next(i for i in range(abstract_heading + 1, len(lines))
                             if lines[i].startswith("## 1."))
        references_heading = next(
            (i for i in range(first_section + 1, len(lines)) if lines[i] == "## References"),
            len(lines),
        )
    except StopIteration as exc:
        raise ValueError("manuscript_body.md must contain Abstract, section 1 and References") from exc
    abstract = "\n".join(lines[abstract_heading + 1:first_section]).strip()
    body = "\n".join(lines[first_section:references_heading]).strip()
    if not abstract or not body:
        raise ValueError("The manuscript body has an empty abstract or main text")
    body = re.sub(r"^## \d+\.\s+", "## ", body, flags=re.MULTILINE)
    body = re.sub(r"^### \d+\.\d+\.\s+", "### ", body, flags=re.MULTILINE)
    return title, abstract, body


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
    if candidate.exists():
        return str(candidate)
    return name


def run(command: list[str], index: int) -> None:
    result = subprocess.run(command, cwd=BUILD, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    write_utf8_lf(BUILD / f"compile_{index}.txt", result.stdout + "\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Compile step failed: {command}; see {BUILD / f'compile_{index}.txt'}")


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    title, abstract, body = read_body()
    figures = re.findall(r"!\[[^\]]*\]\((figures_v\d+/[^)]+)\)", body)
    for figure in figures:
        source = ROOT / "manuscript" / figure
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, BUILD / source.name)
        body = body.replace(figure, source.name)

    conversion = pypandoc.convert_text(
        body, "latex", format="markdown+tex_math_single_backslash",
        extra_args=["--natbib", "--wrap=none", "--shift-heading-level-by=-1"],
    )
    conversion = normalize_newlines(conversion)
    def format_policy_table(match):
        chunk = match.group(0)
        if 'Compound RMSE reduction' in chunk:
            widths = iter(['0.20', '0.10', '0.35', '0.35'])
            chunk = re.sub(r'\\real\{0\.2500\}', lambda _: r'\real{' + next(widths) + '}', chunk, count=4)
            return '\\begingroup\\small\n' + chunk + '\n\\endgroup'
        if 'Scalar full RMSE' in chunk:
            widths = iter(['0.16', '0.04', '0.10', '0.10', '0.18', '0.14', '0.14', '0.14'])
            chunk = re.sub(r'\\real\{0\.1250\}', lambda _: r'\real{' + next(widths) + '}', chunk, count=8)
            chunk = re.sub(r'\\real\{0\.1667\}', lambda _: r'\real{' + next(widths) + '}', chunk, count=8)
            return '\\begingroup\\scriptsize\n' + chunk + '\n\\endgroup'
        if 'Accuracy charge (CNY)' not in chunk:
            return chunk
        widths = iter(['0.10', '0.30', '0.13', '0.25', '0.22'])
        chunk = re.sub(r'\\real\{0\.2000\}', lambda _: r'\real{' + next(widths) + '}', chunk, count=5)
        return '\\begingroup\\small\n' + chunk + '\n\\endgroup'
    conversion = re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}', format_policy_table, conversion, flags=re.DOTALL)
    figure_numbers={Path(path).name:number+1 for number,path in enumerate(figures)}
    seen=set()
    def positioned_figure(match):
        chunk=match.group(0)
        name=re.search(r'\\includegraphics[^\n]*?\{([^{}]+\.pdf)\}',chunk).group(1)
        number=figure_numbers[name];seen.add(number)
        width='.66\\linewidth' if name=='fig08_shared_gasf.pdf' else '\\linewidth'
        chunk=re.sub(r'\\pandocbounded\{\\includegraphics\[.*?\]\{[^{}]+\.pdf\}\}',
                     lambda m:'\\includegraphics[width='+width+']{'+name+'}',chunk,flags=re.DOTALL)
        chunk=chunk.replace('\\begin{figure}','\\begin{figure}[H]')
        chunk=chunk.replace('\\end{figure}',f'\\label{{fig:main{number}}}\n\\end{{figure}}')
        return chunk
    conversion=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',positioned_figure,conversion,flags=re.DOTALL)
    if seen!=set(range(1,len(figures)+1)):raise ValueError('Figure numbering incomplete')
    conversion=re.sub(r'\b(?:Fig\.|Figure)\s+(\d+)\b',lambda m:'Fig.~\\ref{fig:main'+m.group(1)+'}',conversion)
    conversion = conversion.replace("\\section{", "\\FloatBarrier\n\\section{")
    abstract_latex = normalize_newlines(
        pypandoc.convert_text(abstract, "latex", format="markdown", extra_args=["--wrap=none"])
    )

    template = next((ROOT / "manuscript" / "applied_energy" / "official_template").rglob("elsarticle.cls")).parent
    for name in ("elsarticle.cls", "elsarticle-num.bst"):
        shutil.copy2(template / name, BUILD / name)
    bib, keys = bibliography()
    cited = set(re.findall(r"@([A-Za-z0-9_-]+)", abstract + "\n" + body))
    if missing := cited - keys:
        raise ValueError(f"Unresolved citation keys: {sorted(missing)}")
    write_utf8_lf(BUILD / "references.bib", bib)

    metadata = json.loads((ROOT / "manuscript" / "applied_energy" / "frontmatter.json").read_text(encoding="utf-8"))
    authors = "\n".join(
        "\\author[aff1,aff2]{" + author["name"] + (r"\corref{cor1}" if author.get("corresponding") else "") + "}\n"
        + "\\ead{" + author["email"] + "}"
        for author in metadata["authors"]
    )
    authors += "\n\\cortext[cor1]{Corresponding author.}"
    addresses = "\n".join(
        "\\address[aff" + str(index + 1) + "]{" + value + "}"
        for index, value in enumerate(metadata["affiliations"])
    )
    header = r"""\documentclass[preprint,12pt]{elsarticle}
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\usepackage{amsmath,amssymb,booktabs,longtable,array,graphicx,adjustbox,placeins,calc,caption,float,xurl}
\usepackage[unicode,colorlinks=true,allcolors=blue]{hyperref}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\newcounter{none}
\providecommand{\pandocbounded}[1]{\begin{adjustbox}{max width=\linewidth,max totalheight=.65\textheight,keepaspectratio}#1\end{adjustbox}}
\setlength{\emergencystretch}{3em}
\setlength{\intextsep}{10pt plus 2pt minus 2pt}
\makeatletter
\def\ps@pprintTitle{\let\@oddhead\@empty\let\@evenhead\@empty\def\@oddfoot{\footnotesize\itshape Working manuscript, v26\hfill\today}\let\@evenfoot\@oddfoot}
\makeatother
\begin{document}
\begin{frontmatter}
"""
    funding = "\n\\section*{Funding}\n" + metadata["funding"]
    for heading, key in [('Acknowledgements', 'acknowledgements'),
                         ('CRediT authorship contribution statement', 'author_contributions'),
                         ('Declaration of competing interest', 'conflicts')]:
        if metadata.get(key):
            funding += '\n\\section*{' + heading + '}\n' + metadata[key] + '\n'
    declaration_marker = "\\section{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}"
    if declaration_marker in conversion:
        conversion_before_ai, declaration_after_ai = conversion.split(declaration_marker, 1)
        conversion = conversion_before_ai + funding + "\n" + declaration_marker + declaration_after_ai
        funding = ""
    source = (
        header + "\\title{" + title + "}\n" + authors + "\n" + addresses
        + "\n\\begin{abstract}\n" + abstract_latex + "\n\\end{abstract}\n\\end{frontmatter}\n"
        + conversion + funding + "\n\\bibliographystyle{elsarticle-num}\n\\bibliography{references}\n\\end{document}\n"
    )
    write_utf8_lf(BUILD / "main.tex", source)

    commands = [
        [executable("xelatex"), "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        [executable("bibtex"), "main"],
        [executable("xelatex"), "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        [executable("xelatex"), "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ]
    for index, command in enumerate(commands):
        run(command, index)
    log = (BUILD / "main.log").read_text(encoding="utf-8", errors="replace")
    if "There were undefined references" in log or re.search(r"Citation .* undefined", log):
        raise ValueError("Unresolved citations remain in the final LaTeX log")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUILD / "main.pdf", OUT)
    reader = PdfReader(str(OUT))
    report = {
        "status": "COMPILED_LAYOUT_REVIEW_PENDING",
        "pages": len(reader.pages),
        "cited_references": len(cited),
        "bibliography_entries": len(keys),
        "figures": len(figures),
        "linked_figure_references": len(re.findall(r'\\ref\{fig:main\d+\}',conversion)),
        "placement": "H at evidence paragraph; figure08 square export shown at 66 percent text width",
        "pdf": str(OUT),
        "longtable_count": conversion.count(r"\begin{longtable}"),
        "overfull_box_count": log.count("Overfull \\hbox"),
    }
    write_utf8_lf(BUILD / "build_report.json", json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
