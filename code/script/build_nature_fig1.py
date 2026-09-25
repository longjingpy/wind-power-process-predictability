"""Build the Nature-style overview schematic for the process-object story."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path("/mnt/d/projects/WindPowerForcast")
OUT = ROOT / "outputs/next_paper/manuscript_figures/nature_v1"
OUT.mkdir(parents=True, exist_ok=True)
BLUE = "#2166AC"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#5B6470"
TEXT = "#18212B"
LIGHT = "#EEF2F5"


def box(ax, x, y, w, h, label, sub, color, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.014,rounding_size=0.025",
                                facecolor=fill, edgecolor=color, linewidth=1.3))
    ax.text(x + w / 2, y + h * 0.60, label, ha="center", va="center", fontsize=7.1, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=5.8, color=GREY)


def arrow(ax, a, b, color=TEXT, lw=1.1):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=lw, color=color, shrinkA=0, shrinkB=0))


def main():
    for f in ["arial.ttf", "arialbd.ttf"]:
        p = Path("/mnt/c/Windows/Fonts") / f
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.72), gridspec_kw={"wspace": 0.23})
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # a: issue-time boundary
    ax = axes[0]
    ax.text(0.00, 0.98, "a", fontsize=8, fontweight="bold", va="top")
    ax.text(0.08, 0.93, "Issue-time information", fontsize=8, fontweight="bold", va="top")
    box(ax, .06, .69, .84, .13, "Power history", "ends at issue", ORANGE, "#FFF7F0")
    box(ax, .06, .48, .84, .13, "Issued weather", "release time ≤ issue", GREEN, "#F1FAF6")
    box(ax, .06, .27, .84, .13, "Local wind history", "issue-time proxy", BLUE, "#F1F6FC")
    box(ax, .06, .06, .84, .13, "ERA5 / future wind", "retrospective diagnostic", GREY, "#F5F6F7")
    ax.text(.50, .21, "available at t_issue", ha="center", fontsize=6.3, color=TEXT)
    ax.text(.50, .01, "t_issue = T − h", ha="center", fontsize=6.3, color=TEXT)

    # b: shared target
    ax = axes[1]
    ax.text(0.00, 0.98, "b", fontsize=8, fontweight="bold", va="top")
    ax.text(0.08, 0.93, "Shared process target", fontsize=8, fontweight="bold", va="top")
    ax.add_patch(FancyBboxPatch((.06, .20), .86, .62, boxstyle="round,pad=0.02,rounding_size=0.025",
                                facecolor=LIGHT, edgecolor="#AAB5BF", linewidth=1.0))
    x = np.linspace(.13, .86, 17)
    y = .49 + .13 * np.sin(np.linspace(0, 2.2*np.pi, 17)) + .025 * np.sin(np.linspace(0, 8*np.pi, 17))
    ax.plot(x, y, color=BLUE, linewidth=1.5, zorder=3)
    ax.scatter(x, y, s=7, color=BLUE, zorder=4)
    ax.axvline(x[0], color=GREY, linewidth=.7, linestyle="--")
    ax.text(x[0], .15, "T", ha="center", fontsize=6.2)
    ax.text(x[-1], .15, "T+4 h", ha="center", fontsize=6.2)
    ax.text(.49, .73, "illustrative path", ha="center", fontsize=5.8, color=GREY, style="italic")
    ax.text(.49, .10, "17 native 15-min nodes", ha="center", fontsize=6.2, color=TEXT)
    ax.text(.49, .02, "0.2 ordered excursion threshold", ha="center", fontsize=5.5, color=GREY)

    # c: objects and scores
    ax = axes[2]
    ax.text(0.00, 0.98, "c", fontsize=8, fontweight="bold", va="top")
    ax.text(0.08, 0.93, "Process objects", fontsize=8, fontweight="bold", va="top")
    box(ax, .04, .66, .90, .15, "State risk  S", "Brier · occurrence / return", BLUE, "#F1F6FC")
    box(ax, .04, .43, .90, .15, "First passage  τ↓", "conditional RPS · median-MAE", ORANGE, "#FFF7F0")
    box(ax, .04, .20, .90, .15, "Range / path  R, O", "CRPS · coverage · order value", GREEN, "#F1FAF6")
    ax.text(.49, .08, "one target · separate scores", ha="center", fontsize=6.2, color=TEXT)
    ax.text(.49, .02, "supporting objects are not a joint path–time claim", ha="center", fontsize=5.2, color=GREY)
    fig.savefig(OUT / "fig1_process_objects.pdf", bbox_inches="tight", pad_inches=.10)
    fig.savefig(OUT / "fig1_process_objects.svg", bbox_inches="tight", pad_inches=.10)
    fig.savefig(OUT / "fig1_process_objects.png", dpi=450, bbox_inches="tight", pad_inches=.10)
    print(OUT / "fig1_process_objects.pdf")


if __name__ == "__main__":
    main()
