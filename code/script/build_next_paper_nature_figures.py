"""Build the final Nature figure set for the process-predictability paper."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/next_paper"
OUT = DATA / "manuscript_figures/nature_v1"
SOURCES = OUT / "sources"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES.mkdir(parents=True, exist_ok=True)

LEADS = [15, 60, 120, 240, 480, 720]
BLUE = "#2166AC"
ORANGE = "#D55E00"
GREEN = "#009E73"
WONG_ORANGE = "#E69F00"
GREY = "#5B6470"
LIGHT_GREY = "#E8EBEF"
TEXT = "#18212B"


def configure() -> None:
    for name in ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"]:
        path = Path("/mnt/c/Windows/Fonts") / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relpath(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def row(df: pd.DataFrame, **filters: object) -> pd.Series:
    mask = np.ones(len(df), dtype=bool)
    for key, value in filters.items():
        mask &= df[key].astype(str).eq(str(value)).to_numpy()
    out = df.loc[mask]
    if len(out) != 1:
        raise ValueError(f"Expected one row for {filters}, got {len(out)}")
    return out.iloc[0]


def selected(df: pd.DataFrame, **filters: object) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for key, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            mask &= df[key].astype(str).isin([str(v) for v in value]).to_numpy()
        else:
            mask &= df[key].astype(str).eq(str(value)).to_numpy()
    return df.loc[mask].copy()


def source_record(
    figure: str,
    panel: str,
    frame: pd.DataFrame,
    source_paths: list[Path],
    filters: dict[str, object],
    metric: str,
    reference: str,
    sample_n: object = None,
) -> dict[str, object]:
    out_path = SOURCES / f"{figure}_{panel}.csv"
    frame.to_csv(out_path, index=False)
    return {
        "panel": panel,
        "generated_csv": relpath(out_path),
        "generated_rows": int(len(frame)),
        "source_files": [
            {"path": relpath(path), "sha256": digest(path)} for path in source_paths
        ],
        "filters": filters,
        "metric": metric,
        "reference": reference,
        "sample_n": sample_n,
    }


def write_manifest(figure: str, panels: list[dict[str, object]]) -> None:
    manifest = {
        "figure": figure,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": relpath(Path(__file__)),
        "panels": panels,
    }
    (OUT / f"{figure}_source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.13, 1.04, letter, transform=ax.transAxes, fontsize=8,
        fontweight="bold", va="top", ha="left", clip_on=False, color=TEXT,
    )


def clean_axes(ax: plt.Axes, *, xgrid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2, pad=2)
    ax.grid(axis="x" if xgrid else "y", color=LIGHT_GREY, linewidth=0.45)
    ax.set_axisbelow(True)


def lead_axis(ax: plt.Axes) -> None:
    ticks = np.log2(np.asarray(LEADS, dtype=float))
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v) for v in LEADS])
    ax.set_xlabel("Issue lead (min)")


def plot_ci(ax: plt.Axes, values: list[list[float]], color: str, marker: str = "o",
            x: np.ndarray | None = None, label: str | None = None,
            linestyle: str = "-") -> None:
    arr = np.asarray(values, dtype=float)
    if x is None:
        x = np.log2(np.asarray(LEADS, dtype=float))
    ax.errorbar(
        x, arr[:, 0],
        yerr=[arr[:, 0] - arr[:, 1], arr[:, 2] - arr[:, 0]],
        color=color, marker=marker, linestyle=linestyle, linewidth=1.15,
        markersize=3.5, capsize=2.2, capthick=0.7, label=label, zorder=3,
    )


def fig1_process_objects() -> None:
    figure = "fig1_process_objects"
    definitions = pd.DataFrame([
        ["issue", "t=T-h", "issue-time inputs end here"],
        ["target_start", "T", "target start is unobserved at issue"],
        ["target_node_0", "T", "17 native 15-min nodes"],
        ["target_node_16", "T+4h", "17 native 15-min nodes"],
        ["threshold", "0.2 pu", "NP005 and NP009 protocol"],
    ], columns=["object", "notation", "definition"])
    excursion = pd.DataFrame([
        ["up", "max(y_k-min_{j<=k} y_j)", "ordered up excursion >= 0.2 pu", True],
        ["down", "max(max_{j<=k} y_j-y_k)", "ordered down excursion >= 0.2 pu", True],
    ], columns=["object", "operator", "definition", "illustrative_curve"])
    states = pd.DataFrame([
        ["state", "0", "none"],
        ["state", "up", "up excursion"],
        ["state", "down", "down excursion"],
        ["state", "both", "return / both excursions"],
        ["timing", "tau_down", "first ordered down passage; conditional on down event"],
        ["timing", "bins", "15, 30, ..., 240 min and none"],
    ], columns=["object", "notation", "definition"])
    routing = pd.DataFrame([
        ["arm", "X_P", "power history", "timing head"],
        ["arm", "X_W", "issued weather", "included in joint arm"],
        ["arm", "X_J", "X_P + X_W", "state head"],
        ["route", "state=X_J; timing=X_P", "target-aligned route", "frozen source predictions"],
    ], columns=["object", "notation", "definition", "use"])
    p005 = DATA / "np005/protocol.json"
    p009 = DATA / "np009/protocol.json"
    p025 = DATA / "np025/protocol.json"
    p026 = DATA / "np026/protocol.json"
    arr_script = ROOT / "script/next_paper_arrival_time.py"
    lead_script = ROOT / "script/next_paper_leadtime.py"

    panels = [
        source_record(figure, "a", definitions, [p005, p009], {},
                      "17-node target boundary", "NP005/NP009 protocol", None),
        source_record(figure, "b", excursion, [p005, p009, arr_script, lead_script],
                      {"threshold": "0.2"}, "ordered excursion", "NP005/NP009 definitions", None),
        source_record(figure, "c", states, [p009], {},
                      "state and conditional first passage", "NP009 protocol", None),
        source_record(figure, "d", routing, [p025, p026], {},
                      "three information arms and target-aligned routing", "NP025/NP026 protocols", None),
    ]
    write_manifest(figure, panels)

    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.05), constrained_layout=True)
    ax = axes[0, 0]
    panel_letter(ax, "a")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.axvspan(0.04, 0.31, color="#F2F4F7", zorder=0)
    ax.axvspan(0.34, 0.95, color="#EAF1F8", zorder=0)
    ax.annotate("", xy=(0.96, 0.52), xytext=(0.05, 0.52),
                arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.0))
    ax.text(0.16, 0.80, "issue-time information", ha="center", fontsize=7, fontweight="bold")
    ax.text(0.65, 0.80, "shared future target", ha="center", fontsize=7, fontweight="bold")
    ax.text(0.15, 0.64, "t = T − h", ha="center", fontsize=7, color=BLUE, fontweight="bold")
    ax.text(0.15, 0.41, "inputs end at issue", ha="center", fontsize=6, color=GREY)
    ax.text(0.37, 0.64, "T", ha="center", fontsize=7, color=TEXT, fontweight="bold")
    ax.text(0.50, 0.41, "target start is unobserved", ha="center", fontsize=5.8, color=GREY)
    node_x = np.linspace(0.39, 0.93, 17)
    for i, xx in enumerate(node_x):
        ax.plot([xx, xx], [0.47, 0.57], color=BLUE, linewidth=0.75)
        if i in (0, 4, 8, 12, 16):
            ax.text(xx, 0.28, f"{i*15} min", ha="center", fontsize=5.4, color=GREY)
    ax.text(0.66, 0.22, "17 native 15-min nodes: T … T+4 h", ha="center", fontsize=6.2, color=TEXT)
    ax.text(0.15, 0.55, "power / weather / local history", ha="center", fontsize=5.7, color=TEXT)

    ax = axes[0, 1]
    panel_letter(ax, "b")
    ax.set_xlim(-0.4, 16.4); ax.set_ylim(0.23, 0.86)
    x = np.arange(17)
    y = np.array([0.50, 0.55, 0.63, 0.72, 0.68, 0.60, 0.51, 0.40,
                  0.32, 0.39, 0.50, 0.58, 0.65, 0.60, 0.52, 0.45, 0.48])
    ax.plot(x, y, color=BLUE, linewidth=1.5, marker="o", markersize=2.2)
    ax.annotate("up excursion ≥ 0.2 pu", xy=(3, 0.72), xytext=(0.5, 0.81),
                fontsize=5.9, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=0.8))
    ax.annotate("down excursion ≥ 0.2 pu", xy=(8, 0.32), xytext=(8.8, 0.80),
                fontsize=5.9, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=0.8))
    ax.text(0.98, 0.96, "illustrative path (not data)", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color=GREY)
    ax.set_ylabel("normalized power")
    ax.set_xlabel("target node k")
    ax.set_xticks([0, 4, 8, 12, 16])
    clean_axes(ax)

    ax = axes[1, 0]
    panel_letter(ax, "c")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.04, 0.83, "state object", fontsize=7, fontweight="bold")
    for i, label in enumerate(["0", "up", "down", "both"]):
        xx = 0.04 + i * 0.23
        ax.add_patch(FancyBboxPatch((xx, 0.60), 0.18, 0.14,
                                    boxstyle="round,pad=0.015", facecolor="white",
                                    edgecolor=BLUE, linewidth=1.0))
        ax.text(xx + 0.09, 0.67, label, ha="center", va="center", fontsize=6.5)
    ax.annotate("", xy=(0.50, 0.44), xytext=(0.50, 0.58),
                arrowprops=dict(arrowstyle="->", color=GREY, linewidth=0.8))
    ax.text(0.04, 0.36, "conditional first passage", fontsize=7, fontweight="bold")
    ax.text(0.04, 0.22, "τ↓ = first ordered down passage", fontsize=6.2, color=ORANGE)
    ax.text(0.04, 0.10, "conditional on down event · bins 15 … 240 min + none",
            fontsize=5.8, color=GREY)

    ax = axes[1, 1]
    panel_letter(ax, "d")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [
        (0.05, 0.68, 0.34, 0.16, "Joint arm X_J", BLUE),
        (0.05, 0.30, 0.34, 0.16, "Power arm X_P", ORANGE),
        (0.54, 0.68, 0.38, 0.16, "State head S", BLUE),
        (0.54, 0.30, 0.38, 0.16, "Timing head τ↓", ORANGE),
    ]
    for xx, yy, ww, hh, label, color in boxes:
        ax.add_patch(FancyBboxPatch((xx, yy), ww, hh, boxstyle="round,pad=0.02",
                                    facecolor="white", edgecolor=color, linewidth=1.2))
        ax.text(xx + ww/2, yy + hh/2, label, ha="center", va="center",
                fontsize=6.8, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.05, 0.08), 0.34, 0.12, boxstyle="round,pad=0.02",
                                facecolor="#F2F4F7", edgecolor=GREEN, linewidth=1.0,
                                linestyle="--"))
    ax.text(0.22, 0.14, "Issued weather X_W", ha="center", va="center", fontsize=6.3)
    ax.annotate("", xy=(0.54, 0.76), xytext=(0.39, 0.76),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, linewidth=1.2))
    ax.annotate("", xy=(0.54, 0.38), xytext=(0.39, 0.38),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=1.2))
    ax.annotate("", xy=(0.22, 0.68), xytext=(0.22, 0.20),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, linewidth=0.9,
                                linestyle="--"))
    ax.text(0.95, 0.05, "route: state ← X_J; timing ← X_P",
            ha="right", fontsize=5.8, color=GREY)
    ax.text(0.22, 0.88, "X_J = X_P + X_W", ha="center", fontsize=6.0, color=GREY)
    save(fig, figure)


def _load_np005() -> tuple[list[list[float]], pd.DataFrame]:
    df = pd.read_csv(DATA / "np005/paired_intervals.csv")
    sub = selected(df, candidate=[f"power_weather:{h}:joint" for h in LEADS],
                   reference="frequency:0:reference", metric="return_brier",
                   state="all", block_days=7)
    vals = []
    for h in LEADS:
        q = row(df, candidate=f"power_weather:{h}:joint",
                reference="frequency:0:reference", metric="return_brier",
                state="all", block_days=7)
        vals.append([float(q[k]) for k in ["relative_loss_reduction_pct", "low", "high"]])
    return vals, sub


def _load_np009(metric: str) -> tuple[list[list[float]], pd.DataFrame]:
    df = pd.read_csv(DATA / "np009/paired_intervals.csv")
    sub = selected(df, candidate=[f"{h}__down__context" for h in LEADS],
                   reference=[f"{h}__down__frequency" for h in LEADS],
                   metric=metric, state="all", block_days=7)
    vals = []
    for h in LEADS:
        q = row(df, candidate=f"{h}__down__context",
                reference=f"{h}__down__frequency", metric=metric,
                state="all", block_days=7)
        vals.append([float(q[k]) for k in ["relative_loss_reduction_pct", "low", "high"]])
    return vals, sub


def fig2_capability() -> None:
    figure = "fig2_process_capability"
    np005_vals, src_a = _load_np005()
    rps_vals, src_b_rps = _load_np009("conditional_rps")
    mae_vals, src_b_mae = _load_np009("conditional_median_mae_minutes")
    d22_path = DATA / "np022/test_summary.csv"
    d22 = pd.read_csv(d22_path)
    src_c = selected(d22, site=["lahaute", "suining"], lead_minutes=15,
                     task="downward_timing",
                     metric=["conditional_rps", "conditional_median_mae_minutes"],
                     candidate="power_history_extra_trees", reference="training_frequency")
    panels = [
        source_record(figure, "a", src_a, [DATA / "np005/paired_intervals.csv"],
                      {"metric": "return_brier", "block_days": 7},
                      "return-event Brier reduction", "power_weather vs frequency", 6496),
        source_record(figure, "b", pd.concat([src_b_rps, src_b_mae], ignore_index=True),
                      [DATA / "np009/paired_intervals.csv"],
                      {"metrics": ["conditional_rps", "conditional_median_mae_minutes"],
                       "block_days": 7}, "conditional timing reduction",
                      "context vs frequency", 6496),
        source_record(figure, "c", src_c, [d22_path],
                      {"lead_minutes": 15, "metrics": ["conditional_rps", "conditional_median_mae_minutes"]},
                      "external 15-min timing reduction",
                      "power history vs training frequency", {"lahaute": 13912, "suining": 6996}),
    ]
    write_manifest(figure, panels)

    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.75), constrained_layout=True)
    for ax, letter in zip(axes, "abc"):
        panel_letter(ax, letter)
    ax = axes[0]
    plot_ci(ax, np005_vals, BLUE, marker="o")
    ax.axhline(0, color=GREY, linewidth=0.6, linestyle="--")
    lead_axis(ax); ax.set_ylabel("Return-event Brier reduction (%)")
    clean_axes(ax)

    ax = axes[1]
    plot_ci(ax, rps_vals, ORANGE, marker="o", label="RPS")
    plot_ci(ax, mae_vals, WONG_ORANGE, marker="s", label="median MAE")
    ax.axhline(0, color=GREY, linewidth=0.6, linestyle="--")
    lead_axis(ax); ax.set_ylabel("Timing loss reduction (%)")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)
    clean_axes(ax)

    ax = axes[2]
    sites = ["La Haute\nBorne", "Suining"]
    xpos = np.arange(2)
    for metric, marker, color, label in [
        ("conditional_rps", "o", ORANGE, "RPS"),
        ("conditional_median_mae_minutes", "s", WONG_ORANGE, "median MAE"),
    ]:
        vals = []
        lows = []
        highs = []
        for site in ["lahaute", "suining"]:
            q = row(d22, site=site, lead_minutes=15, task="downward_timing",
                    metric=metric, candidate="power_history_extra_trees",
                    reference="training_frequency")
            vals.append(float(q.relative_loss_reduction_pct))
            lows.append(float(q.low)); highs.append(float(q.high))
        arr = np.asarray(vals)
        shift = -0.07 if marker == "o" else 0.07
        ax.errorbar(xpos + shift, arr, yerr=[arr - lows, highs - arr],
                    fmt=marker, color=color, ecolor=color, capsize=2,
                    capthick=0.7, linewidth=0.8, markersize=4, label=label)
    ax.axhline(0, color=GREY, linewidth=0.6, linestyle="--")
    ax.set_xticks(xpos, sites); ax.set_xlabel("External site")
    ax.set_ylabel("15-min timing reduction (%)")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)
    clean_axes(ax)
    save(fig, figure)


def _load_decomposition(path: Path, arm: str, metric: str) -> tuple[list[list[float]], pd.DataFrame]:
    df = pd.read_csv(path)
    task = "event" if metric == "multiclass_brier" else "downward_timing"
    sub = selected(df, lead_minutes=LEADS, arm=arm, metric=metric, task=task)
    vals = []
    for h in LEADS:
        q = row(df, lead_minutes=h, arm=arm, metric=metric, task=task)
        vals.append([float(q[k]) for k in ["relative_loss_reduction_pct", "low", "high"]])
    return vals, sub.sort_values("lead_minutes")


def fig3_information_decomposition() -> None:
    figure = "fig3_information_decomposition"
    p25 = DATA / "np025/test_summary.csv"
    p28 = DATA / "np028/test_summary.csv"
    state25, src_a = _load_decomposition(p25, "joint_increment", "multiclass_brier")
    time25, src_b = _load_decomposition(p25, "power_increment_on_weather", "conditional_rps")
    state28, src_c = _load_decomposition(p28, "joint_increment", "multiclass_brier")
    time28, src_d = _load_decomposition(p28, "power_increment_on_weather", "conditional_rps")
    panels = [
        source_record(figure, "a", src_a, [p25],
                      {"arm": "joint_increment", "candidate": "joint", "reference": "power",
                       "metric": "multiclass_brier"}, "joint vs power four-class state",
                      "joint vs power", 6996),
        source_record(figure, "b", src_c, [p28],
                      {"arm": "joint_increment", "candidate": "joint", "reference": "power",
                       "metric": "multiclass_brier"}, "joint vs power four-class state",
                      "joint vs power", 3772),
        source_record(figure, "c", src_b, [p25],
                      {"arm": "power_increment_on_weather", "candidate": "joint", "reference": "weather",
                       "metric": "conditional_rps"}, "joint vs weather timing RPS",
                      "joint vs weather", 6996),
        source_record(figure, "d", src_d, [p28],
                      {"arm": "power_increment_on_weather", "candidate": "joint", "reference": "weather",
                       "metric": "conditional_rps"}, "joint vs weather timing RPS",
                      "joint vs weather", 3772),
    ]
    write_manifest(figure, panels)

    fig, axes = plt.subplots(2, 2, figsize=(7.3, 4.55), constrained_layout=True)
    for ax, letter in zip(axes.flat, "abcd"):
        panel_letter(ax, letter)
    fig.text(0.29, 0.995, "Historical forecast package · NP025", ha="center",
             va="top", fontsize=7, fontweight="bold")
    fig.text(0.73, 0.995, "Strict previous_day1 package · NP028", ha="center",
             va="top", fontsize=7, fontweight="bold")
    for ax, vals, color, ylabel in [
        (axes[0, 0], state25, BLUE, "Four-class state reduction (%)"),
        (axes[0, 1], state28, BLUE, "Four-class state reduction (%)"),
        (axes[1, 0], time25, ORANGE, "Downward timing RPS reduction (%)"),
        (axes[1, 1], time28, ORANGE, "Downward timing RPS reduction (%)"),
    ]:
        plot_ci(ax, vals, color, marker="o")
        ax.axhline(0, color=GREY, linewidth=0.6, linestyle="--")
        lead_axis(ax); ax.set_ylabel(ylabel)
        clean_axes(ax)
    axes[0, 1].text(0.97, 0.92, "12.53% at 720 min",
                     transform=axes[0, 1].transAxes, ha="right", fontsize=5.8, color=BLUE)
    axes[1, 1].text(0.97, 0.92, "strict 24-h clock",
                     transform=axes[1, 1].transAxes, ha="right", fontsize=5.8, color=ORANGE)
    save(fig, figure)


def route_values(path: Path) -> tuple[list[list[float]], pd.DataFrame]:
    df = pd.read_csv(path)
    sub = selected(df, lead_minutes=LEADS, task="downward_timing", metric="conditional_rps",
                   candidate="target_aligned_route", reference="joint_timing")
    vals = []
    for h in LEADS:
        q = row(df, lead_minutes=h, task="downward_timing", metric="conditional_rps",
                candidate="target_aligned_route", reference="joint_timing")
        vals.append([float(q[k]) for k in ["relative_loss_reduction_pct", "low", "high"]])
    return vals, sub.sort_values("lead_minutes")


def fig4_routing() -> None:
    figure = "fig4_target_aligned_routing"
    p26 = DATA / "np026/routing_summary.csv"
    p29 = DATA / "np029/routing_summary.csv"
    p27 = DATA / "np027/monthly_summary.csv"
    p30 = DATA / "np030/monthly_summary.csv"
    route26, src_a = route_values(p26)
    route29, src_b = route_values(p29)
    src_c = selected(pd.read_csv(p27), lead_minutes=[240, 480], metric="conditional_rps")
    src_d = selected(pd.read_csv(p30), lead_minutes=[15, 60, 120], metric="conditional_rps")
    panels = [
        source_record(figure, "a", src_a, [p26],
                      {"metric": "conditional_rps", "candidate": "target_aligned_route",
                       "reference": "joint_timing"}, "NP026 timing route RPS", "joint timing", 6996),
        source_record(figure, "b", src_b, [p29],
                      {"metric": "conditional_rps", "candidate": "target_aligned_route",
                       "reference": "joint_timing"}, "NP029 strict timing route RPS", "joint timing", 3772),
        source_record(figure, "c", src_c, [p27],
                      {"metric": "conditional_rps", "leads": [240, 480]}, "NP027 monthly RPS stability",
                      "joint timing", "month-specific"),
        source_record(figure, "d", src_d, [p30],
                      {"metric": "conditional_rps", "leads": [15, 60, 120]}, "NP030 strict monthly RPS",
                      "joint timing", "month-specific"),
    ]
    write_manifest(figure, panels)

    fig, axes = plt.subplots(2, 2, figsize=(7.3, 4.65), constrained_layout=True)
    for ax, letter in zip(axes.flat, "abcd"):
        panel_letter(ax, letter)
    fig.text(0.29, 0.995, "NP026 historical forecast package", ha="center",
             va="top", fontsize=7, fontweight="bold")
    fig.text(0.73, 0.995, "NP029 strict previous_day1 package", ha="center",
             va="top", fontsize=7, fontweight="bold")
    for ax, vals, color in [(axes[0, 0], route26, BLUE), (axes[0, 1], route29, GREEN)]:
        plot_ci(ax, vals, color, marker="o")
        ax.axhline(0, color=GREY, linewidth=0.6, linestyle="--")
        lead_axis(ax); ax.set_ylabel("Timing RPS reduction (%)")
        clean_axes(ax)
    axes[0, 0].text(0.02, 0.08, "state head: exact identity", transform=axes[0, 0].transAxes,
                     fontsize=5.8, color=GREY)
    for ax, path, leads, colors in [
        (axes[1, 0], p27, [240, 480], [BLUE, ORANGE]),
        (axes[1, 1], p30, [15, 60, 120], [ORANGE, WONG_ORANGE, GREEN]),
    ]:
        df = pd.read_csv(path)
        months = sorted(df["month"].astype(str).unique())
        month_x = np.arange(len(months))
        for lead, color in zip(leads, colors):
            sub = selected(df, lead_minutes=lead, metric="conditional_rps")
            sub = sub.set_index("month").reindex(months).reset_index()
            vals = sub["relative_loss_reduction_pct"].to_numpy(float)
            lows = sub["low"].to_numpy(float)
            highs = sub["high"].to_numpy(float)
            ax.errorbar(month_x, vals, yerr=[vals-lows, highs-vals],
                        marker="o", color=color, linewidth=1.0, markersize=3.2,
                        capsize=2, capthick=0.6, label=f"{lead} min")
        ax.axhline(0, color=GREY, linewidth=0.6, linestyle="--")
        ax.set_xticks(month_x, months)
        ax.set_xlabel("UTC month")
        ax.set_ylabel("Timing RPS reduction (%)")
        ax.legend(frameon=False, loc="upper left", handlelength=1.0, ncol=2)
        clean_axes(ax)
    save(fig, figure)


def write_captions() -> None:
    text = """# Nature v1 final captions

## Fig. 1 | Process objects and information boundary

a, The issue-time boundary is t=T−h, whereas the shared target begins at T and contains 17 native 15-min nodes through T+4 h; T is unobserved at issue. b, An illustrative path (not an observed record) shows the ordered up and down excursion operators used by the event protocol, each with a 0.2-pu threshold. c, The same path yields a four-class state object and a conditional first-passage object; timing is evaluated only when the corresponding excursion occurs. d, The three information arms are power history X_P, issued weather X_W and their joint arm X_J; the target-aligned route keeps X_J for state risk and X_P for first-passage timing. No performance value is encoded in this schematic.

## Fig. 2 | Process timing skill and external process-object replication

a, Relative reduction in return-event Brier loss for the NP005 power-plus-weather arm relative to the training-frequency reference across six issue leads (n=6,496 windows). b, Relative reductions in conditional first-passage RPS and median arrival-time MAE for the NP009 context arm relative to the frequency arm, with paired 7-day calendar-block intervals (n=6,496). c, Fifteen-minute downward timing reductions from NP022 using only issue-time power history at La Haute Borne and Suining; circles denote RPS and squares median arrival-time MAE. Positive values indicate lower candidate loss. All intervals are paired 7-day calendar-block intervals.

## Fig. 3 | Information gains under two weather packages

a,b, Joint-versus-power relative reductions in the four-class state Brier score for the NP025 historical forecast package and the NP028 strict previous_day1 package, respectively. c,d, Joint-versus-weather relative reductions in downward first-passage RPS for the same packages. Points and bars are paired 7-day calendar-block intervals across all six leads. The NP028 state increment at 720 min is 12.53%; the 12.11% weather-only-versus-power comparison is a different arm and is not plotted. NP025 uses n=6,996 test windows and NP028 n=3,772.

## Fig. 4 | Target-aligned timing routing under two issue-time contracts

a,b, Relative reduction in downward first-passage RPS for the NP026 and NP029 target-aligned routes relative to their joint timing heads across six issue leads. The state head is replay-identical in the route. c, Frozen NP027 monthly RPS reductions for 240- and 480-min leads in June–August 2024. d, Frozen NP030 strict-clock monthly RPS reductions for 15-, 60- and 120-min leads in July–August 2024. Points and bars are paired 7-day calendar-block intervals. Positive and negative estimates remain visible; the panels do not define a forecast horizon.
"""
    (OUT / "captions.md").write_text(text, encoding="utf-8")


def write_bundle_manifest() -> None:
    stems = [
        "fig1_process_objects",
        "fig2_process_capability",
        "fig3_information_decomposition",
        "fig4_target_aligned_routing",
    ]
    outputs = []
    for stem in stems:
        outputs.append({
            "figure": stem,
            "files": {
                ext: {"path": relpath(OUT / f"{stem}.{ext}"),
                      "sha256": digest(OUT / f"{stem}.{ext}")}
                for ext in ["pdf", "svg", "png"]
            },
            "source_manifest": relpath(OUT / f"{stem}_source_manifest.json"),
        })
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": relpath(Path(__file__)),
        "captions": relpath(OUT / "captions.md"),
        "figures": outputs,
    }
    (OUT / "nature_v1_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    configure()
    fig1_process_objects()
    fig2_capability()
    fig3_information_decomposition()
    fig4_routing()
    write_captions()
    write_bundle_manifest()
    print(OUT)


if __name__ == "__main__":
    main()
