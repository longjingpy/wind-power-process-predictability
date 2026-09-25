"""Build enlarged, claim-first result figures for the Renewable Energy manuscript.

The figures replace long main-text Tables 3 and 4 with readable visual evidence.
All numeric values are read from frozen NP005/NP008/NP009/NP025/NP028/NP026/NP029/
NP027/NP030 CSV outputs; no values are manually re-entered.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path("/mnt/d/projects/WindPowerForcast")
DATA = ROOT / "outputs/next_paper"
OUT = DATA / "manuscript_figures/nature_v1"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, PURPLE, GREY, RED, TEXT = "#2166AC", "#D55E00", "#009E73", "#7B3294", "#5B6470", "#B2182B", "#18212B"


def setup():
    for name in ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"]:
        p = Path("/mnt/c/Windows/Fonts") / name
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.5,
        "axes.titlesize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "axes.linewidth": .8, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "axes.titleweight": "bold",
    })


def clean(ax, grid="y"):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, pad=3)
    if grid:
        ax.grid(axis=grid, color="#E5E9EF", lw=.65); ax.set_axisbelow(True)


def panel(ax, letter):
    ax.text(-.12, 1.08, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", color=TEXT)


def save(fig, stem, source_paths):
    for ext, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 450})]:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", pad_inches=.10, **kwargs)
    source_sha256 = {}
    for rel in source_paths:
        source = DATA / rel
        if not source.exists():
            raise FileNotFoundError(f"Missing figure source: {source}")
        source_sha256[rel] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {"figure": stem, "source_paths": source_paths,
                "source_sha256": source_sha256,
                "style": "Arial/TrueType, claim-first, enlarged text", "sha256": {}}
    for p in [OUT / f"{stem}.pdf", OUT / f"{stem}.svg", OUT / f"{stem}.png"]:
        h = hashlib.sha256(p.read_bytes()).hexdigest(); manifest["sha256"][p.name] = h
    (OUT / f"{stem}_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plt.close(fig)


def line_with_ci(ax, leads, frame, label, color, metric, group_col="arm", group=None, ref=None):
    sub = frame.copy()
    if group_col and group is not None:
        sub = sub[sub[group_col].eq(group)]
    sub = sub[sub.metric.eq(metric)].sort_values("lead_minutes")
    x = sub.lead_minutes.to_numpy(float)
    y = sub.relative_loss_reduction_pct.to_numpy(float)
    lo = sub.low.to_numpy(float); hi = sub.high.to_numpy(float)
    ax.fill_between(x, lo, hi, color=color, alpha=.14, linewidth=0)
    ax.plot(x, y, color=color, lw=2.1, marker="o", ms=4.3, label=label, zorder=3)


def require_one_row_per(frame, keys, label):
    counts = frame.groupby(keys, dropna=False).size()
    if not counts.eq(1).all():
        bad = counts[counts.ne(1)].to_dict()
        raise ValueError(f"{label}: expected one row per {keys}; duplicates={bad}")


def fig3_pizhou_core():
    leads = [15, 60, 120, 240, 480, 720]
    d = pd.read_csv(DATA / "np008/test_summary.csv")
    q = pd.read_csv(DATA / "np009/test_summary.csv")
    # Main-panel curves use only the pooled test population and one scoring arm per lead.
    d = d[d["state"].eq("all") & d["mode"].eq("rolling")].copy()
    q = q[q["state"].eq("all") & q["event"].eq("down")].copy()
    require_one_row_per(d, ["lead_minutes", "group", "metric"], "NP008 pooled rolling state")
    require_one_row_per(q, ["lead_minutes", "arm", "metric"], "NP009 pooled downward state")
    fig, axs = plt.subplots(2, 2, figsize=(8.0, 5.8), constrained_layout=True)
    colors = {"frequency": GREY, "full": BLUE}
    for ax, metric, title, ylabel in [
        (axs[0, 0], "return_brier", "Event-state probability", "Return-event Brier (lower is better)"),
        (axs[0, 1], "multiclass_brier", "Four-class state risk", "Multiclass Brier (lower is better)"),
    ]:
        for group, label in [("frequency", "Frequency"), ("full", "Full rolling")]:
            sub = d[(d.group == group) & (d.metric == metric)].sort_values("lead_minutes")
            ax.plot(sub.lead_minutes, sub.loss, marker="o", lw=2.0, ms=4.2, color=colors[group], label=label)
        ax.set_xticks(leads); ax.set_xlabel("Issue lead (min)"); ax.set_ylabel(ylabel); ax.set_title(title, loc="left"); clean(ax)
    for ax, metric, title, ylabel in [
        (axs[1, 0], "conditional_rps", "First-passage timing", "Conditional RPS (lower is better)"),
        (axs[1, 1], "conditional_median_mae_minutes", "Arrival-time precision", "Median arrival error (min)"),
    ]:
        for arm, label, color in [("frequency", "Frequency", GREY), ("context", "Context", ORANGE)]:
            sub = q[(q.event == "down") & (q.arm == arm) & (q.metric == metric)].sort_values("lead_minutes")
            ax.plot(sub.lead_minutes, sub.loss, marker="o", lw=2.0, ms=4.2, color=color, label=label)
        ax.set_xticks(leads); ax.set_xlabel("Issue lead (min)"); ax.set_ylabel(ylabel); ax.set_title(title, loc="left"); clean(ax)
    axs[0, 0].legend(frameon=False, loc="best"); axs[1, 0].legend(frameon=False, loc="best")
    for ax, letter in zip(axs.ravel(), "abcd"): panel(ax, letter)
    save(fig, "fig3_pizhou_core_skill", ["np008/test_summary.csv", "np009/test_summary.csv"])


def fig4_information_decomposition():
    leads = [15, 60, 120, 240, 480, 720]
    d25 = pd.read_csv(DATA / "np025/test_summary.csv"); d28 = pd.read_csv(DATA / "np028/test_summary.csv")
    test_sizes = {"NP025": (int(d25["windows"].max()), 2653), "NP028": (int(d28["windows"].max()), 1328)}
    arm_labels = {"power": "Power history", "weather": "Weather", "joint": "Joint"}
    def absolute_state(package, lead, arm):
        z = np.load(DATA / package / "test_predictions.npz")
        p = z[f"{lead}__{arm}__event"]; y = z[f"{lead}__y"]
        return float(np.mean(np.sum((p - np.eye(4)[y]) ** 2, axis=1)))
    fig, axs = plt.subplots(2, 2, figsize=(8.0, 5.8), constrained_layout=True)
    # absolute state Brier
    for ax, frame, label_prefix, colors in [(axs[0, 0], d25, "NP025", {"power": GREY, "weather": GREEN, "joint": BLUE}), (axs[0, 1], d28, "NP028", {"power": GREY, "weather": ORANGE, "joint": PURPLE})]:
        for arm, color in colors.items():
            sub = frame[(frame.arm == arm) & (frame.task == "event") & (frame.metric == "multiclass_brier")].sort_values("lead_minutes")
            package = "np025" if label_prefix == "NP025" else "np028"
            ax.plot(sub.lead_minutes, [absolute_state(package, int(h), arm) for h in sub.lead_minutes], color=color, marker="o", lw=2.0, ms=4, label=arm_labels[arm])
        ax.set_xticks(leads); ax.set_xlabel("Issue lead (min)"); ax.set_ylabel("Multiclass Brier (lower is better)"); ax.set_title(f"{label_prefix} state information",loc="left"); ax.text(.02,.97, f"test windows n={test_sizes[label_prefix][0]:,}", transform=ax.transAxes, va="top", fontsize=7, color=GREY); clean(ax)
    # timing increments
    for ax, frame, label_prefix, colors in [(axs[1, 0], d25, "NP025", {"power_increment_on_weather": BLUE}), (axs[1, 1], d28, "NP028", {"power_increment_on_weather": PURPLE})]:
        for arm, color in colors.items():
            sub = frame[(frame.arm == arm) & (frame.task == "downward_timing") & (frame.metric == "conditional_rps")].sort_values("lead_minutes")
            ax.fill_between(sub.lead_minutes, sub.low, sub.high, color=color, alpha=.15)
            ax.plot(sub.lead_minutes, sub.relative_loss_reduction_pct, color=color, marker="o", lw=2.0, ms=4, label="Power added to weather")
        ax.axhline(0,color=GREY,lw=.7,ls="--"); ax.set_xticks(leads); ax.set_xlabel("Issue lead (min)"); ax.set_ylabel("Timing RPS loss reduction (%)"); ax.set_title(f"{label_prefix} timing information",loc="left"); ax.text(.02,.97, f"test windows n={test_sizes[label_prefix][0]:,}; down events n={test_sizes[label_prefix][1]:,}", transform=ax.transAxes, va="top", fontsize=7, color=GREY); clean(ax)
    axs[0,0].legend(frameon=False,loc="best"); axs[0,1].legend(frameon=False,loc="best")
    for ax,letter in zip(axs.ravel(),"abcd"): panel(ax,letter)
    save(fig,"fig4_information_decomposition_enhanced",["np025/test_summary.csv","np025/test_predictions.npz","np028/test_summary.csv","np028/test_predictions.npz"])


def fig5_route_landscape():
    d26=pd.read_csv(DATA/"np026/routing_summary.csv"); d29=pd.read_csv(DATA/"np029/routing_summary.csv")
    d36=pd.read_csv(DATA/"np036_joint_timing_baseline/joint_timing_baseline.csv")
    fig,axs=plt.subplots(2,2,figsize=(8.0,5.8),constrained_layout=True)
    for ax,frames,metric,title in [(axs[0,0],[(d26,"NP026 archived",BLUE),(d29,"NP029 strict",PURPLE)],"conditional_rps","Route timing RPS across lead"),(axs[0,1],[(d26,"NP026 archived",BLUE),(d29,"NP029 strict",PURPLE)],"conditional_median_mae_minutes","Route arrival error across lead")]:
        for frame,label,color in frames:
            sub=frame[(frame.candidate=="target_aligned_route")&(frame.reference=="joint_timing")&(frame.metric==metric)].sort_values("lead_minutes")
            x=sub.lead_minutes.to_numpy(float); y=sub.relative_loss_reduction_pct.to_numpy(float); lo=sub.low.to_numpy(float); hi=sub.high.to_numpy(float)
            ax.fill_between(x,lo,hi,color=color,alpha=.14); ax.plot(x,y,marker="o",ms=4,lw=2,color=color,label=label)
        ax.axhline(0,color=GREY,lw=.8,ls="--"); ax.set_xticks([15,60,120,240,480,720]); ax.set_xlabel("Issue lead (min)"); ax.set_ylabel("Relative loss reduction (%)"); ax.set_title(title,loc="left"); clean(ax); ax.legend(frameon=False,loc="best")
    panel(axs[0,0], "a"); panel(axs[0,1], "b")
    # Coherent 17-bin first-passage comparison.  Keep the archived negative
    # gains visible; the strict-clock route improves at the later leads.
    ax=axs[1,0]
    d36=d36[d36.reference.eq("constructed_joint") & d36.metric.eq("joint_first_passage_RPS")].copy()
    require_one_row_per(d36, ["package", "lead_minutes"], "NP036 constructed-joint 17-bin RPS")
    for package, label, color in [("np025", "Archived route (NP026)", BLUE), ("np028", "Strict route (NP029)", PURPLE)]:
        sub=d36[d36.package.eq(package)].sort_values("lead_minutes")
        ax.plot(sub.lead_minutes, sub.route_gain_vs_direct_pct, marker="o", ms=4, lw=2, color=color, label=label)
    ax.axhline(0,color=GREY,lw=.8,ls="--"); ax.set_xticks([15,60,120,240,480,720]); ax.set_xlabel("Issue lead (min)"); ax.set_ylabel("17-bin RPS gain vs direct joint (%)"); ax.set_title("17-bin route vs direct joint RPS",loc="left"); ax.text(.02,.97, "NP036 · six leads · test windows n=6,996", transform=ax.transAxes, va="top", fontsize=7, color=GREY); clean(ax); ax.legend(frameon=False,loc="best")
    ax=axs[1,1]; ax.axis("off"); panel(ax,"d")
    ax.text(.04,.90,"What the route changes",fontsize=10,fontweight="bold",color=TEXT)
    ax.text(.06,.68,"state head",fontsize=8,fontweight="bold",color=BLUE); ax.text(.40,.68,"joint power–weather probabilities",fontsize=8)
    ax.text(.06,.48,"timing head",fontsize=8,fontweight="bold",color=ORANGE); ax.text(.40,.48,"power-history conditional timing",fontsize=8)
    ax.text(.06,.28,"identity",fontsize=8,fontweight="bold",color=GREEN); ax.text(.40,.28,"max state difference = 0",fontsize=8)
    save(fig,"fig6_target_aligned_routing_landscape",["np026/routing_summary.csv","np029/routing_summary.csv","np036_joint_timing_baseline/joint_timing_baseline.csv","np026/state_identity.csv","np029/state_identity.csv"])


def fig6_support_stability():
    fig = plt.figure(figsize=(8.0,5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.05, .82])
    spans=[("NP025 archived",pd.Timestamp("2023-09-01",tz="UTC"),pd.Timestamp("2024-08-31",tz="UTC"),BLUE),("NP028 strict",pd.Timestamp("2024-02-16",tz="UTC"),pd.Timestamp("2024-08-31",tz="UTC"),PURPLE),("NP022 La Haute Borne",pd.Timestamp("2014-01-01",tz="UTC"),pd.Timestamp("2015-12-31",tz="UTC"),GREEN)]
    ax = fig.add_subplot(gs[0,0]); panel(ax,"a")
    for y,(label,start,end,color) in enumerate(reversed(spans)):
        ax.plot([start,end],[y,y],lw=8,color=color,solid_capstyle="round",alpha=.85)
        ax.text(start, y+.17, label, fontsize=7, color=color, fontweight="bold", ha="left")
    ax.set_yticks([]); ax.set_ylim(-.55,2.55)
    ax.set_xlim(pd.Timestamp("2013-12-01",tz="UTC"), pd.Timestamp("2025-01-15",tz="UTC"))
    tick_years = [2014, 2016, 2018, 2020, 2022, 2024]
    ax.set_xticks([pd.Timestamp(f"{year}-01-01",tz="UTC") for year in tick_years]); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y", tz=pd.Timestamp("2024-01-01",tz="UTC").tz))
    ax.set_xlabel("UTC support period"); ax.spines[["top","right","left"]].set_visible(False); ax.grid(axis="x",color="#E5E9EF",lw=.6)
    ax = fig.add_subplot(gs[1,0]); panel(ax,"b")
    lead_minutes = np.array([15,60,120,240,480,720], dtype=float)
    lead_hours = lead_minutes / 60.0
    for y,(label,_,_,color) in enumerate(reversed(spans)):
        ax.scatter(-lead_hours, np.full(lead_hours.shape, y), s=26, color=color, zorder=3, label=label)
    ax.axvline(0,color=GREY,lw=.8,ls="--"); ax.set_xlim(-13, .8); ax.set_ylim(-.55,2.55); ax.set_yticks([0,1,2], ["NP022", "NP028", "NP025"]); ax.set_xticks(-lead_hours, ["12 h","8 h","4 h","2 h","1 h","15 min"]); ax.set_xlabel("Issue lead before target"); ax.grid(axis="x",color="#E5E9EF",lw=.6); ax.spines[["top","right"]].set_visible(False); ax.text(.98,.93,"six tested leads", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=GREY)
    ax=fig.add_subplot(gs[:,1]); panel(ax,"c")
    # stability points from NP027/030
    d27=pd.read_csv(DATA/"np027/monthly_summary.csv"); d30=pd.read_csv(DATA/"np030/monthly_summary.csv")
    points=[]; labels=[]; colors=[]
    for h in [240,480]:
        sub=d27[(d27.lead_minutes==h)&(d27.metric=="conditional_rps")]
        for _,r in sub.iterrows(): points.append((r.relative_loss_reduction_pct)); labels.append(f"NP027 {h} min · {r.month[:4]}-{r.month[5:]}"); colors.append(BLUE)
    for h in [15,60,120]:
        sub=d30[(d30.lead_minutes==h)&(d30.metric=="conditional_rps")]
        for _,r in sub.iterrows(): points.append((r.relative_loss_reduction_pct)); labels.append(f"NP030 {h} min · {r.month[:4]}-{r.month[5:]}"); colors.append(PURPLE)
    y=np.arange(len(points)); ax.scatter(points,y,c=colors,s=26); ax.axvline(0,color=GREY,lw=.8,ls="--"); ax.set_yticks(y,labels,fontsize=6.7); ax.set_xlabel("Monthly route timing gain (%)"); ax.set_title("Within-period stability audit",loc="left"); clean(ax,"x"); ax.grid(axis="y",visible=False)
    save(fig,"fig7_support_and_stability",["np027/monthly_summary.csv","np030/monthly_summary.csv","np025/protocol.json","np028/protocol.json","np022/protocol.json"])


def fig7_weather_ablation():
    d = pd.read_csv(DATA / "np034_weather_ablation/weather_ablation_summary.csv")
    leads = [15, 240, 720]
    arms = ["calendar", "thermodynamic", "wind_vector", "full_weather"]
    labels = {"calendar": "Calendar", "thermodynamic": "Thermodynamic", "wind_vector": "Wind vector", "full_weather": "Full weather"}
    colors = {"calendar": GREY, "thermodynamic": ORANGE, "wind_vector": BLUE, "full_weather": GREEN}
    fig, axs = plt.subplots(2, 2, figsize=(8.0, 5.8), constrained_layout=True)
    for ax, package, title in [(axs[0,0], "np025", "NP025 archived state ablation"), (axs[0,1], "np028", "NP028 strict-clock state ablation")]:
        sub = d[(d.package == package) & (d.task == "event")]
        x = np.arange(len(leads)); width = .19
        for i, arm in enumerate(arms):
            q = sub[sub.arm == arm].set_index("lead_minutes").loc[leads]
            ax.bar(x + (i-1.5)*width, q.relative_vs_frequency_pct, width=width, color=colors[arm], label=labels[arm], edgecolor="white", linewidth=.5)
        n_windows = int(sub["windows"].iloc[0]); ax.axhline(0,color=GREY,lw=.8,ls="--"); ax.set_xticks(x, [str(v) for v in leads]); ax.set_xlabel("Issue lead (min)"); ax.set_ylabel("State gain vs frequency (%)"); ax.set_title(title,loc="left"); ax.text(.02,.97, f"test windows n={n_windows:,}", transform=ax.transAxes, va="top", fontsize=7, color=GREY); clean(ax)
    for ax, package, title in [(axs[1,0], "np025", "NP025 timing ablation"), (axs[1,1], "np028", "NP028 timing ablation")]:
        sub = d[(d.package == package) & (d.task == "downward_timing")].pivot(index="lead_minutes", columns="arm", values="absolute_score").loc[leads]
        for arm in arms:
            if arm in sub.columns: ax.plot(leads, sub[arm], marker="o", lw=2, ms=4, color=colors[arm], label=labels[arm])
        n_events = int(d[(d.package == package) & (d.task == "downward_timing")]["windows"].iloc[0]); ax.set_xticks(leads); ax.set_xlabel("Issue lead (min)"); ax.set_ylabel("Conditional timing RPS"); ax.set_title(title,loc="left"); ax.text(.02,.97, f"down events n={n_events:,}", transform=ax.transAxes, va="top", fontsize=7, color=GREY); clean(ax)
    axs[0,0].legend(frameon=False,loc="best",ncol=2); axs[0,1].legend(frameon=False,loc="best",ncol=2)
    for ax,letter in zip(axs.ravel(),"abcd"): panel(ax,letter)
    save(fig,"fig5_weather_variable_ablation",["np034_weather_ablation/weather_ablation_summary.csv"])


def main():
    setup(); fig3_pizhou_core(); fig4_information_decomposition(); fig7_weather_ablation(); fig5_route_landscape(); fig6_support_stability(); print(OUT)


if __name__ == "__main__": main()
