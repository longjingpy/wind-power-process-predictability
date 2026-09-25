"""Build varied, claim-first Nature v2 figures from verified next-paper CSVs."""
from __future__ import annotations
from pathlib import Path
import hashlib, json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path("/mnt/d/projects/WindPowerForcast")
DATA = ROOT / "outputs/next_paper"
OUT = DATA / "manuscript_figures/nature_v1"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, PURPLE, GREY, TEXT = "#2166AC", "#D55E00", "#009E73", "#7B3294", "#5B6470", "#18212B"
PALE = "#EEF2F5"


def setup():
    for f in ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"]:
        p = Path("/mnt/c/Windows/Fonts") / f
        if p.exists(): font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({"font.family":"Arial", "font.size":7, "axes.labelsize":7,
                         "xtick.labelsize":6.5, "ytick.labelsize":6.5, "legend.fontsize":6.5,
                         "axes.linewidth":.7, "pdf.fonttype":42, "ps.fonttype":42, "svg.fonttype":"none"})


def save(fig, stem):
    for ext, kwargs in [("pdf",{}),("svg",{}),("png",{"dpi":450})]:
        fig.savefig(OUT/f"{stem}.{ext}", bbox_inches="tight", pad_inches=.08, **kwargs)
    plt.close(fig)


def clean(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=2, pad=2); ax.grid(axis="y", color="#E6E9ED", lw=.5); ax.set_axisbelow(True)


def pick(df, **kw):
    m=np.ones(len(df),dtype=bool)
    for k,v in kw.items(): m &= df[k].astype(str).eq(str(v)).to_numpy()
    x=df.loc[m]
    if len(x)!=1: raise ValueError((kw,len(x)))
    return x.iloc[0]


def arr(df, **kw):
    q=pick(df,**kw); return np.array([float(q[k]) for k in ["relative_loss_reduction_pct","low","high"]])


def panel(ax, letter): ax.text(-.08,1.06,letter,transform=ax.transAxes,fontsize=8,fontweight="bold",va="top")


def divnorm(values):
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    lo = min(lo, 0.0); hi = max(hi, 0.0)
    if lo == 0.0: lo = -max(hi * 0.1, 1.0)
    if hi == 0.0: hi = max(-lo * 0.1, 1.0)
    return TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)


def fig2():
    leads=[15,60,120,240,480,720]; x=np.log2(leads)
    d5=pd.read_csv(DATA/"np005/paired_intervals.csv"); d9=pd.read_csv(DATA/"np009/paired_intervals.csv"); d22=pd.read_csv(DATA/"np022/test_summary.csv")
    matrix=np.array([
        [arr(d5,candidate=f"power_weather:{h}:joint",reference="frequency:0:reference",metric="return_brier",state="all",block_days=7)[0] for h in leads],
        [arr(d9,candidate=f"{h}__down__context",reference=f"{h}__down__frequency",metric="conditional_rps",state="all",block_days=7)[0] for h in leads],
        [arr(d9,candidate=f"{h}__down__context",reference=f"{h}__down__frequency",metric="conditional_median_mae_minutes",state="all",block_days=7)[0] for h in leads],
    ])
    fig=plt.figure(figsize=(7.3,3.5),constrained_layout=True); gs=fig.add_gridspec(1,3,width_ratios=[1.2,1,1])
    ax=fig.add_subplot(gs[0,0]); panel(ax,"a")
    norm=divnorm(matrix)
    ax.imshow(matrix,cmap="RdBu_r",norm=norm,aspect="auto")
    ax.set_xticks(range(6),[str(h) for h in leads]); ax.set_yticks(range(3),["Return Brier","Timing RPS","Timing MAE"])
    ax.set_xlabel("Issue lead (min)"); ax.set_title("Object × lead skill",loc="left",fontweight="bold")
    for i in range(3):
        for j in range(6): ax.text(j,i,f"{matrix[i,j]:.1f}",ha="center",va="center",fontsize=5.5,color="white" if abs(matrix[i,j])>7 else TEXT)
    ax.tick_params(length=0); ax.text(0,-.25,"Relative loss reduction (%)",transform=ax.transAxes,fontsize=5.5,color=GREY)
    # dumbbell
    ax=fig.add_subplot(gs[0,1]); panel(ax,"b")
    baseline=54.22; candidate=49.35
    ax.hlines(0,candidate,baseline,color=GREY,lw=2); ax.scatter([baseline,candidate],[0,0],s=45,c=[GREY,ORANGE],zorder=3)
    ax.text(baseline, .11, "frequency\n54.22 min",ha="center",fontsize=6)
    ax.text(candidate,-.17,"context\n49.35 min",ha="center",fontsize=6,color=ORANGE)
    ax.annotate("−4.87 min\nRPS −8.14%",xy=((baseline+candidate)/2,0),xytext=((baseline+candidate)/2,.28),ha="center",arrowprops=dict(arrowstyle="-",color=ORANGE),fontsize=6,color=ORANGE)
    ax.set_xlim(46,57); ax.set_ylim(-.38,.45); ax.set_xticks([47,49,51,53,55,57]); ax.set_yticks([]); ax.set_xlabel("Downward first-passage median error (min)"); ax.set_title("Near-term timing moves earlier",loc="left",fontweight="bold"); clean(ax); ax.grid(False)
    # external dot plot
    ax=fig.add_subplot(gs[0,2]); panel(ax,"c")
    sites=["La Haute\nBorne","Suining"]; y=np.arange(2);
    for j,metric in enumerate(["conditional_rps","conditional_median_mae_minutes"]):
        vals=[]; lows=[]; highs=[]
        for site in ["lahaute","suining"]:
            q=pick(d22,site=site,lead_minutes=15,task="downward_timing",metric=metric)
            vals.append(q.relative_loss_reduction_pct); lows.append(q.low); highs.append(q.high)
        off=(-.10,.10)[j]; ax.errorbar(vals,y+off,xerr=[np.array(vals)-lows,np.array(highs)-np.array(vals)],fmt="o" if j==0 else "s",color=BLUE if j==0 else ORANGE,ecolor=GREY,capsize=2,markersize=4,label="RPS" if j==0 else "median MAE")
    ax.axvline(0,color=GREY,lw=.6,ls="--"); ax.set_yticks(y,sites); ax.set_xlabel("Relative loss reduction (%)"); ax.set_title("Power-history replication",loc="left",fontweight="bold"); ax.legend(frameon=False,loc="upper right"); clean(ax); ax.grid(axis="x",color="#E6E9ED",lw=.5); ax.grid(axis="y",visible=False)
    save(fig,"fig2_process_capability")


def fig3():
    leads=[15,60,120,240,480,720]; x=np.log2(leads)
    d25=pd.read_csv(DATA/"np025/test_summary.csv"); d28=pd.read_csv(DATA/"np028/test_summary.csv")
    # selected 720 state forest
    selected=[("NP025 weather-only / frequency",arr(d25,lead_minutes=720,arm="weather",task="event",metric="multiclass_brier",candidate="weather",reference="frequency"),GREEN),
              ("NP025 joint / power",arr(d25,lead_minutes=720,arm="joint_increment",task="event",metric="multiclass_brier",candidate="joint",reference="power"),BLUE),
              ("NP028 weather-only / power",arr(d28,lead_minutes=720,arm="weather_increment",task="event",metric="multiclass_brier",candidate="weather",reference="power"),ORANGE),
              ("NP028 joint / power",arr(d28,lead_minutes=720,arm="joint_increment",task="event",metric="multiclass_brier",candidate="joint",reference="power"),PURPLE)]
    state25=[]; state28=[]; time25=[]; time28=[]
    for h in leads:
        state25.append(arr(d25,lead_minutes=h,arm="joint_increment",task="event",metric="multiclass_brier",candidate="joint",reference="power")[0])
        state28.append(arr(d28,lead_minutes=h,arm="joint_increment",task="event",metric="multiclass_brier",candidate="joint",reference="power")[0])
        time25.append(arr(d25,lead_minutes=h,arm="power_increment_on_weather",task="downward_timing",metric="conditional_rps",candidate="joint",reference="weather")[0])
        time28.append(arr(d28,lead_minutes=h,arm="power_increment_on_weather",task="downward_timing",metric="conditional_rps",candidate="joint",reference="weather")[0])
    fig=plt.figure(figsize=(7.3,5.1),constrained_layout=True); gs=fig.add_gridspec(2,2,height_ratios=[1,1.1])
    ax=fig.add_subplot(gs[0,0]); panel(ax,"a"); yy=np.arange(len(selected))[::-1]
    for y,(label,q,c) in zip(yy,selected):
        ax.errorbar(q[0],y,xerr=[[q[0]-q[1]],[q[2]-q[0]]],fmt="o",color=c,ecolor=GREY,capsize=2.5,markersize=4)
    ax.axvline(0,color=GREY,lw=.6,ls="--"); ax.set_yticks(yy,[s[0] for s in selected]); ax.set_xlabel("Relative state-risk loss reduction (%)"); ax.set_title("Four-class state gain at 12 h",loc="left",fontweight="bold"); clean(ax); ax.grid(axis="x",color="#E6E9ED",lw=.5); ax.grid(axis="y",visible=False)
    ax=fig.add_subplot(gs[0,1]); panel(ax,"b")
    mat=np.array([state25,state28]); norm=divnorm(mat); ax.imshow(mat,cmap="RdBu_r",norm=norm,aspect="auto")
    ax.set_xticks(np.arange(6),[str(h) for h in leads]); ax.set_yticks([0,1],["NP025 joint→power","NP028 joint→power"]); ax.set_xlabel("Issue lead (min)"); ax.set_title("Added weather in the joint arm",loc="left",fontweight="bold")
    for i in range(2):
        for j in range(6): ax.text(j,i,f"{mat[i,j]:.1f}",ha="center",va="center",fontsize=5.5,color="white" if abs(mat[i,j])>8 else TEXT)
    ax.tick_params(length=0)
    ax=fig.add_subplot(gs[1,0]); panel(ax,"c")
    mat=np.array([time25,time28]); norm=divnorm(mat); ax.imshow(mat,cmap="RdBu_r",norm=norm,aspect="auto")
    ax.set_xticks(np.arange(6),[str(h) for h in leads]); ax.set_yticks([0,1],["NP025 power→weather","NP028 power→weather"]); ax.set_xlabel("Issue lead (min)"); ax.set_title("Added power history in timing",loc="left",fontweight="bold")
    for i in range(2):
        for j in range(6): ax.text(j,i,f"{mat[i,j]:.1f}",ha="center",va="center",fontsize=5.5,color="white" if abs(mat[i,j])>7 else TEXT)
    ax.tick_params(length=0)
    ax=fig.add_subplot(gs[1,1]); panel(ax,"d");
    dates=[pd.Timestamp("2024-01-01",tz="UTC"),pd.Timestamp("2024-02-16",tz="UTC"),pd.Timestamp("2024-06-13",tz="UTC"),pd.Timestamp("2024-07-23",tz="UTC"),pd.Timestamp("2024-08-31",tz="UTC")]
    labels=["Requested","Complete weather","Train cut","Validation cut","Test end"]; colors=["#B7C9E2",GREEN,"#A6A6A6","#D9D9D9",ORANGE]
    ax.plot([dates[0],dates[-1]],[0,0],color=GREY,lw=1); ax.scatter(dates,[0]*len(dates),c=colors,s=35,zorder=3)
    for d,l,c in zip(dates,labels,colors): ax.text(d,.15,l,ha="center",fontsize=5.5,color=c if c!=ORANGE else ORANGE,rotation=35)
    ax.axvline(dates[1],color=GREEN,lw=.8,ls="--"); ax.set_xlim(dates[0],dates[-1]); ax.set_ylim(-.25,.75); ax.set_yticks([]); ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b")); ax.set_xlabel("2024 UTC"); ax.set_title("The strict clock defines support",loc="left",fontweight="bold"); ax.spines[["top","right","left"]].set_visible(False); ax.tick_params(axis="x",length=2)
    save(fig,"fig3_information_decomposition")


def fig4():
    d26=pd.read_csv(DATA/"np026/routing_summary.csv"); d29=pd.read_csv(DATA/"np029/routing_summary.csv"); d27=pd.read_csv(DATA/"np027/monthly_summary.csv"); d30=pd.read_csv(DATA/"np030/monthly_summary.csv")
    selected=[]
    for h in [240,480]:
        for metric,label in [("conditional_rps","NP026 RPS"),("conditional_median_mae_minutes","NP026 median MAE")]: selected.append((f"{label} · {h} min",arr(d26,lead_minutes=h,task="downward_timing",metric=metric,candidate="target_aligned_route",reference="joint_timing"),BLUE if metric=="conditional_rps" else ORANGE))
    for h in [60,120]: selected.append((f"NP029 strict RPS · {h} min",arr(d29,lead_minutes=h,task="downward_timing",metric="conditional_rps",candidate="target_aligned_route",reference="joint_timing"),GREEN))
    fig=plt.figure(figsize=(7.3,4.8),constrained_layout=True); gs=fig.add_gridspec(2,2,height_ratios=[.8,1.2])
    ax=fig.add_subplot(gs[0,0]); panel(ax,"a"); ax.axis("off"); ax.text(.05,.86,"Target-aligned route",fontsize=8,fontweight="bold");
    for y,label,c in [(.60,"Joint state head → state risk",BLUE),(.30,"Power timing head → first passage",ORANGE)]:
        ax.add_patch(FancyBboxPatch((.08,y),.84,.16,boxstyle="round,pad=.015,rounding_size=.025",facecolor="white",edgecolor=c,lw=1.2)); ax.text(.50,y+.08,label,ha="center",va="center",fontsize=6.8,fontweight="bold")
    ax.text(.05,.08,"frozen NP025/NP028 predictions · state identity exact",fontsize=5.7,color=GREY)
    ax=fig.add_subplot(gs[0,1]); panel(ax,"b"); yy=np.arange(len(selected))[::-1]
    for y,(label,q,c) in zip(yy,selected): ax.errorbar(q[0],y,xerr=[[q[0]-q[1]],[q[2]-q[0]]],fmt="o",color=c,ecolor=GREY,capsize=2.5,markersize=4)
    ax.axvline(0,color=GREY,lw=.6,ls="--"); ax.set_yticks(yy,[s[0] for s in selected]); ax.set_xlabel("Relative timing loss reduction (%)"); ax.set_title("Route gains at supported leads",loc="left",fontweight="bold"); clean(ax); ax.grid(axis="x",color="#E6E9ED",lw=.5); ax.grid(axis="y",visible=False)
    ax=fig.add_subplot(gs[1,0]); panel(ax,"c"); rows=[]; labels=[]
    for h in [240,480]:
        for metric,label in [("conditional_rps","RPS"),("conditional_median_mae_minutes","MAE")]:
            sub=d27[(d27.lead_minutes==h)&(d27.metric==metric)]; vals=[float(sub[sub.month==m].relative_loss_reduction_pct.iloc[0]) for m in ["2024-06","2024-07","2024-08"]]; rows.append(vals); labels.append(f"NP027 {h} {label}")
    mat=np.array(rows); norm=divnorm(mat); ax.imshow(mat,cmap="RdBu_r",norm=norm,aspect="auto"); ax.set_xticks(range(3),["Jun","Jul","Aug"]); ax.set_yticks(range(4),labels); ax.tick_params(length=0); ax.set_title("Historical-package monthly audit",loc="left",fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(3): ax.text(j,i,f"{mat[i,j]:.1f}",ha="center",va="center",fontsize=5.5,color="white" if abs(mat[i,j])>8 else TEXT)
    ax=fig.add_subplot(gs[1,1]); panel(ax,"d"); rows=[]; labels=[]
    for h in [15,60,120]:
        sub=d30[(d30.lead_minutes==h)&(d30.metric=="conditional_rps")]; vals=[float(sub[sub.month==m].relative_loss_reduction_pct.iloc[0]) for m in ["2024-07","2024-08"]]; rows.append(vals); labels.append(f"{h} min strict RPS")
    mat=np.array(rows); norm=divnorm(mat); ax.imshow(mat,cmap="RdBu_r",norm=norm,aspect="auto"); ax.set_xticks(range(2),["Jul","Aug"]); ax.set_yticks(range(3),labels); ax.tick_params(length=0); ax.set_title("Strict-clock near-term audit",loc="left",fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(2): ax.text(j,i,f"{mat[i,j]:.1f}",ha="center",va="center",fontsize=5.5,color="white" if abs(mat[i,j])>4 else TEXT)
    save(fig,"fig4_target_aligned_routing")


def main():
    setup(); fig2(); fig3(); fig4(); print(OUT)


if __name__ == "__main__": main()
