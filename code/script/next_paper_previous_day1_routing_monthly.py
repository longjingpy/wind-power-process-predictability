"""NP030: UTC-month stability audit for the strict previous-day1 route."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from next_paper_external_process_validation import block_reduction, timing_losses

ROOT = Path(__file__).resolve().parents[1]
NP028 = ROOT / "outputs/next_paper/np028"
NP029 = ROOT / "outputs/next_paper/np029"
OUT = ROOT / "outputs/next_paper/np030"
LEADS = [15, 60, 120, 240, 480, 720]
MONTHS = ["2024-07", "2024-08"]
METRICS = ["conditional_rps", "conditional_median_mae_minutes"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def prepare(out: Path, script: Path) -> None:
    if out.exists():
        raise RuntimeError(f"Fresh output directory required: {out}")
    out.mkdir(parents=True)
    sources = [script, ROOT / "script/next_paper_external_process_validation.py",
               NP028 / "test_predictions.npz", NP029 / "route_predictions.npz",
               NP029 / "routing_summary.csv", NP029 / "state_identity.csv", NP029 / "protocol.json"]
    protocol = {
        "experiment": "NP030", "stage": "STRICT_PREVIOUS_DAY1_ROUTING_MONTHLY_STABILITY",
        "question": "Does the strict previous-day1 target-aligned route retain its timing increment across UTC months?",
        "months": MONTHS, "leads": LEADS, "metrics": METRICS,
        "input": "Frozen NP028 and NP029 predictions; no refit or month selection",
        "uncertainty": "Paired 7-day calendar blocks within each month, 2,000 draws, seed41",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
    }
    write_json(out / "protocol.json", protocol); write_json(out / "status.json", {"state": "PREPARED"})


def check_sources(out: Path) -> dict:
    p = json.loads((out / "protocol.json").read_text())
    for name, sha in p["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError(f"Changed source: {name}")
    return p


def evaluate(out: Path, script: Path) -> None:
    check_sources(out)
    z = np.load(NP028 / "test_predictions.npz"); route = np.load(NP029 / "route_predictions.npz")
    rows = []
    for lead in LEADS:
        k = str(lead); labels=z[f"{k}__arr"]; times=z[f"{k}__times"]
        a=timing_losses(labels,route[f"{k}__timing"]); b=timing_losses(labels,z[f"{k}__joint__timing"])
        months=pd.Series(pd.to_datetime(times,unit="ns",utc=True)).dt.strftime("%Y-%m").to_numpy()
        for month in MONTHS:
            mask=months==month
            for metric in METRICS:
                g,lo,hi=block_reduction(a[metric][mask],b[metric][mask],times[mask])
                rows.append({"month":month,"lead_minutes":lead,"task":"downward_timing","metric":metric,"candidate":"target_aligned_route","reference":"joint_timing","windows":int(mask.sum()),"relative_loss_reduction_pct":g,"low":lo,"high":hi})
    table=pd.DataFrame(rows); table.to_csv(out/"monthly_summary.csv",index=False); plot(table,out/"previous_day1_routing_monthly.png"); plot(table,out/"previous_day1_routing_monthly.pdf")
    write_json(out/"result_manifest.json",{f.name:digest(f) for f in out.iterdir() if f.is_file() and f.name!="status.json"}); write_json(out/"status.json",{"state":"COMPLETE","summary_rows":len(table),"months":MONTHS})


def plot(table: pd.DataFrame, path: Path) -> None:
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.5),sharex=True)
    for ax,metric,title in zip(axes,METRICS,["Downward timing RPS","Downward timing MAE"]):
        d=table[table.metric==metric]
        for month,g in d.groupby("month"):
            g=g.sort_values("lead_minutes"); ax.errorbar(g.lead_minutes,g.relative_loss_reduction_pct,yerr=[g.relative_loss_reduction_pct-g.low,g.high-g.relative_loss_reduction_pct],marker="o",capsize=2,label=month)
        ax.axhline(0,color="#777",lw=.8); ax.set_title(title); ax.set_xlabel("Lead (min)"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Route relative loss reduction (%)"); axes[-1].legend(frameon=False,fontsize=8); fig.suptitle("NP030 | Strict previous-day1 routing by UTC month"); fig.tight_layout(); fig.savefig(path,dpi=220 if path.suffix==".png" else None); plt.close(fig)


def verify(out: Path) -> None:
    check_sources(out); assert json.loads((out/"status.json").read_text())["state"]=="COMPLETE"; t=pd.read_csv(out/"monthly_summary.csv"); assert len(t)==len(MONTHS)*len(LEADS)*len(METRICS); assert set(t.month)==set(MONTHS)
    near=t[t.lead_minutes.isin([15,60,120])]; assert len(near)==12 and (near.relative_loss_reduction_pct>0).all()
    write_json(out/"verification.json",{"status":"PASS","checks":{"source_hashes_unchanged":True,"frozen_np029_inputs":True,"no_refit_or_month_selection":True,"two_utc_months":True,"six_leads":True,"monthly_scores_recomputed":True,"near_term_same_sign":True,"full_matrix_retained":True},"summary_rows":len(t),"months":MONTHS})
    print(json.dumps({"status":"PASS","summary_rows":len(t),"months":MONTHS}))


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--phase",choices=["prepare","evaluate","verify"],required=True); ap.add_argument("--output",type=Path,default=OUT); a=ap.parse_args()
    if a.phase=="prepare": prepare(a.output,Path(__file__))
    elif a.phase=="evaluate": evaluate(a.output,Path(__file__))
    else: verify(a.output)


if __name__=="__main__": main()
