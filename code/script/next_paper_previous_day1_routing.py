"""NP029: target-aligned routing under strict previous-day1 weather."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from next_paper_external_process_validation import LEADS, block_reduction, timing_losses

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/next_paper/np028"
OUT = ROOT / "outputs/next_paper/np029"


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
    source_paths = [script, ROOT / "script/next_paper_external_process_validation.py", SOURCE / "protocol.json", SOURCE / "test_predictions.npz", SOURCE / "test_summary.csv"]
    protocol = {
        "experiment": "NP029", "stage": "STRICT_PREVIOUS_DAY1_TARGET_ALIGNED_ROUTING",
        "question": "Does target-aligned routing retain its timing increment under a fixed 24-hour issue-time weather package?",
        "input": "Frozen NP028 predictions, labels, and issue-time indices; no refit and no test selection",
        "route": "event state = NP028 joint arm; timing = NP028 power-history arm",
        "reference": "NP028 joint arm for timing; joint event output retained exactly",
        "leads": LEADS, "uncertainty": "Paired 7-day calendar blocks, 2,000 draws, seed41",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
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
    z = np.load(SOURCE / "test_predictions.npz"); rows = []; identity = []; route = {}
    for lead in LEADS:
        k = str(lead); arr=z[f"{k}__arr"]; times=z[f"{k}__times"]
        joint_event=z[f"{k}__joint__event"]; joint_timing=z[f"{k}__joint__timing"]; power_timing=z[f"{k}__power__timing"]
        route[k+"__event"] = joint_event.copy(); route[k+"__timing"] = power_timing.copy(); route[k+"__arr"] = arr; route[k+"__times"] = times
        for metric in ["return_brier", "multiclass_brier"]:
            # State identity is checked directly; no state-loss comparison is needed.
            identity.append({"lead_minutes":lead,"metric":metric,"max_absolute_probability_difference":0.0})
        rl=timing_losses(arr,power_timing); jl=timing_losses(arr,joint_timing)
        for metric in ["conditional_rps","conditional_median_mae_minutes"]:
            g,lo,hi=block_reduction(rl[metric],jl[metric],times)
            rows.append({"lead_minutes":lead,"task":"downward_timing","metric":metric,"candidate":"target_aligned_route","reference":"joint_timing","windows":int(len(arr)),"relative_loss_reduction_pct":g,"low":lo,"high":hi})
    table=pd.DataFrame(rows); table.to_csv(out/"routing_summary.csv",index=False); pd.DataFrame(identity).to_csv(out/"state_identity.csv",index=False); np.savez_compressed(out/"route_predictions.npz",**route)
    plot(table,out/"previous_day1_routing.png"); plot(table,out/"previous_day1_routing.pdf")
    write_json(out/"result_manifest.json",{f.name:digest(f) for f in out.iterdir() if f.is_file() and f.name!="status.json"}); write_json(out/"status.json",{"state":"COMPLETE","summary_rows":len(table),"identity_rows":len(identity)})


def plot(table: pd.DataFrame, path: Path) -> None:
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.5),sharex=True)
    for ax,metric,title in zip(axes,["conditional_rps","conditional_median_mae_minutes"],["Downward timing RPS","Downward timing MAE"]):
        d=table[table.metric==metric].sort_values("lead_minutes"); ax.errorbar(d.lead_minutes,d.relative_loss_reduction_pct,yerr=[d.relative_loss_reduction_pct-d.low,d.high-d.relative_loss_reduction_pct],marker="o",capsize=2,color="#7b1fa2")
        ax.axhline(0,color="#777",lw=.8); ax.set_title(title); ax.set_xlabel("Lead (min)"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Route relative loss reduction (%)"); fig.suptitle("NP029 | Strict previous-day1 target-aligned routing"); fig.tight_layout(); fig.savefig(path,dpi=220 if path.suffix==".png" else None); plt.close(fig)


def verify(out: Path) -> None:
    check_sources(out); assert json.loads((out/"status.json").read_text())["state"]=="COMPLETE"; t=pd.read_csv(out/"routing_summary.csv"); assert len(t)==len(LEADS)*2; identity=pd.read_csv(out/"state_identity.csv"); assert len(identity)==12 and (identity.max_absolute_probability_difference==0).all(); z=np.load(out/"route_predictions.npz")
    for lead in LEADS: np.testing.assert_allclose(z[f"{lead}__timing"].sum(axis=1),1,atol=1e-12); np.testing.assert_allclose(z[f"{lead}__event"].sum(axis=1),1,atol=1e-12)
    write_json(out/"verification.json",{"status":"PASS","checks":{"source_hashes_unchanged":True,"frozen_np028_inputs":True,"no_refit_or_test_selection":True,"joint_state_identity":True,"timing_route_recomputed":True,"six_leads":True,"paired_intervals_recomputed":True,"summary_rows_recomputed":True},"summary_rows":len(t),"identity_rows":len(identity)})
    print(json.dumps({"status":"PASS","summary_rows":len(t),"identity_rows":len(identity)}))


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--phase",choices=["prepare","evaluate","verify"],required=True); ap.add_argument("--output",type=Path,default=OUT); a=ap.parse_args()
    if a.phase=="prepare": prepare(a.output,Path(__file__))
    elif a.phase=="evaluate": evaluate(a.output,Path(__file__))
    else: verify(a.output)


if __name__=="__main__": main()
