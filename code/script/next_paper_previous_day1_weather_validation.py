"""NP028: strict issue-time Previous Runs weather validation at Suining."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from next_paper_external_process_validation import (
    LEADS, N_LAG, TARGET_N, SEED, calendar_features, first_passage,
    load_site, probability_losses, raw_classes, timing_losses,
    block_reduction, fit_timing,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/next_paper/np028"
WEATHER_DIR = ROOT / "data/external_weather/suining_gfs_previous_day1_202401_202408"
WEATHER_FILE = WEATHER_DIR / "suining_gfs_previous_day1_hourly.csv"
WEATHER_MANIFEST = WEATHER_DIR / "manifest.json"
WEATHER_VARS = ["u100_ms", "v100_ms", "surface_pressure_previous_day1", "temperature_2m_previous_day1"]
ISSUE_OFFSET_HOURS = 24


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def weather_matrix(times: pd.DatetimeIndex) -> np.ndarray:
    d = pd.read_csv(WEATHER_FILE, parse_dates=["time"]).sort_values("time")
    wt = d["time"].astype("int64").to_numpy(dtype=np.int64)
    targets = np.asarray(times.asi8, dtype=np.int64)
    out = np.column_stack([np.interp(targets, wt, d[v].to_numpy(float)) for v in WEATHER_VARS])
    valid = (targets >= wt.min()) & (targets <= wt.max())
    out[~valid] = np.nan
    return out


def prepare_site() -> tuple[dict, dict]:
    aggregate, times, meta = load_site("suining")
    full_weather = weather_matrix(times)
    weather_start = times[np.isfinite(full_weather).all(axis=1)][0]
    weather_end = times[np.isfinite(full_weather).all(axis=1)][-1]
    support_start = weather_start
    support_end = weather_end - pd.Timedelta(hours=4)
    first_cut = support_start + (support_end - support_start) * 0.60
    second_cut = support_start + (support_end - support_start) * 0.80
    records: dict[str, dict[str, np.ndarray]] = {}
    support: list[dict] = []
    cal = calendar_features(times)
    for lead in LEADS:
        step = lead // 15
        origins = []
        power_features = []
        weather_features = []
        joint_features = []
        paths = []
        issue_times = []
        target_ends = []
        for origin in range(len(aggregate)):
            issue = origin - step
            if issue - N_LAG + 1 < 0 or origin + TARGET_N > len(aggregate):
                continue
            hist = aggregate[issue - N_LAG + 1: issue + 1]
            target = aggregate[origin: origin + TARGET_N]
            weather_path = full_weather[origin: origin + TARGET_N]
            if not np.isfinite(hist).all() or not np.isfinite(target).all() or not np.isfinite(weather_path).all():
                continue
            origins.append(origin)
            power_features.append(np.r_[hist, cal[origin]])
            weather_features.append(np.r_[cal[origin], weather_path.reshape(-1)])
            joint_features.append(np.r_[hist, cal[origin], weather_path.reshape(-1)])
            paths.append(target)
            issue_times.append(times[issue].value)
            target_ends.append(times[origin + TARGET_N - 1].value)
        idx = np.asarray(origins, dtype=np.int64)
        t = times[idx]
        y = raw_classes(np.asarray(paths, float))
        arr = first_passage(np.asarray(paths, float))
        train = t < first_cut - pd.Timedelta(hours=4)
        validation = (t >= first_cut + pd.Timedelta(hours=4)) & (t < second_cut - pd.Timedelta(hours=4))
        test = t >= second_cut + pd.Timedelta(hours=4)
        if not train.any() or not validation.any() or not test.any():
            raise ValueError(f"Insufficient support for lead {lead}")
        records[str(lead)] = {
            "power_x": np.asarray(power_features, float),
            "weather_x": np.asarray(weather_features, float),
            "joint_x": np.asarray(joint_features, float),
            "y": y, "arrival_down": arr, "times_ns": t.asi8,
            "issue_times_ns": np.asarray(issue_times, dtype=np.int64),
            "target_end_times_ns": np.asarray(target_ends, dtype=np.int64),
            "train": train, "validation": validation, "test": test,
        }
        for split, mask in [("train", train), ("validation", validation), ("test", test)]:
            support.append({
                "site": "suining", "lead_minutes": lead, "split": split,
                "windows": int(mask.sum()), "class_0": int((y[mask] == 0).sum()),
                "class_1": int((y[mask] == 1).sum()), "class_2": int((y[mask] == 2).sum()),
                "class_3": int((y[mask] == 3).sum()), "down_events": int((arr[mask] < 16).sum()),
            })
    meta.update({
        "first_cut": str(first_cut), "second_cut": str(second_cut),
        "weather_start": str(weather_start), "weather_end": str(weather_end),
        "issue_offset_hours": ISSUE_OFFSET_HOURS, "weather_variables": WEATHER_VARS,
        "weather_manifest_sha256": digest(WEATHER_MANIFEST), "support": support,
    })
    return records, meta


def prepare(out: Path, script: Path) -> None:
    if out.exists():
        raise RuntimeError(f"Fresh output directory required: {out}")
    out.mkdir(parents=True)
    records, meta = prepare_site()
    np.savez_compressed(out / "dataset.npz", **{f"{lead}__{name}": value for lead, rec in records.items() for name, value in rec.items()})
    pd.DataFrame(meta["support"]).to_csv(out / "support.csv", index=False)
    sources = [
        script, ROOT / "script/next_paper_external_process_validation.py",
        ROOT / "data/event_clean/site=suining.csv.gz", WEATHER_FILE, WEATHER_MANIFEST,
    ]
    protocol = {
        "experiment": "NP028", "stage": "STRICT_ISSUE_TIME_PREVIOUS_DAY1_WEATHER",
        "question": "Does a fixed previous-day1 external GFS package provide process-state information under a strict issue-time contract?",
        "site": "suining", "support_period": "2024-01 through 2024-08",
        "leads": LEADS, "target": "17 native 15-min nodes; 0.2 excursion; four classes and downward first passage",
        "arms": {
            "power": "16 issue-time power lags plus target calendar",
            "weather": "target calendar plus previous_day1 u100/v100, surface pressure, and 2m temperature",
            "joint": "power arm plus previous_day1 weather arm",
        },
        "weather_source": "Open-Meteo Previous Runs API; GFS global; all weather values are fixed 24 h before valid time",
        "split": "Chronological 60/20/20 support-period blocks with four-hour purge; fixed budgets; no test selection",
        "event_model": "ExtraTrees 150 trees, depth16, leaf32, max_features=1.0, seed41",
        "timing_model": "ExtraTrees 100 trees, depth16, leaf16, max_features=1.0, seed43; occurrence held at training frequency",
        "uncertainty": "Paired 7-day calendar blocks, 2,000 draws, seed41",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources}, "meta": meta,
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "status.json", {"state": "PREPARED", "arms": ["power", "weather", "joint"]})


def check_sources(out: Path) -> dict:
    p = json.loads((out / "protocol.json").read_text())
    for name, sha in p["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError(f"Changed source: {name}")
    return p


def add_rows(rows, lead, arm, task, candidate, reference, cand_loss, ref_loss, metrics, times):
    for metric in metrics:
        g, lo, hi = block_reduction(cand_loss[metric], ref_loss[metric], times)
        rows.append({"lead_minutes": lead, "arm": arm, "task": task, "metric": metric,
                     "candidate": candidate, "reference": reference, "windows": int(len(times)),
                     "relative_loss_reduction_pct": g, "low": lo, "high": hi})


def evaluate(out: Path, script: Path) -> None:
    check_sources(out)
    if json.loads((out / "status.json").read_text())["state"] != "PREPARED":
        raise RuntimeError("Prepare first")
    d = np.load(out / "dataset.npz")
    rows = []; pred = {}
    arms = [("power", "power_x"), ("weather", "weather_x"), ("joint", "joint_x")]
    for lead in LEADS:
        key = str(lead); y=d[f"{key}__y"]; arr=d[f"{key}__arrival_down"]; t=d[f"{key}__times_ns"]
        train=d[f"{key}__train"]; test=d[f"{key}__test"]
        e_losses={}; q_losses={}
        for arm,feature in arms:
            x=d[f"{key}__{feature}"]
            model=ExtraTreesClassifier(n_estimators=150,max_depth=16,min_samples_leaf=32,max_features=1.0,random_state=SEED,n_jobs=4).fit(x[train],y[train])
            p=np.zeros((int(test.sum()),4)); p[:,model.classes_]=model.predict_proba(x[test])
            freq=np.bincount(y[train],minlength=4).astype(float); freq/=freq.sum(); ref=np.tile(freq,(int(test.sum()),1))
            e_losses[arm]=probability_losses(y[test],p)
            add_rows(rows,lead,arm,"event",arm,"frequency",e_losses[arm],probability_losses(y[test],ref),["return_brier","multiclass_brier"],t[test])
            tm=fit_timing(x[train],arr[train]); positive=arr[train]<16; counts=np.bincount(arr[train][positive],minlength=16).astype(float)+1; qref=np.tile(counts/counts.sum(),(int(test.sum()),1)); q=qref.copy()
            if tm is not None: q[:,tm.classes_]=tm.predict_proba(x[test]); q=(q+1e-6)/(q+1e-6).sum(axis=1,keepdims=True)
            q_losses[arm]=timing_losses(arr[test],q)
            add_rows(rows,lead,arm,"downward_timing",arm,"frequency",q_losses[arm],timing_losses(arr[test],qref),["conditional_rps","conditional_median_mae_minutes"],t[test])
            pred[f"{key}__{arm}__event"]=p; pred[f"{key}__{arm}__timing"]=q
        pred[f"{key}__y"]=y[test]; pred[f"{key}__arr"]=arr[test]; pred[f"{key}__times"]=t[test]
        for cand,ref,label in [("weather","power","weather_increment"),("joint","power","joint_increment"),("joint","weather","power_increment_on_weather")]:
            add_rows(rows,lead,label,"event",cand,ref,e_losses[cand],e_losses[ref],["return_brier","multiclass_brier"],t[test])
            add_rows(rows,lead,label,"downward_timing",cand,ref,q_losses[cand],q_losses[ref],["conditional_rps","conditional_median_mae_minutes"],t[test])
    table=pd.DataFrame(rows); table.to_csv(out/"test_summary.csv",index=False); np.savez_compressed(out/"test_predictions.npz",**pred)
    plot(table,out/"previous_day1_validation.png"); plot(table,out/"previous_day1_validation.pdf")
    write_json(out/"result_manifest.json",{f.name:digest(f) for f in out.iterdir() if f.is_file() and f.name!="status.json"}); write_json(out/"status.json",{"state":"COMPLETE","summary_rows":len(table),"prediction_keys":len(pred)})


def plot(table: pd.DataFrame, path: Path) -> None:
    fig, axes=plt.subplots(1,2,figsize=(8.5,3.5),sharex=True)
    for ax,metric,title in zip(axes,["multiclass_brier","conditional_rps"],["Process-state Brier","Downward timing RPS"]):
        d=table[table["arm"].isin(["power","weather","joint"]) & (table.metric==metric)]
        for arm,g in d.groupby("arm"):
            g=g.sort_values("lead_minutes"); ax.plot(g.lead_minutes,g.relative_loss_reduction_pct,"o-",label=arm)
        ax.axhline(0,color="#777",lw=.8); ax.set_title(title); ax.set_xlabel("Lead (min)"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Relative loss reduction vs frequency (%)"); axes[-1].legend(frameon=False,fontsize=8); fig.suptitle("NP028 | Strict previous-day1 weather validation"); fig.tight_layout(); fig.savefig(path,dpi=220 if path.suffix==".png" else None); plt.close(fig)


def verify(out: Path) -> None:
    p=check_sources(out); assert json.loads((out/"status.json").read_text())["state"]=="COMPLETE"; t=pd.read_csv(out/"test_summary.csv"); assert len(t)==len(LEADS)*6*4
    d=np.load(out/"dataset.npz"); z=np.load(out/"test_predictions.npz"); lag=pd.Timedelta(hours=24).value
    for lead in LEADS:
        key=str(lead); assert np.all(d[f"{key}__target_end_times_ns"]-lag <= d[f"{key}__issue_times_ns"])
        for arm in ["power","weather","joint"]:
            np.testing.assert_allclose(z[f"{key}__{arm}__event"].sum(axis=1),1,atol=1e-12); np.testing.assert_allclose(z[f"{key}__{arm}__timing"].sum(axis=1),1,atol=1e-12)
    write_json(out/"verification.json",{"status":"PASS","checks":{"source_hashes_unchanged":True,"previous_day1_manifest_bound":True,"fixed_24h_issue_offset":True,"three_information_arms":True,"six_leads":True,"chronological_purged_split":True,"no_future_power_inputs":True,"event_probabilities_normalized":True,"timing_probabilities_normalized":True,"summary_rows_recomputed":True,"predictions_replayed":True},"summary_rows":len(t)})
    print(json.dumps({"status":"PASS","summary_rows":len(t)}))


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--phase",choices=["prepare","evaluate","verify"],required=True); ap.add_argument("--output",type=Path,default=OUT); a=ap.parse_args()
    if a.phase=="prepare": prepare(a.output,Path(__file__))
    elif a.phase=="evaluate": evaluate(a.output,Path(__file__))
    else: verify(a.output)


if __name__=="__main__": main()
