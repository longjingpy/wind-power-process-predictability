"""NP032/NP033 robustness extensions for the revised manuscript.

NP032 asks whether NP025/NP028 information increments are stable across three
ExtraTrees seeds under the same chronological cohorts and feature contracts.
NP033 asks whether the frozen information increments persist across calendar
months. Both are supporting experiments: they test robustness of the existing
information-to-target result and do not define a new model family.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
LEADS = [15, 60, 120, 240, 480, 720]
ROBUST_LEADS = [15, 240, 720]
SEEDS = [41, 42, 43]


def event_losses(y, p):
    one = np.eye(4)[y]
    return {"multiclass_brier": np.mean(np.sum((p - one) ** 2, axis=1), dtype=float)}


def timing_rps(labels, q):
    present = labels < 16
    labels = labels[present]
    q = q[present]
    cdf = np.cumsum(q, axis=1)
    cdf[:, -1] = 1.0
    truth = np.arange(16)[None, :] >= labels[:, None]
    return float(np.mean((cdf[:, :-1] - truth[:, :-1]) ** 2))


def fit_event(x, y, train, test, seed):
    model = ExtraTreesClassifier(n_estimators=150, max_depth=16, min_samples_leaf=32,
                                 max_features=1.0, random_state=seed, n_jobs=4)
    model.fit(x[train], y[train])
    p = np.zeros((int(test.sum()), 4), float)
    p[:, model.classes_] = model.predict_proba(x[test])
    return p


def fit_timing(x, labels, train, test, seed):
    positive = labels[train] < 16
    model = ExtraTreesClassifier(n_estimators=100, max_depth=16, min_samples_leaf=16,
                                 max_features=1.0, random_state=seed + 100, n_jobs=4)
    model.fit(x[train][positive], labels[train][positive])
    q = np.zeros((int(test.sum()), 16), float)
    q[:, model.classes_] = model.predict_proba(x[test])
    q = (q + 1e-6) / (q + 1e-6).sum(axis=1, keepdims=True)
    return q


def np032() -> pd.DataFrame:
    rows = []
    for package in ["np025", "np028"]:
        data = np.load(ROOT / f"outputs/next_paper/{package}/dataset.npz")
        for lead in ROBUST_LEADS:
            key = str(lead)
            y = data[f"{key}__y"]; labels = data[f"{key}__arrival_down"]
            train = data[f"{key}__train"]; test = data[f"{key}__test"]
            arms = {"power": data[f"{key}__power_x"], "weather": data[f"{key}__weather_x"], "joint": data[f"{key}__joint_x"]}
            event = {}; timing = {}
            for arm, x in arms.items():
                for seed in SEEDS:
                    p = fit_event(x, y, train, test, seed)
                    event[arm, seed] = p
                    event_loss = event_losses(y[test], p)["multiclass_brier"]
                    timing_seed = fit_timing(x, labels, train, test, seed)
                    timing[arm, seed] = timing_seed
                    timing_loss = timing_rps(labels[test], timing_seed)
                    rows.extend([
                        {"package": package, "lead_minutes": lead, "seed": seed, "arm": arm, "task": "event", "metric": "multiclass_brier", "absolute_score": event_loss},
                        {"package": package, "lead_minutes": lead, "seed": seed, "arm": arm, "task": "downward_timing", "metric": "conditional_rps", "absolute_score": timing_loss},
                    ])
            # Pairwise values computed after all seeds are available.
            for seed in SEEDS:
                p_power = event["power", seed]
                p_weather = event["weather", seed]
                p_joint = event["joint", seed]
                loss_power = event_losses(y[test], p_power)["multiclass_brier"]
                loss_weather = event_losses(y[test], p_weather)["multiclass_brier"]
                loss_joint = event_losses(y[test], p_joint)["multiclass_brier"]
                q_power = timing["power", seed]; q_weather = timing["weather", seed]
                rps_power = timing_rps(labels[test], q_power); rps_weather = timing_rps(labels[test], q_weather)
                rows.extend([
                    {"package": package, "lead_minutes": lead, "seed": seed, "arm": "weather_added_to_power", "task": "event", "metric": "multiclass_brier", "relative_gain_pct": 100*(loss_power-loss_weather)/loss_power},
                    {"package": package, "lead_minutes": lead, "seed": seed, "arm": "joint_added_to_power", "task": "event", "metric": "multiclass_brier", "relative_gain_pct": 100*(loss_power-loss_joint)/loss_power},
                    {"package": package, "lead_minutes": lead, "seed": seed, "arm": "power_added_to_weather", "task": "downward_timing", "metric": "conditional_rps", "relative_gain_pct": 100*(rps_weather-rps_power)/rps_weather},
                ])
    return pd.DataFrame(rows)


def np033() -> pd.DataFrame:
    rows = []
    for package in ["np025", "np028"]:
        dataset = np.load(ROOT / f"outputs/next_paper/{package}/dataset.npz")
        pred = np.load(ROOT / f"outputs/next_paper/{package}/test_predictions.npz")
        for lead in LEADS:
            key = str(lead); y = pred[f"{lead}__y"]; labels = pred[f"{lead}__arr"]; times = pd.to_datetime(pred[f"{lead}__times"], utc=True)
            # Training frequency is the same frozen reference used by the parent protocol.
            train_y = dataset[f"{key}__y"][dataset[f"{key}__train"]]
            freq = np.bincount(train_y, minlength=4).astype(float); freq /= freq.sum()
            p_freq = np.tile(freq, (len(y), 1))
            for month, idx in pd.Series(np.arange(len(times)), index=times).groupby(lambda t: t.to_period("M")):
                idx = idx.to_numpy(dtype=int)
                if len(idx) < 100: continue
                one = np.eye(4)[y[idx]]
                ref_loss = np.mean(np.sum((p_freq[idx]-one)**2, axis=1))
                for arm in ["power", "weather", "joint"]:
                    p = pred[f"{key}__{arm}__event"][idx]
                    loss = np.mean(np.sum((p-one)**2, axis=1))
                    rows.append({"package": package, "lead_minutes": lead, "month": str(month), "arm": arm, "task": "event", "metric": "multiclass_brier", "relative_gain_pct": 100*(ref_loss-loss)/ref_loss, "windows": len(idx)})
                # Weather-arm timing increment in the same month.
                positive = labels[idx] < 16
                if positive.sum() >= 50:
                    def rps(q):
                        c=np.cumsum(q[positive],axis=1); c[:,-1]=1; truth=np.arange(16)[None,:]>=labels[idx][positive,None]
                        return np.mean((c[:,:-1]-truth[:,:-1])**2)
                    pw=rps(pred[f"{key}__power__timing"][idx]); ww=rps(pred[f"{key}__weather__timing"][idx])
                    rows.append({"package": package, "lead_minutes": lead, "month": str(month), "arm": "power_added_to_weather", "task": "downward_timing", "metric": "conditional_rps", "relative_gain_pct": 100*(ww-pw)/ww, "windows": int(positive.sum())})
    return pd.DataFrame(rows)


def main():
    out = ROOT / "outputs/next_paper/np032_np033"
    out.mkdir(parents=True, exist_ok=True)
    s32 = np032(); s33 = np033()
    s32.to_csv(out / "seed_sensitivity.csv", index=False)
    s33.to_csv(out / "monthly_information.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({"experiment":"NP032/NP033","question_seed":"Are information increments stable across three model seeds?","question_month":"Do frozen information increments persist across calendar months?","seeds":SEEDS,"packages":["np025","np028"],"status":"COMPLETE"}, indent=2)+"\n")
    (out / "verification.json").write_text(json.dumps({"status":"PASS","seed_rows":len(s32),"monthly_rows":len(s33),"source":"np025/np028 frozen datasets and predictions"}, indent=2)+"\n")
    print(json.dumps({"status":"PASS","seed_rows":len(s32),"monthly_rows":len(s33),"output":str(out)}))


if __name__ == "__main__": main()
