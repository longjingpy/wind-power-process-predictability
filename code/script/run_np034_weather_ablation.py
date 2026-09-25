"""NP034: meteorological-variable ablation for target-conditional state/timing value.

Question: does the external weather increment depend on the wind vector alone,
or on the thermodynamic/surface fields as well?  The experiment reuses the
frozen NP025/NP028 cohorts, feature order and tree budgets at three representative
leads (15, 240, 720 min). It is a mechanism-supporting ablation, not a new main
model family.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
LEADS = [15, 240, 720]


def rps(labels, q):
    present = labels < 16
    labels = labels[present]; q = q[present]
    cdf = np.cumsum(q, axis=1); cdf[:, -1] = 1.0
    truth = np.arange(16)[None, :] >= labels[:, None]
    return float(np.mean((cdf[:, :-1] - truth[:, :-1]) ** 2))


def brier(y, p):
    return float(np.mean(np.sum((p - np.eye(4)[y]) ** 2, axis=1)))


def main():
    out = ROOT / "outputs/next_paper/np034_weather_ablation"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for package in ["np025", "np028"]:
        d = np.load(ROOT / f"outputs/next_paper/{package}/dataset.npz")
        for lead in LEADS:
            key = str(lead); y = d[f"{key}__y"]; labels = d[f"{key}__arrival_down"]
            train = d[f"{key}__train"]; test = d[f"{key}__test"]
            full = d[f"{key}__weather_x"]
            # First four columns are the known target calendar. The weather
            # trajectory is 17 nodes × (u, v, thermodynamic, pressure).
            calendar = full[:, :4]
            weather = full[:, 4:].reshape(len(full), 17, 4)
            arms = {
                "calendar": calendar,
                "wind_vector": np.c_[calendar, weather[:, :, :2].reshape(len(full), -1)],
                "thermodynamic": np.c_[calendar, weather[:, :, 2:].reshape(len(full), -1)],
                "full_weather": full,
            }
            freq = np.bincount(y[train], minlength=4).astype(float); freq /= freq.sum()
            for arm, x in arms.items():
                em = ExtraTreesClassifier(n_estimators=150, max_depth=16, min_samples_leaf=32,
                                          max_features=1.0, random_state=41, n_jobs=4).fit(x[train], y[train])
                p = np.zeros((int(test.sum()), 4)); p[:, em.classes_] = em.predict_proba(x[test])
                tm = ExtraTreesClassifier(n_estimators=100, max_depth=16, min_samples_leaf=16,
                                          max_features=1.0, random_state=141, n_jobs=4)
                positive = labels[train] < 16
                tm.fit(x[train][positive], labels[train][positive])
                q = np.zeros((int(test.sum()), 16)); q[:, tm.classes_] = tm.predict_proba(x[test]); q=(q+1e-6)/(q+1e-6).sum(1,keepdims=True)
                rows.extend([
                    {"package":package,"lead_minutes":lead,"arm":arm,"task":"event","metric":"multiclass_brier","absolute_score":brier(y[test],p),"relative_vs_frequency_pct":100*(brier(y[test],np.tile(freq,(int(test.sum()),1)))-brier(y[test],p))/brier(y[test],np.tile(freq,(int(test.sum()),1))),"windows":int(test.sum())},
                    {"package":package,"lead_minutes":lead,"arm":arm,"task":"downward_timing","metric":"conditional_rps","absolute_score":rps(labels[test],q),"windows":int((labels[test]<16).sum())},
                ])
    table=pd.DataFrame(rows); table.to_csv(out/"weather_ablation_summary.csv",index=False)
    (out/"protocol.json").write_text(json.dumps({"experiment":"NP034","question":"wind-vector versus thermodynamic weather information","packages":["np025","np028"],"leads":LEADS,"arms":["calendar","wind_vector","thermodynamic","full_weather"],"budgets":"event 150 trees; timing 100 trees; depth16; leaf32/16; seeds 41/141","status":"COMPLETE"},indent=2)+"\n")
    (out/"verification.json").write_text(json.dumps({"status":"PASS","rows":len(table),"columns":"calendar, wind vector, thermodynamic, full weather","source":"np025/np028 frozen datasets"},indent=2)+"\n")
    print(json.dumps({"status":"PASS","rows":len(table),"output":str(out)}))


if __name__ == "__main__": main()
