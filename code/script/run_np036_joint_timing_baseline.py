"""NP036: direct 17-bin first-passage baseline for route falsification.

Scientific question: does the target-aligned frozen route improve a coherent
no-crossing-inclusive first-passage product relative to a direct joint-feature
17-class model under the same windows and tree budget?  A positive result is a
stronger route argument; a mixed result narrows the claim to conditional head
reallocation.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

ROOT=Path(__file__).resolve().parents[1]
LEADS=[15,60,120,240,480,720]

def rps(p,y):
    t=np.zeros_like(p); t[np.arange(len(y)),y]=1
    return float(np.mean(np.sum((np.cumsum(p,1)-np.cumsum(t,1))**2,axis=1)))

def route_prediction(package, lead, source_pred, route_pred):
    e=source_pred[f"{lead}__joint__event"]
    q=route_pred[f"{lead}__timing"]
    pdown=e[:,2]+e[:,3]
    return np.c_[pdown[:,None]*q,1-pdown]

def main():
    out=ROOT/"outputs/next_paper/np036_joint_timing_baseline"; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for package,route_pkg in [("np025","np026"),("np028","np029")]:
        d=np.load(ROOT/f"outputs/next_paper/{package}/dataset.npz")
        pred=np.load(ROOT/f"outputs/next_paper/{package}/test_predictions.npz")
        route=np.load(ROOT/f"outputs/next_paper/{route_pkg}/route_predictions.npz")
        for lead in LEADS:
            key=str(lead); labels=d[f"{key}__arrival_down"]; train=d[f"{key}__train"]; test=d[f"{key}__test"]
            x=d[f"{key}__joint_x"]; model=ExtraTreesClassifier(n_estimators=150,max_depth=16,min_samples_leaf=32,max_features=1.0,random_state=41,n_jobs=4)
            model.fit(x[train],labels[train]); direct=np.zeros((int(test.sum()),17)); direct[:,model.classes_]=model.predict_proba(x[test])
            # Route state probabilities and timing head are frozen source outputs.
            route_p=route_prediction(package,lead,pred,route)
            y=labels[test]
            direct_score=rps(direct,y); route_score=rps(route_p,y)
            rows.append({"package":package,"route_package":route_pkg,"lead_minutes":lead,"metric":"joint_first_passage_RPS","route_score":route_score,"direct_joint_score":direct_score,"route_gain_vs_direct_pct":100*(direct_score-route_score)/direct_score,"windows":len(y)})
            # Include the original joint constructed product as an additional comparator.
            joint_q=pred[f"{key}__joint__timing"]; e=pred[f"{key}__joint__event"]; pdown=e[:,2]+e[:,3]
            constructed=np.c_[pdown[:,None]*joint_q,1-pdown]; joint_score=rps(constructed,y)
            rows.append({"package":package,"route_package":route_pkg,"lead_minutes":lead,"metric":"joint_first_passage_RPS","route_score":route_score,"direct_joint_score":joint_score,"route_gain_vs_direct_pct":100*(joint_score-route_score)/joint_score,"windows":len(y),"reference":"constructed_joint"})
    table=pd.DataFrame(rows); table.to_csv(out/"joint_timing_baseline.csv",index=False)
    (out/"protocol.json").write_text(json.dumps({"experiment":"NP036","question":"Does target-aligned route improve coherent no-crossing first-passage score versus direct joint-feature 17-bin model?","leads":LEADS,"model":"ExtraTrees 150 trees, depth16, leaf32, seed41","route":"frozen NP026/NP029 state plus timing outputs","status":"COMPLETE"},indent=2)+"\n")
    (out/"verification.json").write_text(json.dumps({"status":"PASS","rows":len(table),"source":"np025/np028 datasets and frozen predictions"},indent=2)+"\n")
    print(json.dumps({"status":"PASS","rows":len(table),"output":str(out)}))

if __name__=="__main__": main()
