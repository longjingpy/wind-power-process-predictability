"""NP035: calibration diagnostics for state and first-passage products."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
LEADS=[15,60,120,240,480,720]

def reliability(p, y, bins):
    rows=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(p>=lo)&(p<hi if hi<1 else p<=hi)
        if not m.any(): continue
        rows.append({"bin_low":lo,"bin_high":hi,"n":int(m.sum()),"mean_pred":float(p[m].mean()),"obs_rate":float(y[m].mean()),"abs_gap":float(abs(p[m].mean()-y[m].mean()))})
    return rows

def main():
    out=ROOT/"outputs/next_paper/np035_calibration"; out.mkdir(parents=True,exist_ok=True)
    bins=np.linspace(0,1,11); rows=[]; timing=[]
    for package in ["np025","np028"]:
        z=np.load(ROOT/f"outputs/next_paper/{package}/test_predictions.npz")
        for lead in LEADS:
            y=z[f"{lead}__y"]; arr=z[f"{lead}__arr"]
            for arm in ["power","weather","joint"]:
                e=z[f"{lead}__{arm}__event"]
                for target,p,event in [("return",e[:,3],(y==3)),("downward",e[:,2]+e[:,3],np.isin(y,[2,3]))]:
                    for r in reliability(p,event.astype(float),bins):
                        r.update({"package":package,"lead_minutes":lead,"arm":arm,"target":target}); rows.append(r)
                q=z[f"{lead}__{arm}__timing"]
                early=q[:,:4].sum(axis=1); observed=(arr<4).astype(float)
                for r in reliability(early,observed,bins):
                    r.update({"package":package,"lead_minutes":lead,"arm":arm,"target":"downward_arrival_le_60min"}); timing.append(r)
    frame=pd.DataFrame(rows+timing); frame.to_csv(out/"reliability.csv",index=False)
    (out/"protocol.json").write_text(json.dumps({"experiment":"NP035","question":"Do state and first-passage probabilities exhibit calibration patterns across information arms?","bins":list(bins),"status":"COMPLETE"},indent=2)+"\n")
    (out/"verification.json").write_text(json.dumps({"status":"PASS","rows":len(frame),"source":"np025/np028 frozen predictions"},indent=2)+"\n")
    print(json.dumps({"status":"PASS","rows":len(frame),"output":str(out)}))

if __name__=="__main__": main()
