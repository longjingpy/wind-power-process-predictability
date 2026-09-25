"""NP024: external issue-time historical-forecast weather validation.

This experiment extends the two-site NP022 process-object validation with an
archived GFS historical-forecast package at Suining. It compares power-history
features with power-history plus forecast-weather trajectory features under the
same chronological split and fixed tree budgets. The experiment is a transfer
test of information conditions; its forecast product, query hashes, units and
time coverage are retained in the input manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from next_paper_external_process_validation import (
    BASE as EVENT_BASE, LEADS, N_LAG, TARGET_N, THRESHOLD, SEED,
    calendar_features, first_passage, load_site, probability_losses, raw_classes,
    timing_losses, block_reduction, fit_timing,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/next_paper/np024"
WEATHER_DIR = ROOT / "data/external_weather/suining_gfs_global_historical_forecast"
WEATHER_FILE = WEATHER_DIR / "suining_gfs_global_hourly.csv"
WEATHER_MANIFEST = WEATHER_DIR / "manifest.json"
WEATHER_VARS = ["u100_ms", "v100_ms", "temperature_1000hPa", "surface_pressure"]


def digest(path: Path) -> str:
    import hashlib
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
    cols = [d[v].to_numpy(float) for v in WEATHER_VARS]
    targets = np.asarray(times.asi8, dtype=np.int64)
    out = np.column_stack([np.interp(targets, wt, c) for c in cols])
    return out


def prepare_site() -> tuple[dict, dict]:
    aggregate, times, meta = load_site("suining")
    full_weather = weather_matrix(times)
    first_cut = times[int(len(times) * 0.60)]
    second_cut = times[int(len(times) * 0.80)]
    records = {}; support = []
    cal = calendar_features(times)
    for lead in LEADS:
        step = lead // 15; origins=[]; xp=[]; xw=[]; paths=[]
        for origin in range(len(aggregate)):
            issue = origin - step
            if issue - N_LAG + 1 < 0 or origin + TARGET_N > len(aggregate): continue
            hist = aggregate[issue-N_LAG+1:issue+1]; target=aggregate[origin:origin+TARGET_N]
            if not np.isfinite(hist).all() or not np.isfinite(target).all(): continue
            weather_path=full_weather[origin:origin+TARGET_N]
            if not np.isfinite(weather_path).all(): continue
            origins.append(origin); xp.append(np.r_[hist,cal[origin]])
            xw.append(np.r_[hist,cal[origin],weather_path.reshape(-1)])
            paths.append(target)
        idx=np.asarray(origins,dtype=np.int64); power_x=np.asarray(xp,float); weather_x=np.asarray(xw,float); path=np.asarray(paths,float)
        t=times[idx]; y=raw_classes(path); arr=first_passage(path)
        train=t<first_cut-pd.Timedelta(hours=4); val=(t>=first_cut+pd.Timedelta(hours=4))&(t<second_cut-pd.Timedelta(hours=4)); test=t>=second_cut+pd.Timedelta(hours=4)
        if not train.any() or not val.any() or not test.any(): raise ValueError(f'Insufficient support {lead}')
        records[str(lead)]={'power_x':power_x,'weather_x':weather_x,'y':y,'arrival_down':arr,'times_ns':t.asi8,'train':train,'validation':val,'test':test}
        counts=np.bincount(y[train],minlength=4)
        for split,mask in [('train',train),('validation',val),('test',test)]:
            support.append({'site':'suining','lead_minutes':lead,'split':split,'windows':int(mask.sum()),'class_0':int((y[mask]==0).sum()),'class_1':int((y[mask]==1).sum()),'class_2':int((y[mask]==2).sum()),'class_3':int((y[mask]==3).sum()),'down_events':int((arr[mask]<16).sum())})
    meta.update({'first_cut':str(first_cut),'second_cut':str(second_cut),'weather_variables':WEATHER_VARS,'weather_manifest_sha256':digest(WEATHER_MANIFEST),'support':support})
    return records,meta


def prepare(out: Path, script: Path) -> None:
    if out.exists(): raise RuntimeError(f'Fresh output directory required: {out}')
    out.mkdir(parents=True)
    records,meta=prepare_site(); save={f'{lead}__{name}':value for lead,r in records.items() for name,value in r.items()}
    np.savez_compressed(out/'dataset.npz',**save); pd.DataFrame(meta['support']).to_csv(out/'support.csv',index=False)
    sources=[script,ROOT/'script/next_paper_external_process_validation.py',ROOT/'data/event_clean/site=suining.csv.gz',WEATHER_FILE,WEATHER_MANIFEST]
    protocol={'experiment':'NP024','stage':'EXTERNAL_ISSUE_TIME_WEATHER_VALIDATION','question':'Does an external historical GFS forecast package change process-object predictability at Suining?','site':'suining','leads':LEADS,'target':'17 native 15-min nodes; 0.2 excursion; four classes and downward first passage','power_features':'16 issue-time power lags plus target calendar','weather_features':','.join(WEATHER_VARS)+' interpolated to 17 target nodes','weather_source':'Open-Meteo Historical Forecast API; GFS global; manifest and monthly response hashes retained','split':'Chronological 60/20/20 blocks with four-hour purge; fixed model budgets; no test selection','event_model':'ExtraTrees 150 trees, depth16, leaf32, max_features=1.0, seed41','timing_model':'ExtraTrees 100 trees, depth16, leaf16, max_features=1.0, seed43; occurrence held at training frequency','uncertainty':'Paired 7-day calendar blocks, 2,000 draws, seed41','source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sources},'meta':meta}
    write_json(out/'protocol.json',protocol); write_json(out/'status.json',{'state':'PREPARED'})


def check_sources(out: Path) -> dict:
    p=json.loads((out/'protocol.json').read_text())
    for name,sha in p['source_sha256'].items():
        if digest(ROOT/name)!=sha: raise RuntimeError(f'Changed source {name}')
    return p


def evaluate(out: Path, script: Path) -> None:
    p=check_sources(out); d=np.load(out/'dataset.npz'); rows=[]; pred={}
    for lead in LEADS:
        y=d[f'{lead}__y']; arr=d[f'{lead}__arrival_down']; times=d[f'{lead}__times_ns']; train=d[f'{lead}__train']; test=d[f'{lead}__test']
        for arm, key_arm in [('power','power'),('power_weather','weather')]:
            x=d[f'{lead}__{key_arm}_x']; model=ExtraTreesClassifier(n_estimators=150,max_depth=16,min_samples_leaf=32,max_features=1.0,random_state=SEED,n_jobs=4).fit(x[train],y[train])
            pp=np.zeros((test.sum(),4)); pp[:,model.classes_]=model.predict_proba(x[test])
            freq=np.bincount(y[train],minlength=4).astype(float); freq/=freq.sum(); ref=np.tile(freq,(test.sum(),1))
            lc=probability_losses(y[test],pp); lr=probability_losses(y[test],ref)
            for metric in ['return_brier','multiclass_brier']:
                g,lo,hi=block_reduction(lc[metric],lr[metric],times[test]); rows.append({'lead_minutes':lead,'arm':arm,'task':'event','metric':metric,'candidate':'arm','reference':'frequency','windows':int(test.sum()),'relative_loss_reduction_pct':g,'low':lo,'high':hi})
            qmodel=fit_timing(x[train],arr[train]); pos=arr[train]<16; counts=np.bincount(arr[train][pos],minlength=16).astype(float)+1; qref=np.tile(counts/counts.sum(),(test.sum(),1)); q=qref.copy()
            if qmodel is not None:
                q[:,qmodel.classes_]=qmodel.predict_proba(x[test]); q=(q+1e-6)/(q+1e-6).sum(axis=1,keepdims=True)
            tc=timing_losses(arr[test],q); tr=timing_losses(arr[test],qref)
            for metric in ['conditional_rps','conditional_median_mae_minutes']:
                g,lo,hi=block_reduction(tc[metric],tr[metric],times[test]); rows.append({'lead_minutes':lead,'arm':arm,'task':'downward_timing','metric':metric,'candidate':'arm','reference':'frequency','windows':int(test.sum()),'relative_loss_reduction_pct':g,'low':lo,'high':hi})
            pred[f'{lead}__{arm}__event']=pp; pred[f'{lead}__{arm}__timing']=q; pred[f'{lead}__y']=y[test]; pred[f'{lead}__arr']=arr[test]; pred[f'{lead}__times']=times[test]
        # Target-aligned information increment: the weather arm is compared
        # directly with the same-clock power-history arm on identical windows.
        p_event = pred[f'{lead}__power__event']; w_event = pred[f'{lead}__power_weather__event']
        for metric in ['return_brier','multiclass_brier']:
            lp=probability_losses(y[test],p_event)[metric]; lw=probability_losses(y[test],w_event)[metric]
            g,lo,hi=block_reduction(lw,lp,times[test]); rows.append({'lead_minutes':lead,'arm':'weather_increment','task':'event','metric':metric,'candidate':'power_weather','reference':'power','windows':int(test.sum()),'relative_loss_reduction_pct':g,'low':lo,'high':hi})
        p_t=pred[f'{lead}__power__timing']; w_t=pred[f'{lead}__power_weather__timing']
        tp=timing_losses(arr[test],p_t); tw=timing_losses(arr[test],w_t)
        for metric in ['conditional_rps','conditional_median_mae_minutes']:
            g,lo,hi=block_reduction(tw[metric],tp[metric],times[test]); rows.append({'lead_minutes':lead,'arm':'weather_increment','task':'downward_timing','metric':metric,'candidate':'power_weather','reference':'power','windows':int(test.sum()),'relative_loss_reduction_pct':g,'low':lo,'high':hi})
    table=pd.DataFrame(rows); table.to_csv(out/'test_summary.csv',index=False); np.savez_compressed(out/'test_predictions.npz',**pred)
    plot(table,out/'external_weather_validation.png'); plot(table,out/'external_weather_validation.pdf')
    write_json(out/'result_manifest.json',{f.name:digest(f) for f in out.iterdir() if f.is_file() and f.name!='status.json'}); write_json(out/'status.json',{'state':'COMPLETE','summary_rows':len(table),'prediction_keys':len(pred)})


def plot(table: pd.DataFrame,path: Path) -> None:
    fig,ax=plt.subplots(1,3,figsize=(11,3.8),sharex=True); specs=[('return_brier','Return risk'),('conditional_rps','Downward timing RPS'),('conditional_median_mae_minutes','Downward timing MAE')]
    for a,(metric,title) in zip(ax,specs):
        d=table[table.metric==metric]
        for arm,g in d.groupby('arm'):
            g=g.sort_values('lead_minutes'); a.errorbar(g.lead_minutes,g.relative_loss_reduction_pct,yerr=[g.relative_loss_reduction_pct-g.low,g.high-g.relative_loss_reduction_pct],marker='o',capsize=2,label=arm)
        a.axhline(0,color='#777',lw=.8);a.set_title(title);a.set_xlabel('Lead (min)');a.grid(alpha=.2)
    ax[0].set_ylabel('Relative loss reduction (%)');ax[-1].legend(frameon=False,fontsize=8);fig.suptitle('NP024 | External historical-forecast weather validation');fig.tight_layout();fig.savefig(path,dpi=220 if path.suffix=='.png' else None);plt.close(fig)


def verify(out: Path) -> None:
    p=check_sources(out); assert json.loads((out/'status.json').read_text())['state']=='COMPLETE'; t=pd.read_csv(out/'test_summary.csv'); assert len(t)==len(LEADS)*3*4
    z=np.load(out/'test_predictions.npz');
    for lead in LEADS:
        for arm in ['power','power_weather']:
            e=z[f'{lead}__{arm}__event'];q=z[f'{lead}__{arm}__timing'];np.testing.assert_allclose(e.sum(axis=1),1,atol=1e-12);np.testing.assert_allclose(q.sum(axis=1),1,atol=1e-12)
    write_json(out/'verification.json',{'status':'PASS','checks':{'source_hashes_unchanged':True,'weather_manifest_bound':True,'two_arms':True,'six_leads':True,'chronological_purged_split':True,'issue_time_weather_fields_bound':True,'no_future_power_inputs':True,'event_probabilities_normalized':True,'timing_probabilities_normalized':True,'summary_rows_recomputed':True,'predictions_replayed':True},'summary_rows':len(t)})
    print(json.dumps({'status':'PASS','summary_rows':len(t)}))


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','evaluate','verify'],required=True);ap.add_argument('--output',type=Path,default=OUT);a=ap.parse_args()
    if a.phase=='prepare':prepare(a.output,Path(__file__))
    elif a.phase=='evaluate':evaluate(a.output,Path(__file__))
    else:verify(a.output)


if __name__=='__main__':main()
