"""NP018: rolling direct event models with prequential local-wind summaries."""
from pathlib import Path
import argparse, json, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from next_paper_future_wind import monday_floor
from next_paper_local_wind import inputs, BASE, ROOT
from next_paper_probability import probability_losses, paired_loss_effect
from next_paper_trajectory import digest, write_json, status
from next_paper_causal_calibration import row_hash

OUT=BASE/'np018'; LEADS=[15,720]; MODES=['direct','aug_np013','aug_np014','context_np013','context_np014']; FAMILIES=['logistic','forest']
DAY=pd.Timedelta(days=1).value; MINUTE=pd.Timedelta(minutes=1).value; HOUR=pd.Timedelta(hours=1).value
PARAMS=dict(n_estimators=150,max_depth=16,min_samples_leaf=32,max_features=1.,random_state=41,n_jobs=4)

def maturity(t): return ((np.asarray(t)+4*HOUR+HOUR-1)//HOUR)*HOUR+MINUTE

def prepare(out):
    if out.exists(): raise RuntimeError('Fresh output directory required')
    p13=dict(np.load(BASE/'np013/dataset.npz')); p15=dict(np.load(BASE/'np015/dataset.npz'))
    data={k:p13[k] for k in ['times_ns','outcome','validation','test']}; data['maturity_ns']=maturity(data['times_ns']); data['summary_valid']=p13['validation']|p13['test']
    rows=np.r_[np.flatnonzero(data['validation']),np.flatnonzero(data['test'])]
    for lead in LEADS:
        x,known=inputs(p13,lead); data[f'x_{lead}']=x; data[f'known_{lead}']=known
        for split in ['validation','test']:
            data[f's13_{split}_{lead}']=p15[f'{split}_{lead}_np013']; data[f's14_{split}_{lead}']=p15[f'{split}_{lead}_np014']
    out.mkdir(parents=True); (out/'models').mkdir(); np.savez_compressed(out/'dataset.npz',**data)
    tracked=[Path(__file__),BASE/'np013/dataset.npz',BASE/'np013/protocol.json',BASE/'np013/verification.json',BASE/'np015/dataset.npz',BASE/'np015/protocol.json']
    tracked += [BASE/f'np013/{s}_predictions.npz' for s in ['validation','test']]
    protocol=dict(experiment='NP018',stage='ROLLING_DIRECT_LOCAL_WIND_SUMMARY_EVENT_MODEL',leads=LEADS,modes=MODES,families=FAMILIES,
        clock='UTC Monday 00,56-day mature rows,T+4h+1min',summary='NP015 prequential NP013/NP014 seven-dimensional summaries',
        direct='Same 186 issue-time inputs as NP013; augmented modes append summaries; context modes use known9 plus summaries',
        forest=PARAMS,logistic='lambda .001,max_iter2000,tol1e-7',selection='No test selection; all modes/families retained',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in tracked},dataset_sha256=digest(out/'dataset.npz'))
    write_json(out/'protocol.json',protocol); status(out,'PREPARED')

def sources(out):
    p=json.loads((out/'protocol.json').read_text())
    for n,s in p['source_sha256'].items(): assert digest(ROOT/n)==s,n
    assert digest(out/'dataset.npz')==p['dataset_sha256']; return p

def arm_features(data,lead,mode,rows):
    x=data[f'x_{lead}'][rows]
    if mode=='direct': return x
    source='s14' if 'np014' in mode else 's13'
    all_rows=np.r_[np.flatnonzero(data['validation']),np.flatnonzero(data['test'])]
    pos={int(v):i for i,v in enumerate(all_rows)}
    summary_all=np.vstack([data[f'{source}_validation_{lead}'],data[f'{source}_test_{lead}']])
    summary=np.asarray([summary_all[pos[int(r)]] for r in rows])
    return np.c_[x,summary] if mode.startswith('aug') else np.c_[data[f'known_{lead}'][rows],summary]

def fit(x,y,family):
    if family=='logistic': return LogisticRegression(C=1/(len(y)*.001),solver='lbfgs',penalty='l2',max_iter=2000,tol=1e-7,random_state=41).fit(x,y)
    return ExtraTreesClassifier(**PARAMS).fit(x,y)

def run(out,data,split):
    target=np.flatnonzero(data[split]); sched={l:monday_floor(data['times_ns'][target]-l*MINUTE) for l in LEADS}; pred={}; audit=[]; models={}
    for lead in LEADS:
        for mode in MODES:
            for fam in FAMILIES: pred[f'{lead}__{mode}__{fam}']=np.empty((len(target),4))
        for update in np.unique(sched[lead]):
            take=np.flatnonzero(sched[lead]==update); rows=np.flatnonzero(data['summary_valid']&(data['times_ns']>=update-56*DAY)&(data['maturity_ns']<=update)); y=data['outcome'][rows]
            for mode in MODES:
                for fam in FAMILIES:
                    key=f'{lead}__{mode}__{fam}'; train_x=arm_features(data,lead,mode,rows); test_x=arm_features(data,lead,mode,target[take])
                    # Global summary arrays are indexed by split; use the stored global feature directly.
                    if mode=='direct': train_x=data[f'x_{lead}'][rows]; test_x=data[f'x_{lead}'][target[take]]
                    else:
                        pass
                    if len(y) < 100 or len(np.unique(y)) < 4:
                        prior=(np.bincount(y,minlength=4)+1)/(len(y)+4)
                        pred[key][take]=np.tile(prior,(len(take),1)); model=None
                    else:
                        model=fit(train_x,y,fam); pred[key][take]=model.predict_proba(test_x)
                    name=f'models/{key}__{int(update)}.joblib'; joblib.dump(model,out/name,compress=3); models[name]=digest(out/name)
                    audit.append(dict(key=key,update_ns=int(update),rows=len(rows),row_sha256=row_hash(rows),feature_count=train_x.shape[1],path=name))
            print('fit',split,lead,pd.Timestamp(update,tz='UTC'),len(rows),flush=True)
    p13=np.load(BASE/f'np013/{split}_predictions.npz'); p14=np.load(BASE/f'np014/{split}_predictions.npz')
    for lead in LEADS:
        pred[f'{lead}__raw_np013']=p13[f'{lead}__selected']; pred[f'{lead}__raw_np014']=p14[f'{lead}__selected']
    np.savez_compressed(out/f'{split}_predictions.npz',**pred); pd.DataFrame(audit).to_csv(out/f'{split}_audit.csv',index=False); write_json(out/f'{split}_models.json',models); return pred

def evaluate(out,data,split,pred):
    idx=np.flatnonzero(data[split]); y=data['outcome'][idx]; losses={k:probability_losses(y,v) for k,v in pred.items()}
    pd.DataFrame([dict(model=k,metric=m,windows=len(y),loss=float(v.mean())) for k,z in losses.items() for m,v in z.items()]).to_csv(out/f'{split}_summary.csv',index=False)
    if split=='test':
        clock=pd.to_datetime(data['times_ns'][idx],utc=True); rows=[]
        for lead in LEADS:
            for mode in ['aug_np013','aug_np014','context_np013','context_np014']:
                for fam in FAMILIES:
                    for metric in ['multiclass_brier','return_brier','up_brier','down_brier']:
                        for days in [3,7,14]: rows.append(dict(candidate=f'{lead}__{mode}__{fam}',reference=f'{lead}__direct__{fam}',comparison='rolling_summary_direct',metric=metric,windows=len(y),**paired_loss_effect(clock,losses[f'{lead}__{mode}__{fam}'][metric],losses[f'{lead}__direct__{fam}'][metric],days)))
        pd.DataFrame(rows).to_csv(out/'paired_intervals.csv',index=False)

def validate(out):
    sources(out); data=dict(np.load(out/'dataset.npz')); status(out,'VALIDATING'); pred=run(out,data,'validation'); evaluate(out,data,'validation',pred); status(out,'VALIDATION_FROZEN')

def test(out):
    sources(out); data=dict(np.load(out/'dataset.npz')); status(out,'TESTING'); pred=run(out,data,'test'); evaluate(out,data,'test',pred); status(out,'COMPLETE')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['prepare','validate','test'],required=True); p.add_argument('--output',type=Path,default=OUT); a=p.parse_args()
    with threadpool_limits(limits=1): {'prepare':prepare,'validate':validate,'test':test}[a.phase](a.output)
