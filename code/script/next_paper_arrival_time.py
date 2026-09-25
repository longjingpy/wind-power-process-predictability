"""NP009: first-threshold timing with event occurrence held fixed.

Gneiting & Raftery (2007), Strictly Proper Scoring Rules, Prediction, and
Estimation, doi:10.1198/016214506000001437, Section4 / threshold-CDF scores:
average binary CDF losses across ordered categories (loss orientation).
Here 16 arrival bins and a no-event category have a shared terminal event
probability, so changes in ranked probability score isolate time allocation.

The project NP007 multinomial fit and mature-label policy are reused in
generalized form for16 conditional time bins. The deterministic uniform
mixture16/(n+16) keeps unobserved bins possible; it is a smoothing rule, not
a guarantee of calibration. The ordered/sorted comparison retains forecast
values and endpoints, following the project's temporal-information controls.
NP006 complete trajectories supply a separate nonlinear timing reference.

First passage within a fixed future window is not catalogue event onset.
Separate upward/downward time marginals are not a complete joint power path.
"""
from pathlib import Path
import argparse
import json
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from next_paper_context_sources import OUT as PARENT, frozen as check_parent
from next_paper_causal_calibration import (ROOT, BASE, LEADS, STATES, DAY, MINUTE, MATURITY,
    training_rows, row_hash, read_inputs, digest, write_json, status)
from next_paper_leadtime import read_data
from next_paper_probability import paired_loss_effect

OUT = ROOT / 'outputs/next_paper/np009'
DOMAIN = ROOT / 'outputs/next_paper/np006'
EVENTS = ['up', 'down']
ARMS = ['frequency', 'context', 'ordered', 'sorted']
LAMBDAS = [.0003, .001, .003]
K = 16


def first_passage(path, threshold=.2):
    path = np.asarray(path)
    if path.ndim not in [2,3] or path.shape[-1] != 17 or not np.isfinite(path).all():
        raise ValueError('Finite17-node paths required')
    excursion = [path-np.minimum.accumulate(path,axis=-1), np.maximum.accumulate(path,axis=-1)-path]
    return np.stack([np.where((value[...,1:] >= threshold-1e-12).any(axis=-1),
        (value[...,1:] >= threshold-1e-12).argmax(axis=-1), K) for value in excursion])


def features(context, profile, arm):
    if arm == 'frequency':
        return np.empty((len(context),0))
    if arm == 'context':
        return context
    if arm == 'sorted':
        profile = np.sort(profile, axis=-1)
    return np.c_[context, profile.reshape(len(context),-1)]


def target_calendar(times_ns):
    """Known target time controls daily/annual cycles and grid phase in every arm."""
    times=pd.to_datetime(times_ns,utc=True).tz_convert('Asia/Shanghai')
    hour=times.hour+times.minute/60
    phase=times.minute//15
    return np.c_[np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*times.dayofyear/365.25),np.cos(2*np.pi*times.dayofyear/365.25),
        phase==1,phase==2,phase==3]


def fit_time(x, y, strength, frequency=False):
    counts = np.bincount(y,minlength=K)
    if frequency or len(y)<100 or len(np.unique(y))<3:
        return dict(kind='frequency' if frequency else 'fallback', counts=counts, n=len(y),
                    probability=(counts+1)/(len(y)+K), n_iter=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model=LogisticRegression(C=1/(len(y)*strength),solver='lbfgs',max_iter=2000,
            tol=1e-7,random_state=41).fit(x,y)
    return dict(kind='logistic',counts=counts,n=len(y),classes=model.classes_,
        coefficient=model.coef_,intercept=model.intercept_,C=model.C,n_iter=int(model.n_iter_.max()))


def predict_time(record, x):
    if record['kind']!='logistic':
        return np.tile(record['probability'],(len(x),1))
    p=np.zeros((len(x),K))
    p[:,record['classes']]=softmax(x@record['coefficient'].T+record['intercept'],axis=1)
    return (record['n']*p+1)/(record['n']+K)


def timing_rows(times, labels, update, days):
    rows=training_rows(times,update,days)
    return rows[np.asarray(labels)[rows]<K]


def timing_losses(label, conditional, occurrence):
    q=np.asarray(conditional); p=np.asarray(occurrence); label=np.asarray(label,int)
    if q.shape!=(len(label),K) or not np.isfinite(q).all() or (q<0).any() or not np.allclose(q.sum(axis=1),1):
        raise ValueError('Normalized16-bin conditional probabilities required')
    if ((p<0)|(p>1)).any() or not np.isfinite(p).all() or ((label<0)|(label>K)).any():
        raise ValueError('Valid occurrence probabilities and arrival labels required')
    cdf=np.cumsum(q,axis=1); cdf[:,-1]=1.
    joint=cdf*p[:,None]; joint[:,-1]=p
    truth=np.arange(K)[None,:]>=label[:,None]
    squared=(joint-truth)**2; present=label<K
    conditional_rps=np.full(len(label),np.nan); mae=conditional_rps.copy()
    conditional_rps[present]=np.mean((cdf[present,:-1]-truth[present,:-1])**2,axis=1)
    median=(cdf>=.5).argmax(axis=1)
    mae[present]=15*np.abs(median[present]-label[present])
    probability=np.where(present,p*q[np.arange(len(label)),np.minimum(label,K-1)],1-p)
    return dict(joint_rps=squared.mean(axis=1),early_cdf_loss=squared[:,:-1].mean(axis=1),
        occurrence_brier=squared[:,-1],joint_log_loss=-np.log(np.maximum(probability,1e-12)),
        conditional_rps=conditional_rps,conditional_median_mae_minutes=mae)


def prepare(out):
    parent_protocol=check_parent(PARENT)
    if out.exists() or json.loads((PARENT/'status.json').read_text())['state']!='COMPLETE':
        raise RuntimeError('Complete parent and fresh output required')
    inherited=read_inputs(PARENT); base=read_data(BASE)
    pool=base['validation']|base['test']; weather=base['weather'][pool].reshape(-1,17,8)
    np.testing.assert_array_equal(inherited['times_ns'],base['times_ns'][pool])
    speed=np.stack([np.hypot(weather[:,:,4],weather[:,:,5]),np.hypot(weather[:,:,0],weather[:,:,1])],axis=1)
    profile=(speed[:,:,1:-1]-speed[:,:,0,None])/5
    actual=base['actual'][pool]; labels=first_passage(actual)
    expected=(labels[0]<K).astype(int)+2*(labels[1]<K).astype(int)
    np.testing.assert_array_equal(expected,inherited['outcome'])
    occurrence=np.full((2,6,len(actual)),np.nan)
    tracked=[Path(__file__),ROOT/'script/next_paper_causal_calibration.py',ROOT/'script/next_paper_probability.py',
             ROOT/'src/wind_events/paired_probability.py',BASE/'dataset.npz',PARENT/'dataset.npz',
             PARENT/'protocol.json',PARENT/'freeze.json',DOMAIN/'protocol.json',ROOT/'outputs/next_paper/site_metadata.json']
    for split in ['validation','test']:
        mask=inherited[split]
        source=PARENT/f'{split}_predictions.npz'
        with np.load(source) as values:
            for j,lead in enumerate(LEADS):
                p=values[f'{lead}__full__rolling']
                occurrence[0,j,mask]=p[:,1]+p[:,3]
                occurrence[1,j,mask]=p[:,2]+p[:,3]
                tracked.append(BASE/f'{split}_{lead}_power_weather.npz')
        tracked.append(source)
    data={key:inherited[key] for key in ['times_ns','state','validation','test','original_validation']}
    calendar=target_calendar(inherited['times_ns'])
    context=np.concatenate([inherited['context'][:,:,:7],np.broadcast_to(calendar[None],(6,*calendar.shape))],axis=2)
    data.update(context=context,profile=profile,arrival=labels,occurrence=occurrence)
    out.mkdir(parents=True)
    np.savez_compressed(out/'dataset.npz',**data)
    support=[]
    for split in ['original_validation','validation','test']:
        mask=data[split]
        for e,event in enumerate(EVENTS):
            count=np.bincount(labels[e,mask],minlength=K+1)
            support.append(dict(split=split,event=event,windows=int(mask.sum()),events=int(count[:K].sum()),
                no_event=int(count[K]),**{f'bin_{15*(k+1)}min':int(count[k]) for k in range(K)}))
    pd.DataFrame(support).to_csv(out/'support.csv',index=False)
    protocol=dict(experiment='NP009',stage='EXPLORATORY_TIMING_WITH_FIXED_OCCURRENCE',
        question='Does forecast temporal ordering improve arrival-time allocation beyond occurrence risk and background?',
        target='First ordered drawup/down>=0.2capacity within unchanged17-point[T,T+4h] target; bins15..240min and none',
        events=EVENTS,leads=LEADS,window_days=parent_protocol['window_days'],lambda_candidates=LAMBDAS,
        occurrence='Frozen NP008 full rolling four-class marginals; same p for every timing method',
        arms=ARMS,features={'context':14,'ordered':44,'sorted':44},
        calendar='Same known target-T daily/annual sine-cosine and3quarter-hour indicators in all covariate arms',
        curves='GFS/JMA speed at15 interior quarter-hours minus forecast start, /5; endpoints in context; hourly-origin interpolated archive',
        sorting='Sort each model interior multiset independently; retain endpoints/context/dimensions',
        fitting='Only matured positive windows;16-time multinomial with mean-loss L2,C=1/(n*lambda); same causal56day rule',
        smoothing='Predictive mixture with uniform mass16/(n+16); empirical frequency adds1 to each bin; <100positive or<3classes fallback',
        selection='One lambda minimizes mean positive-event conditional RPS across2directions*6leads*3covariate arms on validation',
        primary_score='17-category normalized RPS = mean16CDF Brier terms; shared terminal p; losses lower-is-better',
        secondary_scores='Early15CDF mean, shared terminalBrier,joint log loss,positive-only conditionalRPS and median-timeMAE',
        trajectory_reference='NP005 fixed450neighbors with NP006 bounds; conditional time counts smoothed by1; combine the same occurrence p',
        trajectory_bounds=json.loads((DOMAIN/'protocol.json').read_text())['training_observed_bounds'],
        caveats='Not catalogue onset or continuous time; separate directional marginals not full coherent joint paths; previously studied calendar',
        inference='Paired3/7/14day target blocks,2000draws,seed41; primary7d; generated forecasts held fixed',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in tracked},dataset_sha256=digest(out/'dataset.npz'))
    write_json(out/'protocol.json',protocol)
    status(out,'PREPARED')
    print(pd.DataFrame(support)[['split','event','windows','events','no_event']].to_string(index=False),flush=True)


def sources(out):
    p=json.loads((out/'protocol.json').read_text())
    for name,sha in p['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError(f'Changed source:{name}')
    if digest(out/'dataset.npz')!=p['dataset_sha256']:raise RuntimeError('Changed prepared data')
    return p


def run_timing(out,data,split,days,strength,tag):
    target=np.flatnonzero(data[split]); n=len(target)
    predictions,records={},{}
    for j,lead in enumerate(LEADS):
        issue=data['times_ns'][target]-lead*MINUTE; update=issue//DAY*DAY
        for e,event in enumerate(EVENTS):
            for arm in ARMS:
                key=f'{lead}__{event}__{arm}'; x=features(data['context'][j],data['profile'],arm)
                predictions[key]=np.empty((n,K))
                for stamp in np.unique(update):
                    rows=timing_rows(data['times_ns'],data['arrival'][e],int(stamp),days)
                    record=fit_time(x[rows],data['arrival'][e,rows],strength,frequency=arm=='frequency')
                    record.update(update_ns=int(stamp),row_sha256=row_hash(rows),
                        latest_available_ns=int((data['times_ns'][rows]+MATURITY).max()) if len(rows) else None)
                    take=np.flatnonzero(update==stamp)
                    assert stamp<=issue[take].min() and (not len(rows) or record['latest_available_ns']<=stamp)
                    predictions[key][take]=predict_time(record,x[target[take]])
                    records[f'{key}__{stamp}']=record
        print(datetime.now(timezone.utc).isoformat(),tag,'lead',lead,'snapshots',len(records),flush=True)
    np.savez_compressed(out/f'{tag}_predictions.npz',**predictions)
    joblib.dump(records,out/f'{tag}_snapshots.joblib',compress=3)
    return predictions


def trajectory_timing(out,data,protocol,split):
    base=read_data(BASE); tr=base['train']; target=np.flatnonzero(data[split])
    old_times=base['times_ns'][base[split]]; positions=np.searchsorted(old_times,data['times_ns'][target])
    np.testing.assert_array_equal(old_times[positions],data['times_ns'][target])
    result={}; member_counts={}
    for j,lead in enumerate(LEADS):
        with np.load(BASE/f'{split}_{lead}_power_weather.npz') as values:
            neighbors=values['neighbors'][positions]
        delta=base['actual'][tr]-base['anchor'][j,tr,None]
        anchor=base['anchor'][j,base[split]][positions]
        counts=np.zeros((2,len(target),K),int)
        for start in range(0,len(target),256):
            stop=min(start+256,len(target))
            paths=np.clip(anchor[start:stop,None,None]+delta[neighbors[start:stop]],*protocol['trajectory_bounds'])
            label=first_passage(paths)
            for e in range(2):
                counts[e,start:stop]=np.stack([(label[e]==k).sum(axis=1) for k in range(K)],axis=1)
        for e,event in enumerate(EVENTS):
            total=counts[e].sum(axis=1)
            result[f'{lead}__{event}__trajectory']=(counts[e]+1)/(total[:,None]+K)
            member_counts[f'{lead}__{event}']=total
    np.savez_compressed(out/f'{split}_trajectory.npz',**result)
    np.savez_compressed(out/f'{split}_trajectory_member_counts.npz',**member_counts)
    return result


def validate(out):
    p=sources(out)
    if json.loads((out/'status.json').read_text())['state']!='PREPARED':raise RuntimeError('Fresh run required')
    data=read_inputs(out); target=data['validation']; candidates=[]; details=[]
    trajectory_timing(out,data,p,'validation')
    for strength in LAMBDAS:
        tag=f'validation_l{strength:g}'
        prediction=run_timing(out,data,'validation',p['window_days'],strength,tag)
        objective=[]
        for key,q in prediction.items():
            lead,event,arm=key.split('__'); e=EVENTS.index(event); j=LEADS.index(int(lead))
            scores=timing_losses(data['arrival'][e,target],q,data['occurrence'][e,j,target])
            values={m:float(np.nanmean(v)) for m,v in scores.items()}
            details.append(dict(lambda_=strength,lead_minutes=int(lead),event=event,arm=arm,**values))
            if arm!='frequency':objective.append(values['conditional_rps'])
        candidates.append(dict(lambda_=strength,conditional_rps=float(np.mean(objective)),compared_modes=len(objective)))
        print('candidate',candidates[-1],flush=True)
    winner=min(candidates,key=lambda r:(r['conditional_rps'],-r['lambda_']))
    pd.DataFrame(candidates).to_csv(out/'validation_candidates.csv',index=False)
    pd.DataFrame(details).to_csv(out/'validation_summary.csv',index=False)
    write_json(out/'selection.json',dict(state='POLICY_FROZEN_BEFORE_TEST',selected=winner,
        frozen_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=digest(out/'protocol.json'),
        validation_sha256={p.name:digest(p) for p in out.glob('validation_*')}))
    status(out,'VALIDATION_FROZEN',selected=winner)


def frozen(out):
    p=sources(out); selection=json.loads((out/'selection.json').read_text())
    assert digest(out/'protocol.json')==selection['protocol_sha256']
    for name,sha in selection['validation_sha256'].items():assert digest(out/name)==sha
    return p,selection


def evaluate(out,data,prediction):
    mask=data['test']; times=pd.to_datetime(data['times_ns'][mask],utc=True)
    losses={}; summary=[]; pairs=[]
    for key,q in prediction.items():
        lead,event,arm=key.split('__'); j=LEADS.index(int(lead));e=EVENTS.index(event)
        losses[key]=timing_losses(data['arrival'][e,mask],q,data['occurrence'][e,j,mask])
    for state in ['all',*STATES]:
        group=np.ones(int(mask.sum()),bool) if state=='all' else data['state'][mask]==state
        for key,metrics in losses.items():
            lead,event,arm=key.split('__')
            for metric,v in metrics.items():
                take=group&np.isfinite(v)
                summary.append(dict(lead_minutes=int(lead),event=event,arm=arm,state=state,metric=metric,
                    windows=int(take.sum()),loss=float(v[take].mean())))
        for lead in LEADS:
            for event in EVENTS:
                for a,b in [('context','frequency'),('ordered','context'),('ordered','sorted'),('ordered','frequency'),('trajectory','frequency'),('ordered','trajectory')]:
                    candidate,reference=f'{lead}__{event}__{a}',f'{lead}__{event}__{b}'
                    for metric in ['joint_rps','early_cdf_loss','conditional_rps','conditional_median_mae_minutes']:
                        va,vb=losses[candidate][metric],losses[reference][metric]
                        take=group&np.isfinite(va)&np.isfinite(vb)
                        for days in ([3,7,14] if state=='all' else [7]):
                            pairs.append(dict(candidate=candidate,reference=reference,state=state,metric=metric,windows=int(take.sum()),
                                **paired_loss_effect(times[take],va[take],vb[take],days)))
    pd.DataFrame(summary).to_csv(out/'test_summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'paired_intervals.csv',index=False)
    print(pd.DataFrame(summary).query("state=='all' and lead_minutes==720").to_string(index=False),flush=True)


def test(out):
    p,selection=frozen(out)
    if json.loads((out/'status.json').read_text())['state']!='VALIDATION_FROZEN' or list(out.glob('test_*')):
        raise RuntimeError('Frozen untested policy required')
    data=read_inputs(out)
    prediction=run_timing(out,data,'test',p['window_days'],selection['selected']['lambda_'],'test')
    prediction.update(trajectory_timing(out,data,p,'test'))
    evaluate(out,data,prediction)
    status(out,'COMPLETE',test_windows=int(data['test'].sum()),selected=selection['selected'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['prepare','validate','test'],required=True)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        {'prepare':prepare,'validate':validate,'test':test}[args.phase](args.output)
