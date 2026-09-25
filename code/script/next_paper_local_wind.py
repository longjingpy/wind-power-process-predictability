"""NP013: available-input local wind scenarios and process probabilities.

Meinshausen (2006), Quantile Regression Forests, JMLR7:983-999, Sections2-3,
https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf:
leaf weights define an empirical conditional response distribution. We reuse
NP004's ExtraTrees/multioutput/whole-path sampling adaptation, here fitting
summaries of local-wind forecast errors. These are empirical site/height
corrections, not physical height extrapolation or the original theorem.

NP012 supplies the29-feature multinomial wind-response map. Integrating its
probabilities over predicted wind paths approximates the process law; it may
discard history effects remaining after issue power/calendar/wind. Direct
matched and all-label classifiers test that approximation. Point and
unconditional-error controls isolate uncertainty and conditional information.
Gneiting&Raftery2007 doi:10.1198/016214506000001437 supplies the proper-score
foundation; inherited shared calendar blocks compare fixed predictions.
"""
from pathlib import Path
import argparse
import json
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from threadpoolctl import threadpool_limits

from next_paper_future_wind import monday_floor,features as features12,predict as predict12
from next_paper_causal_calibration import fit_map,apply_map,row_hash
from next_paper_arrival_time import target_calendar
from next_paper_atmospheric_context import ROOT,BASE,DAY,MINUTE,HOUR
from next_paper_probability import (build_leaf_bank,draw_neighbors,probability_from_classes,
    probability_losses,empirical_crps,paired_loss_effect)
from next_paper_trajectory import digest,write_json,status

OUT=BASE/'np013';LEADS=[15,720]
PARAMS=dict(n_estimators=150,max_depth=16,min_samples_leaf=32,max_features=1.,random_state=41,n_jobs=4)
ARMS=['direct_all','direct_matched','mixture','point','unconditional','neighbor_event','issued_response']
WEIGHTS=[0.,.25,.5,.75,1.]
WIND_ARMS=['conditional','unconditional','gfs','persistence']
WIND_METRICS=['profile_crps','profile_mse',*[f'{q}_{m}' for q in ['mean','range','net'] for m in ['crps','mae','coverage80','width80']]]


def residual_summary(error):
    return np.column_stack([error.mean(axis=1),error[:,0],error[:,-1],error.min(axis=1),
                            error.max(axis=1),error[:,-1]-error[:,0],np.ptp(error,axis=1)])/5


def response_features(known,wind):
    wind=np.asarray(wind);shape=wind.shape[:-1]
    if wind.shape[-1]!=17 or wind.ndim not in [2,3]:raise ValueError('17-point wind paths required')
    if wind.ndim==3:known=np.broadcast_to(known[:,None,:],(*shape,9)).reshape(-1,9)
    w=wind.reshape(-1,17)
    return np.c_[known,w/20,w.mean(axis=1)/20,(w[:,-1]-w[:,0])/5,np.ptp(w,axis=1)/10]


def response(teacher,known,wind):
    return apply_map(teacher,response_features(known,wind)).reshape(*wind.shape[:-1],4)


def inputs(data,lead):
    j=LEADS.index(lead)
    x=np.c_[data['history'][j],data['weather'],data['onsite_history'][j]]
    known=np.c_[data['anchor'][j],data['anchor'][j]**2,data['calendar']]
    return x,known


def train_rows(data,update,wind=False):
    take=(data['times_ns']>=update-56*DAY)&(data['times_ns']+241*MINUTE<=update)
    if wind:take &= data['wind_valid']
    return np.flatnonzero(take)


def prepare(out):
    if out.exists():raise RuntimeError('Fresh output directory required')
    base=np.load(BASE/'np005/dataset.npz');wind=np.load(BASE/'np006/wind_reference.npz')
    profile=base['weather'].reshape(-1,17,8)
    data=dict(times_ns=base['times_ns'],outcome=base['outcome'],weather=base['weather'],
        history=base['history'][[0,5]],onsite_history=base['onsite'][[0,5]],anchor=base['anchor'][[0,5]],
        calendar=target_calendar(base['times_ns']),gfs=np.hypot(profile[:,:,4],profile[:,:,5]),
        wind=wind['wind'],wind_valid=wind['valid'],original_validation=base['validation'],
        validation=base['validation']&(base['times_ns']>=pd.Timestamp('2024-09-01T00:00Z').value),test=base['test'])
    np.testing.assert_array_equal(data['wind_valid'],np.isfinite(data['wind']).all(axis=1))
    for lead in LEADS:
        x,known=inputs(data,lead);assert x.shape[1]==186 and known.shape[1]==9 and np.isfinite(x).all()
    out.mkdir(parents=True);(out/'models').mkdir()
    np.savez_compressed(out/'dataset.npz',**data)
    support=[]
    for split in ['validation','test']:
        take=data[split];y=data['outcome'][take]
        support.append(dict(split=split,forecast_windows=int(take.sum()),wind_diagnostic_windows=int((take&data['wind_valid']).sum()),
            **{f'class{i}':int((y==i).sum()) for i in range(4)}))
    pd.DataFrame(support).to_csv(out/'support.csv',index=False)
    tracked=[Path(__file__),BASE/'np005/dataset.npz',BASE/'np006/wind_reference.npz',BASE/'np008/dataset.npz',BASE/'np012/dataset.npz']
    tracked += [BASE/f'np008/{s}_predictions.npz' for s in ['validation','test']]
    tracked += [BASE/f'np012/{s}_models.json' for s in ['validation','test']]
    tracked += [BASE/f'np006/{s}_{l}_power_weather.npz' for s in ['validation','test'] for l in LEADS]
    tracked += [ROOT/f'script/{name}.py' for name in ['next_paper_probability','next_paper_future_wind','next_paper_causal_calibration','next_paper_arrival_time']]
    protocol=dict(experiment='NP013',stage='AVAILABLE_INPUT_LOCAL_WIND_PROBABILITY_TRANSFER',
        leads=LEADS,arms=ARMS,forest_parameters=PARAMS,teacher_lambda=.001,inputs=186,
        fit_pool='All NP00518653targets; past56days, UTC Monday update; T+4h+1min matured',
        matched_rows='Wind regression, response and matched classifier share complete-wind rows; all classifier uses all matured power labels',
        targets='Seven summaries(mean,start,end,min,max,net,range)of onsite140m minusGFS100m path, all/5',
        scenarios='One full residual path per regression tree leaf; add current GFS, clip only below0;150members',
        response='NP01229features: known9context +17wind/20 +mean/20,net/5,range/10; multinomial lambda=.001',
        hypothesis='Approximate event law through wind; remaining history effects can invalidate the factorization',
        selection='One shared weight minimizes mean validation returnBrier across leads among candidates with mean multiclassBrier<=direct_all;ties smaller weight',
        weights=WEIGHTS,blend='(1-w)*direct_all+w*mixture; oracle cannot enter blend or selection',
        inference='Fixed predictions;3/7/14day paired calendar blocks,2000draws seed41; no refitting or resampling uncertainty in intervals',
        reference='Frozen NP008,NP006 plus NP012 issued_gfs forest extended to all target windows; no future-input source used',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in tracked},dataset_sha256=digest(out/'dataset.npz'))
    write_json(out/'protocol.json',protocol);status(out,'PREPARED',support=support)
    print(pd.DataFrame(support).to_string(index=False),flush=True)


def sources(out):
    p=json.loads((out/'protocol.json').read_text())
    for name,sha in p['source_sha256'].items():assert digest(ROOT/name)==sha,name
    assert digest(out/'dataset.npz')==p['dataset_sha256']
    return p


def fit_pack(data,x,known,update):
    all_rows=train_rows(data,update);rows=train_rows(data,update,True)
    y=data['outcome'];error=data['wind'][rows]-data['gfs'][rows];target=residual_summary(error)
    reg=ExtraTreesRegressor(**PARAMS).fit(x[rows],target)
    bank,leaf_error=build_leaf_bank([(41,reg,None)],x[rows],target)
    teacher=fit_map(response_features(known[rows],data['wind'][rows]),y[rows],.001)
    matched=ExtraTreesClassifier(**PARAMS).fit(x[rows],y[rows])
    direct=ExtraTreesClassifier(**PARAMS).fit(x[all_rows],y[all_rows])
    for model in [matched,direct]:np.testing.assert_array_equal(model.classes_,np.arange(4))
    return dict(regressor=reg,bank=bank,response=teacher,direct_matched=matched,direct_all=direct,
        wind_rows=rows,all_rows=all_rows,wind_rows_sha256=row_hash(rows),all_rows_sha256=row_hash(all_rows),
        error_paths=error,event_labels=y[rows],update_ns=int(update),leaf_target_error=leaf_error,
        latest_matured_ns=int((data['times_ns'][all_rows]+241*MINUTE).max()))


def operational_predict(pack,x,known,gfs,seed):
    """Only known issue-time arrays and fitted historical state enter this call."""
    neighbors=draw_neighbors([(41,pack['regressor'],None)],pack['bank'],x,seed)
    uncond=np.random.default_rng(seed+1).integers(0,len(pack['wind_rows']),size=neighbors.shape,dtype=np.int32)
    conditional=np.maximum(gfs[:,None,:]+pack['error_paths'][neighbors],0)
    unconditional=np.maximum(gfs[:,None,:]+pack['error_paths'][uncond],0)
    teacher=pack['response']
    probabilities=dict(direct_all=pack['direct_all'].predict_proba(x),
        direct_matched=pack['direct_matched'].predict_proba(x),
        mixture=response(teacher,known,conditional).mean(axis=1),
        point=response(teacher,known,conditional.mean(axis=1)),
        unconditional=response(teacher,known,unconditional).mean(axis=1),
        neighbor_event=probability_from_classes(pack['event_labels'][neighbors]),
        issued_response=response(teacher,known,gfs))
    return probabilities,conditional,unconditional,neighbors,uncond


def wind_metrics(samples,truth):
    mean=samples.mean(axis=1)
    result=dict(profile_crps=empirical_crps(samples,truth).mean(axis=1),profile_mse=((mean-truth)**2).mean(axis=1))
    for name,op in [('mean',lambda a:a.mean(axis=-1)),('range',lambda a:np.ptp(a,axis=-1)),('net',lambda a:a[...,-1]-a[...,0])]:
        particle=op(samples);actual=op(truth)
        lo,hi=np.quantile(particle,[.1,.9],axis=1)
        result.update({f'{name}_crps':empirical_crps(particle[:,:,None],actual[:,None]).ravel(),
            f'{name}_mae':np.abs(particle.mean(axis=1)-actual),f'{name}_coverage80':((actual>=lo)&(actual<=hi)).astype(float),
            f'{name}_width80':hi-lo})
    return result


def reference_predictions(data,split):
    target=np.flatnonzero(data[split]);times=data['times_ns'][target];result={}
    parent=np.load(BASE/'np008/dataset.npz');p8=np.load(BASE/f'np008/{split}_predictions.npz')
    idx8=pd.Index(parent['times_ns'][parent[split]]).get_indexer(times);assert (idx8>=0).all()
    mask6=data['original_validation'] if split=='validation' else data['test']
    idx6=pd.Index(data['times_ns'][mask6]).get_indexer(times)
    d12=dict(np.load(BASE/'np012/dataset.npz'));idx12=pd.Index(d12['times_ns']).get_indexer(times);assert (idx12>=0).all()
    hashes=json.loads((BASE/f'np012/{split}_models.json').read_text())
    for lead in LEADS:
        result[f'{lead}__frozen_np008']=p8[f'{lead}__full__rolling'][idx8]
        result[f'{lead}__frozen_np006']=np.load(BASE/f'np006/{split}_{lead}_power_weather.npz')['joint'][idx6]
        x=features12(d12,lead,'issued_gfs')[idx12];assert np.isfinite(x).all()
        schedule=monday_floor(times-lead*MINUTE);pred=np.empty((len(times),4))
        for update in np.unique(schedule):
            name=f'models/{split}__{lead}__issued_gfs__forest__{int(update)}.joblib'
            assert digest(BASE/'np012'/name)==hashes[name]
            take=np.flatnonzero(schedule==update);pack=joblib.load(BASE/'np012'/name)
            pred[take]=predict12(pack,x[take])
        result[f'{lead}__frozen_np012_gfs']=pred
    return result


def run_phase(out,data,split):
    target=np.flatnonzero(data[split]);n=len(target);predictions={};neighbors={};diagnostics={};oracles={}
    hashes={};audit=[]
    for j,lead in enumerate(LEADS):
        x,known=inputs(data,lead);schedule=monday_floor(data['times_ns'][target]-lead*MINUTE)
        for arm in ARMS:predictions[f'{lead}__{arm}']=np.empty((n,4))
        for name in ['conditional','unconditional']:neighbors[f'{lead}__{name}']=np.empty((n,150),np.int32)
        for arm in WIND_ARMS:
            for metric in WIND_METRICS:diagnostics[f'{lead}__{arm}__{metric}']=np.full(n,np.nan)
        oracles[str(lead)]=np.full((n,4),np.nan)
        for update in np.unique(schedule):
            pack=fit_pack(data,x,known,int(update));assert pack['latest_matured_ns']<=update
            take=np.flatnonzero(schedule==update);assert update<=np.min(data['times_ns'][target[take]]-lead*MINUTE)
            seed=41+int(update//DAY)+lead+(100000 if split=='test' else 0)
            # Batch bounds prevent large wind-response design matrices.
            for start in range(0,len(take),128):
                loc=take[start:start+128];rows=target[loc]
                prob,wind,unc,ids,unids=operational_predict(pack,x[rows],known[rows],data['gfs'][rows],seed+start)
                for arm,v in prob.items():predictions[f'{lead}__{arm}'][loc]=v
                neighbors[f'{lead}__conditional'][loc]=pack['wind_rows'][ids]
                neighbors[f'{lead}__unconditional'][loc]=pack['wind_rows'][unids]
                good=data['wind_valid'][rows];wind_rows=rows[good]
                if good.any():
                    truth=data['wind'][wind_rows]
                    particles={'conditional':wind[good],'unconditional':unc[good],
                        'gfs':data['gfs'][wind_rows,None,:],
                        'persistence':np.broadcast_to(data['onsite_history'][j,wind_rows,0,None,None]*20,(len(wind_rows),1,17))}
                    for arm,sample in particles.items():
                        for metric,v in wind_metrics(sample,truth).items():diagnostics[f'{lead}__{arm}__{metric}'][loc[good]]=v
                    # Kept separate: observed future wind must never enter operating forecasts.
                    oracles[str(lead)][loc[good]]=response(pack['response'],known[wind_rows],truth)
            name=f'models/{split}__{lead}__{int(update)}.joblib';joblib.dump(pack,out/name,compress=3);hashes[name]=digest(out/name)
            audit.append(dict(lead=lead,update_ns=int(update),path=name,wind_rows=len(pack['wind_rows']),all_rows=len(pack['all_rows']),
                wind_rows_sha256=pack['wind_rows_sha256'],all_rows_sha256=pack['all_rows_sha256'],seed=seed,
                latest_matured_ns=pack['latest_matured_ns'],predictions=len(take),leaf_target_error=pack['leaf_target_error']))
            print(datetime.now(timezone.utc).isoformat(),split,lead,str(pd.Timestamp(update,tz='UTC')),'rows',len(pack['wind_rows']),len(pack['all_rows']),flush=True)
    predictions.update(reference_predictions(data,split))
    np.savez_compressed(out/f'{split}_predictions.npz',**predictions)
    np.savez_compressed(out/f'{split}_neighbors.npz',**neighbors)
    np.savez_compressed(out/f'{split}_wind_scores.npz',**diagnostics)
    np.savez_compressed(out/f'{split}_oracle.npz',**oracles)
    pd.DataFrame(audit).to_csv(out/f'{split}_audit.csv',index=False)
    write_json(out/f'{split}_models.json',hashes)
    return predictions


def select_weight(y,predictions):
    rows=[]
    for weight in WEIGHTS:
        scores=[]
        for lead in LEADS:
            p=(1-weight)*predictions[f'{lead}__direct_all']+weight*predictions[f'{lead}__mixture']
            loss=probability_losses(y,p);scores.append([loss['return_brier'].mean(),loss['multiclass_brier'].mean()])
        rows.append(dict(weight=weight,return_brier=float(np.mean(scores,axis=0)[0]),multiclass_brier=float(np.mean(scores,axis=0)[1])))
    baseline=rows[0]['multiclass_brier']
    for r in rows:r['eligible']=bool(r['multiclass_brier']<=baseline+1e-12)
    selected=min([r for r in rows if r['eligible']],key=lambda r:(r['return_brier'],r['weight']))
    return selected,rows


def evaluate(out,data,split,predictions,weight):
    take=data[split];times=pd.to_datetime(data['times_ns'][take],utc=True);y=data['outcome'][take]
    for lead in LEADS:predictions[f'{lead}__selected']=(1-weight)*predictions[f'{lead}__direct_all']+weight*predictions[f'{lead}__mixture']
    np.savez_compressed(out/f'{split}_predictions.npz',**predictions)
    losses={k:probability_losses(y,v) for k,v in predictions.items()};summary=[];paired=[]
    for key,value in losses.items():
        for metric,v in value.items():summary.append(dict(model=key,metric=metric,windows=len(y),loss=float(v.mean())))
    oracle=np.load(out/f'{split}_oracle.npz');good=data['wind_valid'][take]
    for lead in LEADS:
        for metric,v in probability_losses(y[good],oracle[str(lead)][good]).items():
            summary.append(dict(model=f'{lead}__oracle_DIAGNOSTIC',metric=metric,windows=int(good.sum()),loss=float(v.mean())))
    pd.DataFrame(summary).to_csv(out/f'{split}_summary.csv',index=False)
    wind=np.load(out/f'{split}_wind_scores.npz');wrows=[];wpairs=[]
    for key,value in wind.items():
        lead,arm,metric=key.split('__');finite=np.isfinite(value)
        wrows.append(dict(lead=int(lead),arm=arm,metric=metric,windows=int(finite.sum()),value=float(value[finite].mean())))
    pd.DataFrame(wrows).to_csv(out/f'{split}_wind_summary.csv',index=False)
    if split=='test':
        comparisons=[('selected','direct_all'),('mixture','direct_all'),('mixture','direct_matched'),
            ('mixture','point'),('mixture','unconditional'),('mixture','neighbor_event'),('mixture','issued_response'),('direct_matched','direct_all')]
        comparisons += [(a,b) for a in ['selected','direct_all','mixture'] for b in ['frozen_np008','frozen_np006','frozen_np012_gfs']]
        for lead in LEADS:
            for a,b in comparisons:
                for metric in ['multiclass_brier','return_brier','up_brier','down_brier']:
                    for days in [3,7,14]:
                        ka,kb=f'{lead}__{a}',f'{lead}__{b}'
                        paired.append(dict(candidate=ka,reference=kb,metric=metric,windows=len(y),**paired_loss_effect(times,losses[ka][metric],losses[kb][metric],days)))
            for ref in ['gfs','persistence','unconditional']:
                for metric in ['profile_crps','profile_mse','mean_crps','range_crps','net_crps']:
                    va,vb=wind[f'{lead}__conditional__{metric}'],wind[f'{lead}__{ref}__{metric}'];valid=np.isfinite(va)&np.isfinite(vb)
                    for days in [3,7,14]:wpairs.append(dict(lead=lead,reference=ref,metric=metric,windows=int(valid.sum()),**paired_loss_effect(times[valid],va[valid],vb[valid],days)))
        pd.DataFrame(paired).to_csv(out/'paired_intervals.csv',index=False)
        pd.DataFrame(wpairs).to_csv(out/'wind_intervals.csv',index=False)


def validate(out):
    sources(out)
    if json.loads((out/'status.json').read_text())['state']!='PREPARED':raise RuntimeError('Prepared run required')
    data=dict(np.load(out/'dataset.npz'));status(out,'VALIDATING')
    pred=run_phase(out,data,'validation');selected,candidates=select_weight(data['outcome'][data['validation']],pred)
    evaluate(out,data,'validation',pred,selected['weight']);pd.DataFrame(candidates).to_csv(out/'blend_selection.csv',index=False)
    write_json(out/'freeze.json',dict(selected=selected,protocol_sha256=digest(out/'protocol.json'),
        validation_sha256={f.name:digest(f) for f in out.glob('validation_*') if f.is_file()},selection_sha256=digest(out/'blend_selection.csv')))
    status(out,'VALIDATION_FROZEN',selected=selected);print('SELECTED',selected,flush=True)


def test(out):
    sources(out);freeze=json.loads((out/'freeze.json').read_text())
    assert digest(out/'protocol.json')==freeze['protocol_sha256'] and digest(out/'blend_selection.csv')==freeze['selection_sha256']
    for name,sha in freeze['validation_sha256'].items():assert digest(out/name)==sha
    if json.loads((out/'status.json').read_text())['state']!='VALIDATION_FROZEN':raise RuntimeError('Frozen run required')
    data=dict(np.load(out/'dataset.npz'));status(out,'TESTING')
    pred=run_phase(out,data,'test');evaluate(out,data,'test',pred,freeze['selected']['weight'])
    status(out,'COMPLETE',test_windows=int(data['test'].sum()),selected_weight=freeze['selected']['weight'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','validate','test'],required=True)
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    with threadpool_limits(limits=1):{'prepare':prepare,'validate':validate,'test':test}[args.phase](args.output)
