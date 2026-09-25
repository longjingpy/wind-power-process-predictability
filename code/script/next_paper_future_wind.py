"""NP012: retrospective future-wind information and mapping-capacity diagnostic.

Geurts et al. (2006), Extremely randomized trees, doi:10.1007/s10994-006-6226-1,
Section2: randomized tree ensembles provide a nonlinear mapping reference.
The150-tree/depth16/leaf32 budget is inherited from one NP005 forest; it is
not a new algorithm. NP007 supplies mean-loss L2 multinomial logistic fitting.
Gneiting & Raftery2007, doi:10.1198/016214506000001437, and the existing
calendar-block adaptation provide paired proper-score evaluation.

ERA5 and future SCADA wind are deliberately retrospective inputs. Their
scores diagnose usable information for these mappings, NOT operational skill
or a theoretical predictability bound. All arms share complete cases, weekly
updates and event-clock maturation; ERA5 publication latency is not simulated.
100m ERA5 versus100m GFS isolates that source comparison;140m operating wind
differs in height, location, averaging and instrumentation simultaneously.
"""
from pathlib import Path
import argparse
import json
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits

from next_paper_atmospheric_context import BASE,ROOT,HOUR,DAY,MINUTE,window_physics
from next_paper_causal_calibration import fit_map,apply_map,row_hash
from next_paper_arrival_time import target_calendar
from next_paper_probability import probability_losses,paired_loss_effect
from next_paper_trajectory import digest,write_json,status

OUT=BASE/'np012'
LEADS=[15,720]
ARMS=['power','issued_gfs','issued_jma','era5_admin','era5_turbine','onsite','onsite_hourly','onsite_sorted']
FAMILIES=['logistic','forest']
POWER=ROOT/'outputs/protocol_benchmark_v24/economics/jiangsu_native/farm_15min.parquet'
PAIRS=[(arm,'power') for arm in ARMS[1:]]+[
    ('era5_admin','issued_gfs'),('era5_turbine','era5_admin'),('onsite','era5_turbine'),
    ('onsite','onsite_hourly'),('onsite','onsite_sorted')]


def monday_floor(times):
    # Unix day4 was Monday1970-01-05; retain the UTC weekly boundary.
    times=np.asarray(times,dtype=np.int64)
    return ((times-4*DAY)//(7*DAY))*7*DAY+4*DAY


def maturity(times):
    return ((np.asarray(times,dtype=np.int64)+4*HOUR+HOUR-1)//HOUR)*HOUR+MINUTE


def hourly_profile(frame,times):
    clock=frame.index.asi8;hourly=clock%HOUR==0
    assert np.all(np.diff(clock[hourly])==HOUR)
    nodes=times[:,None]+np.arange(17)[None]*15*MINUTE
    assert nodes.min()>=clock[hourly][0] and nodes.max()<=clock[hourly][-1]
    return np.interp(nodes.ravel(),clock[hourly],frame.wind.to_numpy()[hourly]).reshape(-1,17)


def sorted_interior(wind):
    copy=wind.copy();copy[:,1:-1]=np.sort(copy[:,1:-1],axis=1);return copy


def features(data,lead,arm):
    j=LEADS.index(lead);anchor=data['anchor'][j]
    x=np.c_[anchor,anchor**2,data['calendar']]
    if arm!='power':
        wind=data[arm]
        x=np.c_[x,wind/20,wind.mean(axis=1)/20,(wind[:,-1]-wind[:,0])/5,np.ptp(wind,axis=1)/10]
    return x


def training_rows(data,update):
    return np.flatnonzero(data['common']&(data['times_ns']>=update-56*DAY)&(data['maturity_ns']<=update))


def prepare(out):
    if out.exists():raise RuntimeError('Fresh output directory required')
    base=np.load(BASE/'np005/dataset.npz');pool=base['validation']|base['test']
    times=base['times_ns'][pool];clock=pd.to_datetime(times,utc=True)
    old_leads=[15,60,120,240,480,720]
    data=dict(times_ns=times,outcome=base['outcome'][pool],
        anchor=base['anchor'][[old_leads.index(l) for l in LEADS]][:,pool],
        calendar=target_calendar(times),maturity_ns=maturity(times))
    weather=base['weather'][pool].reshape(-1,17,8)
    data['issued_gfs']=np.hypot(weather[:,:,4],weather[:,:,5])
    data['issued_jma']=np.hypot(weather[:,:,0],weather[:,:,1])
    hourly=pd.read_parquet(BASE/'np010/hourly.parquet')
    for arm,site in [('era5_admin','administrative'),('era5_turbine','turbine_mean')]:
        _,fields=window_physics(hourly.loc[hourly.site==site],times)
        data[arm]=np.hypot(fields['u100'],fields['v100'])
    source=np.load(BASE/'np006/wind_reference.npz');data['onsite']=source['wind'][pool]
    frame=pd.read_parquet(POWER)
    data['onsite_hourly']=hourly_profile(frame,times)
    data['onsite_sorted']=sorted_interior(data['onsite'])
    complete=np.logical_and.reduce([np.isfinite(data[arm]).all(axis=1) for arm in ARMS[1:]])
    data.update(common=complete,original_validation=base['validation'][pool],original_test=base['test'][pool],
        validation=complete&base['validation'][pool]&(times>=pd.Timestamp('2024-09-01T00:00Z').value),
        test=complete&base['test'][pool])
    inventory=[]
    for split in ['original_validation','validation','test']:
        original=(base['validation'][pool]&(times>=pd.Timestamp('2024-09-01T00:00Z').value)) if split=='validation' else (base['test'][pool] if split=='test' else base['validation'][pool])
        take=original&complete
        inventory.append(dict(split=split,original_windows=int(original.sum()),common_windows=int(take.sum()),
            removed_windows=int((original&~complete).sum()),first_target=str(clock[take][0]),last_target=str(clock[take][-1]),
            **{f'class{i}':int((data['outcome'][take]==i).sum()) for i in range(4)}))
    out.mkdir(parents=True);(out/'models').mkdir()
    np.savez_compressed(out/'dataset.npz',**data)
    pd.DataFrame(inventory).to_csv(out/'support.csv',index=False)
    tracked=[Path(__file__),POWER,BASE/'np005/dataset.npz',BASE/'np006/wind_reference.npz',
        BASE/'np010/hourly.parquet',BASE/'np010/protocol.json',BASE/'site_metadata.json',BASE/'np008/dataset.npz']
    tracked += [BASE/f'np008/{s}_predictions.npz' for s in ['validation','test']]
    tracked += [BASE/f'np006/{s}_{l}_power_weather.npz' for s in ['validation','test'] for l in LEADS]
    tracked += [ROOT/f'script/{n}.py' for n in ['next_paper_causal_calibration','next_paper_atmospheric_context','next_paper_arrival_time','next_paper_probability']]
    protocol=dict(experiment='NP012',stage='RETROSPECTIVE_FUTURE_INFORMATION_DIAGNOSTIC',
        question='How much process probability performance can future regional or onsite wind information unlock under two mappings?',
        sources=ARMS,leads=LEADS,families=FAMILIES,dimensions={'power':9,'wind_arms':29},
        comparisons=PAIRS,parameter_selection='None; fixed before validation; no test selection',
        logistic_lambda=.001,forest=dict(n_estimators=150,max_depth=16,min_samples_leaf=32,random_state=41,n_jobs=4),
        window_days=56,update='UTC Monday00; same rows for all sources/families at a given lead/update',
        maturity='ceil_hour(T+4h)+1minute; latest bracketing wind source is accounted for',
        target='Original17-node power path classes; no reconstruction or outcome filtering beyond shared wind completeness',
        calendar='Same7known T daily/annual/quarter-hour terms; same issue-power and square',
        wind_features='17speeds/20,mean/20,net/5,range/10; same dimensions for every wind source',
        onsite='140m operating SCADA;17 trailing15min means ending one minute before target nodes',
        hourly_control='Keep original SCADA wind values at UTC whole hours, then interpolate; no hour-average or height conversion',
        sorted_control='Keep endpoints and interior15-value multiset; sort only internal values',
        information_boundary='ERA5 and future SCADA are retrospective, not operationally available; actual publication latency is not simulated',
        reference_boundary='Frozen NP008 full rolling and NP006 weather joint scores restricted to common targets; not retrained under the new sampling',
        inference='Paired3/7/14daycalendar blocks;2000draws seed41; scores on fixed sequential predictions',
        source_sha256={str(f.relative_to(ROOT)):digest(f) for f in tracked},dataset_sha256=digest(out/'dataset.npz'))
    write_json(out/'protocol.json',protocol);status(out,'PREPARED',support=inventory)
    print(pd.DataFrame(inventory).to_string(index=False),flush=True)


def sources(out):
    p=json.loads((out/'protocol.json').read_text())
    for name,sha in p['source_sha256'].items():assert digest(ROOT/name)==sha,name
    assert digest(out/'dataset.npz')==p['dataset_sha256']
    return p


def fit(data,x,rows,family,p):
    y=data['outcome'][rows];counts=np.bincount(y,minlength=4)
    if family=='logistic' or len(rows)<100 or (counts==0).any():
        return fit_map(x[rows],y,p['logistic_lambda'])
    model=ExtraTreesClassifier(**p['forest']).fit(x[rows],y)
    np.testing.assert_array_equal(model.classes_,np.arange(4))
    return dict(kind='forest',model=model,class_counts=counts,n_iter=0)


def predict(record,x):
    return record['model'].predict_proba(x) if record['kind']=='forest' else apply_map(record,x)


def references(data,split):
    parent=np.load(BASE/'np008/dataset.npz');np.testing.assert_array_equal(parent['times_ns'],data['times_ns'])
    old=np.load(BASE/f'np008/{split}_predictions.npz');result={}
    for lead in LEADS:
        result[f'{lead}__frozen_np008']=old[f'{lead}__full__rolling'][data[split][parent[split]]]
        original=data['original_validation'] if split=='validation' else data['original_test']
        probability=np.load(BASE/f'np006/{split}_{lead}_power_weather.npz')['joint']
        result[f'{lead}__frozen_np006']=probability[data[split][original]]
    return result


def run_phase(out,data,p,split):
    target=np.flatnonzero(data[split]);predictions={};audit=[];model_hashes={}
    for lead in LEADS:
        schedule=monday_floor(data['times_ns'][target]-lead*MINUTE)
        for arm in ARMS:
            x=features(data,lead,arm)
            for family in FAMILIES:
                key=f'{lead}__{arm}__{family}';probability=np.empty((len(target),4))
                for update in np.unique(schedule):
                    rows=training_rows(data,int(update));take=np.flatnonzero(schedule==update)
                    record=fit(data,x,rows,family,p)
                    record.update(rows=len(rows),row_sha256=row_hash(rows),update_ns=int(update),
                        latest_maturity_ns=int(data['maturity_ns'][rows].max()),feature_count=x.shape[1],
                        training_rows=rows,lead=lead,arm=arm,family=family)
                    assert record['latest_maturity_ns']<=update<=np.min(data['times_ns'][target[take]]-lead*MINUTE)
                    probability[take]=predict(record,x[target[take]])
                    name=f'models/{split}__{key}__{int(update)}.joblib'
                    joblib.dump(record,out/name,compress=3);model_hashes[name]=digest(out/name)
                    audit.append(dict(key=key,path=name,update_ns=int(update),kind=record['kind'],
                        rows=len(rows),row_sha256=record['row_sha256'],latest_maturity_ns=record['latest_maturity_ns'],
                        feature_count=x.shape[1],n_iter=record['n_iter'],prediction_rows=len(take)))
                predictions[key]=probability
                print(datetime.now(timezone.utc).isoformat(),split,key,'fits',len(audit),flush=True)
        # Checkpoint complete lead predictions; a crash remains explicitly incomplete.
        np.savez_compressed(out/f'{split}_predictions.partial.npz',**predictions)
    predictions.update(references(data,split))
    np.savez_compressed(out/f'{split}_predictions.npz',**predictions)
    (out/f'{split}_predictions.partial.npz').unlink()
    pd.DataFrame(audit).to_csv(out/f'{split}_update_audit.csv',index=False)
    write_json(out/f'{split}_models.json',model_hashes)
    return predictions


def evaluate(out,data,split,predictions):
    target=data[split];y=data['outcome'][target];times=pd.to_datetime(data['times_ns'][target],utc=True)
    losses={key:probability_losses(y,v) for key,v in predictions.items()}
    summary=[];paired=[]
    for key,metrics in losses.items():
        for metric,v in metrics.items():summary.append(dict(model=key,metric=metric,windows=len(y),loss=float(v.mean())))
    if split=='test':
        contrasts=[]
        for lead in LEADS:
            for family in FAMILIES:
                contrasts += [(f'{lead}__{a}__{family}',f'{lead}__{b}__{family}','information') for a,b in PAIRS]
                contrasts += [(f'{lead}__{a}__{family}',f'{lead}__{ref}','frozen_reference') for a in ['issued_gfs','onsite'] for ref in ['frozen_np008','frozen_np006']]
            contrasts += [(f'{lead}__{a}__forest',f'{lead}__{a}__logistic','mapping') for a in ARMS]
        for a,b,kind in contrasts:
            for metric in ['multiclass_brier','return_brier','up_brier','down_brier']:
                for days in [3,7,14]:
                    paired.append(dict(candidate=a,reference=b,comparison=kind,metric=metric,windows=len(y),
                        **paired_loss_effect(times,losses[a][metric],losses[b][metric],days)))
        pd.DataFrame(paired).to_csv(out/'paired_intervals.csv',index=False)
    pd.DataFrame(summary).to_csv(out/f'{split}_summary.csv',index=False)


def validate(out):
    p=sources(out)
    if json.loads((out/'status.json').read_text())['state']!='PREPARED':raise RuntimeError('Prepared run required')
    data=dict(np.load(out/'dataset.npz'));status(out,'VALIDATING')
    pred=run_phase(out,data,p,'validation');evaluate(out,data,'validation',pred)
    write_json(out/'freeze.json',dict(protocol_sha256=digest(out/'protocol.json'),
        validation_sha256={f.name:digest(f) for f in out.glob('validation_*') if f.is_file()},
        selection='No search; all preregistered sources and both mappings retained'))
    status(out,'VALIDATION_FROZEN',predictions=len(pred))


def test(out):
    p=sources(out);frozen=json.loads((out/'freeze.json').read_text())
    assert digest(out/'protocol.json')==frozen['protocol_sha256']
    for name,sha in frozen['validation_sha256'].items():assert digest(out/name)==sha
    if json.loads((out/'status.json').read_text())['state']!='VALIDATION_FROZEN':raise RuntimeError('Frozen validation required')
    data=dict(np.load(out/'dataset.npz'));status(out,'TESTING')
    pred=run_phase(out,data,p,'test');evaluate(out,data,'test',pred)
    status(out,'COMPLETE',test_windows=int(data['test'].sum()),predictions=len(pred))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','validate','test'],required=True)
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    with threadpool_limits(limits=1):{'prepare':prepare,'validate':validate,'test':test}[args.phase](args.output)
