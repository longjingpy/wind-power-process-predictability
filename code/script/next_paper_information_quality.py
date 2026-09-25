"""NP011: atmospheric information errors within power-shape matched pairs.

This is descriptive outcome matching, not causal inference. Match membership
depends only on truth paths and calendars, before orienting pairs by weather
error. The level/net/endpoint-detrended geometry decomposition extends the
existing manuscript S26 and NP002/NP010 conventions (project designs; no DOI).
It separates mean wind error from evolution error to explain task-specific
information value. ERA5/Hersbach2020, doi:10.1002/qj.3803, supplies retrospective
regional reference winds, not operational inputs or140m observation truth.

Scores reuse NP004/009: Gneiting & Raftery2007, doi:10.1198/016214506000001437.
Shared calendar blocks follow the project's Kunsch1989 adaptation,
doi:10.1214/aos/1176347265. Three prespecified primary effects additionally use
the maximum centered bootstrap deviation, standardized by bootstrap SD;
this is an approximate simultaneous interval conditional on matched members.
No claim of a novel scoring, matching or inference algorithm is made.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from next_paper_atmospheric_context import (ROOT,BASE,DAY,SITES,LEADS,
    shape_features,balance,load_losses)
from next_paper_probability import probability_losses
from next_paper_trajectory import digest,write_json,status

OUT=BASE/'np011'
QUALITY=['level','net','geometry','path']
MIN_GAP={'level':.25,'net':.25,'geometry':.10,'path':.25}
PRIMARY=[('net','720__occurrence__full-minus-power__down_brier'),
         ('geometry','720__occurrence__full-minus-power__return_brier'),
         ('geometry','15__down__context-minus-frequency__conditional_rps')]


def shape_pairs(path,times,outcome):
    _,month,hour=shape_features(path,times);candidates=[]
    for group in np.unique(month*4+outcome):
        idx=np.flatnonzero(month*4+outcome==group)
        a,b=np.triu_indices(len(idx),1);a=idx[a];b=idx[b]
        dh=np.abs(hour[a]-hour[b]);keep=(np.minimum(dh,24-dh)<=3)&(times[b]-times[a]>=DAY)
        a,b=a[keep],b[keep]
        for value in [path[:,0],path[:,-1]-path[:,0],np.ptp(path,axis=1)]:
            keep=np.abs(value[a]-value[b])<=.10;a,b=a[keep],b[keep]
        dist=np.sqrt(np.mean((path[a]-path[b])**2,axis=1));keep=dist<=.075
        candidates.extend(zip(dist[keep],a[keep],b[keep]))
    used=set();pairs=[]
    for dist,a,b in sorted(candidates):
        if a in used or b in used:continue
        used.update([a,b]);pairs.append((a,b,dist,int(outcome[a])))
    return pd.DataFrame(pairs,columns=['a','b','path_rmse','outcome'])


def weather_error(forecast,reference):
    if forecast.shape!=reference.shape or forecast.shape[-1]!=17:
        raise ValueError('Aligned17-node speed paths required')
    error=forecast-reference
    line=error[:,:1]+(error[:,-1:]-error[:,:1])*np.linspace(0,1,17)[None]
    return {'level':np.abs(error.mean(axis=1)),
            'net':np.abs(error[:,-1]-error[:,0]),
            'geometry':np.sqrt(np.mean((error[:,1:-1]-line[:,1:-1])**2,axis=1)),
            'path':np.sqrt(np.mean(error**2,axis=1))}


def orient(pairs,error,cohort='events',condition='all'):
    choose=np.ones(len(pairs),bool)
    if cohort=='events':choose &= pairs.outcome.to_numpy()!=0
    if condition=='up':choose &= pairs.outcome.isin([1,3]).to_numpy()
    if condition=='down':choose &= pairs.outcome.isin([2,3]).to_numpy()
    a=pairs.a.to_numpy(int)[choose];b=pairs.b.to_numpy(int)[choose]
    keep=np.abs(error[a]-error[b])>1e-9;a,b=a[keep],b[keep]
    high=np.where(error[a]>error[b],a,b);low=np.where(error[a]>error[b],b,a)
    return high,low


def support_record(path,times,high,low,error,quality):
    frame=pd.DataFrame({'high':high,'low':low})
    bal=balance(path,times,frame);smd=float(bal.smd.abs().max())
    blocks=[len(np.unique(times[rows]//(7*DAY))) for rows in [high,low]]
    gap=float(np.median(error[high]-error[low]))
    accepted=len(high)>=100 and min(blocks)>=8 and smd<=.2 and gap>=MIN_GAP[quality]
    return dict(pairs=len(high),high_blocks=blocks[0],low_blocks=blocks[1],
        max_abs_smd=smd,median_quality_gap=gap,accepted=accepted),bal


def prepare(out):
    if out.exists():raise RuntimeError('Fresh output directory required')
    parent=BASE/'np010';data=np.load(parent/'dataset.npz')
    base=np.load(BASE/'np005/dataset.npz');take=base['test']
    times=data['times_ns'];path=data['path'];outcome=data['outcome']
    np.testing.assert_array_equal(times,base['times_ns'][take])
    pairs=shape_pairs(path,times,outcome)
    weather=base['weather'][take].reshape(-1,17,8)
    errors={};support=[];balances=[]
    for site in SITES:
        for product,offset,height in [('gfs',4,100),('jma',0,10)]:
            forecast=np.hypot(weather[:,:,offset],weather[:,:,offset+1])
            reference=np.hypot(data[f'{site}__u{height}'],data[f'{site}__v{height}'])
            for quality,error in weather_error(forecast,reference).items():
                key=f'{site}__{product}__{quality}';errors[key]=error
                for cohort in ['all','events']:
                    for condition in ['all','up','down']:
                        high,low=orient(pairs,error,cohort,condition)
                        record,bal=support_record(path,times,high,low,error,quality)
                        metadata=dict(site=site,product=product,quality=quality,cohort=cohort,condition=condition)
                        support.append({**metadata,**record});balances.append(bal.assign(**metadata))
    out.mkdir(parents=True)
    np.savez_compressed(out/'dataset.npz',times_ns=times,path=path,outcome=outcome)
    np.savez_compressed(out/'weather_errors.npz',**errors)
    pairs.to_csv(out/'shape_pairs.csv',index=False)
    pd.DataFrame(support).to_csv(out/'support.csv',index=False)
    pd.concat(balances).to_csv(out/'balance.csv',index=False)
    count=np.bincount(pairs.outcome,minlength=4)
    tracked=[Path(__file__),parent/'dataset.npz',parent/'protocol.json',parent/'verification.json',
        BASE/'np005/dataset.npz',BASE/'np008/dataset.npz',BASE/'np008/test_predictions.npz',
        BASE/'np009/dataset.npz',BASE/'np009/test_predictions.npz',BASE/'np009/test_trajectory.npz']
    tracked += [BASE/f'np006/test_{lead}_{arm}.npz' for lead in LEADS for arm in ['power','power_weather']]
    tracked += [ROOT/f'script/{name}.py' for name in ['next_paper_atmospheric_context','next_paper_probability','next_paper_arrival_time']]
    protocol=dict(experiment='NP011',stage='EXPLORATORY_MATCHED_INFORMATION_QUALITY',
        question='Do level/net/internal atmospheric forecast errors correspond to task-specific information value for similar power paths?',
        pairing='NP010 physical path/calendar calipers, no weather condition in matching; greedy nearest eligible edge without reuse',
        target='Same6496test targets; occurrence class retained; primary cohort has at least one threshold event',
        quality='GFS100m/JMA10m against same-height ERA5; abs mean error,abs net error,endpoint-detrended interiorRMSE,fullRMSE',
        ordering='Orient frozen shape pairs by each meteorological error; remove equal errors<=1e-9; no power loss used',
        primary=PRIMARY,minimum_median_quality_gap=MIN_GAP,matched_pairs_by_class=count.tolist(),
        support='>=100pairs,>=8seven-dayblocks on both sides,max abs shape/calendarSMD<=.20,minimum quality gap',
        uncertainty='Fixed pair members;common3/7/14daycalendar weights;2000multinomialdraws seed41;three-primary max standardized centered deviation interval',
        interpretation='Descriptive matched association on explored calendar; ERA5 is not online input or independent groundtruth; no new forecasts fitted',
        sources={str(p.relative_to(ROOT)):digest(p) for p in tracked},
        artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file()})
    write_json(out/'protocol.json',protocol)
    status(out,'PREPARED',pairs=len(pairs),pair_class_counts=count.tolist(),protocol_sha256=digest(out/'protocol.json'))
    print('Matched pairs by class:',count.tolist(),flush=True)
    print(pd.DataFrame(support).query("site=='administrative' and product=='gfs' and cohort=='events'").to_string(index=False),flush=True)


def all_losses():
    result=load_losses();ds=np.load(BASE/'np005/dataset.npz');y=ds['outcome'][ds['test']]
    for lead in LEADS:
        for arm in ['power','power_weather']:
            p=np.load(BASE/f'np006/test_{lead}_{arm}.npz')['joint']
            for metric,value in probability_losses(y,p).items():
                if metric!='log_loss':result[f'{lead}__occurrence__joint_{arm}__{metric}']=value
        for metric in ['multiclass_brier','up_brier','down_brier','return_brier']:
            result[f'{lead}__occurrence__joint_weather-minus-power__{metric}']=result[f'{lead}__occurrence__joint_power_weather__{metric}']-result[f'{lead}__occurrence__joint_power__{metric}']
    return result


def block_effect(times,high,low,values,days):
    if values.ndim==1:values=values[:,None]
    blocks,idx=np.unique(times//(days*DAY),return_inverse=True)
    w=np.random.default_rng(41).multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=2000)
    means=[];draws=[]
    for rows in [high,low]:
        x=values[rows];assert np.isfinite(x).all()
        sums=np.zeros((len(blocks),x.shape[1]));counts=np.zeros(len(blocks))
        np.add.at(sums,idx[rows],x);np.add.at(counts,idx[rows],1)
        den=(w@counts)[:,None]
        draws.append(np.divide(w@sums,den,out=np.full((2000,x.shape[1]),np.nan),where=den>0))
        means.append(x.mean(axis=0))
    return means,draws[0]-draws[1]


def simultaneous(point,replicates):
    valid=np.isfinite(replicates).all(axis=1);r=replicates[valid]
    scale=r.std(axis=0,ddof=1)
    if np.any(scale<=0):raise ValueError('Nonzero bootstrap variability required')
    critical=float(np.quantile(np.max(np.abs((r-point)/scale),axis=1),.95))
    return np.stack([point-critical*scale,point+critical*scale]),critical,int(valid.sum())


def evaluate(out):
    p=json.loads((out/'protocol.json').read_text())
    if json.loads((out/'status.json').read_text())['state']!='PREPARED':raise RuntimeError('Prepared unscored run required')
    for name,sha in p['sources'].items():assert digest(ROOT/name)==sha,name
    for name,sha in p['artifacts'].items():assert digest(out/name)==sha,name
    ds=np.load(out/'dataset.npz');times=ds['times_ns'];pairs=pd.read_csv(out/'shape_pairs.csv')
    errors=np.load(out/'weather_errors.npz');support=pd.read_csv(out/'support.csv')
    losses=all_losses();np.savez_compressed(out/'losses.npz',**losses)
    rows=[];primary_draws={days:{} for days in [3,7,14]};primary_points={days:{} for days in [3,7,14]}
    for key,error in errors.items():
        site,product,quality=key.split('__')
        for cohort in ['all','events']:
            for condition in ['all','up','down']:
                high,low=orient(pairs,error,cohort,condition)
                keys=[k for k in losses if (k.split('__')[1]=='occurrence' if condition=='all' else k.split('__')[1]==condition)]
                matrix=np.column_stack([losses[k] for k in keys])
                info=support.loc[(support.site==site)&(support['product']==product)&(support.quality==quality)&(support.cohort==cohort)&(support.condition==condition)].iloc[0]
                for days in [3,7,14]:
                    means,rep=block_effect(times,high,low,matrix,days)
                    diff=means[0]-means[1];ci=np.nanquantile(rep,[.025,.975],axis=0)
                    for i,k in enumerate(keys):
                        lead,task,arm,metric=k.split('__')
                        rows.append(dict(site=site,product=product,quality=quality,cohort=cohort,condition=condition,
                            block_days=days,lead_minutes=int(lead),task=task,arm=arm,metric=metric,
                            pairs=len(high),high_mean=means[0][i],low_mean=means[1][i],high_minus_low=diff[i],
                            ci_low=ci[0,i],ci_high=ci[1,i],valid_replicates=int(np.isfinite(rep[:,i]).sum()),
                            accepted=bool(info.accepted),median_quality_gap=info.median_quality_gap,max_abs_smd=info.max_abs_smd))
                        if site=='administrative' and product=='gfs' and cohort=='events' and (quality,k) in PRIMARY:
                            name=f'{quality}__{k}';primary_draws[days][name]=rep[:,i];primary_points[days][name]=diff[i]
    table=pd.DataFrame(rows);table.to_csv(out/'conditional_contrasts.csv',index=False)
    primary=[];critical={}
    for days in [3,7,14]:
        keys=[f'{q}__{k}' for q,k in PRIMARY]
        point=np.array([primary_points[days][k] for k in keys]);rep=np.column_stack([primary_draws[days][k] for k in keys])
        ci,c,n=simultaneous(point,rep);critical[str(days)]=c
        for i,(q,k) in enumerate(PRIMARY):
            lead,task,arm,metric=k.split('__')
            r=table.loc[(table.site=='administrative')&(table['product']=='gfs')&(table.quality==q)&(table.cohort=='events')&
                (table.block_days==days)&(table.lead_minutes==int(lead))&(table.task==task)&(table.arm==arm)&(table.metric==metric)].iloc[0].to_dict()
            r.update(simultaneous_low=ci[0,i],simultaneous_high=ci[1,i],family_size=3,simultaneous_valid_replicates=n)
            primary.append(r)
    pd.DataFrame(primary).to_csv(out/'primary_contrasts.csv',index=False)
    write_json(out/'inference.json',dict(primary_family=PRIMARY,critical_values=critical,
        procedure='maximum absolute centered standardized bootstrap deviation across3fixed primary effects; fixed pairs, common calendar blocks'))
    write_json(out/'result_manifest.json',{f.name:digest(f) for f in out.iterdir() if f.is_file() and f.name!='status.json'})
    status(out,'COMPLETE',pairs=len(pairs),contrast_rows=len(table),new_forecast_fits=0)
    print(pd.DataFrame(primary).query('block_days==7').to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','evaluate'],required=True)
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    with threadpool_limits(limits=1):{'prepare':prepare,'evaluate':evaluate}[args.phase](args.output)
