"""NP010: retrospective atmospheric context for frozen process forecasts.

ERA5: Hersbach et al. (2020), The ERA5 global reanalysis,
doi:10.1002/qj.3803. ECMWF data documentation, 'Mean rates/fluxes and
accumulations': hourly energy is for the hour ending at validity time.
Divide by3600 and integrate overlaps, never deaccumulate these hourly values.
https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation
ARCO source/provenance: https://github.com/google-research/arco-era5
Spatial alias/bounds handling is adapted from pizhou_surface_context.py.

Power-path matching below is an explicitly designed descriptive comparison,
not a causal estimator or a published matching-method reproduction. Match on
observed outcomes to ask about similar realized processes, never to generate
forecasts. Matching never sees forecast losses. Kunsch (1989),
doi:10.1214/aos/1176347265 motivates inherited calendar-block resampling;
the shared endpoint-weight adaptation holds matched membership fixed.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from next_paper_trajectory import digest, write_json, status
from next_paper_probability import probability_losses
from next_paper_arrival_time import timing_losses
from next_paper_causal_calibration import ROOT, LEADS, DAY, MINUTE

OUT = ROOT/'outputs/next_paper/np010'
BASE = ROOT/'outputs/next_paper'
HOUR = 60*MINUTE
SITES = ['administrative', 'turbine_mean']
REGIMES = ['shear', 'heat']
ALIASES = {'u10':'10m_u_component_of_wind', 'v10':'10m_v_component_of_wind',
           'u100':'100m_u_component_of_wind', 'v100':'100m_v_component_of_wind',
           't2m':'2m_temperature', 'sp':'surface_pressure',
           'sshf':'surface_sensible_heat_flux'}
UNITS = {k:'m s**-1' for k in ['u10','v10','u100','v100']}
UNITS.update(t2m='K',sp='Pa',sshf='J m**-2')


def coordinates():
    site=json.loads((BASE/'site_metadata.json').read_text())
    table=pd.read_csv(ROOT/'data/real/whereTurbins.csv')
    def dms(value):
        d,m,s=map(float,str(value).split()); return d+m/60+s/3600
    lat=np.r_[site['coordinates']['latitude'],table['纬度'].map(dms)]
    lon=np.r_[site['coordinates']['longitude'],table['经度'].map(dms)]
    return lat,lon


def extract_hourly():
    lat,lon=coordinates(); frames=[]; manifest=[]
    for path in sorted((ROOT/'data/era5/raw').glob('era5_20*.nc')):
        with xr.open_dataset(path) as data:
            tc='time' if 'time' in data.coords else 'valid_time'
            time=pd.to_datetime(data[tc].values,utc=True)
            assert np.all(np.diff(time.asi8)==HOUR)
            assert data.latitude.min()<=lat.min()<=lat.max()<=data.latitude.max()
            assert data.longitude.min()<=lon.min()<=lon.max()<=data.longitude.max()
            arrays={}; names={}
            for key,alias in ALIASES.items():
                name=key if key in data else alias
                assert data[name].attrs.get('units')==UNITS[key]
                values=data[name].interp(latitude=xr.DataArray(lat,dims='point'),
                    longitude=xr.DataArray(lon,dims='point')).transpose(tc,'point').values
                assert np.isfinite(values).all()
                arrays[key]=np.stack([values[:,0],values[:,1:].mean(axis=1)])
                names[key]=name
            for s,site in enumerate(SITES):
                frames.append(pd.DataFrame({'time':time,'site':site,
                    **{k:v[s] for k,v in arrays.items()}}))
            manifest.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path),
                latitude=data.latitude.values.tolist(),longitude=data.longitude.values.tolist(),
                names=names,units=UNITS,first=str(time[0]),last=str(time[-1])))
    frame=pd.concat(frames,ignore_index=True).sort_values(['site','time'])
    for _,g in frame.groupby('site'):
        assert g.time.is_unique and np.all(np.diff(g.time.astype('int64'))==HOUR)
    return frame,manifest


def window_physics(hourly,times):
    """Use instantaneous vector interpolation and exact quarter-hour flux overlap."""
    clock=hourly.time.astype('int64').to_numpy(); times=np.asarray(times,np.int64)
    nodes=times[:,None]+np.arange(17)[None]*15*MINUTE
    assert nodes.min()>=clock[0] and nodes.max()<=clock[-1]
    fields={k:np.interp(nodes.ravel(),clock,hourly[k]).reshape(nodes.shape)
            for k in ALIASES if k!='sshf'}
    # Each midpoint lies in one hourly accumulation interval; all T use15min.
    assert not np.any(times%(15*MINUTE))
    middle=nodes[:,:-1]+int(7.5*MINUTE)
    ends=np.searchsorted(clock,middle,side='left')
    assert np.all(clock[ends]-HOUR<middle) and np.all(middle<clock[ends])
    flux=hourly.sshf.to_numpy()[ends].mean(axis=1)/3600
    u,v=fields['u100'],fields['v100']; speed=np.hypot(u,v)
    shear=np.hypot(u-fields['u10'],v-fields['v10'])
    def average(x):return np.trapezoid(x,dx=1/16,axis=1)
    result={'shear':average(shear),'heat':flux,'speed100':average(speed),
        'wind_delta':speed[:,-1]-speed[:,0],
        'vector_delta':np.hypot(u[:,-1]-u[:,0],v[:,-1]-v[:,0]),
        'temperature_delta':fields['t2m'][:,-1]-fields['t2m'][:,0],
        'pressure_delta_hpa':(fields['sp'][:,-1]-fields['sp'][:,0])/100}
    return result,fields


def shape_features(path,times):
    local=pd.to_datetime(times,utc=True).tz_convert('Asia/Shanghai')
    hour=local.hour.to_numpy()+local.minute.to_numpy()/60
    x=np.column_stack([path,path[:,-1]-path[:,0],np.ptp(path,axis=1),
                       np.sin(hour*2*np.pi/24),np.cos(hour*2*np.pi/24)])
    return x,local.year.to_numpy()*100+local.month.to_numpy(),hour


def match_paths(path,times,outcome,regime):
    """Greedy nearest eligible path pairs, without replacement or loss access."""
    _,month,hour=shape_features(path,times); candidates=[]
    for group in np.unique(month*4+outcome):
        a=np.flatnonzero((month*4+outcome==group)&(regime==1))
        b=np.flatnonzero((month*4+outcome==group)&(regime==-1))
        if not len(a) or not len(b):continue
        d=np.sqrt(cdist(path[a],path[b],metric='sqeuclidean')/17)
        dt=np.abs(times[a,None]-times[b][None,:])
        dh=np.abs(hour[a,None]-hour[b][None,:]); dh=np.minimum(dh,24-dh)
        net=path[:,-1]-path[:,0]; span=np.ptp(path,axis=1)
        ok=(d<=.075)&(dt>=DAY)&(dh<=3)
        for feature in [path[:,0],net,span]:
            ok &= np.abs(feature[a,None]-feature[b][None,:])<=.10
        ia,ib=np.where(ok)
        candidates.extend(zip(d[ia,ib],a[ia],b[ib]))
    used=set(); pairs=[]
    for dist,a,b in sorted(candidates):
        if a in used or b in used:continue
        used.update([a,b]); pairs.append((a,b,dist))
    return pd.DataFrame(pairs,columns=['high','low','path_rmse'])


def balance(path,times,pairs):
    x,_,_=shape_features(path,times)
    names=[f'power_{15*i}' for i in range(17)]+['net','range','hour_sin','hour_cos']
    a=pairs.high.to_numpy(int); b=pairs.low.to_numpy(int)
    scale=np.sqrt((x[a].var(axis=0)+x[b].var(axis=0))/2)
    diff=x[a].mean(axis=0)-x[b].mean(axis=0)
    smd=np.divide(diff,scale,out=np.zeros_like(diff),where=scale>1e-12)
    smd[(scale<=1e-12)&(np.abs(diff)>1e-12)]=np.inf
    return pd.DataFrame({'feature':names,'high_mean':x[a].mean(axis=0),
                         'low_mean':x[b].mean(axis=0),'smd':smd})


def prepare(out):
    if out.exists():raise RuntimeError('Fresh directory required')
    base=np.load(BASE/'np005/dataset.npz'); take=base['test']
    times=base['times_ns'][take]; path=base['actual'][take]; outcome=base['outcome'][take]
    hourly,source=extract_hourly(); physics=[]; thresholds={}; fields={}
    cutoff=pd.Timestamp('2024-10-20T16:00Z').value
    for site in SITES:
        h=hourly.loc[hourly.site.eq(site)]
        reference=h.time.astype('int64').to_numpy()
        reference=reference[reference+4*HOUR<cutoff]
        training,_=window_physics(h,reference)
        low,high=np.quantile(training['shear'],[1/3,2/3]);thresholds[site]=[float(low),float(high)]
        value,fields[site]=window_physics(h,times)
        weather=base['weather'][take].reshape(-1,17,8)
        for model,offset,height in [('gfs',4,100),('jma',0,10)]:
            observed=np.hypot(fields[site][f'u{height}'],fields[site][f'v{height}'])
            forecast=np.hypot(weather[:,:,offset],weather[:,:,offset+1])
            value[f'{model}_speed_rmse']=np.sqrt(np.mean((forecast-observed)**2,axis=1))
            value[f'{model}_delta_error']=(forecast[:,-1]-forecast[:,0])-(observed[:,-1]-observed[:,0])
        value['shear_group']=np.where(value['shear']<=low,-1,np.where(value['shear']>=high,1,0))
        value['heat_group']=np.where(value['heat']<=-20,-1,np.where(value['heat']>=20,1,0))
        physics.append(pd.DataFrame({'row':np.arange(len(times)),'time':pd.to_datetime(times,utc=True),
                                    'site':site,**value}))
    physical=pd.concat(physics,ignore_index=True)
    matches=[]; balances=[]; supports=[]
    for site in SITES:
        h=physical.loc[physical.site.eq(site)]
        for regime in REGIMES:
            pairs=match_paths(path,times,outcome,h[f'{regime}_group'].to_numpy())
            if len(pairs)==0:raise RuntimeError(f'No eligible pairs: {site}/{regime}')
            bal=balance(path,times,pairs)
            a=pairs.high.to_numpy(int);b=pairs.low.to_numpy(int)
            blocks=[len(np.unique(times[idx]//(7*DAY))) for idx in [a,b]]
            max_smd=float(bal.smd.abs().max())
            accepted=len(pairs)>=100 and min(blocks)>=8 and max_smd<=.20
            supports.append(dict(site=site,regime=regime,pairs=len(pairs),
                all_high=int((h[f'{regime}_group']==1).sum()),all_low=int((h[f'{regime}_group']==-1).sum()),
                high_blocks=blocks[0],low_blocks=blocks[1],max_abs_smd=max_smd,
                accepted=accepted,median_path_rmse=float(pairs.path_rmse.median())))
            matches.append(pairs.assign(site=site,regime=regime))
            balances.append(bal.assign(site=site,regime=regime))
    out.mkdir(parents=True)
    hourly.to_parquet(out/'hourly.parquet',index=False)
    physical.to_csv(out/'physical_context.csv',index=False)
    pd.concat(matches).to_csv(out/'matches.csv',index=False)
    pd.concat(balances).to_csv(out/'balance.csv',index=False)
    pd.DataFrame(supports).to_csv(out/'support.csv',index=False)
    np.savez_compressed(out/'dataset.npz',times_ns=times,path=path,outcome=outcome,
        **{f'{s}__{k}':v for s,f in fields.items() for k,v in f.items()})
    tracked=[BASE/'site_metadata.json',ROOT/'data/real/whereTurbins.csv',Path(__file__),
        BASE/'np005/dataset.npz',BASE/'np008/dataset.npz',BASE/'np008/test_predictions.npz',
        BASE/'np009/dataset.npz',BASE/'np009/test_predictions.npz',BASE/'np009/test_trajectory.npz']
    tracked += [ROOT/f'script/{name}.py' for name in ['next_paper_probability','next_paper_arrival_time']]
    protocol=dict(experiment='NP010',stage='EXPLORATORY_RETROSPECTIVE_MATCHED_CONTEXT',
        question='Does forecast quality differ between atmospheric backgrounds for similar realized power paths?',
        threshold_reference='Hourly4h windows from2024-07-01 with end<2024-10-20T16Z; no prediction losses',
        shear_thresholds=thresholds,heat_thresholds=[-20,20],sites=SITES,regimes=REGIMES,
        physical_definitions='Target4h mean vector shear10-100m(m/s); downward-positive sensible heat(W/m2)',
        matching='Exact process class/month; circular hour<=3; time distance>=24h; pathRMSE<=.075; start/net/range<=.10; greedy sorted distance no replacement',
        acceptance='>=100pairs;>=8seven-dayblocks each side;max absolute SMD<=.20 for17pathnodes/net/range/hour sin-cos',
        uncertainty='Common calendar-block weights on both sides; fixed matched membership;2000draws seed41;3/7/14day;exploratory pointwise intervals',
        forecast_use='Frozen NP008/009 only; no reanalysis inputs or refitting',
        caveats='Outcome matching defines a descriptive subpopulation; not causal, stability truth, forecast availability or independent validation',
        era5_source=source,sources={str(p.relative_to(ROOT)):digest(p) for p in tracked},
        artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file()},
        documentation={'ecmwf':'https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation',
                       'arco':'https://github.com/google-research/arco-era5'})
    write_json(out/'protocol.json',protocol)
    status(out,'PREPARED',windows=len(times),matched_conditions=len(supports),protocol_sha256=digest(out/'protocol.json'))
    print(pd.DataFrame(supports).to_string(index=False),flush=True)


def load_losses():
    dataset=np.load(BASE/'np008/dataset.npz'); take=dataset['test']
    occurrence=np.load(BASE/'np008/test_predictions.npz');losses={}
    timing=np.load(BASE/'np009/dataset.npz');tp=np.load(BASE/'np009/test_predictions.npz')
    trajectory=np.load(BASE/'np009/test_trajectory.npz')
    np.testing.assert_array_equal(dataset['times_ns'][take],timing['times_ns'][timing['test']])
    for j,lead in enumerate(LEADS):
        for arm in ['full','power','frequency']:
            p=occurrence[f'{lead}__{arm}__rolling']
            for metric,v in probability_losses(dataset['outcome'][take],p).items():
                if metric!='log_loss':losses[f'{lead}__occurrence__{arm}__{metric}']=v
        for e,event in enumerate(['up','down']):
            for arm in ['context','frequency','trajectory']:
                source=trajectory if arm=='trajectory' else tp
                v=timing_losses(timing['arrival'][e,timing['test']],source[f'{lead}__{event}__{arm}'],
                                 timing['occurrence'][e,j,timing['test']])
                for metric in ['conditional_rps','conditional_median_mae_minutes']:
                    losses[f'{lead}__{event}__{arm}__{metric}']=v[metric]
    # Negative increments indicate added information lowered loss.
    for key in list(losses):
        lead,event,arm,metric=key.split('__')
        if arm in ['full','context','trajectory']:
            refs=['power','frequency'] if arm=='full' else ['frequency']
            for ref in refs:
                losses[f'{lead}__{event}__{arm}-minus-{ref}__{metric}']=losses[key]-losses[f'{lead}__{event}__{ref}__{metric}']
    return losses


def contrast(times,high,low,values,days):
    """Bootstrap both matched sides with shared calendar-block multiplicities."""
    values=np.asarray(values);high=np.asarray(high,int);low=np.asarray(low,int)
    if values.ndim==1:values=values[:,None]
    blocks,inverse=np.unique(times//(days*DAY),return_inverse=True)
    weights=np.random.default_rng(41).multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=2000)
    means=[];resampled=[];counts=[]
    for rows in [high,low]:
        v=values[rows]; valid=np.isfinite(v)
        sums=np.zeros((len(blocks),v.shape[1]));count=np.zeros_like(sums)
        np.add.at(sums,inverse[rows],np.where(valid,v,0));np.add.at(count,inverse[rows],valid.astype(int))
        means.append(np.nanmean(v,axis=0));counts.append(valid.sum(axis=0))
        den=weights@count
        resampled.append(np.divide(weights@sums,den,out=np.full_like(den,np.nan),where=den>0))
    dif=resampled[0]-resampled[1];ci=np.nanquantile(dif,[.025,.975],axis=0)
    return means,counts,ci,np.isfinite(dif).sum(axis=0)


def evaluate(out):
    p=json.loads((out/'protocol.json').read_text())
    if json.loads((out/'status.json').read_text())['state']!='PREPARED':raise RuntimeError('Prepared fresh analysis required')
    for name,sha in p['sources'].items():assert digest(ROOT/name)==sha,name
    for name,sha in p['artifacts'].items():assert digest(out/name)==sha,name
    data=np.load(out/'dataset.npz');times=data['times_ns']; physical=pd.read_csv(out/'physical_context.csv')
    matches=pd.read_csv(out/'matches.csv');support=pd.read_csv(out/'support.csv')
    losses=load_losses();np.savez_compressed(out/'losses.npz',**losses)
    keys=list(losses);matrix=np.column_stack(list(losses.values()))
    rows=[];descriptive=[]
    for site in SITES:
        ctx=physical.loc[physical.site.eq(site)].reset_index(drop=True)
        for regime in REGIMES:
            pair=matches.loc[matches.site.eq(site)&matches.regime.eq(regime)]
            eligible=bool(support.loc[support.site.eq(site)&support.regime.eq(regime),'accepted'].iloc[0])
            a=pair.high.to_numpy(int);b=pair.low.to_numpy(int)
            for population,high,low in [('all',np.flatnonzero(ctx[f'{regime}_group']==1),np.flatnonzero(ctx[f'{regime}_group']==-1)),('matched',a,b)]:
                for sign,idx in [('high',high),('low',low)]:
                    for field in ['shear','heat','speed100','wind_delta','vector_delta','temperature_delta','pressure_delta_hpa','gfs_speed_rmse','jma_speed_rmse','gfs_delta_error','jma_delta_error']:
                        descriptive.append(dict(site=site,regime=regime,population=population,group=sign,field=field,windows=len(idx),mean=float(ctx.loc[idx,field].mean())))
                for days in ([3,7,14] if population=='matched' else [7]):
                    means,counts,ci,valid=contrast(times,high,low,matrix,days)
                    for k,key in enumerate(keys):
                        lead,event,arm,metric=key.split('__')
                        rows.append(dict(site=site,regime=regime,population=population,block_days=days,
                            lead_minutes=int(lead),event=event,arm=arm,metric=metric,
                            high_n=int(counts[0][k]),low_n=int(counts[1][k]),high_mean=means[0][k],low_mean=means[1][k],
                            high_minus_low=means[0][k]-means[1][k],ci_low=ci[0,k],ci_high=ci[1,k],
                            valid_replicates=int(valid[k]),matched_support_accepted=eligible))
    pd.DataFrame(rows).to_csv(out/'conditional_contrasts.csv',index=False)
    pd.DataFrame(descriptive).to_csv(out/'physical_descriptions.csv',index=False)
    write_json(out/'result_manifest.json',{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='status.json'})
    status(out,'COMPLETE',windows=len(times),contrasts=len(rows),new_forecast_fits=0)
    print(f'Complete: {len(rows)} contrasts; no new forecast fits',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','evaluate'],required=True)
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    with threadpool_limits(limits=1):{'prepare':prepare,'evaluate':evaluate}[args.phase](args.output)
