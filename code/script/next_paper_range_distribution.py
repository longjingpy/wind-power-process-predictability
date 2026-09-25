"""NP014: forecast wind range directly and transfer it to frozen wind scenarios.

Meinshausen2006, JMLR7:983-999, Sections2-3 supplies the inherited leaf empirical
distribution idea. Schefzik, Thorarinsdottir & Gneiting2013, Uncertainty
Quantification in Complex Simulation Models Using Ensemble Copula Coupling,
doi:10.1214/13-STS443, Section4.1 eq(4.3), motivates rank assignment.
https://arxiv.org/pdf/1302.7149

Here rank assignment concerns a derived range, followed by an affine path
transformation and nonnegative projection. It is not standard multivariate
ECC and does not automatically retain its full dependence guarantees.
The physical constraint can lift the old mean; all lifts and fallbacks are
reported. Same-marginal shuffling tests coupling without changing range draws.
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
from next_paper_local_wind import (OUT as PARENT,BASE,ROOT,LEADS,PARAMS,DAY,MINUTE,
    inputs,train_rows,response,wind_metrics,WIND_METRICS)
from next_paper_probability import build_leaf_bank,draw_neighbors,probability_losses,paired_loss_effect
from next_paper_trajectory import digest,write_json,status

OUT=BASE/'np014'
ARMS=['coupled','shuffled','unconditional','point']
OLD_ARMS={'direct150':'direct_all','old_mixture':'mixture','old_selected':'selected',
          'frozen_np008':'frozen_np008','frozen_gfs':'frozen_np012_gfs'}
TRANSFORM=['flat_fraction','linear_fraction','lift_fraction','mean_lift','max_lift']
WEIGHTS=[0.,.25,.5,.75,1.]


def rank_assign(old_range,samples):
    if old_range.shape!=samples.shape:raise ValueError('Same member shape required')
    order=np.argsort(old_range,axis=1,kind='stable')
    ranks=np.argsort(order,axis=1,kind='stable')
    return np.take_along_axis(np.sort(samples,axis=1),ranks,axis=1)


def reshape_range(paths,desired,donors):
    """Preserve range exactly; preserve mean when compatible with nonnegative speed."""
    if paths.shape!=donors.shape or paths.shape[:2]!=desired.shape or paths.shape[-1]!=17:
        raise ValueError('Aligned17-node paths, donor paths and range samples required')
    if not np.isfinite(desired).all() or np.any(desired<0):raise ValueError('Finite nonnegative ranges required')
    old_range=np.ptp(paths,axis=2);flat=old_range<=1e-12
    template=np.where(flat[:,:,None],donors,paths)
    spread=np.ptp(template,axis=2);linear=spread<=1e-12
    template=np.where(linear[:,:,None],np.linspace(0,1,17)[None,None,:],template)
    spread=np.ptp(template,axis=2)
    z=(template-template.min(axis=2,keepdims=True))/spread[:,:,None]
    old_mean=paths.mean(axis=2);lower=np.maximum(old_mean-desired*z.mean(axis=2),0)
    transformed=lower[:,:,None]+desired[:,:,None]*z
    lift=transformed.mean(axis=2)-old_mean
    stats={'flat_fraction':flat.mean(axis=1),'linear_fraction':linear.mean(axis=1),
        'lift_fraction':(lift>1e-12).mean(axis=1),'mean_lift':lift.mean(axis=1),'max_lift':lift.max(axis=1)}
    return transformed,stats


def prepare(out):
    if out.exists():raise RuntimeError('Fresh output directory required')
    assert json.loads((PARENT/'verification.json').read_text())['status']=='PASS'
    tracked=[Path(__file__),PARENT/'dataset.npz',PARENT/'protocol.json',PARENT/'verification.json']
    tracked += [PARENT/f'{s}_{name}' for s in ['validation','test'] for name in ['predictions.npz','neighbors.npz','wind_scores.npz','models.json','audit.csv']]
    tracked += [ROOT/f'script/{n}.py' for n in ['next_paper_local_wind','next_paper_probability','next_paper_causal_calibration']]
    data=np.load(PARENT/'dataset.npz');out.mkdir(parents=True);(out/'models').mkdir()
    pd.DataFrame([dict(split=s,forecast_windows=int(data[s].sum()),wind_windows=int((data[s]&data['wind_valid']).sum())) for s in ['validation','test']]).to_csv(out/'support.csv',index=False)
    p=dict(experiment='NP014',stage='DIRECT_RANGE_DISTRIBUTION_AND_SCENARIO_CONSTRAINT',
        input_dataset=str((PARENT/'dataset.npz').relative_to(ROOT)),leads=LEADS,arms=ARMS,
        range_parameters=PARAMS,direct_parameters={**PARAMS,'n_estimators':450},
        inherited='NP013issue-time inputs/training clocks and fixed150-member wind paths/response; no parent refit',
        target='Actual native17-point onsite140m range/5; native ranges sampled from prediction leaves',
        clock='Same56day UTC Monday update;T+4h+1min matured;full forecast cohort',
        transformation='Rank-assigned range; preserve normalized temporal shape;mean-preserving shift if nonnegative,otherwise lift mean;flat paths use historical donor or linear fallback',
        controls='shuffled same conditional range multiset;unconditional native range draws;point conditional range mean',
        selection='Min mean validation returnBrier among candidates with mean multiclass<=frozenGFS; ties lower tree count',
        candidates='five coupled/direct150 weights plus direct450 and frozenGFS;controls not selectable',weights=WEIGHTS,
        tree_budget='coupled300trees; interior-weight blend450; direct450 counts total trees, not exact compute/parameter equality',
        inference='3/7/14day paired calendar blocks;2000draws seed41;exploratory fixed-model intervals',
        rank_source='Schefzik et al2013 doi:10.1214/13-STS443 Section4.1 eq4.3; aggregate-range adaptation, not full ECC equivalence',
        source_sha256={str(f.relative_to(ROOT)):digest(f) for f in tracked})
    write_json(out/'protocol.json',p);status(out,'PREPARED')


def sources(out):
    p=json.loads((out/'protocol.json').read_text())
    for name,sha in p['source_sha256'].items():assert digest(ROOT/name)==sha,name
    return p


def fit_pack(data,x,update):
    rows=train_rows(data,update,True);all_rows=train_rows(data,update)
    ranges=np.ptp(data['wind'][rows],axis=1)
    reg=ExtraTreesRegressor(**PARAMS).fit(x[rows],ranges/5)
    bank,err=build_leaf_bank([(41,reg,None)],x[rows],ranges[:,None]/5)
    direct=ExtraTreesClassifier(**{**PARAMS,'n_estimators':450}).fit(x[all_rows],data['outcome'][all_rows])
    return dict(range_model=reg,bank=bank,range_values=ranges,wind_rows=rows,all_rows=all_rows,
        direct450=direct,update_ns=int(update),leaf_error=err)


def predict_ranges(pack,x,base_paths,seed):
    local=draw_neighbors([(41,pack['range_model'],None)],pack['bank'],x,seed)
    uniform=np.random.default_rng(seed+1).integers(0,len(pack['wind_rows']),size=local.shape,dtype=np.int32)
    conditional=pack['range_values'][local];unconditional=pack['range_values'][uniform]
    old_range=np.ptp(base_paths,axis=2)
    permutation=np.argsort(np.random.default_rng(seed+2).random(local.shape),axis=1)
    desired={'coupled':rank_assign(old_range,conditional),
        'shuffled':np.take_along_axis(conditional,permutation,axis=1),
        'unconditional':rank_assign(old_range,unconditional),
        'point':np.broadcast_to(conditional.mean(axis=1,keepdims=True),conditional.shape)}
    return desired,local,uniform


def run_phase(out,data,split):
    target=np.flatnonzero(data[split]);n=len(target)
    parent_pred=np.load(PARENT/f'{split}_predictions.npz');parent_neighbors=np.load(PARENT/f'{split}_neighbors.npz')
    parent_audit=pd.read_csv(PARENT/f'{split}_audit.csv');parent_hashes=json.loads((PARENT/f'{split}_models.json').read_text())
    pred={f'{l}__{a}':parent_pred[f'{l}__{b}'] for l in LEADS for a,b in OLD_ARMS.items()}
    ids={};wind_scores={};transforms={};audit=[];hashes={}
    for lead in LEADS:
        x,known=inputs(data,lead)
        for arm in [*ARMS,'direct450']:pred[f'{lead}__{arm}']=np.empty((n,4))
        for arm in ['conditional','unconditional']:ids[f'{lead}__{arm}']=np.empty((n,150),np.int32)
        for arm in ARMS:
            for metric in WIND_METRICS:wind_scores[f'{lead}__{arm}__{metric}']=np.full(n,np.nan)
            for name in TRANSFORM:transforms[f'{lead}__{arm}__{name}']=np.empty(n)
        for rec in parent_audit.loc[parent_audit.lead==lead].itertuples():
            update=int(rec.update_ns)
            assert digest(PARENT/rec.path)==parent_hashes[rec.path]
            parent_pack=joblib.load(PARENT/rec.path);pack=fit_pack(data,x,update)
            np.testing.assert_array_equal(pack['wind_rows'],parent_pack['wind_rows'])
            np.testing.assert_array_equal(pack['all_rows'],parent_pack['all_rows'])
            issue=pd.to_datetime(data['times_ns'][target]-lead*MINUTE,utc=True)
            schedule=(issue.normalize()-pd.to_timedelta(issue.dayofweek,unit='D')).asi8
            take=np.flatnonzero(schedule==update)
            seed=41+int(update//DAY)+lead+(300000 if split=='test' else 200000)
            for start in range(0,len(take),128):
                loc=take[start:start+128];rows=target[loc]
                old_ids=parent_neighbors[f'{lead}__conditional'][loc]
                donors=data['wind'][old_ids]
                base_paths=np.maximum(data['gfs'][rows,None,:]+(donors-data['gfs'][old_ids]),0)
                desired,local,uniform=predict_ranges(pack,x[rows],base_paths,seed+start)
                ids[f'{lead}__conditional'][loc]=pack['wind_rows'][local]
                ids[f'{lead}__unconditional'][loc]=pack['wind_rows'][uniform]
                pred[f'{lead}__direct450'][loc]=pack['direct450'].predict_proba(x[rows])
                good=data['wind_valid'][rows]
                for arm in ARMS:
                    paths,stats=reshape_range(base_paths,desired[arm],donors)
                    pred[f'{lead}__{arm}'][loc]=response(parent_pack['response'],known[rows],paths).mean(axis=1)
                    for name,value in stats.items():transforms[f'{lead}__{arm}__{name}'][loc]=value
                    if good.any():
                        for metric,value in wind_metrics(paths[good],data['wind'][rows[good]]).items():
                            wind_scores[f'{lead}__{arm}__{metric}'][loc[good]]=value
            name=f'models/{split}__{lead}__{update}.joblib';joblib.dump(pack,out/name,compress=3);hashes[name]=digest(out/name)
            audit.append(dict(lead=lead,update_ns=update,path=name,parent_path=rec.path,wind_rows=len(pack['wind_rows']),all_rows=len(pack['all_rows']),seed=seed,predictions=len(take)))
            print(datetime.now(timezone.utc).isoformat(),split,lead,str(pd.Timestamp(update,tz='UTC')),flush=True)
    np.savez_compressed(out/f'{split}_predictions.npz',**pred)
    np.savez_compressed(out/f'{split}_range_neighbors.npz',**ids)
    np.savez_compressed(out/f'{split}_wind_scores.npz',**wind_scores)
    np.savez_compressed(out/f'{split}_transforms.npz',**transforms)
    pd.DataFrame(audit).to_csv(out/f'{split}_audit.csv',index=False);write_json(out/f'{split}_models.json',hashes)
    return pred


def candidates():
    result=[dict(name=f'blend_{w:g}',kind='blend',weight=w,trees=150 if w==0 else (300 if w==1 else 450)) for w in WEIGHTS]
    return result+[dict(name='direct450',kind='direct450',weight=None,trees=450),dict(name='frozen_gfs',kind='frozen_gfs',weight=None,trees=150)]


def candidate_probability(pred,lead,choice):
    if choice['kind']=='blend':return (1-choice['weight'])*pred[f'{lead}__direct150']+choice['weight']*pred[f'{lead}__coupled']
    return pred[f"{lead}__{choice['kind']}"]


def select_candidate(y,pred):
    table=[]
    limit=np.mean([probability_losses(y,pred[f'{l}__frozen_gfs'])['multiclass_brier'].mean() for l in LEADS])
    for choice in candidates():
        losses=[probability_losses(y,candidate_probability(pred,l,choice)) for l in LEADS]
        row={**choice,'return_brier':float(np.mean([v['return_brier'].mean() for v in losses])),
             'multiclass_brier':float(np.mean([v['multiclass_brier'].mean() for v in losses]))}
        row['eligible']=bool(row['multiclass_brier']<=limit+1e-12);table.append(row)
    selected=min([r for r in table if r['eligible']],key=lambda r:(r['return_brier'],r['trees'],r['name']))
    return selected,table


def evaluate(out,data,split,pred,selected):
    take=data[split];times=pd.to_datetime(data['times_ns'][take],utc=True);y=data['outcome'][take]
    for lead in LEADS:pred[f'{lead}__selected']=candidate_probability(pred,lead,selected)
    np.savez_compressed(out/f'{split}_predictions.npz',**pred)
    losses={key:probability_losses(y,v) for key,v in pred.items()}
    summary=[dict(model=key,metric=m,windows=len(y),loss=float(v.mean())) for key,metrics in losses.items() for m,v in metrics.items()]
    pd.DataFrame(summary).to_csv(out/f'{split}_summary.csv',index=False)
    wind=dict(np.load(out/f'{split}_wind_scores.npz'));old=np.load(PARENT/f'{split}_wind_scores.npz')
    for lead in LEADS:
        for arm in ['conditional','unconditional']:
            for metric in WIND_METRICS:wind[f'{lead}__old_{arm}__{metric}']=old[f'{lead}__{arm}__{metric}']
    ws=[]
    for key,v in wind.items():
        lead,arm,metric=key.split('__');good=np.isfinite(v)
        ws.append(dict(lead=int(lead),arm=arm,metric=metric,windows=int(good.sum()),value=float(v[good].mean())))
    pd.DataFrame(ws).to_csv(out/f'{split}_wind_summary.csv',index=False)
    trans=np.load(out/f'{split}_transforms.npz');ts=[]
    for key,v in trans.items():
        lead,arm,metric=key.split('__');ts.append(dict(lead=int(lead),arm=arm,metric=metric,mean=float(v.mean()),maximum=float(v.max())))
    pd.DataFrame(ts).to_csv(out/f'{split}_transform_summary.csv',index=False)
    if split=='test':
        comparisons=[('coupled',b) for b in ['old_mixture','shuffled','unconditional','point','direct450','frozen_gfs']]
        comparisons += [('selected',b) for b in ['direct450','frozen_gfs','old_selected','direct150']]
        comparisons += [('direct450',b) for b in ['direct150','frozen_gfs']]
        effects=[];weffects=[]
        for lead in LEADS:
            for a,b in comparisons:
                for metric in ['multiclass_brier','return_brier','up_brier','down_brier']:
                    for days in [3,7,14]:effects.append(dict(candidate=f'{lead}__{a}',reference=f'{lead}__{b}',metric=metric,windows=len(y),
                        **paired_loss_effect(times,losses[f'{lead}__{a}'][metric],losses[f'{lead}__{b}'][metric],days)))
            for ref in ['old_conditional','old_unconditional','unconditional','shuffled','point']:
                for metric in ['profile_crps','mean_crps','range_crps','net_crps']:
                    a,b=wind[f'{lead}__coupled__{metric}'],wind[f'{lead}__{ref}__{metric}'];good=np.isfinite(a)&np.isfinite(b)
                    for days in [3,7,14]:weffects.append(dict(lead=lead,reference=ref,metric=metric,windows=int(good.sum()),
                        **paired_loss_effect(times[good],a[good],b[good],days)))
        pd.DataFrame(effects).to_csv(out/'paired_intervals.csv',index=False)
        pd.DataFrame(weffects).to_csv(out/'wind_intervals.csv',index=False)


def validate(out):
    sources(out)
    if json.loads((out/'status.json').read_text())['state']!='PREPARED':raise RuntimeError('Prepared run required')
    data=dict(np.load(PARENT/'dataset.npz'));status(out,'VALIDATING');pred=run_phase(out,data,'validation')
    selected,table=select_candidate(data['outcome'][data['validation']],pred)
    evaluate(out,data,'validation',pred,selected);pd.DataFrame(table).to_csv(out/'selection.csv',index=False)
    write_json(out/'freeze.json',dict(selected=selected,protocol_sha256=digest(out/'protocol.json'),selection_sha256=digest(out/'selection.csv'),
        validation_sha256={f.name:digest(f) for f in out.glob('validation_*') if f.is_file()}))
    status(out,'VALIDATION_FROZEN',selected=selected);print('SELECTED',selected,flush=True)


def test(out):
    sources(out);freeze=json.loads((out/'freeze.json').read_text())
    assert digest(out/'protocol.json')==freeze['protocol_sha256'] and digest(out/'selection.csv')==freeze['selection_sha256']
    for name,sha in freeze['validation_sha256'].items():assert digest(out/name)==sha
    if json.loads((out/'status.json').read_text())['state']!='VALIDATION_FROZEN':raise RuntimeError('Frozen run required')
    data=dict(np.load(PARENT/'dataset.npz'));status(out,'TESTING');pred=run_phase(out,data,'test')
    evaluate(out,data,'test',pred,freeze['selected']);status(out,'COMPLETE',selected=freeze['selected'],test_windows=int(data['test'].sum()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','validate','test'],required=True)
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    with threadpool_limits(limits=1):{'prepare':prepare,'validate':validate,'test':test}[args.phase](args.output)
