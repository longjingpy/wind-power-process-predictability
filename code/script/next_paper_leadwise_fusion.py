"""NP020: lead-specific fusion of frozen NP013 and NP014 probabilities."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from next_paper_probability import probability_losses,paired_loss_effect
from next_paper_trajectory import digest,write_json,status

ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'outputs/next_paper'; OUT=BASE/'np020'; LEADS=[15,720]; WEIGHTS=[0.,.25,.5,.75,1.]

def prepare(out):
    if out.exists():raise RuntimeError('Fresh output directory required')
    p=np.load(BASE/'np013/dataset.npz');out.mkdir(parents=True);np.savez_compressed(out/'dataset.npz',times_ns=p['times_ns'],outcome=p['outcome'],validation=p['validation'],test=p['test'])
    tracked=[Path(__file__),BASE/'np013/dataset.npz',BASE/'np013/protocol.json',BASE/'np013/verification.json',BASE/'np014/protocol.json',BASE/'np014/verification.json',BASE/'np019/protocol.json',BASE/'np019/verification.json']
    tracked += [BASE/f'{n}/{s}_predictions.npz' for n in ['np013','np014','np019'] for s in ['validation','test']]
    protocol=dict(experiment='NP020',stage='LEAD_SPECIFIC_FROZEN_PROBABILITY_FUSION',leads=LEADS,weights=WEIGHTS,
        candidates=['raw_np013','raw_np014'],selection='Each lead chooses a weight on validation subject to multiclass Brier no worse than the better raw source, then minimizes return Brier; no test selection',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in tracked},dataset_sha256=digest(out/'dataset.npz'))
    write_json(out/'protocol.json',protocol);status(out,'PREPARED')

def sources(out):
    p=json.loads((out/'protocol.json').read_text())
    for n,s in p['source_sha256'].items():assert digest(ROOT/n)==s,n
    assert digest(out/'dataset.npz')==p['dataset_sha256'];return p

def make(split):
    a=np.load(BASE/f'np013/{split}_predictions.npz');b=np.load(BASE/f'np014/{split}_predictions.npz');r={}
    for l in LEADS:r[f'{l}__raw_np013']=a[f'{l}__selected'];r[f'{l}__raw_np014']=b[f'{l}__selected']
    return r

def choose(pred,y):
    rows=[];chosen={}
    for l in LEADS:
        p13=pred[f'{l}__raw_np013'];p14=pred[f'{l}__raw_np014']; limit=min(probability_losses(y,p13)['multiclass_brier'].mean(),probability_losses(y,p14)['multiclass_brier'].mean())
        for w in WEIGHTS:
            p=(1-w)*p13+w*p14;z=probability_losses(y,p);rows.append(dict(lead=l,weight=w,multiclass_brier=z['multiclass_brier'].mean(),return_brier=z['return_brier'].mean(),eligible=z['multiclass_brier'].mean()<=limit+1e-12))
        group=pd.DataFrame([x for x in rows if x['lead']==l]);best=group[group.eligible].sort_values(['return_brier','weight']).iloc[0].to_dict();chosen[str(l)]=best
    return chosen,pd.DataFrame(rows)

def blend(pred,chosen):
    r=dict(pred)
    for l in LEADS:
        w=chosen[str(l)]['weight'];r[f'{l}__selected']=(1-w)*pred[f'{l}__raw_np013']+w*pred[f'{l}__raw_np014']
    return r

def run(out,phase):
    sources(out);p=make(phase);d=np.load(out/'dataset.npz');y=d['outcome'][d[phase]]
    if phase=='validation':
        chosen,table=choose(p,y);table.to_csv(out/'selection.csv',index=False);write_json(out/'freeze.json',dict(protocol_sha256=digest(out/'protocol.json'),dataset_sha256=digest(out/'dataset.npz'),selection=chosen,selection_sha256=digest(out/'selection.csv')));pred=blend(p,chosen);np.savez_compressed(out/'validation_predictions.npz',**pred);status(out,'VALIDATION_FROZEN',selection=chosen)
    else:
        f=json.loads((out/'freeze.json').read_text());assert digest(out/'selection.csv')==f['selection_sha256'];pred=blend(p,f['selection']);np.savez_compressed(out/'test_predictions.npz',**pred)
        clock=pd.to_datetime(d['times_ns'][d['test']],utc=True);loss={k:probability_losses(y,v) for k,v in pred.items()};rows=[]
        for l in LEADS:
            for ref in ['raw_np013','raw_np014']:
                for m in ['multiclass_brier','return_brier','up_brier','down_brier']:
                    for days in [3,7,14]:rows.append(dict(candidate=f'{l}__selected',reference=f'{l}__{ref}',metric=m,windows=len(y),**paired_loss_effect(clock,loss[f'{l}__selected'][m],loss[f'{l}__{ref}'][m],days)))
        pd.DataFrame(rows).to_csv(out/'paired_intervals.csv',index=False);status(out,'COMPLETE',selection=f['selection'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','validation','test'],required=True);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();{'prepare':prepare,'validation':lambda x:run(x,'validation'),'test':lambda x:run(x,'test')}[a.phase](a.output)
