"""NP019: validation-frozen fusion of rolling summary models and raw probabilities."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from next_paper_probability import probability_losses,paired_loss_effect
from next_paper_trajectory import digest,write_json,status

ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'outputs/next_paper'; OUT=BASE/'np019'; LEADS=[15,720]
WEIGHTS=[0.,.25,.5,.75,1.]

def prepare(out):
    if out.exists():raise RuntimeError('Fresh output directory required')
    out.mkdir(parents=True,exist_ok=True)
    p13=np.load(BASE/'np013/dataset.npz'); np.savez_compressed(out/'dataset.npz',times_ns=p13['times_ns'],outcome=p13['outcome'],validation=p13['validation'],test=p13['test'])
    tracked=[Path(__file__),BASE/'np013/dataset.npz',BASE/'np013/protocol.json',BASE/'np013/verification.json',BASE/'np018/protocol.json',BASE/'np018/verification.json']
    tracked += [BASE/f'np013/{s}_predictions.npz' for s in ['validation','test']]
    tracked += [BASE/f'np014/{s}_predictions.npz' for s in ['validation','test']]
    tracked += [BASE/f'np018/{s}_predictions.npz' for s in ['validation','test']]
    protocol=dict(experiment='NP019',stage='ROLLING_SUMMARY_PROBABILITY_FUSION',leads=LEADS,weights=WEIGHTS,
        candidates=['raw_np013','raw_np014','aug_np013__forest','aug_np014__forest','context_np013__logistic','context_np014__logistic'],
        selection='One candidate and one global weight selected on validation: multiclass Brier no worse than the better of raw_np013/raw_np014, then minimum mean return Brier; ties lower weight',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in tracked},dataset_sha256=digest(out/'dataset.npz'))
    write_json(out/'protocol.json',protocol);status(out,'PREPARED')

def sources(out):
    p=json.loads((out/'protocol.json').read_text())
    for n,s in p['source_sha256'].items():assert digest(ROOT/n)==s,n
    assert digest(out/'dataset.npz')==p['dataset_sha256'];return p

def make(split):
    p13=np.load(BASE/f'np013/{split}_predictions.npz');p14=np.load(BASE/f'np014/{split}_predictions.npz');p18=np.load(BASE/f'np018/{split}_predictions.npz')
    result={}
    for lead in LEADS:
        result[f'{lead}__raw_np013']=p13[f'{lead}__selected']; result[f'{lead}__raw_np014']=p14[f'{lead}__selected']
        for name in ['aug_np013__forest','aug_np014__forest','context_np013__logistic','context_np014__logistic']:
            result[f'{lead}__{name}']=p18[f'{lead}__{name}']
    return result

def evaluate(out,split,pred):
    d=np.load(out/'dataset.npz');mask=d[split];y=d['outcome'][mask];rows=[]
    for k,v in pred.items():
        for m,x in probability_losses(y,v).items():rows.append(dict(model=k,metric=m,loss=float(x.mean()),windows=len(y)))
    pd.DataFrame(rows).to_csv(out/f'{split}_summary.csv',index=False)

def select(validation):
    d=np.load(OUT/'dataset.npz');y=d['outcome'][d['validation']];items=[];base=[]
    for lead in LEADS:base.append(probability_losses(y,validation[f'{lead}__raw_np013']))
    limit=min(np.mean([probability_losses(y,validation[f'{l}__{source}'])['multiclass_brier'].mean() for l in LEADS])
              for source in ['raw_np013','raw_np014'])
    for source in ['raw_np013','raw_np014','aug_np013__forest','aug_np014__forest','context_np013__logistic','context_np014__logistic']:
        for w in WEIGHTS:
            values=[]
            for lead in LEADS:
                raw=validation[f'{lead}__raw_np013']; candidate=validation[f'{lead}__{source}']; p=(1-w)*raw+w*candidate
                values.append(probability_losses(y,p))
            items.append(dict(source=source,weight=w,multiclass_brier=float(np.mean([x['multiclass_brier'].mean() for x in values])),return_brier=float(np.mean([x['return_brier'].mean() for x in values])),eligible=np.mean([x['multiclass_brier'].mean() for x in values])<=limit+1e-12))
    table=pd.DataFrame(items);chosen=table.loc[table.eligible].sort_values(['return_brier','weight','source']).iloc[0].to_dict();return chosen,table

def blend(pred,choice):
    out=dict(pred)
    for lead in LEADS:out[f'{lead}__selected']=(1-choice['weight'])*pred[f'{lead}__raw_np013']+choice['weight']*pred[f'{lead}__{choice["source"]}']
    return out

def run(out,phase):
    sources(out);pred=make(phase)
    if phase=='validation':
        chosen,table=select(pred);table.to_csv(out/'selection.csv',index=False);write_json(out/'freeze.json',dict(protocol_sha256=digest(out/'protocol.json'),dataset_sha256=digest(out/'dataset.npz'),selection=chosen,selection_sha256=digest(out/'selection.csv')))
        pred=blend(pred,chosen);np.savez_compressed(out/'validation_predictions.npz',**pred);evaluate(out,phase,pred);status(out,'VALIDATION_FROZEN',selection=chosen)
    else:
        f=json.loads((out/'freeze.json').read_text());assert digest(out/'protocol.json')==f['protocol_sha256'] and digest(out/'selection.csv')==f['selection_sha256'];pred=blend(pred,f['selection']);np.savez_compressed(out/'test_predictions.npz',**pred);evaluate(out,phase,pred)
        d=np.load(out/'dataset.npz');y=d['outcome'][d['test']];clock=pd.to_datetime(d['times_ns'][d['test']],utc=True);losses={k:probability_losses(y,v) for k,v in pred.items()};rows=[]
        for lead in LEADS:
            for ref in ['raw_np013','raw_np014','aug_np013__forest','aug_np014__forest']:
                for m in ['multiclass_brier','return_brier','up_brier','down_brier']:
                    for days in [3,7,14]:rows.append(dict(candidate=f'{lead}__selected',reference=f'{lead}__{ref}',metric=m,windows=len(y),**paired_loss_effect(clock,losses[f'{lead}__selected'][m],losses[f'{lead}__{ref}'][m],days)))
        pd.DataFrame(rows).to_csv(out/'paired_intervals.csv',index=False);status(out,'COMPLETE',selection=f['selection'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','validation','test'],required=True);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    {'prepare':prepare,'validation':lambda x:run(x,'validation'),'test':lambda x:run(x,'test')}[a.phase](a.output)
