"""Exact paired calendar-block scores, including efficient weighted AUROC.

Kunsch (1989), doi:10.1214/aos/1176347265, dependent-data resampling motivates
shared temporal blocks. The empirical AUC is its weighted positive-negative
concordance statistic, with half weight for ties. Preaggregating concordance
by block pair makes repeated AUC evaluation exact without resampling events
independently or sorting the full event vector thousands of times.
"""
import numpy as np
import pandas as pd


def block_design(times, days=7, repetitions=2000, seed=41):
    ns=pd.DatetimeIndex(pd.to_datetime(times,utc=True)).as_unit('ns').asi8
    _,index=np.unique(ns//pd.Timedelta(days=days).value,return_inverse=True)
    blocks=int(index.max()+1)
    weights=np.random.default_rng(seed).multinomial(blocks,np.full(blocks,1/blocks),size=repetitions)
    return index,weights.astype(float)


def auc_draws(y, score, block, weights):
    y=np.asarray(y,dtype=int);score=np.asarray(score,float);block=np.asarray(block,int)
    nblocks=weights.shape[1]
    _,group=np.unique(score,return_inverse=True)
    pos=np.zeros((group.max()+1,nblocks));neg=np.zeros_like(pos)
    np.add.at(pos,(group[y==1],block[y==1]),1)
    np.add.at(neg,(group[y==0],block[y==0]),1)
    kernel=pos.T@(np.cumsum(neg,axis=0)-.5*neg)
    numerator=np.einsum('ij,ij->i',weights@kernel,weights)
    denominator=(weights@pos.sum(axis=0))*(weights@neg.sum(axis=0))
    return np.divide(numerator,denominator,out=np.full(len(weights),np.nan),where=denominator>0)


def probability_scores(y,p,times,days=7,repetitions=2000):
    y=np.asarray(y,int);p=np.asarray(p,float)
    if p.shape!=(len(y),3) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Aligned finite three-class probabilities required')
    index,weights=block_design(times,days,repetitions)
    both=np.vstack([np.ones((1,weights.shape[1])),weights])
    counts=np.bincount(index,minlength=both.shape[1])
    loss=-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))
    brier=((p-np.eye(3)[y])**2).sum(axis=1)
    scores={}
    for name,value in [('log_loss',loss),('brier',brier)]:
        scores[name]=(both@np.bincount(index,weights=value,minlength=both.shape[1]))/(both@counts)
    transition=y!=1
    direction=p[transition,2]/np.maximum(p[transition,0]+p[transition,2],1e-12)
    scores['direction_auroc']=auc_draws(y[transition]==2,direction,index[transition],both)
    return scores,{'events':len(y),'direction_events':int(transition.sum()),'occupied_blocks':weights.shape[1]}


def paired_rows(frame,probabilities,comparisons,days=7,repetitions=2000):
    scores={};summary=[];effects=[]
    for name,p in probabilities.items():
        result,support=probability_scores(frame.outcome,p,frame.time_start,days,repetitions)
        scores[name]=result
        for metric,values in result.items():
            lo,hi=np.nanquantile(values[1:],[.025,.975])
            summary.append({'model':name,'metric':metric,'estimate':values[0],'low':lo,'high':hi,
                'block_days':days,'valid_draws':int(np.isfinite(values[1:]).sum()),**support})
    for candidate,reference in comparisons:
        for metric in scores[candidate]:
            sign=1 if metric=='direction_auroc' else -1
            delta=sign*(scores[candidate][metric]-scores[reference][metric])
            lo,hi=np.nanquantile(delta[1:],[.025,.975])
            effects.append({'candidate':candidate,'reference':reference,'metric':metric,'gain':delta[0],
                'low':lo,'high':hi,'block_days':days,'valid_draws':int(np.isfinite(delta[1:]).sum()),**support})
    return summary,effects
