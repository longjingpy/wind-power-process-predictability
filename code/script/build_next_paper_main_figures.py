"""Build unified manuscript figure drafts from frozen NP005/008/009/014/018/020 tables."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/next_paper/manuscript_figures'; OUT.mkdir(parents=True,exist_ok=True)
BLUE='#2166ac'; ORANGE='#b35806'; GREY='#666666'; GREEN='#1b7837'

def save(fig,name):
    fig.tight_layout(rect=[0,.04,1,.95])
    fig.savefig(OUT/f'{name}.png',dpi=220,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def fig2():
    d=pd.read_csv(ROOT/'outputs/next_paper/np005/paired_intervals.csv')
    leads=[15,60,120,240,480,720]; vals=[]
    for lead in leads:
        q=d[(d.candidate==f'power_weather:{lead}:joint')&(d.reference=='frequency:0:reference')&
            (d.metric=='return_brier')&(d.state=='all')&(d.block_days==7)].iloc[0]
        vals.append([q.relative_loss_reduction_pct,q.low,q.high])
    vals=np.asarray(vals)
    d8=pd.read_csv(ROOT/'outputs/next_paper/np008/paired_intervals.csv');weather=[]
    for lead in leads:
        q=d8[(d8.candidate==f'{lead}__weather__rolling')&(d8.reference==f'{lead}__frequency__rolling')&
             (d8.metric=='multiclass_brier')&(d8.state=='all')&(d8.block_days==7)].iloc[0]
        weather.append([q.relative_loss_reduction_pct,q.low,q.high])
    weather=np.asarray(weather)
    fig,ax=plt.subplots(figsize=(7.6,4.6))
    x=np.arange(6);ax.errorbar(x,vals[:,0],yerr=[vals[:,0]-vals[:,1],vals[:,2]-vals[:,0]],fmt='o-',color=BLUE,capsize=4,label='Power + weather: return-event Brier')
    ax.errorbar(x,weather[:,0],yerr=[weather[:,0]-weather[:,1],weather[:,2]-weather[:,0]],fmt='s--',color=ORANGE,capsize=4,label='Weather increment: four-class Brier')
    ax.axhline(0,color=GREY,lw=1,ls='--');ax.set_xticks(x,[str(v) for v in leads]);ax.set_xlabel('Issue lead (min)');ax.set_ylabel('Relative loss reduction (%)')
    ax.set_title('Predictability depends on process target and issue lead',loc='left');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
    save(fig,'fig2_lead_time_process_skill')

def fig3():
    d=pd.read_csv(ROOT/'outputs/next_paper/np009/paired_intervals.csv')
    leads=[15,60,120,240,480,720];metrics=[('conditional_rps','Conditional timing RPS reduction (%)'),('conditional_median_mae_minutes','Median arrival-time MAE reduction (%)')]
    fig,axes=plt.subplots(1,2,figsize=(9,4.2))
    for ax,(metric,label) in zip(axes,metrics):
        y=[]
        for lead in leads:
            q=d[(d.candidate==f'{lead}__down__context')&(d.reference==f'{lead}__down__frequency')&
                (d.metric==metric)&(d.state=='all')&(d.block_days==7)].iloc[0]
            y.append([q.relative_loss_reduction_pct,q.low,q.high])
        y=np.asarray(y);x=np.arange(6);ax.errorbar(x,y[:,0],yerr=[y[:,0]-y[:,1],y[:,2]-y[:,0]],fmt='o-',color=BLUE,capsize=4)
        ax.axhline(0,color=GREY,lw=1,ls='--');ax.set_xticks(x,[str(v) for v in leads]);ax.set_xlabel('Issue lead (min)');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Occurrence probability and first-passage timing separate',x=.05,ha='left',fontsize=13)
    save(fig,'fig3_occurrence_vs_arrival_time')

def fig4():
    d=pd.read_csv(ROOT/'outputs/next_paper/np014/wind_intervals.csv');s=pd.read_csv(ROOT/'outputs/next_paper/np014/test_wind_summary.csv')
    fig,axes=plt.subplots(1,3,figsize=(10.8,4.2));arms=['old_conditional','old_unconditional','coupled'];labels=['Joint residual','Unconditional errors','Direct range']
    for ax,metric,title in zip(axes,['range_crps','range_coverage80','range_width80'],['Range CRPS','80% coverage (%)','80% width (m/s)']):
        for lead,color,label in [(15,BLUE,'15 min'),(720,ORANGE,'12 h')]:
            vals=[float(s[(s.lead==lead)&(s.arm==a)&(s.metric==metric)].value.iloc[0]) for a in arms]
            if metric=='range_coverage80': vals=np.asarray(vals)*100
            ax.plot(range(3),vals,'o-',color=color,label=label)
        if metric=='range_coverage80':ax.axhline(80,color=GREY,lw=1,ls='--')
        ax.set_xticks(range(3),labels,fontsize=8);ax.set_title(title,loc='left');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8);fig.suptitle('Direct range prediction improves an upstream target',x=.05,ha='left',fontsize=13);save(fig,'fig4_range_target_alignment')

def fig5():
    d18=pd.read_csv(ROOT/'outputs/next_paper/np018/paired_intervals.csv')
    d20=pd.read_csv(ROOT/'outputs/next_paper/np020/paired_intervals.csv')
    d26=pd.read_csv(ROOT/'outputs/next_paper/np026/routing_summary.csv')
    rows=[]
    for label,df,lead,metric,cand,ref in [
        ('Target-aligned route',d26,240,'conditional_rps','target_aligned_route','joint_timing'),
        ('Target-aligned route',d26,480,'conditional_rps','target_aligned_route','joint_timing'),
        ('Summary vs direct',d18,720,'multiclass_brier','720__aug_np013__forest','720__direct__forest'),
        ('Summary vs direct',d18,720,'return_brier','720__aug_np013__forest','720__direct__forest'),
        ('Fusion vs NP014',d20,720,'multiclass_brier','720__selected','720__raw_np014'),
        ('Fusion vs NP014',d20,720,'return_brier','720__selected','720__raw_np014')]:
        subset=df[(df.candidate==cand)&(df.reference==ref)&(df.metric==metric)]
        if 'lead_minutes' in df.columns:
            subset=subset[pd.to_numeric(subset.lead_minutes,errors='coerce')==lead]
        if 'block_days' in df.columns:
            subset=subset[subset.block_days==7]
        q=subset.iloc[0]
        rows.append((label,lead,metric,q.relative_loss_reduction_pct,q.low,q.high))
    fig,ax=plt.subplots(figsize=(8.8,5.2)); y=np.arange(len(rows)); vals=np.array([r[3] for r in rows]); low=np.array([r[4] for r in rows]); high=np.array([r[5] for r in rows]); colors=['#762a83','#762a83',BLUE,ORANGE,GREEN,'#b2abd2']
    ax.errorbar(vals,y,xerr=[vals-low,high-vals],fmt='none',ecolor='#555',capsize=4,lw=1)
    ax.scatter(vals,y,c=colors,s=48,zorder=3)
    ax.axvline(0,color=GREY,lw=1,ls='--')
    labels=[]
    for a,lead,m,_,_,_ in rows:
        if m=='conditional_rps': pretty='Downward timing RPS'
        elif m=='multiclass_brier': pretty='Four-class Brier'
        else: pretty='Return-event Brier'
        labels.append(f'{a} ({lead} min)\n{pretty}')
    ax.set_yticks(y,labels);ax.set_xlabel('Relative loss reduction (%)');ax.set_title('Target-aligned information improves process-specific forecasts',loc='left');ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False);save(fig,'fig5_target_aligned_information')

if __name__=='__main__':
    fig2();fig3();fig4();fig5();print(OUT)
