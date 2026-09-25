"""Build event-aligned lifecycle vectors: pre, transition, post."""
from pathlib import Path
import numpy as np,pandas as pd,sys
R=Path(__file__).resolve().parents[1]; B=R/'data/event_clean'; event_file=sys.argv[1] if len(sys.argv)>1 else 'candidate_events.csv.gz'; ev=pd.read_csv(B/event_file); ev.t_start=pd.to_datetime(ev.t_start,utc=True); ev.t_end=pd.to_datetime(ev.t_end,utc=True); rows=[]
for site in ('pizhou','suining','yandun','lahaute'):
 d=pd.read_csv(B/f'site={site}.csv.gz'); d.timestamp=pd.to_datetime(d.timestamp,utc=True)
 for tid,eg in ev[ev.site.eq(site)].groupby('turbine_id'):
  g=d[d.turbine_id.astype(str).eq(str(tid))].sort_values('timestamp').set_index('timestamp'); valid=g.usable_power & ~g.missing_observation; scale=g.loc[valid,'power_kw'].quantile(.995) if valid.any() else np.nan
  if not np.isfinite(scale) or scale<=0: continue
  z=(g.power_kw/scale).astype(float)
  for _,r in eg.iterrows():
   pre=pd.date_range(r.t_start-pd.Timedelta('1h'),r.t_start-pd.Timedelta('15min'),freq='15min',tz='UTC'); post=pd.date_range(r.t_end+pd.Timedelta('15min'),r.t_end+pd.Timedelta('1h'),freq='15min',tz='UTC'); mid=pd.date_range(r.t_start,r.t_end,freq='15min',tz='UTC')
   a=z.reindex(pre).to_numpy(); b=z.reindex(mid).to_numpy(); c=z.reindex(post).to_numpy()
   if len(b)<2 or not np.isfinite(np.r_[a,b,c]).all(): continue
   tm=np.linspace(0,1,9); bm=np.interp(tm,np.linspace(0,1,len(b)),b); w=np.r_[a,bm,c]; w=(w-w[0])/(np.max(np.abs(w-w[0])) or 1)
   rows.append({'site':site,'turbine_id':str(tid),'t_start':r.t_start,'t_end':r.t_end,'amplitude_norm':r.amplitude_norm,'duration_min':r.duration_min,'shape_vector':' '.join(f'{x:.6f}' for x in w),'shape_total_variation':float(np.abs(np.diff(w)).sum()),'shape_reversal_count':int(np.sum(np.diff(np.sign(np.diff(w)))!=0))})
out=pd.DataFrame(rows); name='lifecycle_shape_features_common.csv.gz' if event_file.startswith('common_') else 'lifecycle_shape_features.csv.gz'; out.to_csv(B/name,index=False,compression='gzip'); print({'rows':len(out),'sites':out.groupby('site').size().to_dict(),'output':name})
