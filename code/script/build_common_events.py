"""Build events persistent across all three detector thresholds."""
from pathlib import Path
import json,pandas as pd
R=Path(__file__).resolve().parents[1]; B=R/'data/event_clean'; ds={s:pd.read_csv(B/f'candidate_events{suffix}.csv.gz') for s,suffix in [('010','_threshold_0.10'),('020',''),('030','_threshold_0.30')]}
for d in ds.values(): d.t_start=pd.to_datetime(d.t_start,utc=True)
base=ds['020'].sort_values('t_start').copy(); report={}
for s in ('010','030'):
 q=ds[s].sort_values('t_start').assign(found=1); m=pd.merge_asof(base[['site','turbine_id','t_start']].sort_values('t_start'),q[['site','turbine_id','t_start','found']].sort_values('t_start'),on='t_start',by=['site','turbine_id'],tolerance=pd.Timedelta('2h'),direction='nearest'); base[f'found_{s}']=m.found.notna().to_numpy()
common=base[base.found_010&base.found_030].drop(columns=['found_010','found_030']); common.to_csv(B/'common_threshold_events.csv.gz',index=False,compression='gzip'); report={'total':len(common),'sites':common.groupby('site').size().to_dict(),'base_total':len(base)}; (B/'common_threshold_events.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
