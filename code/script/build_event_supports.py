"""Build disjoint event-support components before dependence-aware evaluation."""
from pathlib import Path
import pandas as pd
import json
B=Path(__file__).resolve().parents[1]/'outputs/dynamic_events_v3'
d=pd.read_csv(B/'windows.csv.gz',dtype={'turbine':str})
d['time']=pd.to_datetime(d.time,utc=True)
e=d[d.methods.ne('random_control')].copy(); rows=[]
for (site,tid,split),g in e.groupby(['site','turbine','split']):
    g=g.sort_values('time'); groups=g.time.diff().gt(pd.Timedelta('4h')).cumsum()
    for _,part in g.groupby(groups):
        rows.append({'site':site,'turbine':tid,'split':split,'start':str(part.time.min()-pd.Timedelta('2h')),'end':str(part.time.max()+pd.Timedelta('2h')),'anchor_count':len(part),'methods':';'.join(sorted(set(';'.join(part.methods).split(';'))))})
f=pd.DataFrame(rows); f.to_csv(B/'disjoint_event_supports.csv.gz',index=False)
summary={'anchor_rows':len(e),'disjoint_support_components':len(f),'multi_anchor_components':int(f.anchor_count.gt(1).sum()),'per_site':f.groupby('site').size().to_dict(),'meaning':'Connected overlapping four-hour window supports, not inferred physical onset/recovery; models still use original anchors, so component-aware evaluation remains required.'}
(B/'event_support_report.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

