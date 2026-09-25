"""Build a blinded, stratified, offline multirater packet from real SCADA."""
from pathlib import Path
import json,hashlib,zipfile,re,subprocess
import numpy as np
import pandas as pd
from prepare_annotation_v9 import load_sources, prepare_series
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"annotation_multirater_v15"
QUOTAS={"steady_screen":20,"rising_screen":5,"falling_screen":5,"turning_screen":10,"random_screen":10}
SEED=20260915

def candidates(frame,scale,rng):
    q=frame.power.resample("30min").mean()/scale
    center=q.rolling(5,center=True,min_periods=5)
    spread=center.max()-center.min()
    net=q.shift(-2)-q.shift(2)
    masks={
      "steady_screen":(spread<=.04)&(center.mean()>.1),
      "rising_screen":net>=.20,
      "falling_screen":net<=-.20,
      "turning_screen":(spread>=.20)&(net.abs()<=.35*spread),
      "random_screen":q.notna(),
    }
    lower=frame.index[0]+.6*(frame.index[-1]-frame.index[0])
    allowed=(q.index>lower+pd.Timedelta(hours=4))&(q.index<frame.index[-1]-pd.Timedelta(hours=3))
    result=[]
    for stratum,mask in masks.items():
        indices=np.flatnonzero(mask.fillna(False).to_numpy()&allowed)
        for i in rng.permutation(indices)[:100]:
            result.append((stratum,q.index[i]))
    return result

def main():
    rng=np.random.default_rng(SEED); pool=[]; provenance=[]
    for site,turbine,path,raw,clock,units in load_sources():
        frame=prepare_series(raw)
        cutoff=frame.index[0]+.6*(frame.index[-1]-frame.index[0])
        scale=float(frame.loc[frame.index<cutoff,"power"].quantile(.995))
        if not np.isfinite(scale) or scale<=0:continue
        nominal=float(frame.index.to_series().diff().dt.total_seconds().median()/60)
        for stratum,anchor in candidates(frame,scale,rng):
            w=frame.loc[anchor-pd.Timedelta(hours=2):anchor+pd.Timedelta(hours=2)].copy()
            relative=(w.index-anchor).total_seconds()/60
            focus=(relative>=-60)&(relative<=60)
            gaps=w.index.to_series().diff().dt.total_seconds().dropna()
            if len(w)<9 or focus.sum()<5 or not w.power.notna().all():continue
            if relative.min()>-120 or relative.max()<120 or (gaps>nominal*90).any():continue
            values=w.power.to_numpy(float)/scale
            if not np.isfinite(values).all() or (values<-.05).any() or (values>1.5).any():continue
            channels={}
            for key in ["wind","rpm","pitch"]:
                if key in w:
                    channels[key]=[float(v) if np.isfinite(v) else None for v in w[key].to_numpy(float)]
            pool.append({"site":site,"turbine":turbine,"stratum":stratum,"source":str(path.relative_to(ROOT)),
                         "source_clock":clock,"anchor":anchor,"scale":scale,"minutes":relative.tolist(),
                         "power":values.tolist(),"channels":channels,"nominal_minutes":nominal})
        provenance.append({"site":site,"turbine":turbine,"source":str(path.relative_to(ROOT))})
    selected=[]; used={}
    for site in ["pizhou","yandun","greece","sdwpf"]:
        for stratum,target in QUOTAS.items():
            options=[item for item in pool if item["site"]==site and item["stratum"]==stratum]
            count=0
            for i in rng.permutation(len(options)):
                item=options[i]; key=(site,item["turbine"])
                if any(abs((item["anchor"]-a).total_seconds())<4*3600 for a in used.get(key,[])):continue
                used.setdefault(key,[]).append(item["anchor"]);selected.append(item);count+=1
                if count==target:break
            if count<target:raise RuntimeError(f"Real-data sampling shortfall: {site}/{stratum}: {count}/{target}")
    rng.shuffle(selected); windows=[]; audit=[]
    for i,item in enumerate(selected,1):
        wid=f"M{i:03d}"; a=item["anchor"]
        windows.append({"window_id":wid,"target_start":str(a-pd.Timedelta(hours=1)),
                        "target_end":str(a+pd.Timedelta(hours=1)),"source_clock":item["source_clock"],
                        "minutes":item["minutes"],"power":item["power"],"wind":item["channels"].get("wind",[]),
                        "nominal_minutes":item["nominal_minutes"]})
        audit.append({"window_id":wid,"site":item["site"],"turbine":item["turbine"],
                      "stratum":item["stratum"],"anchor":str(a),"source":item["source"],
                      "scale":item["scale"],"source_clock":item["source_clock"]})
    digest=hashlib.sha256(json.dumps(windows,sort_keys=True).encode()).hexdigest()
    packet={"schema":"wind_multirater_v15","packet_id":"wind-v15-"+digest[:12],
            "target_minutes":[-60,60],"display_minutes":[-120,120],"seed":SEED,"windows":windows}
    OUT.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(packet,ensure_ascii=False,allow_nan=False)
    template=(ROOT/"script/templates/multirater_v15.html").read_text()
    assert template.count("__PACKET__")==1
    javascript=re.search(r"<script>\s*(.*?)</script>",template,re.S)[1]
    subprocess.run(["node","--check"],input=javascript,text=True,check=True)
    (OUT/"index.html").write_text(template.replace("__PACKET__",encoded.replace("<","\\u003c")),encoding="utf8")
    (OUT/"packet.json").write_text(encoded,encoding="utf8")
    pd.DataFrame(audit).to_csv(OUT/"sampling_manifest.csv",index=False)
    counts=pd.DataFrame(audit).groupby(["site","stratum"]).size().reset_index(name="windows")
    counts.to_csv(OUT/"sampling_balance.csv",index=False)
    readme="""# 多标注者风功率区域复核 v15

双击 index.html 即可离线作答；也可通过发布网页作答。每位参与者使用自己的匿名ID，不填写姓名或邮箱。
只标橙色两小时区域；灰色区域只作背景。完成后导出JSON并交给研究负责人。

窗口按四个站点等量抽取，每站20个平稳筛选、5个上升筛选、5个下降筛选、10个单转折筛选、10个随机筛选。
这是抽样层平衡，最终人工阳性比例由独立作答决定，不能把机器筛选当真值。
界面不显示检测器、抽样层、旧人工答案或原因标签。研究负责人分析前保持独立作答。

新协议中事件presence表示目标段内可辨识动态变化。持续低功率本身是状态；只有出现转入/转出变化时选择相应动态形态。
这与v9的presence口径不同，禁止直接合并为同一版本。原v9标签保持不变。
人工边界为可选项，未标记时按区域级标签分析，不推算精确起点延迟或全目录recall。

来源映射、原始时钟和目标区域见sampling_manifest.csv/packet.json。Greek/SDWPF保留源时钟，不强行声明UTC。
框内功率用各原始序列早期q99.5归一化，纵轴保留幅度。所有轨迹来自现有SCADA；无生成或补插值轨迹。

合并工具：python script/merge_multirater_v15.py --packet annotation_multirater_v15/packet.json --inputs results/*.json --output outputs/multirater_v15
只有至少两名独立标注者的交集才产生一致性结果；不确定与数据问题独立统计。
代码和本轮经授权的标注包使用MIT许可，外部数据原许可仍适用。
"""
    (OUT/"README.md").write_text(readme,encoding="utf8")
    (OUT/"protocol.json").write_text(json.dumps({"schema":packet["schema"],"packet_id":packet["packet_id"],
      "windows":len(windows),"target_hours":2,"context_hours":4,"selection_truth":False,
      "sites":4,"strata_per_site":QUOTAS,"legacy_labels_used":False,
      "independent_raters_received":0,"sha256_windows":digest},ensure_ascii=False,indent=2))
    with zipfile.ZipFile(OUT/"wind_multirater_v15.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ["index.html","packet.json","README.md","protocol.json"]:
            archive.write(OUT/name,name)
    print(packet["packet_id"],len(windows))
    print(counts.to_string(index=False))
if __name__=="__main__":main()
