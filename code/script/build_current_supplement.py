"""Build the paper-specific supplement from frozen NP022--NP031 outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = Path("/mnt/c/Users/admin/Desktop/next-paper")
OUT_MD = DOCS / "next_paper_current_supplement.md"
LEADS = [15, 60, 120, 240, 480, 720]


def md_table(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 4) -> str:
    table = frame.copy()
    if columns:
        table = table[columns]
    for c in table.columns:
        if pd.api.types.is_float_dtype(table[c]):
            table[c] = table[c].map(lambda x: f"{x:.{digits}f}")
    headers = [str(c) for c in table.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in table.columns) + " |")
    return "\n".join(lines)


def rps(p: np.ndarray, y: np.ndarray) -> float:
    t = np.zeros_like(p)
    t[np.arange(len(y)), y] = 1
    return float(np.mean(np.sum((np.cumsum(p, axis=1) - np.cumsum(t, axis=1)) ** 2, axis=1)))


def conditional_rps(q: np.ndarray, y: np.ndarray) -> float:
    cdf = np.cumsum(q, axis=1)
    cdf[:, -1] = 1.0
    truth = np.arange(16)[None, :] >= y[:, None]
    return float(np.mean((cdf[:, :-1] - truth[:, :-1]) ** 2))


def raw_scores(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = ROOT / "outputs/next_paper" / tag
    z = np.load(base / "test_predictions.npz")
    rows: list[dict] = []
    joint_rows: list[dict] = []
    for lead in LEADS:
        y = z[f"{lead}__y"]
        arr = z[f"{lead}__arr"]
        idx = arr < 16
        for arm in ["power", "weather", "joint"]:
            e = z[f"{lead}__{arm}__event"]
            q = z[f"{lead}__{arm}__timing"]
            state = np.eye(4)[y]
            pdown = e[:, 2] + e[:, 3]
            joint = np.c_[pdown[:, None] * q, 1 - pdown]
            rows.extend([
                {"package": tag, "lead_min": lead, "arm": arm, "metric": "multiclass Brier", "score": float(np.mean(np.sum((e - state) ** 2, axis=1))), "n": len(y)},
                {"package": tag, "lead_min": lead, "arm": arm, "metric": "return Brier", "score": float(np.mean((e[:, 3] - (y == 3)) ** 2)), "n": len(y)},
                {"package": tag, "lead_min": lead, "arm": arm, "metric": "conditional timing RPS", "score": conditional_rps(q[idx], arr[idx]), "n": int(idx.sum())},
                {"package": tag, "lead_min": lead, "arm": arm, "metric": "joint first-passage RPS", "score": rps(joint, arr), "n": len(y)},
            ])
            if arm in {"power", "weather", "joint"}:
                joint_rows.append({"package": tag, "lead_min": lead, "arm": arm, "joint_RPS": rps(joint, arr), "event_Brier": float(np.mean(np.sum((e - state) ** 2, axis=1))), "conditional_RPS_positive": conditional_rps(q[idx], arr[idx]), "all_windows": len(y), "down_events": int(idx.sum())})
    return pd.DataFrame(rows), pd.DataFrame(joint_rows)


def protocol_summary(tag: str) -> dict:
    return json.loads((ROOT / "outputs/next_paper" / tag / "protocol.json").read_text(encoding="utf-8"))


def compact_protocol(tag: str) -> str:
    p = protocol_summary(tag)
    meta = p.get("meta", {})
    weather_start = meta.get("weather_start")
    weather_end = meta.get("weather_end")
    support_text = f"{weather_start} to {weather_end}" if weather_start and weather_end else "all requested rows in the archived combined file"
    lines = [
        f"**{tag.upper()}**: {p.get('stage')}",
        f"- Question: {p.get('question')}",
        f"- Site and period: {p.get('site')}; {meta.get('grid_start', p.get('support_period', 'see protocol'))} to {meta.get('grid_end', '')}.",
        f"- Leads: {', '.join(str(x) for x in p.get('leads', []))} min; target: {p.get('target')}.",
        f"- Arms: {json.dumps(p.get('arms', {}), ensure_ascii=False)}.",
        f"- Split: {p.get('split')}; event model: {p.get('event_model')}; timing model: {p.get('timing_model')}.",
        f"- Weather source: {p.get('weather_source')}; variables: {', '.join(meta.get('weather_variables', []))}; complete support: {support_text}.",
    ]
    return "\n".join(lines)


def pizhou_core_summary() -> tuple[str, str]:
    p5 = protocol_summary("np005")
    p8 = protocol_summary("np008")
    p9 = protocol_summary("np009")
    d5 = pd.read_csv(ROOT / "outputs/next_paper/np005/test_summary.csv")
    d8 = pd.read_csv(ROOT / "outputs/next_paper/np008/test_summary.csv")
    d9 = pd.read_csv(ROOT / "outputs/next_paper/np009/test_summary.csv")
    core = "\n".join([
        "### S2.1 Pizhou core process-state protocol (NP005/NP008)",
        f"- NP005 question: {p5['question']}",
        "- Pizhou target: 17 native quarter-hour power nodes from T to T+4 h; 0.2 of the provider-supplied 87.45-MW farm capacity; power divided by 87.45 MW; no clipping of recorded support values.",
        f"- Leads: {', '.join(str(x) for x in p5['lead_minutes'])} min; test cohort: 6,496 windows; arms: {', '.join(p5['arms'])}; model: ExtraTrees 150 trees, depth 16, leaf 32; no test selection.",
        f"- Clock/split: {p5['input_clock']}; UTC boundaries {', '.join(p5['boundaries_utc'])}; target support {p5['support']}.",
        f"- Maturity and update: {p8['maturity']}; NP008 reuses frozen NP007 rolling predictions and fixes lambda {p8['lambda_']} before test scoring.",
        f"- NP008 features: {', '.join(p8['features'])}; state classes: {p8['target']}; source hashes and freeze receipt are stored in `outputs/next_paper/np008/protocol.json` and `freeze.json`.",
        "",
        "### S2.2 Pizhou timing protocol (NP009)",
        f"- Question: {p9['question']}",
        f"- Timing bins: {p9['target']}; occurrence: {p9['occurrence']}; arms: {', '.join(p9['arms'])}; feature counts: {json.dumps(p9['features'], ensure_ascii=False)}; lambda selection: {p9['selection']}.",
        f"- Arm construction: context uses endpoint/context covariates; ordered uses the ordered NWP interior trajectory; sorted independently sorts the same interior values and preserves endpoint/context features. Fitting: {p9['fitting']}; smoothing: {p9['smoothing']}.",
        "- The validation-frozen timing policy selects lambda=0.001 before test scoring; `selection.json` records the frozen policy and validation hash. NP009 is a 16-time multinomial timing model, not the ExtraTrees timing model used by NP025/NP028.",
        "- NP009 uses the frozen NP008 full rolling occurrence marginals. At 15 min, the downward timing denominator is 2,339 windows; frequency conditional RPS/median-MAE are 0.161377/54.215 min and context values are 0.148245/49.348 min.",
        "",
        "### S2.3 Pizhou headline absolute scores",
        md_table(pd.DataFrame([
            {"package": "NP008", "lead_min": 720, "arm": "frequency rolling", "metric": "return Brier", "score": float(d8[(d8.lead_minutes==720)&(d8.group=='frequency')&(d8['mode']=='rolling')&(d8.metric=='return_brier')].loss.iloc[0]), "n": 6496},
            {"package": "NP008", "lead_min": 720, "arm": "full rolling", "metric": "return Brier", "score": float(d8[(d8.lead_minutes==720)&(d8.group=='full')&(d8['mode']=='rolling')&(d8.metric=='return_brier')].loss.iloc[0]), "n": 6496},
            {"package": "NP008", "lead_min": 720, "arm": "frequency rolling", "metric": "multiclass Brier", "score": float(d8[(d8.lead_minutes==720)&(d8.group=='frequency')&(d8['mode']=='rolling')&(d8.metric=='multiclass_brier')].loss.iloc[0]), "n": 6496},
            {"package": "NP008", "lead_min": 720, "arm": "full rolling", "metric": "multiclass Brier", "score": float(d8[(d8.lead_minutes==720)&(d8.group=='full')&(d8['mode']=='rolling')&(d8.metric=='multiclass_brier')].loss.iloc[0]), "n": 6496},
            {"package": "NP009", "lead_min": 15, "arm": "frequency", "metric": "conditional timing RPS", "score": float(d9[(d9.lead_minutes==15)&(d9.event=='down')&(d9.arm=='frequency')&(d9.metric=='conditional_rps')].loss.iloc[0]), "n": 2339},
            {"package": "NP009", "lead_min": 15, "arm": "context", "metric": "conditional timing RPS", "score": float(d9[(d9.lead_minutes==15)&(d9.event=='down')&(d9.arm=='context')&(d9.metric=='conditional_rps')].loss.iloc[0]), "n": 2339},
            {"package": "NP009", "lead_min": 15, "arm": "frequency", "metric": "conditional median MAE (min)", "score": float(d9[(d9.lead_minutes==15)&(d9.event=='down')&(d9.arm=='frequency')&(d9.metric=='conditional_median_mae_minutes')].loss.iloc[0]), "n": 2339},
            {"package": "NP009", "lead_min": 15, "arm": "context", "metric": "conditional median MAE (min)", "score": float(d9[(d9.lead_minutes==15)&(d9.event=='down')&(d9.arm=='context')&(d9.metric=='conditional_median_mae_minutes')].loss.iloc[0]), "n": 2339},
        ]), ["package", "lead_min", "arm", "metric", "score", "n"], 6),
    ])
    return core, ""


def main() -> None:
    raw = []
    joint = []
    for tag in ["np025", "np028"]:
        a, b = raw_scores(tag)
        raw.append(a)
        joint.append(b)
    raw_df = pd.concat(raw, ignore_index=True)
    joint_df = pd.concat(joint, ignore_index=True)
    cal = pd.read_csv(ROOT / "outputs/next_paper/np031_calendar_control/calendar_control_summary.csv")
    cal_show = cal.rename(columns={"source": "package", "lead_minutes": "lead_min", "relative_loss_reduction_pct": "relative_gain_pct", "absolute_candidate": "calendar_only_score", "absolute_reference": "frequency_reference_score"})
    cal_show = cal_show[["package", "lead_min", "task", "metric", "relative_gain_pct", "low", "high", "calendar_only_score", "frequency_reference_score", "windows"]]

    route = []
    for tag in ["np026", "np029"]:
        frame = pd.read_csv(ROOT / "outputs/next_paper" / tag / "routing_summary.csv")
        frame = frame[(frame["candidate"] == "target_aligned_route") & (frame["reference"] == "joint_timing")].copy()
        frame.insert(0, "package", tag)
        route.append(frame)
    route_df = pd.concat(route, ignore_index=True)
    route_show = route_df.copy()
    route_show["metric"] = route_show["metric"].map({
        "conditional_rps": "timing RPS",
        "conditional_median_mae_minutes": "timing median MAE",
    }).fillna(route_show["metric"])
    route_show["gain (95% CI) %"] = route_show.apply(
        lambda r: f"{r['relative_loss_reduction_pct']:.2f} ({r['low']:.2f}, {r['high']:.2f})", axis=1
    )
    route_show = route_show[["package", "lead_minutes", "metric", "gain (95% CI) %", "windows"]]
    support_rows = []
    for tag in ["np025", "np028"]:
        p = protocol_summary(tag)
        for row in p["meta"]["support"]:
            if row["split"] == "test" and row["lead_minutes"] == 15:
                row = dict(row); row["package"] = tag; support_rows.append(row)
    support = pd.DataFrame(support_rows)
    seed_ext = pd.read_csv(ROOT / "outputs/next_paper/np032_np033/seed_sensitivity.csv")
    seed_pair = seed_ext[seed_ext["relative_gain_pct"].notna()].groupby(["package", "lead_minutes", "arm", "metric"], as_index=False)["relative_gain_pct"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    month_ext = pd.read_csv(ROOT / "outputs/next_paper/np032_np033/monthly_information.csv")
    month_sum = month_ext.groupby(["package", "arm", "task", "metric"], as_index=False)["relative_gain_pct"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    cal_ext = pd.read_csv(ROOT / "outputs/next_paper/np035_calibration/reliability.csv")
    cal_sum = cal_ext.groupby(["package", "target"], as_index=False)["abs_gap"].agg(["mean", "max", "count"]).reset_index()
    joint_baseline = pd.read_csv(ROOT / "outputs/next_paper/np036_joint_timing_baseline/joint_timing_baseline.csv")
    ablation = pd.read_csv(ROOT / "outputs/next_paper/np034_weather_ablation/weather_ablation_summary.csv")
    external_np022 = pd.read_csv(ROOT / "outputs/next_paper/np022/test_summary.csv")
    external_show = external_np022[
        (external_np022["lead_minutes"] == 15)
        & (external_np022["metric"].isin(["conditional_rps", "conditional_median_mae_minutes"]))
    ][["site", "lead_minutes", "metric", "relative_loss_reduction_pct", "low", "high", "windows", "valid_windows"]]
    range_show = pd.DataFrame([
        {"lead_min": 15, "task": "range CRPS", "candidate": 0.550995, "reference": 0.676670, "gain_pct": 18.57, "CI_low": 12.94, "CI_high": 24.24, "n": 6059},
        {"lead_min": 720, "task": "range CRPS", "candidate": 0.564087, "reference": 0.676414, "gain_pct": 16.61, "CI_low": 11.61, "CI_high": 22.05, "n": 6059},
        {"lead_min": 720, "task": "return Brier; shuffled order", "candidate": 0.120999, "reference": 0.122366, "gain_pct": 1.12, "CI_low": 0.20, "CI_high": 2.13, "n": 6496},
    ])
    threshold = pd.read_csv(ROOT / "outputs/next_paper/np037_threshold_sensitivity/threshold_sensitivity.csv")
    threshold_state = threshold[
        (threshold["task"] == "state")
        & (threshold["arm"] == "weather")
        & (threshold["threshold"].isin([0.15, 0.25]))
    ][["package", "threshold", "lead_minutes", "relative_vs_frequency_pct", "windows"]]
    threshold_timing = threshold[threshold["task"] == "downward_timing"].pivot_table(
        index=["package", "threshold", "lead_minutes"], columns="arm", values="absolute_score"
    ).reset_index()
    threshold_timing["power_added_to_weather_pct"] = 100 * (
        threshold_timing["weather"] - threshold_timing["power"]
    ) / threshold_timing["weather"]
    threshold_timing = threshold_timing[
        ["package", "threshold", "lead_minutes", "power_added_to_weather_pct"]
    ]
    cap_summary = pd.read_csv(ROOT / "outputs/next_paper/np038_capacity_matched_information/capacity_matched_summary.csv")
    cap_comp = pd.read_csv(ROOT / "outputs/next_paper/np038_capacity_matched_information/comparisons.csv")
    cap_summary_show = cap_comp[
        (cap_comp["run_type"] == "primary")
        & (cap_comp["task"] == "state")
        & (cap_comp["metric"] == "multiclass_brier")
        & cap_comp["comparison"].isin(["weather_summary20_vs_noise20", "power_weather_summary40_vs_power_noise40"])
    ][["package", "lead_minutes", "comparison", "candidate_score", "reference_score", "reduction_pct", "reduction_low", "reduction_high", "valid_windows"]]
    cap_summary_agg = (
        cap_comp[
            (cap_comp["run_type"] == "primary")
            & (cap_comp["task"] == "state")
            & (cap_comp["metric"] == "multiclass_brier")
            & cap_comp["comparison"].isin(["weather_summary20_vs_noise20", "power_weather_summary40_vs_power_noise40"])
        ]
        .groupby(["package", "comparison"])["reduction_pct"]
        .agg(["min", "max", "mean"])
        .reset_index()
    )
    route_global = pd.read_csv(ROOT / "outputs/next_paper/np040_route_global_summary/global_summary.csv")
    decision_all = pd.read_csv(ROOT / "outputs/next_paper/np041_all_lead_decision_sensitivity/decision_summary.csv")
    decision_all_show = decision_all[["lead_minutes", "cost_ratio", "value", "low", "high", "threshold", "windows"]]
    decision_cmp = pd.read_csv(ROOT / "outputs/next_paper/np041_all_lead_decision_sensitivity/np023_comparison.csv")
    cal_bins = cal_ext[(cal_ext["lead_minutes"].isin([15, 720])) & (cal_ext["arm"] == "joint")][["package", "lead_minutes", "target", "bin_low", "bin_high", "n", "mean_pred", "obs_rate", "abs_gap"]]

    core_text, _ = pizhou_core_summary()
    text = r"""# Supplementary material: Target-conditional information horizons in wind-power event forecasting

This supplement belongs to the Renewable Energy manuscript *Different information horizons shape wind-power event predictability*. It replaces the legacy event-representation supplement and documents the current NP005–NP037 evidence package.

## S1. Study object and information boundary

The target begins at $T$ and contains 17 native 15-min power nodes through $T+4\,\mathrm{h}$. Pizhou power is divided by the provider-supplied 87.45-MW farm capacity; Suining and the external process sites use their site-specific training-period q99.5 scales. All sites use a 0.2 normalized excursion rule, with the physical scale declared separately. The four state classes are neither, upward, downward and both. The downward first-passage label has 16 crossing bins (15–240 min) and a 17th no-crossing bin. Power history ends before the issue time. The strict-clock weather package uses a value fixed 24 h before the valid target time. Future onsite wind and ERA5 are retrospective diagnostics or matured labels and do not enter an operating issue-time row.

The administrative reference for Pizhou is 34.3403°N, 118.0068°E and the supplied onsite wind reference height is 140 m. The Suining aggregate contains 14 turbines and retains a row when at least 80% of turbines are valid. Each turbine is scaled by its first chronological training-period q99.5 value. No arbitrary height conversion is applied; the weather products retain their source heights and are interpreted as issued product information.

## S2. Issue-time contracts and chronological support

""" + core_text + "\n\n" + compact_protocol("np025") + "\n\n" + compact_protocol("np028") + r"""

For NP025, the historical-forecast trajectory is the Open-Meteo GFS global historical-forecast package. Each row is selected by the target-valid trajectory and its archived product timestamp; the row-level rule is that all requested fields must be present in the historical package before the forecast row is accepted. The archive does not expose a model release timestamp, so NP025 is reported as a retrospective information decomposition and is not used as the strict real-time release test. NP028 uses the Open-Meteo Previous Runs GFS global package and fixes every weather value 24 h before target valid time. Complete NP028 weather support begins on 2024-02-16. The NP025 and NP028 weather manifests retain monthly query URLs, response hashes, coordinates, units and returned timestamps.

The chronological split is 60/20/20 with a four-hour purge at each boundary. The event model is ExtraTrees with 150 trees, maximum depth 16, minimum leaf size 32, all available features and seed 41. The conditional timing model uses ExtraTrees with 100 trees, depth 16, leaf size 16 and seed 43 on matured downward-event labels; the occurrence probability is held at the training frequency for the conditional timing product. The matched arms use identical windows and budgets within each package.

## S3. Label and score definitions

For a target path $Y_T$, the upward and downward excursions are running-extrema functionals. A downward crossing label $\\tau_{T,\\downarrow}$ is the first 15-min node at which the downward excursion reaches 0.2 on the declared site-specific normalized scale; the no-crossing label is bin 16. Conditional timing RPS and median arrival-time error are evaluated on the crossing subset, whose denominator is reported in every table. Event-state Brier scores use all test windows.

The unconditional joint first-passage distribution used as a diagnostic is formed from the event probability $p_{\\downarrow}=p_{\\mathrm{downward}}+p_{\\mathrm{both}}$ and the conditional timing distribution $q$:

$$
\tilde q_{0:15}=p_{\downarrow}q_{0:15},\qquad \tilde q_{16}=1-p_{\downarrow}.
$$

The joint first-passage RPS is reported as an accounting score that retains the occurrence and timing components in one no-crossing-inclusive distribution. The main paper keeps conditional RPS as the primary timing object because the route is designed to improve the timing distribution given a process occurrence.

Relative loss reduction is $100(1-L_\\mathrm{candidate}/L_\\mathrm{reference})$. Intervals are paired seven-day calendar-block bootstrap percentile intervals with 2,000 draws and seed 41; the same target blocks are used for the compared arms. The fitted models are held fixed in these intervals.

## S4. External power-history process-object validation (NP022)

NP022 uses 15-min SCADA from La Haute Borne (4 turbines, 13,912 test windows and 3,966 downward events at 15 min; 2014–2015) and Suining (14 turbines, 6,996 test windows and 2,653 downward events at 15 min; 2023–2024), with a 17-node, four-hour target and the same normalized excursion rule on each site's training-period q99.5 scale. Each site uses 16 power-history lags ending at issue time, known calendar terms, chronological 60/20/20 splits and a four-hour purge. The result validates the occurrence-versus-first-passage distinction for a power-history forecast object; it does not validate the weather-source increment. Suining is reused as the site for NP024–NP030 information-arm experiments, while La Haute Borne supplies the independent external farm check. The exact 15-min timing results are shown below; intervals are paired seven-day calendar-block bootstrap intervals.

""" + md_table(external_show, None, 3) + r"""

## S5. Three-arm information decomposition (NP025)

The three matched arms are power history plus calendar, weather trajectory plus calendar, and their concatenation. The following tables give absolute scores, denominators and the complete six-lead matrix.

### S5.1 Absolute scores

""" + md_table(raw_df, ["package", "lead_min", "arm", "metric", "score", "n"], 6) + r"""

### S5.2 No-crossing-inclusive joint accounting

`joint_RPS` is the 17-bin score on all windows. `conditional_RPS_positive` is the original positive-event conditional score and uses `down_events` as its denominator; it is not an all-window score.

""" + md_table(joint_df, ["package", "lead_min", "arm", "joint_RPS", "event_Brier", "conditional_RPS_positive", "all_windows", "down_events"], 6) + r"""

## S6. Calendar-only information control (NP031)

NP031 fits the same ExtraTrees event and timing budgets on only the four known target-calendar variables extracted from the frozen NP025 and NP028 datasets. It uses the same train/test masks, target rows and seven-day paired bootstrap. The control is deliberately reported against the frequency reference: it shows whether the headline increments require power or weather information beyond calendar seasonality.

""" + md_table(cal_show, None, 4) + r"""

The calendar-only control has negative event-state relative reductions for both packages and negative conditional timing reductions at the primary leads. The information-arm gains therefore include information beyond the shared calendar columns.

## S7. Target-aligned routing (NP026 and NP029)

NP026 and NP029 are deterministic transformations of frozen test predictions. The route keeps the joint event-state probabilities and assigns the power-history timing probabilities to downward first passage. No model is refit, no test lead is selected, and no test month is used to define the mapping. State identity is checked by maximum absolute probability difference.

""" + md_table(route_show, None, 4) + r"""

The complete matrices are retained in `outputs/next_paper/np026/routing_summary.csv` and `outputs/next_paper/np029/routing_summary.csv`. The primary route endpoints are 240 and 480 min for the historical package and 60 and 120 min for the strict-clock package; all six leads remain in the supplement.

## S8. Monthly stability audits (NP027 and NP030)

NP027 recomputes the frozen NP026 route separately for UTC June, July and August 2024. NP030 recomputes the strict-clock route for July and August 2024. No month, model or lead is selected during the audit. The complete monthly tables, including mixed signs at secondary leads, are retained in the corresponding output directories and figure source manifests. The main text reports the pre-specified supported endpoints; the audit is an auxiliary stability check.

## S9. Reproducibility and source traceability

The current evidence package is generated from the following frozen output directories: `np005`, `np007`, `np008`, `np009`, `np013`, `np014`, `np016`, `np017`, `np018`, `np019`, `np020`, `np022`, `np024`, `np025`, `np026`, `np027`, `np028`, `np029`, `np030`, `np031_calendar_control`, `np032_np033`, `np034_weather_ablation`, `np035_calibration` and `np036_joint_timing_baseline`. Each directory contains a protocol or manifest, test predictions or summary, and an independent verification record. NP005/NP007/NP008/NP009 bind the Pizhou cohort, feature definitions, frozen occurrence predictions, timing predictions and validation-frozen lambda. NP013/NP014/NP016–NP020 are retained supporting range/calibration/fusion outputs referenced by the main text and their protocols remain in the corresponding output directories. NP025 and NP028 bind their scripts, SCADA source, weather file and weather manifest by SHA-256. NP026 and NP029 bind their routing scripts to the frozen source prediction manifests. NP031 records the dataset source and the four-column calendar extraction. NP032/NP033 record the seed and calendar robustness extensions; NP034/NP035 record the weather-variable ablation and calibration diagnostics.

NP037 adds the threshold reconstruction check under outputs/next_paper/np037_threshold_sensitivity, and the exact NP022 external timing rows and NP014 range/path rows are reproduced in S4 and S15.

Raw private SCADA is not redistributed. The minimum reproducibility index consists of the protocol JSON, support CSV, prediction summary CSV, verification JSON and source-hash manifest in each listed output directory; these files are sufficient to replay the reported score arithmetic after the private source is replaced by the permitted derived arrays. The released replication package will include public weather files or source URLs, feature/label arrays after privacy review, prediction summaries, source hashes and the exact scripts used to produce the tables. All timestamp handling is UTC.

## S10. Secondary matrices and scope

The full six-lead, arm-by-metric relative-loss matrices, all monthly rows, support counts and negative or mixed increments remain part of the machine-readable output directories. The main text uses the pre-specified positive endpoints to explain the information-to-target rule; these secondary matrices provide the complete audit trail for the interpretation.

## S11. Seed and month robustness extensions (NP032/NP033)

NP032 refits the NP025 and NP028 event and timing models at seeds 41, 42 and 43 for the 15-, 240- and 720-min endpoints. The central state increments remain stable: at 720 min, joint-versus-power multiclass Brier gains average 9.56% (SD 0.48%) for NP025 and 12.61% (SD 0.57%) for NP028. The full seed matrix is retained below.

""" + md_table(seed_pair, None, 3) + r"""

NP033 recalculates frozen information increments by calendar month for all six leads. The summary below reports mean, SD, minimum, maximum and count across supported months; the complete row matrix is stored in `outputs/next_paper/np032_np033/monthly_information.csv`.

""" + md_table(month_sum, None, 3) + r"""

## S12. Calibration diagnostics (NP035)

NP035 bins frozen state probabilities and the probability of a downward arrival within 60 min into ten probability bins. It reports bin count, mean predicted probability, observed frequency and absolute calibration gap for NP025/NP028, all leads and all arms. These diagnostics separate information value from calibration: a lower forecast loss can coexist with a calibration gap, and the main paper therefore uses proper losses as its primary comparison while retaining reliability rows here.

""" + md_table(cal_sum, None, 4) + r"""

The representative bin-level rows below expose the calibration calculation for the 15-min and 12-h joint arms; the complete reliability table is stored in `outputs/next_paper/np035_calibration/reliability.csv`.

""" + md_table(cal_bins, None, 4) + r"""

The protocol and verification receipts bind the reliability rows to the frozen NP025/NP028 predictions.

## S13. Joint first-passage baseline and route falsification (NP036)

NP036 fits a direct 17-class first-passage model with the joint feature arm under the same chronological windows and ExtraTrees budget. The route is evaluated both against this direct joint model and against the constructed joint distribution from the original joint event/timing heads. The direct baseline is a stronger end-to-end comparator; the constructed comparator isolates the value of replacing the conditional timing head while preserving the state head. The full six-lead matrix is stored in `outputs/next_paper/np036_joint_timing_baseline/joint_timing_baseline.csv`.

""" + md_table(joint_baseline, None, 4) + r"""

## S14. Weather-variable ablation (NP034)

NP034 compares calendar-only, thermodynamic-only, wind-vector-only and full-weather feature sets at 15, 240 and 720 min under the same tree budgets. The wind-vector channel supplies the positive state-risk contribution in both packages; thermodynamic-only features remain below the frequency reference in the tested ablation.

""" + md_table(ablation, None, 5) + r"""

## S15. Range and path supporting results (NP014)

NP014 supplies a separate amplitude and path-order channel. The first two rows compare the direct range arm with the old conditional range reference; the third compares the path-order assignment with the same range marginal after order shuffling. The table gives the exact headline values used in the main text; the full range, coverage and event-value matrices remain in the NP014 output package.

""" + md_table(range_show, None, 3) + r"""

## S16. Excursion-threshold sensitivity (NP037)

NP037 reconstructs the Suining aggregate from the archived SCADA and reruns the fixed power, weather and joint feature contracts at thresholds 0.15, 0.20 and 0.25. The 0.20 labels reproduce the frozen NP025/NP028 labels exactly for every tested lead and package. At the two off-design thresholds, the weather-only state point gain remains positive at all six package–lead combinations; the timing table reports the added value of power history relative to weather.

### S16.1 Weather state increment at thresholds 0.15 and 0.25

""" + md_table(threshold_state, None, 3) + r"""

### S16.2 Power-history timing increment across thresholds

""" + md_table(threshold_timing, None, 3) + r"""

## S17. Mathematical synthesis and operational interface

The forecast family used in the paper is

$$
\mathcal{Y}_{q,T}=\Phi_q(Y_T),\qquad
\widehat F_{q,h}^{\,a}=f_q(X_{T-h}^{\,a},h),\qquad
\mathcal{L}_{q,h}^{\,a}=\mathbb{E}\,S_q(\widehat F_{q,h}^{\,a},\mathcal{Y}_{q,T}).
$$

The empirical information value is

$$
\widehat{\\mathrm{IV}}_q(A\\rightarrow B;h)
=100\\frac{\\widehat{\\mathcal{L}}_{q,h}^{\,A}-\\widehat{\\mathcal{L}}_{q,h}^{\,B}}
{\\widehat{\\mathcal{L}}_{q,h}^{\,A}}.
$$

For a downward occurrence probability $p_{\\downarrow}$ and a conditional first-passage distribution $q_\\tau$, the no-crossing-inclusive accounting distribution is

$$
\\widetilde q_k=p_{\\downarrow}q_{\\tau,k}\\quad(k=0,\\ldots,15),\\qquad
\\widetilde q_{16}=1-p_{\\downarrow}.
$$

The target-aligned route preserves the joint state vector and replaces only the conditional timing channel. Therefore, the constructed joint-score change is attributable to the timing-channel replacement under the fixed state identity; NP036 supplies the direct 17-bin joint comparison. The operational interface uses the return-event probability for a state alert and the conditional timing distribution for timing allocation. NP023 varies the alert threshold for a stated false-negative/false-positive loss while keeping forecast probabilities frozen.

## S18. Capacity-matched information controls (NP038)

NP038 compares the original feature arms with compact weather summaries, a fixed projection, timestamp-matched noise, and power-plus-weather/noise arms at matched dimensions. Weather summaries use the four calendar columns plus mean, standard deviation, first node and last node for each issued weather variable. The projection is standardized on training rows before a fixed QR map; the noise arm is generated from timestamp-hashed random seeds. The complete 432-row score table, 648-row seed table, feature maps, labels, predictions and tree-complexity records are released in the NP038 directory.

""" + md_table(cap_summary_agg, None, 3) + r"""

The rows below provide the primary matched state-score comparisons; interval bounds are paired seven-day blocks with 2,000 draws.

""" + md_table(cap_summary_show, None, 3) + r"""

NP038 also retains projection and timing comparisons whose signs vary by package or lead. These rows remain part of the release and are not promoted to the main narrative.

## S19. All-lead route uncertainty and endpoint policy (NP039/NP040)

NP039 recomputes route, constructed-joint and direct-joint 17-bin products with a shared UTC 7-day block universe, 2,000 draws and fixed models. Conditional timing rows use the downward-event mask; state identity is checked against the frozen joint event head. NP040 first averages the six lead-wise ratios within each bootstrap draw and then takes the interval, avoiding interval averaging.

""" + md_table(route_global[["package", "reference", "metric", "equal_weight_mean_gain_pct", "global_low", "global_high", "min_lead_gain_pct", "max_lead_gain_pct", "min_valid_windows"]], None, 3) + r"""

The endpoint policy makes all six leads primary. The 240/480-min archived and 60/120-min strict patterns are descriptive local peaks, not test-selected endpoints.

## S20. All-lead decision sensitivity and NP023 comparison (NP041)

NP041 selects alert thresholds only on validation probabilities and applies them once to the test cohort. The full six-lead table includes no-alert, always-alert, validation-frequency and validation-constant policy costs, alert rates, saved loss arrays and paired intervals. The central forecast-product result is the full-rolling cost reduction relative to no alert; the two-lead NP023 comparison is retained separately because NP014/NP019/NP020 are available only at 15 and 720 min.

""" + md_table(decision_all_show, None, 3) + r"""

### S20.1 NP023 fusion comparison

""" + md_table(decision_cmp, None, 3) + r"""

## S21. Literature gap and anonymous reproduction

The verified literature matrix separates NWP ensemble ramps, joint time-distribution ramps, timing/intensity scoring, weather-regime conditioning and probabilistic calibration. The full comparison and official metadata are in temp/literature_gap_verified.md and literature_matrix_30plus.md. The public repository contains the processed aggregate/features/labels, frozen predictions, protocols, score tables, figure manifests and verification scripts. Raw SCADA, turbine-level coordinates, operational-status records and credentials are excluded. A reader can clone the repository anonymously and run the verification scripts without a GitHub account.
"""
    tick = chr(96)
    old_index = "np035_calibration" + tick + " and " + tick + "np036_joint_timing_baseline"
    new_index = "np035_calibration" + tick + ", " + tick + "np036_joint_timing_baseline" + tick + " and " + tick + "np037_threshold_sensitivity"
    text = text.replace(old_index, new_index)
    # The raw template keeps LaTeX commands readable in Python source; Markdown
    # math needs one command slash after interpolation.
    text = text.replace("\\\\", "\\")
    OUT_MD.write_text(text, encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()
