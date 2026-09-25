"""Build the editorial preparation package from existing evidence; no experiments.

Uses the downloaded Elsevier class and numeric bibliography style. The source
bundle is flat because the publisher's Editorial Manager guidance requires it.
"""
from pathlib import Path
import csv
import json
import re
import os
import shutil
import subprocess
import zipfile
import hashlib
import math
import pypandoc

TABLE_LOG = []

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript/applied_energy"
BUILD = ROOT / "temp/applied_energy_build"
SOURCES = OUT / "submission_source"
DATA_OUT = Path(os.environ.get("WPF_EVIDENCE_OUT", str(ROOT / "outputs/dynamic_events_v6")))
DOWN_OUT = ROOT / "outputs/downstream_v6"


def latex(text):
    text = re.sub(r"^(#+)\s+\d+(?:\.\d+)*\.\s+", r"\1 ", text, flags=re.M)
    try:
        return subprocess.run(
            ["pandoc", "-f", "markdown+tex_math_single_backslash", "-t", "latex", "--natbib", "--no-highlight", "--wrap=none"],
            input=text, text=True, capture_output=True, check=True,
        ).stdout.strip()
    except FileNotFoundError:
        return pypandoc.convert_text(text, "latex", format="markdown+tex_math_single_backslash",
                                     extra_args=["--natbib", "--no-highlight", "--wrap=none"]).strip()


def read_rows(name):
    with (DATA_OUT / name).open(newline="") as f:
        return list(csv.DictReader(f))


TABLE_SOURCES = {
  "1": "outputs/dynamic_events_v6/representation_summary_primary.csv",
  "2": "outputs/dynamic_events_v6/block_intervals.csv",
  "3": "outputs/dynamic_events_v6/external_greece/summary_by_sample_size.csv",
  "4": "outputs/dynamic_events_v6/weather_mechanism/risk_by_horizon.csv",
  "5": "outputs/dynamic_events_v6/weather_mechanism/time_shift_placebo.csv",
  "6": "outputs/dynamic_events_v6/weather_mechanism/mechanism_chain_recovery_effects.csv",
  "7": "outputs/real_decision_v9_signed/forecast_summary.csv",
  "8": "outputs/real_decision_v9_signed/economic_summary.csv",
  "9": "outputs/detection_benchmark_v9/protocol_ablation_metrics.csv",
  "10": "outputs/annotation_v9/sampling_support.csv",
  "11": "outputs/user_labels_v9/window_detector_metrics.csv",
  "12": "outputs/cv_benchmark_v7_100_complete/cv_benchmark_metrics.csv",
  "13": "outputs/sdwpf_v7/learned_forward_metrics.csv",
  "14": "outputs/matching_coverage_v12/pair_coverage_summary.csv",
  "15": "outputs/weather_uncertainty_v12/risk_block_intervals.csv",
  "16": "outputs/event_matching_v14/event_matching_summary.csv",
  "17": "outputs/yandun_sampling_v14/anchor_detection_summary.csv",
  "18": "outputs/weight_sensitivity_v13/weight_sensitivity.csv",
  "19": "outputs/weather_adjusted_v14/adjusted_weather_association.json",
  "20": "outputs/external_local_v13/greece_local_raw25_pairs.csv",
  "21": "outputs/external_transfer_v14/summary.csv",
  "22": "outputs/detection_benchmark_v9_100/metrics.csv",
  "23": "outputs/storage_policy_v15/capacity_price_surface.csv",
  "24": "outputs/storage_policy_v15/validation_selected_designs.csv",
  "25": "outputs/storage_policy_v15/oracle_bounds.csv",
  "26": "outputs/storage_policy_v15/conditional_advantage.csv",
  "27": "outputs/detection_benchmark_v9_100/compute_cost_summary.csv",
  "28": "outputs/detection_hpo_v16/selected_summary.csv",
  "29": "outputs/detection_confirmation_v16/metrics.csv",
  "30": "outputs/detection_confirmation_v16/paired_gain_intervals.csv",
  "31": "outputs/detection_confirmation_v16/compute.csv",
  "32": "outputs/matched_composition_v16/site_composition_summary.csv",
  "33": "outputs/conditional_agreement_v17/paper_summary.csv",
  "34": "outputs/oracle_mean_rule_v17/oracle_summary.csv",
  "35": "outputs/noise_calibration_v17/summary.csv",
  "36": "outputs/noise_calibration_v17/paired_intervals.csv",
  "37": "outputs/storage_audit_v17/paired_mse_event_weighted.csv",
  "38": "outputs/storage_audit_v17/selected_test_cost_decomposition.csv",
  "39": "outputs/storage_audit_v17/real_trace_costs.csv",
  "40": "outputs/weather_diagnostics_v17/diagnostics.json",
  "41": "outputs/detection_hpo_v16/summary.csv",
  "42": "outputs/oracle_mean_rule_v17/all_rule_candidates.csv",
  "43": "outputs/yandun_resolution_v17/resolution_pair_summary.csv"
}


def table(title, headers, rows):
    number = re.search(r"Table S(\d+)\.", title)[1]
    assert len(rows) > 0, title
    source = ROOT / TABLE_SOURCES[number]
    TABLE_LOG.append({"table": int(number), "rows": len(rows), "source": str(source.relative_to(ROOT)),
                      "source_bytes": source.stat().st_size, "status": "GENERATED"})
    escape = lambda x: str(x).replace("|", "/").replace("\n", " ")
    return "\n\n### " + title + "\n\n" + "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(escape(v) for v in row) + " |" for row in rows) + "\n"


def supplement_tables():
    text = "\n\n## Supplementary result tables\n\nThe tables present the evaluation populations and measures defined in Supplementary Methods S1–S12. Calendar-block intervals summarize temporal variation within each farm. Seed and batch summaries describe their stated computational units.\n"
    def detector_name(value, site=None):
        if site == 'greece':
            for lag, minutes in [('1h', '20 min'), ('2h', '40 min'), ('4h', '80 min')]:
                value = value.replace('_' + lag, '_' + minutes)
        return (value.replace("endpoint_corridor", "Corridor")
                     .replace("adaptive_corridor", "Adaptive corridor")
                     .replace("financial_tail", "Tail")
                     .replace("mean_shift", "Mean shift")
                     .replace("threshold", "Threshold")
                     .replace("opsda_cui2015", "OpSDA")
                     .replace("sda_florita2013", "SDA")
                     .replace("_", " "))
    f3 = lambda v: f"{float(v):.3f}"
    names = {"hill": "Hill of Towie", "lahaute": "La Haute Borne", "pizhou": "Pizhou", "suining": "Suining", "yandun": "Yandun", "greece": "Greece"}
    rows = read_rows("representation_summary_primary.csv")
    text += table("Table S1. Primary representation comparison (k = 4)", ["Site/group", "Representation", "NMI", "ARI", "Agreement"], [[names.get(r["group"], r["group"]), r["representation"].replace("_", " "), f3(r["nmi"]), f3(r["ari"]), f3(r["agreement"])] for r in rows])
    rows = read_rows("block_intervals.csv")
    text += table("Table S2. Raw25 block-length sensitivity", ["Site/group", "Days", "Occupied blocks", "Paired records", "NMI [95% interval]"], [[names.get(r["group"], r["group"]), r["block_days"], r["occupied_blocks"], r["pairs"], f'{f3(r["nmi"])} [{f3(r["nmi_low"])}, {f3(r["nmi_high"])}]'] for r in rows if r["representation"] == "raw25"])
    rows = read_rows("external_greece/summary_by_sample_size.csv")
    text += table("Table S3. Greek configuration-pair support strata", ["Matches per pair", "Configuration pairs", "Median NMI", "Minimum", "Maximum"], [[r["sample_bin"], r["configuration_pairs"], f3(r["median_nmi"]), f3(r["min_nmi"]), f3(r["max_nmi"])] for r in rows])
    rows = read_rows("weather_mechanism/risk_by_horizon.csv") + read_rows("weather_mechanism/noaa_ground_proxy_risk.csv")
    text += table("Table S4. Descriptive weather contrasts", ["Source", "Horizon (h)", "Exposed n", "Control n", "Risk exposed", "Risk control", "Difference"], [["NOAA proxy" if r["exposure"].startswith("noaa") else "ERA5", r["horizon_h"], r["treated_n"], r["control_n"], f3(r["treated_risk"]), f3(r["control_risk"]), f3(r["risk_difference"])] for r in rows])
    rows = read_rows("weather_mechanism/time_shift_placebo.csv")
    text += table("Table S5. Archived time-displacement diagnostics (4 h outcome)", ["Displacement", "Exposed n", "Control n", "Risk difference"], [[r["exposure"].replace("_treatment", ""), r["treated_n"], r["control_n"], f3(r["risk_difference"])] for r in rows])
    rows = read_rows("weather_mechanism/mechanism_chain_recovery_effects.csv")
    text += table("Table S6. Scaled post-event power index and residual-power contrasts", ["Measure", "Events", "Exposed minus control", "95% block interval"], [[{"recovery_fraction_4h": "Scaled post-event power index", "residual_4h": "Residual power", "response_abs": "Absolute power response"}.get(r["metric"], r["metric"].replace("_", " ")), r["n"], f3(r["difference_exposed_minus_unexposed"]), f'[{f3(r["ci95_low"])}, {f3(r["ci95_high"])}]'] for r in rows])
    decision = ROOT / "outputs/real_decision_v9_signed"
    if (decision / "forecast_summary.csv").exists():
        rows = list(csv.DictReader((decision / "forecast_summary.csv").open(newline="")))
        text += table("Table S7. Forecasts on common chronological targets", ["Site", "Model", "Training", "Subset", "n", "nMAE (%)"], [[r["site"], r["model"], r["training"].replace("_", " "), r["subset"].replace("_", " "), r["n"], f3(r["nmae"])] for r in rows])
    if (decision / "economic_summary.csv").exists():
        rows = list(csv.DictReader((decision / "economic_summary.csv").open(newline="")))
        text += table("Table S8. Storage costs under the legacy annual-charge assumptions", ["Site", "Model", "Training", "Scenario", "P (MW/MW)", "E (MWh/MW)", "Cost (CNY per norm. MW)"], [[r["site"], r["model"], r["training"].replace("_", " "), r["scenario"], f3(r["power"]), f3(r["energy"]), f3(r["cost"])] for r in rows])
    controlled = ROOT / "outputs/detection_benchmark_v9/protocol_ablation_metrics.csv"
    if controlled.exists():
        rows = list(csv.DictReader(controlled.open(newline="")))
        rows = [r for r in rows if r["split"] == "test"]
        text += table("Table S9. Controlled synthetic event detection (IoU >= 0.3)", ["Model", "Seed", "Policy", "Precision", "Recall", "F1"], [[r["model"], r["seed"], r["policy"].replace("_", " "), f3(r["precision"]), f3(r["recall"]), f3(r["f1"])] for r in rows])
    annotation = ROOT / "outputs/annotation_v9/sampling_support.csv"
    if annotation.exists():
        rows = list(csv.DictReader(annotation.open(newline="")))
        text += table("Table S10. Human-review sampling support", ["Site", "Sampling stratum", "Windows"], [[r["site"], r["stratum"].replace("_", " "), r["actual"]] for r in rows])
    human_metrics = ROOT / "outputs/user_labels_v9/window_detector_metrics.csv"
    if human_metrics.exists():
        rows = list(csv.DictReader(human_metrics.open(newline="")))
        text += table("Table S11. Human-labelled centre-region detector agreement", ["Site", "Detector", "Windows", "TP", "FP", "FN", "Precision", "Recall", "F1"], [[r["site"], detector_name(r["config"], r["site"]), r["n_windows"], r["tp"], r["fp"], r["fn"], f3(r["precision"]), f3(r["recall"]), f3(r["f1"])] for r in rows])
    learned = pd.read_csv(ROOT / "outputs/cv_benchmark_v7_100_complete/cv_benchmark_metrics.csv")
    farm_groups = ["pairs:" + name for name in ["pizhou", "suining", "yandun", "lahaute", "hill"]]
    learned = learned[(learned.k == 4) & learned.group.isin(farm_groups)]
    text += table("Table S12. Learned representations by farm, averaged over three seeds (k = 4)",
                  ["Representation", "Farm", "NMI", "ARI"],
                  [[rep.replace("_", " "), names.get(group.replace("pairs:", ""), group),
                    f3(part.nmi.mean()), f3(part.ari.mean())]
                   for (rep, group), part in learned.groupby(["representation", "group"])])
    weighted = pd.read_csv(ROOT / "outputs/matching_coverage_v12/coverage_weighted_agreement.csv")
    iou = pd.read_csv(ROOT / "outputs/matching_coverage_v12/iou_threshold_sensitivity.csv")
    pair = pd.read_csv(ROOT / "outputs/matching_coverage_v12/pair_coverage_summary.csv")
    comp_rows = [[names.get(r["group"], r["group"]), f3(r["pair_weighted_nmi"]), f3(r["pair_weighted_ari"]), str(int(r["total_pairs"])),
                  f3(pair[pair.site == r["group"]].left_coverage.mean()), f3(pair[pair.site == r["group"]].right_coverage.mean()),
                  f3(iou[iou.site == r["group"]].fraction_iou_ge_030.iloc[0]),
                  f3(iou[iou.site == r["group"]].fraction_iou_ge_050.iloc[0]),
                  f3(iou[iou.site == r["group"]].fraction_iou_ge_070.iloc[0])]
                 for _, r in weighted.iterrows()]
    text += table("Table S14. Matched-count-weighted agreement and detector-occupancy overlap",
                  ["Site", "Weighted NMI", "Weighted ARI", "Pairs", "Left cov.", "Right cov.", "IoU >= 0.3", "IoU >= 0.5", "IoU >= 0.7"], comp_rows)
    external = pd.read_csv(ROOT / "outputs/sdwpf_v7/learned_forward_metrics.csv")
    text += table("Table S13. SDWPF learned-model transfer across seven batches and three seeds",
                  ["Representation", "k", "Runs", "Mean NMI", "SD NMI", "Mean ARI"],
                  [[rep, k, len(part), f3(part.nmi.mean()), f3(part.nmi.std()), f3(part.ari.mean())]
                   for (rep, k), part in external.groupby(["representation", "k"])])
    rematch = pd.read_csv(ROOT / "outputs/event_matching_v14/event_matching_summary.csv")
    text += table("Table S16. Event-level IoU and matching-strategy sensitivity",
                  ["Site", "IoU", "Matcher", "Pairs", "NMI", "ARI", "W. NMI", "W. ARI", "Left cov.", "Right cov."],
                  [[names.get(r.site, r.site), f"{r.cutoff:.1f}", {"greedy": "Greedy", "maximum_iou": "Max. IoU"}[r.method],
                    str(int(r.pairs)), f3(r.nmi), f3(r.ari), f3(r.weighted_nmi), f3(r.weighted_ari), f3(r.left_coverage), f3(r.right_coverage)]
                   for r in rematch.itertuples()])
    text += "\nMax. IoU denotes maximum-total-IoU assignment. NMI and ARI are equal-configuration-pair means over pairs with matched events; W. NMI and W. ARI are weighted by matched-event counts. Left and right coverage are mean matched fractions of the respective detector catalogues across all configuration pairs.\n"
    weather = pd.read_csv(ROOT / "outputs/weather_uncertainty_v12/risk_block_intervals.csv")
    text += table("Table S15. Observational weather contrasts with shared calendar-block intervals",
                  ["Source", "Horizon (h)", "Block days", "Blocks", "Risk difference", "95% interval"],
                  [[r.source, r.horizon_h, r.block_days, r.occupied_blocks, f3(r.risk_difference),
                    f"[{f3(r.ci95_low)}, {f3(r.ci95_high)}]"] for r in weather.itertuples()])
    budget = pd.read_csv(ROOT / "outputs/detection_benchmark_v9_100/metrics.csv")
    budget = budget[(budget.split == "test") & budget.iou_cutoff.eq(0.3)]
    bsum = budget.groupby(["model", "protocol"])[["precision", "recall", "f1", "optimizer_steps"]].mean().reset_index()
    text += table("Table S22. 100-epoch upper-budget controlled detection sensitivity",
                  ["Model", "Protocol", "Precision", "Recall", "F1", "Optimizer steps"],
                  [[r.model, r.protocol, f3(r.precision), f3(r.recall), f3(r.f1), f3(r.optimizer_steps)]
                   for r in bsum.itertuples()])
    adjusted = json.loads((ROOT / "outputs/weather_adjusted_v14/adjusted_weather_association.json").read_text(encoding="utf8"))
    assert all(math.isfinite(v) for key in ["standardized_rd_ci95", "odds_ratio_ci95"] for v in adjusted[key])
    assert all(len(adjusted[key]) == 2 for key in ["standardized_rd_ci95", "odds_ratio_ci95"])
    text += table("Table S19. Adjusted weather association model",
                  ["Records", "Blocks", "Risk difference", "95% interval", "Odds ratio", "OR interval"],
                  [[adjusted["rows"], adjusted["blocks"], f3(adjusted["standardized_risk_difference"]),
                    f"[{f3(adjusted["standardized_rd_ci95"][0])}, {f3(adjusted["standardized_rd_ci95"][1])}]",
                    f3(adjusted["odds_ratio"]), f"[{f3(adjusted["odds_ratio_ci95"][0])}, {f3(adjusted["odds_ratio_ci95"][1])}]"]])
    local = json.loads((ROOT / "outputs/external_local_v13/manifest.json").read_text(encoding="utf8"))
    frozen = pd.read_csv(ROOT / "outputs/dynamic_events_v6/external_greece/representation_summary.csv")
    text += table("Table S20. Greek local-training and frozen-transfer comparison",
                  ["Training population", "Config. pairs", "NMI", "ARI", "Agreement"],
                  [["Greek local", local["configuration_pairs"], f3(local["mean_nmi"]), f3(local["mean_ari"]), f3(local["mean_agreement"])],
                   ["Pizhou frozen", len(frozen), f3(frozen.nmi.mean()), f3(frozen.ari.mean()), f3(frozen.agreement.mean())]])
    fine = pd.read_csv(ROOT / "outputs/external_transfer_v14/summary.csv")
    text += table("Table S21. Same-encoder Greek frozen and adapted TCN comparison",
                  ["Mode", "Pairs", "Equal NMI", "Equal ARI", "Pooled NMI", "Pooled ARI", "Weighted NMI", "Weighted ARI"],
                  [[r.mode.replace("_", " "), r.event_pairs, f3(r.equal_pair_nmi),
                    f3(r.equal_pair_ari), f3(r.pooled_nmi), f3(r.pooled_ari), f3(r.weighted_nmi), f3(r.weighted_ari)] for r in fine.itertuples()])
    sampling = pd.read_csv(ROOT / "outputs/yandun_sampling_v14/anchor_detection_summary.csv")
    text += table("Table S17. Yandun physical-horizon and sampling sensitivity",
                  ["Minutes", "Horizon (h)", "Clock", "Eligible anchors", "Positive anchors", "Positive rate"],
                  [[r.minutes, r.horizon_h, r.clock.replace("_", " "), r.eligible_anchors,
                    r.positive_anchors, f3(r.positive_rate)] for r in sampling.itertuples()])
    weight = pd.read_csv(ROOT / "outputs/weight_sensitivity_v13/weight_sensitivity.csv")
    weight_test = weight[weight.split.eq("test")]
    wsummary = weight_test.groupby(["site", "factor"])[["all_nmae_pct", "ramp_nmae_pct", "ramp_n"]].mean().reset_index()
    econ = weight[weight.split.str.startswith("economics_base")].groupby(["site", "factor"])[["test_cost", "selected_power"]].mean().reset_index()
    wsummary = wsummary.merge(econ, on=["site", "factor"])
    text += table("Table S18. Event-weight sensitivity under the legacy annual-charge assumptions",
                  ["Site", "Factor", "Ramp weight", "All nMAE (%)", "Ramp nMAE (%)", "Base cost"],
                  [[r.site, r.factor, r.factor + 1, f3(r.all_nmae_pct), f3(r.ramp_nmae_pct), f3(r.test_cost)] for r in wsummary.itertuples()])
    policy = pd.read_csv(ROOT / "outputs/storage_policy_v15/capacity_price_surface.csv")
    fixed = policy[policy.storage_fraction.eq(.1) & policy.duration_h.eq(2)].copy()
    rates = fixed.groupby(["site", "model", "price_case", "tail_premium", "training"]).total_cny_per_installed_mw.mean().unstack("training").reset_index()
    hours = fixed.groupby(["site", "model", "price_case", "tail_premium"]).calendar_h.first().reset_index()
    rates = rates.merge(hours, on=["site", "model", "price_case", "tail_premium"])
    text += table("Table S23. Fixed 10%-power and 2-hour storage costs normalized by test-calendar hours",
                  ["Site", "Model", "Prices", "Tail premium", "Calendar h", "MSE CNY/MW/h", "Event-weighted CNY/MW/h", "Difference"],
                  [[r.site, r.model, r.price_case, r.tail_premium, f3(r.calendar_h), f3(r.mse/r.calendar_h),
                    f3(r.event_weighted/r.calendar_h), f3((r.event_weighted-r.mse)/r.calendar_h)] for r in rates.itertuples()])
    text += "\nTable S23 compares training objectives within each site. Calendar-hour normalization makes Pizhou and Yandun exposure lengths explicit; all estimated contrasts are within-site.\n"
    designs = pd.read_csv(ROOT / "outputs/storage_policy_v15/validation_selected_designs.csv")
    counts = designs.groupby(["regime", "fraction", "duration_h"]).size().reset_index(name="configurations")
    text += table("Table S24. Validation-selected storage designs across model-seed-price configurations",
                  ["Design regime", "Power fraction", "Duration (h)", "Configurations"],
                  [[r.regime.replace("_", " "), r.fraction, r.duration_h, r.configurations] for r in counts.itertuples()])
    bounds = pd.read_csv(ROOT / "outputs/storage_policy_v15/oracle_bounds.csv").groupby(["site", "model", "training", "case"]).mean(numeric_only=True).reset_index()
    text += table("Table S25. Perfect-information and feasible online operating costs per installed MW",
                  ["Site", "Model", "Training", "Prices", "Oracle bound", "Online control"],
                  [[r.site, r.model, r.training.replace("_", " "), r.case,
                    f3(r.oracle_operating_cny_per_installed_mw), f3(r.online_operating_cny_per_installed_mw)] for r in bounds.itertuples()])
    conditional = pd.read_csv(ROOT / "outputs/storage_policy_v15/conditional_advantage.csv")
    conditional = conditional[conditional.mean_difference < 0].sort_values("mean_difference")
    text += table("Table S26. Exploratory test-grid cells with lower event-weighted storage cost",
                  ["Site", "Model", "Prices", "Tail premium", "Power fraction", "Duration (h)", "Mean difference", "Seeds better"],
                  [[r.site, r.model, r.price_case, r.tail_premium, r.storage_fraction, r.duration_h,
                    f3(r.mean_difference), f"{int(r.better_seeds)}/{int(r.seeds)}"] for r in conditional.itertuples()])
    text += "\nMean difference is event-weighted minus MSE total cost in CNY per installed MW, averaged over three seeds. These cells are a post-hoc diagnostic over the test grid; validation-selected policy comparisons appear in Table S37.\n"
    compute = pd.read_csv(ROOT / "outputs/detection_benchmark_v9_100/compute_cost_summary.csv")
    compute_labels = {"mean_rule": "Mean rule", "timesnet": "TimesNet", "kanad": "KAN-AD",
                      "tcn_ae": "TCN-AE", "transformer_ae": "Transformer-AE"}
    text += table("Table S27. Controlled detection compute budget and model size",
                  ["Model", "Seeds", "Parameters", "Max steps", "Max epochs", "Test n", "CKPT bytes"],
                  [[compute_labels.get(r.model, r.model), r.seeds, r.parameters, r.train_optimizer_steps, r.train_epochs, r.test_sequences, r.checkpoint_bytes]
                   for r in compute.itertuples()])
    text += "\nOptimizer steps, epochs and checkpoint sizes are recorded from the upper-budget benchmark; the mean-change rule is analytic and has no fitted parameters.\n"
    hpo = pd.read_csv(ROOT / "outputs/detection_hpo_v16/selected_summary.csv")
    hpo_names = {"h08_z08": "size 8", "h16_z16": "size 16", "sequence_h08_z08": "width 8, z 8", "mean_w4": "window 4"}
    text += table("Table S28. Validation-selected detection hyperparameters and held-out performance",
                  ["Model", "Selected size", "Validation F1", "Test F1 (SD)", "Precision", "Recall", "Delay", "Seeds"],
                  [["Trans.-AE" if r.model == "transformer_ae" else compute_labels.get(r.model, r.model), hpo_names.get(r.selected_config, r.selected_config), f3(r.validation_f1_mean),
                    f'{f3(r.test_f1_mean)} ({f3(r.test_f1_sd)})' if pd.notna(r.test_f1_sd) else f3(r.test_f1_mean),
                    f3(r.test_precision_mean), f3(r.test_recall_mean), f3(r.test_delay_mean_steps), r.seeds]
                   for r in hpo.itertuples()])
    text += "\nTrans.-AE denotes the positional Transformer autoencoder. Size denotes TimesNet d_model or KAN-AD Fourier order; width and z are encoder width and latent size for the sequence autoencoders. SD is across training seeds. Model selection uses mean validation F1; the simple rule selects its mean window on the same validation population.\n"
    confirmation = pd.read_csv(ROOT / "outputs/detection_confirmation_v16/metrics.csv")
    confirm_rows = []
    for (condition, model), group in confirmation.groupby(["condition", "model"]):
        value = f'{f3(group.f1.mean())} ({f3(group.f1.std())})' if len(group) > 1 else f3(group.f1.iloc[0])
        confirm_rows.append([condition.replace('_', ' '), compute_labels[model], len(group), value,
                             f3(group.precision.mean()), f3(group.recall.mean()), f3(group.mean_delay_steps.mean())])
    text += table("Table S29. Fresh post-selection synthetic confirmation", ["Condition", "Model", "Seeds", "F1 (SD)", "Precision", "Recall", "Delay"], confirm_rows)
    text += "\nEach condition contains 1,600 fresh sequences, including 800 episodes. Delay is in generator steps, and SD summarizes the three frozen training seeds.\n"
    gains = pd.read_csv(ROOT / "outputs/detection_confirmation_v16/paired_gain_intervals.csv")
    text += table("Table S30. Paired F1 differences from the validation-selected mean rule",
                  ["Condition", "Model", "F1 difference", "95% interval"],
                  [[r.condition.replace('_', ' '), compute_labels[r.model], f3(r.mean_f1_gain_vs_mean_rule),
                    f'[{f3(r.ci95_low)}, {f3(r.ci95_high)}]'] for r in gains.itertuples()])
    text += "\nIntervals use 2,000 shared resamples of complete synthetic sequences, conditional on the frozen trained models. They characterize sampling uncertainty separately from between-seed variation.\n"
    runtime = pd.read_csv(ROOT / "outputs/detection_confirmation_v16/compute.csv")
    text += table("Table S31. CPU score-generation cost of the selected configurations",
                  ["Model", "Parameters", "Median ms/sequence", "Range", "Batch"],
                  [[compute_labels[r.model], r.parameters, f'{r.cpu_ms_per_sequence_median:.6f}',
                    f'[{r.cpu_ms_per_sequence_min:.6f}, {r.cpu_ms_per_sequence_max:.6f}]', r.batch_size]
                   for r in runtime.itertuples()])
    text += "\nTiming uses an AMD Ryzen 7 9700X under WSL2, two CPU threads, one warm-up and five runs of 800 sequences. Normalization and score generation are timed; interval postprocessing is excluded. The simple rule uses a NumPy cumulative-sum implementation.\n"
    composition = pd.read_csv(ROOT / "outputs/matched_composition_v16/site_composition_summary.csv")
    feature_labels = {"duration_hours": "Duration (h)", "amplitude": "Signed change", "absolute_amplitude": "Abs. change",
                      "power_range": "Power range", "power_start": "Start power", "upward_fraction": "Upward share",
                      "wind_prewindow_ms": "Wind (m/s)"}
    text += table("Table S32. Matched and unmatched composition within detector-pair sides",
                  ["Site", "Feature", "Sides used / all", "Matched mean", "Unmatched mean", "Difference"],
                  [[names[r.site], feature_labels[r.feature], f'{r.supported_pair_sides}/{r.pair_sides}',
                    f3(r.equal_pair_mean_matched), f3(r.equal_pair_mean_unmatched), f3(r.equal_pair_mean_difference)]
                   for r in composition.itertuples()])
    text += "\nMeans give equal weight to supported detector-pair sides with available feature values in both populations. Power and amplitude use the archived normalization; direction is the fraction of upward events. Earlier wind averages four complete half-hour bins strictly before event start. The detailed companion CSV includes population sizes, valid wind counts and quartiles for every side.\n"
    conditional = pd.read_csv(ROOT / "outputs/conditional_agreement_v17/paper_summary.csv")
    conditional = conditional[conditional.representation.eq("raw25")]
    text += table("Table S33. Conditional raw25 agreement and retained support",
                  ["Site", "Conditioning", "Retained / original", "Catalogue cov. L", "Catalogue cov. R", "Scored fraction", "W. NMI", "W. ARI"],
                  [[names[r.site], r.conditioning, f"{r.retained_pairs}/{r.original_matched_pairs}", f3(r.conditional_left_coverage),
                    f3(r.conditional_right_coverage), f3(r.scored_fraction_of_retained), f3(r.pair_weighted_nmi_n_ge_30),
                    f3(r.pair_weighted_ari_n_ge_30)] for r in conditional.itertuples()])
    text += "\nConditional raw25 scores are computed within detector-pair strata. Direction retains most temporal matches and gives lower information agreement. Amplitude and duration conditioning retain higher within-stratum agreement at lower support. Both-constant strata are excluded from informative scores, and one-side-constant strata contribute zero. Scored fractions refer to retained pairs in defined strata with at least 30 matches.\n"
    oracle = pd.read_csv(ROOT / "outputs/oracle_mean_rule_v17/oracle_summary.csv")
    text += table("Table S34. Test-label oracle ceilings for adjacent-mean windows",
                  ["Condition", "Window grid", "Oracle mode", "Window", "F1", "Label access"],
                  [[r.condition.replace("_", " "), r.grid.replace("_", " "), r.mode.replace("_", " "), r.window, f3(r.f1), "test-label oracle"] for r in oracle.itertuples()])
    noise = pd.read_csv(ROOT / "outputs/noise_calibration_v17/summary.csv")
    text += table("Table S35. Equal-validation noise calibration on a fresh high-noise draw",
                  ["Model", "Policy", "F1", "F1 SD", "Precision", "Recall", "Seeds"],
                  [[compute_labels[r.model], r.policy.replace("_", " "), f3(r.f1), f3(r.f1_sd) if pd.notna(r.f1_sd) else "-", f3(r.precision), f3(r.recall), r.seeds] for r in noise.itertuples()])
    noise_gains = pd.read_csv(ROOT / "outputs/noise_calibration_v17/paired_intervals.csv")
    text += table("Table S36. Paired gains after source or target-noise calibration",
                  ["Model", "Policy", "F1 gain", "95% interval"],
                  [[compute_labels[r.model], r.policy.replace("_", " "), f3(r.f1_gain_vs_mean_rule), f"[{f3(r.ci95_low)}, {f3(r.ci95_high)}]"] for r in noise_gains.itertuples()])
    audit = pd.read_csv(ROOT / "outputs/storage_audit_v17/paired_mse_event_weighted.csv")
    decomp = pd.read_csv(ROOT / "outputs/storage_audit_v17/selected_test_cost_decomposition.csv")
    dmean = decomp.groupby(["site", "model", "price_case", "tail_premium", "regime", "training"]).total_cny_per_installed_mw_h.mean().reset_index()
    piv = dmean.pivot_table(index=["site", "model", "price_case", "tail_premium", "regime"], columns="training", values="total_cny_per_installed_mw_h").reset_index()
    pairs_count = audit.groupby(["site", "model", "price_case", "tail_premium", "regime"]).agg(seeds=("seed", "size"), lower_event_weighted=("event_weighted_minus_mse_cny_per_installed_mw", lambda x: int((x < 0).sum()))).reset_index()
    policy_summary = piv.merge(pairs_count, on=["site", "model", "price_case", "tail_premium", "regime"])
    text += table("Table S37. Validation-selected policy cost comparison",
                  ["Site", "Model", "Prices", "Tail", "Regime", "MSE CNY/MW/h", "Event-weighted CNY/MW/h", "Difference", "Lower EW / seeds"],
                  [[r.site, r.model, r.price_case, r.tail_premium, r.regime.replace("_", " "), f3(r.mse), f3(r.event_weighted), f3(r.event_weighted-r.mse), f"{r.lower_event_weighted}/{r.seeds}"] for r in policy_summary.itertuples()])
    representative = decomp[(decomp.model.eq("tcn")) & (decomp.seed.eq(41)) & decomp.regime.eq("at_least_10pct_2h") & decomp.price_case.isin(["low", "base", "high"]) & decomp.tail_premium.isin([0., 5000.])]
    text += table("Table S38. Representative validation-selected cost decomposition",
                  ["Site", "Training", "Prices", "Tail", "Capital", "Throughput", "Short", "Surplus", "Tail cost", "Inventory", "Total"],
                  [[r.site, r.training.replace("_", " "), r.price_case, r.tail_premium, f3(r.capital_cny_per_installed_mw_h), f3(r.throughput_cny_per_installed_mw_h), f3(r.shortfall_cny_per_installed_mw_h), f3(r.surplus_cny_per_installed_mw_h), f3(r.tail_premium_cny_per_installed_mw_h), f3(r.inventory_cny_per_installed_mw_h), f3(r.total_cny_per_installed_mw_h)] for r in representative.itertuples()])
    traces = pd.read_csv(ROOT / "outputs/storage_audit_v17/real_trace_costs.csv")
    traces["online_rate"] = traces.online_total_cny_per_installed_mw / traces.calendar_h
    traces["lp_rate"] = traces.lp_total_cny_per_installed_mw / traces.calendar_h
    text += table("Table S39. Real fixed 10%-power/2-hour online and LP traces",
                  ["Site", "Training", "Scenario", "Calendar h", "Online CNY/MW/h", "LP CNY/MW/h", "LP minus online", "Online severe steps", "LP severe steps"],
                  [[r.site, r.training.replace("_", " "), r.scenario, f3(r.calendar_h), f3(r.online_rate), f3(r.lp_rate), f3(r.lp_rate-r.online_rate), r.online_severe_steps, r.lp_severe_steps] for r in traces.itertuples()])
    diag = json.loads((ROOT / "outputs/weather_diagnostics_v17/diagnostics.json").read_text(encoding="utf8"))
    coefficient = pd.read_csv(ROOT / "outputs/weather_diagnostics_v17/coefficients.csv")
    exposure = coefficient[coefficient.term.eq("a")].iloc[0]
    text += table("Table S40. Weather association model diagnostics",
                  ["Observations", "Blocks", "IRLS iterations", "Converged", "Exposure estimate", "Cluster SE", "Block-t p", "BH q"],
                  [[diag["observations"], diag["blocks"], diag["iterations"], diag["converged"], f3(exposure.estimate), f3(exposure.cluster_se), f3(exposure.block_t_pvalue), f3(exposure.bh_qvalue_block_t)]])
    all_hpo = pd.read_csv(ROOT / "outputs/detection_hpo_v16/summary.csv")
    all_hpo = all_hpo.merge(pd.read_csv(ROOT / "outputs/detection_hpo_v16/validation_summary.csv"), on=["model", "config"], validate="one_to_one")
    text += table("Table S41. All neural model-size validation and test curves",
                  ["Model", "Configuration", "Validation F1", "Validation SD", "Test F1", "Test SD"],
                  [[compute_labels.get(r.model, r.model), r.config, f3(r.validation_f1_mean), f3(r.validation_f1_sd), f3(r.f1), f3(r.f1_sd)] for r in all_hpo.itertuples()])
    all_oracle = pd.read_csv(ROOT / "outputs/oracle_mean_rule_v17/all_rule_candidates.csv")
    text += table("Table S42. All adjacent-mean candidate curves",
                  ["Condition", "Window", "Validation F1", "Frozen test F1", "Oracle test F1"],
                  [[r.condition.replace("_", " "), r.window, f3(r.validation_f1), f3(r.frozen_test_f1), f3(r.oracle_test_f1)] for r in all_oracle.itertuples()])
    text += "\nAll candidate rows in S41 and S42 remain available in CSV form with their full selection settings. Oracle rows use test labels and provide an upper-bound diagnostic. Deployable selection uses validation labels. S37 uses validation-selected storage designs; S26 remains a post-hoc test-grid diagnostic. Cost components are in CNY per installed MW per test-calendar hour; unrounded values remain in the source CSV.\n"
    resolution = pd.read_csv(ROOT / "outputs/yandun_resolution_v17/resolution_pair_summary.csv")
    text += table("Table S43. Yandun raw25 agreement across native and aggregated resolutions",
                  ["Left min", "Right min", "Left test", "Right test", "Pairs", "Left cov.", "Right cov.", "NMI", "ARI"],
                  [[r.left_minutes, r.right_minutes, r.left_test_events, r.right_test_events, r.matched_pairs,
                    f3(r.left_coverage), f3(r.right_coverage), f3(r.nmi), f3(r.ari)] for r in resolution.itertuples()])
    text += "\nEach resolution fits its own training standardizer and k=4 prototypes. Test intervals match within turbine and split at IoU 0.5. The source is Yandun's native 15-min SCADA; 30- and 60-min series use complete arithmetic means and preserve missing-bin boundaries.\n"
    assert sorted(row["table"] for row in TABLE_LOG) == list(range(1, 44))
    (OUT / "supplementary_table_manifest.json").write_text(json.dumps(sorted(TABLE_LOG, key=lambda row: row["table"]), indent=2), encoding="utf8")
    first = text.index("\n\n### Table S")
    parts = re.split(r"(?=\n\n### Table S\d+\.)", text[first:])
    parts = [part for part in parts if part.strip()]
    return text[:first] + "".join(sorted(parts, key=lambda part: int(re.search(r"Table S(\d+)", part)[1])))


PREAMBLE = r"""\documentclass[preprint,12pt]{elsarticle}
\usepackage[a4paper,left=22mm,right=22mm,top=22mm,bottom=22mm]{geometry}
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\setsansfont{TeX Gyre Heros}
\usepackage{amsmath,amssymb,booktabs,longtable,array,calc,graphicx}
\usepackage{float}
\usepackage{needspace}
\usepackage{xurl}
\let\OriginalPath\path
\renewcommand{\path}[1]{\mbox{\OriginalPath{#1}}}
\usepackage{fvextra}
\DefineVerbatimEnvironment{verbatim}{Verbatim}{breaklines=true,fontsize=\footnotesize}
\usepackage[colorlinks=true,allcolors=black,breaklinks=true]{hyperref}
\usepackage{lineno}
\modulolinenumbers[5]
\biboptions{sort&compress}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
\newcounter{none}
\setlength{\emergencystretch}{3em}
\sloppy
\journal{Applied Energy}
\begin{document}
"""


def compile_tex(name):
    subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", f"-outdir={BUILD}", name + ".tex"], cwd=SOURCES, check=True, stdout=(BUILD / f"{name}_build_stdout.txt").open("w"), stderr=subprocess.STDOUT)
    shutil.copy2(BUILD / f"{name}.pdf", OUT / f"{name}.pdf")
    bbl = BUILD / f"{name}.bbl"
    if bbl.exists():
        shutil.copy2(bbl, SOURCES / bbl.name)


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir(parents=True, exist_ok=True)
    TABLE_LOG.clear()
    meta = json.loads((OUT / "frontmatter.json").read_text(encoding="utf8"))
    assert 3 <= len(meta["highlights"]) <= 5
    assert all(len(s) <= 85 for s in meta["highlights"])
    bundle = OUT / "official_template/elsarticle/elsarticle"
    for name in ["elsarticle.cls", "elsarticle-num.bst"]:
        shutil.copy2(bundle / name, SOURCES / name)
    shutil.copy2(OUT / "references.bib", SOURCES / "references.bib")
    for i in range(1, 9):
        shutil.copy2(OUT / f"figures/fig{i}.pdf", SOURCES / f"fig{i}.pdf")
    caption_text = (OUT / "figure_captions.md").read_text(encoding="utf8")
    captions = re.findall(r"^Fig\. (\d+)\. (.*?)(?=\n\nFig\.|\Z)", caption_text, flags=re.M | re.S)
    assert len(captions) == 8
    body = latex((OUT / "manuscript_body.md").read_text(encoding="utf8"))
    # Place each figure beside the result it explains.
    cap_map = {number: caption.strip() for number, caption in captions}
    def inline_fig(number, anchor):
        figure = (r"\setcounter{figure}{" + str(int(number)-1) + "}\n" + r"\begin{figure}[H]\centering" + "\n" +
                  r"\includegraphics[width=0.92\linewidth]{fig" + number + ".pdf}\n" +
                  r"\caption{" + latex(cap_map[number]) + "}\n" +
                  r"\label{fig:" + number + "}" + "\n" + r"\end{figure}" + "\n")
        nonlocal body
        if anchor not in body:
            raise RuntimeError("inline anchor missing: " + anchor)
        body = body.replace(anchor, anchor + "\n\n" + figure, 1)
    inline_fig("1", "The framework supports a compact reporting record for future ramp studies and gives flexibility analyses a traceable measurement layer.")
    inline_fig("2", "Together, support and coverage give the reader both the amount of comparative evidence and its reach within the catalogue (Figs. 1 and 2).")
    inline_fig("3", "Figure 3 displays the fixed-feature comparison.")
    inline_fig("4", "These comparisons describe sensitivity to grouping resolution (Fig. 4).")
    inline_fig("5", "The NMI describes retained shape grouping, and the coverage identifies the fraction of candidate support reached by temporal matching (Fig. 5).")
    inline_fig("6", "Each comparison describes its own observation coverage and atmospheric measurement scale.")
    inline_fig("7", "Figure 7 places these external estimates beside the primary learned-model comparison.")
    inline_fig("8", "A local mean-change rule reaches F1 0.741 on this generator.")
    body = body.replace("Table 1. Primary SCADA population and half-hour input eligibility.", "\\needspace{8\\baselineskip}\\textbf{Table 1.} Primary SCADA population and half-hour input eligibility.")
    body = body.replace(r"\begin{longtable}", r"\begingroup\small\setlength{\tabcolsep}{3pt}" + "\n" + r"\begin{longtable}")
    body = body.replace(r"\end{longtable}", r"\end{longtable}" + "\n" + r"\endgroup")
    body = re.sub(r"(?m)^Table (\d+)\. (.+)$",
                  lambda match: r"\needspace{9\baselineskip}\textbf{Table " + match[1] + ".} " + match[2], body)
    (SOURCES / "body.tex").write_text(body, encoding="utf8")
    author_lines = []
    for a in meta["authors"]:
        mark = a["superscripts"]
        corr = r"\corref{cor1}" if a.get("corresponding") else ""
        email = r"\ead{" + a["email"] + "}"
        orcid = r"\textsuperscript{ORCID: " + a["orcid"] + "}"
        author_lines.append(r"\author[aff" + mark.replace(",", ",aff") + "]{" + a["name"] + corr + " " + orcid + "}" + email)
    affiliation_lines = []
    for i, aff in enumerate(meta["affiliations"], start=1):
        affiliation_lines.append(r"\address[aff" + str(i) + "]{" + aff + "}")
    front = r"\begin{frontmatter}" + "\n" + r"\title{" + latex(meta["title"]) + r"}" + "\n" + "\n".join(author_lines + affiliation_lines) + "\n" + r"\cortext[cor1]{Correspondence: Chaoxia Yuan, chaoxia.yuan@nuist.edu.cn}" + "\n" + r"\begin{abstract}" + "\n" + latex(meta["abstract"]) + "\n" + r"\end{abstract}" + "\n" + r"\begin{keyword}" + "\n" + " \\sep ".join(latex(k) for k in meta["keywords"]) + "\n" + r"\end{keyword}\end{frontmatter}" + "\n"
    declarations = "\n"
    for heading, key in [
        ("Data availability", "data_availability"),
        ("Funding", "funding"),
        ("Author contributions", "author_contributions"),
        ("Acknowledgements", "acknowledgements"),
        ("Competing interests", "conflicts"),
    ]:
        declarations += "\\section*{" + heading + "}\n" + latex(meta[key]) + "\n"
    main_tex = PREAMBLE + front + "\\linenumbers\n\\input{body}\n" + declarations + "\n\\nolinenumbers\\bibliographystyle{elsarticle-num}\\bibliography{references}\n\\end{document}\n"
    (SOURCES / "main.tex").write_text(main_tex, encoding="utf8")
    supp_md = (OUT / "supplementary_methods.md").read_text(encoding="utf8") + supplement_tables()
    (OUT / "supplementary_complete.md").write_text(supp_md, encoding="utf8")
    supp_latex = latex(supp_md)
    for heading in ["section", "subsection", "subsubsection"]:
        supp_latex = supp_latex.replace("\\" + heading + "{", "\\" + heading + "*{")
    supp_latex = supp_latex.replace(
        r"\subsubsection*{Table S",
        r"\needspace{9\baselineskip}\subsubsection*{Table S",
    )
    supp_latex = supp_latex.replace(
        r"\needspace{9\baselineskip}\subsubsection*{Table S29.",
        r"\clearpage\subsubsection*{Table S29.",
    )
    # Give the detector field enough room while preserving readable numeric columns.
    start = supp_latex.index("Table S11.")
    end = supp_latex.index("Table S12.", start)
    fragment = supp_latex[start:end]
    widths = iter([0.12, 0.24, 0.10, 0.06, 0.06, 0.06, 0.12, 0.12, 0.12])
    fragment = re.sub(r"\\real\{0\.1111\}", lambda _: r"\real{" + f"{next(widths):.4f}" + "}", fragment)
    supp_latex = supp_latex[:start] + fragment + supp_latex[end:]
    supp_tex = PREAMBLE + r"\begin{frontmatter}\title{Supplementary material: " + latex(meta["title"]) + r"}\begin{abstract}Methods and result tables for the seven-archive study, including learned representations, external transfer, human-reviewed regions, controlled episode localization and chronological forecasting.\end{abstract}\end{frontmatter}" + "\n\\scriptsize\\setlength{\\tabcolsep}{2pt}\n" + supp_latex + "\n\\clearpage\n\\bibliographystyle{elsarticle-num}\\bibliography{references}\n\\end{document}\n"
    (SOURCES / "supplementary.tex").write_text(supp_tex, encoding="utf8")
    compile_tex("main")
    compile_tex("supplementary")
    with zipfile.ZipFile(OUT / "latex_source_flat.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(SOURCES.iterdir()):
            if file.is_file():
                archive.write(file, arcname=file.name)
    checks = {"abstract_words": len(meta["abstract"].split()), "highlight_characters": [len(s) for s in meta["highlights"]], "source_files_flat": True, "evidence_refit": False}
    (BUILD / "build_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
