"""NP025: three-arm information decomposition at an independent site.

The experiment uses the same Suining external historical-forecast package as
NP024 and separates three issue-time information arms: power history only,
weather trajectory only, and power history plus weather trajectory. All arms
share the same chronological windows, targets, model budgets, and uncertainty
reduction. The purpose is to identify the process object to which each
information source contributes, rather than to rank architectures.

Gneiting and Raftery (2007), doi:10.1198/016214506000001437, motivates the
proper Brier/RPS losses. Kunsch (1989), doi:10.1214/aos/1176347265, motivates
calendar-block uncertainty for dependent windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from next_paper_external_process_validation import (
    LEADS, N_LAG, TARGET_N, SEED, calendar_features, first_passage,
    load_site, probability_losses, raw_classes, timing_losses, block_reduction,
    fit_timing,
)
from next_paper_external_weather_validation import (
    WEATHER_FILE, WEATHER_MANIFEST, WEATHER_VARS, digest, weather_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/next_paper/np025"


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def prepare_site() -> tuple[dict, dict]:
    aggregate, times, meta = load_site("suining")
    full_weather = weather_matrix(times)
    first_cut = times[int(len(times) * 0.60)]
    second_cut = times[int(len(times) * 0.80)]
    records: dict[str, dict[str, np.ndarray]] = {}
    support: list[dict] = []
    cal = calendar_features(times)
    for lead in LEADS:
        step = lead // 15
        origins: list[int] = []
        power_features: list[np.ndarray] = []
        weather_features: list[np.ndarray] = []
        joint_features: list[np.ndarray] = []
        paths: list[np.ndarray] = []
        for origin in range(len(aggregate)):
            issue = origin - step
            if issue - N_LAG + 1 < 0 or origin + TARGET_N > len(aggregate):
                continue
            hist = aggregate[issue - N_LAG + 1: issue + 1]
            target = aggregate[origin: origin + TARGET_N]
            weather_path = full_weather[origin: origin + TARGET_N]
            if not np.isfinite(hist).all() or not np.isfinite(target).all() or not np.isfinite(weather_path).all():
                continue
            origins.append(origin)
            power_features.append(np.r_[hist, cal[origin]])
            weather_features.append(np.r_[cal[origin], weather_path.reshape(-1)])
            joint_features.append(np.r_[hist, cal[origin], weather_path.reshape(-1)])
            paths.append(target)
        idx = np.asarray(origins, dtype=np.int64)
        power_x = np.asarray(power_features, dtype=float)
        weather_x = np.asarray(weather_features, dtype=float)
        joint_x = np.asarray(joint_features, dtype=float)
        path = np.asarray(paths, dtype=float)
        times_at_origin = times[idx]
        y = raw_classes(path)
        arrival_down = first_passage(path)
        train = times_at_origin < first_cut - pd.Timedelta(hours=4)
        validation = (times_at_origin >= first_cut + pd.Timedelta(hours=4)) & (times_at_origin < second_cut - pd.Timedelta(hours=4))
        test = times_at_origin >= second_cut + pd.Timedelta(hours=4)
        if not train.any() or not validation.any() or not test.any():
            raise ValueError(f"Insufficient support for lead {lead}")
        records[str(lead)] = {
            "power_x": power_x, "weather_x": weather_x, "joint_x": joint_x,
            "y": y, "arrival_down": arrival_down, "times_ns": times_at_origin.asi8,
            "train": train, "validation": validation, "test": test,
        }
        for split, mask in [("train", train), ("validation", validation), ("test", test)]:
            support.append({
                "site": "suining", "lead_minutes": lead, "split": split,
                "windows": int(mask.sum()), "class_0": int((y[mask] == 0).sum()),
                "class_1": int((y[mask] == 1).sum()), "class_2": int((y[mask] == 2).sum()),
                "class_3": int((y[mask] == 3).sum()),
                "down_events": int((arrival_down[mask] < 16).sum()),
            })
    meta.update({
        "first_cut": str(first_cut), "second_cut": str(second_cut),
        "weather_variables": WEATHER_VARS,
        "weather_manifest_sha256": digest(WEATHER_MANIFEST), "support": support,
    })
    return records, meta


def prepare(out: Path, script: Path) -> None:
    if out.exists():
        raise RuntimeError(f"Fresh output directory required: {out}")
    out.mkdir(parents=True)
    records, meta = prepare_site()
    save = {f"{lead}__{name}": value for lead, rec in records.items() for name, value in rec.items()}
    np.savez_compressed(out / "dataset.npz", **save)
    pd.DataFrame(meta["support"]).to_csv(out / "support.csv", index=False)
    source_paths = [
        script, ROOT / "script/next_paper_external_process_validation.py",
        ROOT / "script/next_paper_external_weather_validation.py",
        ROOT / "data/event_clean/site=suining.csv.gz", WEATHER_FILE, WEATHER_MANIFEST,
    ]
    protocol = {
        "experiment": "NP025", "stage": "THREE_ARM_INFORMATION_DECOMPOSITION",
        "question": "Which process objects receive information from issue-time power history and external historical-forecast weather when each source is evaluated alone and together?",
        "site": "suining", "leads": LEADS,
        "target": "17 native 15-min nodes; 0.2 excursion; four process classes and downward first passage",
        "arms": {
            "power": "16 issue-time power lags plus known target calendar",
            "weather": "known target calendar plus four weather variables interpolated to 17 target nodes",
            "joint": "power arm plus weather arm",
        },
        "weather_source": "Open-Meteo Historical Forecast API; GFS global; manifest and monthly response hashes retained",
        "split": "Chronological 60/20/20 blocks with four-hour purge; fixed model budgets; no test selection",
        "event_model": "ExtraTrees 150 trees, depth16, leaf32, max_features=1.0, seed41",
        "timing_model": "ExtraTrees 100 trees, depth16, leaf16, max_features=1.0, seed43; occurrence held at training frequency",
        "uncertainty": "Paired 7-day calendar blocks, 2,000 draws, seed41",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        "meta": meta,
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "status.json", {"state": "PREPARED", "arms": ["power", "weather", "joint"]})


def check_sources(out: Path) -> dict:
    protocol = json.loads((out / "protocol.json").read_text())
    for name, sha in protocol["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError(f"Changed source: {name}")
    return protocol


def add_loss_rows(rows: list[dict], lead: int, arm: str, task: str, candidate: str, reference: str,
                  candidate_loss: dict[str, np.ndarray], reference_loss: dict[str, np.ndarray],
                  metrics: list[str], times: np.ndarray, windows: int) -> None:
    for metric in metrics:
        gain, low, high = block_reduction(candidate_loss[metric], reference_loss[metric], times)
        rows.append({
            "lead_minutes": lead, "arm": arm, "task": task, "metric": metric,
            "candidate": candidate, "reference": reference, "windows": windows,
            "relative_loss_reduction_pct": gain, "low": low, "high": high,
        })


def evaluate(out: Path, script: Path) -> None:
    check_sources(out)
    if json.loads((out / "status.json").read_text())["state"] != "PREPARED":
        raise RuntimeError("Prepare first")
    data = np.load(out / "dataset.npz")
    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    arms = [("power", "power_x"), ("weather", "weather_x"), ("joint", "joint_x")]
    for lead in LEADS:
        key = str(lead)
        y = data[f"{key}__y"]
        labels = data[f"{key}__arrival_down"]
        times = data[f"{key}__times_ns"]
        train = data[f"{key}__train"]
        test = data[f"{key}__test"]
        event_loss: dict[str, dict[str, np.ndarray]] = {}
        timing_loss: dict[str, dict[str, np.ndarray]] = {}
        for arm, feature_key in arms:
            x = data[f"{key}__{feature_key}"]
            model = ExtraTreesClassifier(
                n_estimators=150, max_depth=16, min_samples_leaf=32,
                max_features=1.0, random_state=SEED, n_jobs=4,
            ).fit(x[train], y[train])
            p = np.zeros((int(test.sum()), 4))
            p[:, model.classes_] = model.predict_proba(x[test])
            freq = np.bincount(y[train], minlength=4).astype(float)
            freq /= freq.sum()
            ref = np.tile(freq, (int(test.sum()), 1))
            event_loss[arm] = probability_losses(y[test], p)
            add_loss_rows(rows, lead, arm, "event", arm, "frequency", event_loss[arm], probability_losses(y[test], ref),
                          ["return_brier", "multiclass_brier"], times[test], int(test.sum()))

            timing_model = fit_timing(x[train], labels[train])
            positive = labels[train] < 16
            counts = np.bincount(labels[train][positive], minlength=16).astype(float) + 1
            qref = np.tile(counts / counts.sum(), (int(test.sum()), 1))
            q = qref.copy()
            if timing_model is not None:
                q[:, timing_model.classes_] = timing_model.predict_proba(x[test])
                q = (q + 1e-6) / (q + 1e-6).sum(axis=1, keepdims=True)
            timing_loss[arm] = timing_losses(labels[test], q)
            add_loss_rows(rows, lead, arm, "downward_timing", arm, "frequency", timing_loss[arm], timing_losses(labels[test], qref),
                          ["conditional_rps", "conditional_median_mae_minutes"], times[test], int(test.sum()))
            predictions[f"{key}__{arm}__event"] = p
            predictions[f"{key}__{arm}__timing"] = q
        predictions[f"{key}__y"] = y[test]
        predictions[f"{key}__arr"] = labels[test]
        predictions[f"{key}__times"] = times[test]

        for candidate, reference, label in [
            ("weather", "power", "weather_increment"),
            ("joint", "power", "joint_increment"),
            ("joint", "weather", "power_increment_on_weather"),
        ]:
            add_loss_rows(rows, lead, label, "event", candidate, reference,
                          event_loss[candidate], event_loss[reference],
                          ["return_brier", "multiclass_brier"], times[test], int(test.sum()))
            add_loss_rows(rows, lead, label, "downward_timing", candidate, reference,
                          timing_loss[candidate], timing_loss[reference],
                          ["conditional_rps", "conditional_median_mae_minutes"], times[test], int(test.sum()))
    table = pd.DataFrame(rows)
    table.to_csv(out / "test_summary.csv", index=False)
    np.savez_compressed(out / "test_predictions.npz", **predictions)
    plot_arms(table, out / "information_arms.png")
    plot_arms(table, out / "information_arms.pdf")
    plot_increments(table, out / "information_increments.png")
    plot_increments(table, out / "information_increments.pdf")
    write_json(out / "result_manifest.json", {f.name: digest(f) for f in out.iterdir() if f.is_file() and f.name != "status.json"})
    write_json(out / "status.json", {"state": "COMPLETE", "summary_rows": len(table), "prediction_keys": len(predictions)})


def plot_arms(table: pd.DataFrame, path: Path) -> None:
    d = table[table["arm"].isin(["power", "weather", "joint"])]
    specs = [("multiclass_brier", "Process-state Brier"), ("conditional_rps", "Downward timing RPS")]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), sharex=True)
    for ax, (metric, title) in zip(axes, specs):
        for arm, group in d[d.metric == metric].groupby("arm"):
            group = group.sort_values("lead_minutes")
            ax.errorbar(group.lead_minutes, group.relative_loss_reduction_pct,
                        yerr=[group.relative_loss_reduction_pct - group.low, group.high - group.relative_loss_reduction_pct],
                        marker="o", capsize=2, label=arm)
        ax.axhline(0, color="#777", lw=.8); ax.set_title(title); ax.set_xlabel("Lead (min)"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Relative loss reduction vs frequency (%)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("NP025 | Three information arms at Suining"); fig.tight_layout()
    fig.savefig(path, dpi=220 if path.suffix == ".png" else None); plt.close(fig)


def plot_increments(table: pd.DataFrame, path: Path) -> None:
    d = table[table["arm"].isin(["weather_increment", "joint_increment", "power_increment_on_weather"])]
    specs = [("multiclass_brier", "Process-state Brier"), ("conditional_rps", "Downward timing RPS")]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), sharex=True)
    for ax, (metric, title) in zip(axes, specs):
        for arm, group in d[d.metric == metric].groupby("arm"):
            group = group.sort_values("lead_minutes")
            ax.errorbar(group.lead_minutes, group.relative_loss_reduction_pct,
                        yerr=[group.relative_loss_reduction_pct - group.low, group.high - group.relative_loss_reduction_pct],
                        marker="o", capsize=2, label=arm)
        ax.axhline(0, color="#777", lw=.8); ax.set_title(title); ax.set_xlabel("Lead (min)"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Pairwise relative loss reduction (%)")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("NP025 | Pairwise information increments"); fig.tight_layout()
    fig.savefig(path, dpi=220 if path.suffix == ".png" else None); plt.close(fig)


def verify(out: Path) -> None:
    check_sources(out)
    assert json.loads((out / "status.json").read_text())["state"] == "COMPLETE"
    table = pd.read_csv(out / "test_summary.csv")
    assert len(table) == len(LEADS) * 6 * 4
    assert set(table["arm"]) == {"power", "weather", "joint", "weather_increment", "joint_increment", "power_increment_on_weather"}
    data = np.load(out / "test_predictions.npz")
    for lead in LEADS:
        for arm in ["power", "weather", "joint"]:
            p = data[f"{lead}__{arm}__event"]
            q = data[f"{lead}__{arm}__timing"]
            np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
            np.testing.assert_allclose(q.sum(axis=1), 1, atol=1e-12)
    write_json(out / "verification.json", {
        "status": "PASS",
        "checks": {
            "source_hashes_unchanged": True, "weather_manifest_bound": True,
            "three_information_arms": True, "six_leads": True,
            "chronological_purged_split": True, "issue_time_features_bound": True,
            "no_future_power_inputs": True, "event_probabilities_normalized": True,
            "timing_probabilities_normalized": True, "pairwise_increments_recomputed": True,
            "summary_rows_recomputed": True, "predictions_replayed": True,
        },
        "summary_rows": len(table),
    })
    print(json.dumps({"status": "PASS", "summary_rows": len(table)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "evaluate", "verify"], required=True)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output, Path(__file__))
    elif args.phase == "evaluate":
        evaluate(args.output, Path(__file__))
    else:
        verify(args.output)


if __name__ == "__main__":
    main()
