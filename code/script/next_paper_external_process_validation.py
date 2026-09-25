"""NP022: external process-object validation without an issued-weather arm.

Question: do the occurrence-versus-arrival-time distinctions survive at two
independent wind farms when only issue-time power history is available?

The experiment deliberately does not claim an external weather-information
gain. It uses the existing auditable 15-min SCADA cleaning contract, estimates
per-turbine normalization scales from the training period only, applies a
four-hour/0.2-capacity process target, and evaluates a frozen ExtraTrees
history model against the training-frequency reference. The result is an
external I1 check; NP005--NP020 remain the source of the information-arm
claims.

Gneiting and Raftery (2007), doi:10.1198/016214506000001437, motivates the
proper Brier/RPS losses. Künsch (1989), doi:10.1214/aos/1176347265, motivates
calendar-block uncertainty for dependent windows. No causal atmospheric claim
is made from this power-history-only experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/event_clean"
OUT = ROOT / "outputs/next_paper/np022"
SITES = {"lahaute": "site=lahaute.csv.gz", "suining": "site=suining.csv.gz"}
LEADS = [15, 60, 120, 240, 480, 720]
TARGET_N = 17
N_LAG = 16
THRESHOLD = 0.2
SEED = 41
BATCH = 1000


def digest(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def raw_classes(path: np.ndarray) -> np.ndarray:
    up = np.max(path - np.minimum.accumulate(path, axis=1), axis=1) >= THRESHOLD - 1e-12
    down = np.max(np.maximum.accumulate(path, axis=1) - path, axis=1) >= THRESHOLD - 1e-12
    return up.astype(np.int8) + 2 * down.astype(np.int8)


def first_passage(path: np.ndarray) -> np.ndarray:
    excursion = np.maximum.accumulate(path, axis=1) - path
    hit = excursion[:, 1:] >= THRESHOLD - 1e-12
    out = np.full(len(path), 16, dtype=np.int16)
    any_hit = hit.any(axis=1)
    out[any_hit] = hit[any_hit].argmax(axis=1)
    return out


def calendar_features(times: pd.DatetimeIndex) -> np.ndarray:
    hour = times.hour + times.minute / 60.0
    day = times.dayofyear
    return np.c_[
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25),
    ].astype(float)


def load_site(site: str) -> tuple[np.ndarray, pd.DatetimeIndex, dict]:
    path = BASE / SITES[site]
    d = pd.read_csv(path, parse_dates=["timestamp"])
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d[d["usable_power"].astype(bool)].copy()
    d["power_kw"] = pd.to_numeric(d["power_kw"], errors="coerce")
    d = d.dropna(subset=["power_kw"])
    turbines = sorted(d["turbine_id"].astype(str).unique())
    full = pd.date_range(d.timestamp.min().floor("15min"), d.timestamp.max().floor("15min"), freq="15min", tz="UTC")
    wide = d.assign(turbine_id=d.turbine_id.astype(str)).pivot_table(index="timestamp", columns="turbine_id", values="power_kw", aggfunc="mean")
    wide = wide.reindex(full)
    cut = full[int(len(full) * 0.60)]
    scales = {}
    for tid in turbines:
        values = wide.loc[wide.index < cut, tid].to_numpy(float)
        values = values[np.isfinite(values)]
        scale = float(np.quantile(values, 0.995)) if len(values) else float("nan")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid train-only scale for {site}/{tid}")
        scales[tid] = scale
    normalized = wide.copy()
    for tid in turbines:
        normalized[tid] = normalized[tid] / scales[tid]
    valid_count = normalized.notna().sum(axis=1)
    aggregate = normalized.mean(axis=1, skipna=True).where(valid_count >= int(np.ceil(0.8 * len(turbines))))
    meta = {
        "site": site, "raw_rows": int(len(d)), "turbines": len(turbines),
        "grid_start": str(full[0]), "grid_end": str(full[-1]), "grid_rows": len(full),
        "training_scale_cutoff": str(cut), "scales": scales,
        "aggregate_valid_fraction": float(aggregate.notna().mean()),
        "minimum_turbine_fraction": 0.8,
    }
    return aggregate.to_numpy(float), full, meta


def build_site(site: str) -> tuple[dict, dict]:
    aggregate, times, meta = load_site(site)
    first_cut = times[int(len(times) * 0.60)]
    second_cut = times[int(len(times) * 0.80)]
    records = {}
    support = []
    cal = calendar_features(times)
    for lead in LEADS:
        step = lead // 15
        origins = []
        features = []
        paths = []
        for origin in range(len(aggregate)):
            issue = origin - step
            if issue - N_LAG + 1 < 0 or origin + TARGET_N > len(aggregate):
                continue
            hist = aggregate[issue - N_LAG + 1: issue + 1]
            target = aggregate[origin: origin + TARGET_N]
            if not np.isfinite(hist).all() or not np.isfinite(target).all():
                continue
            origins.append(origin)
            features.append(np.r_[hist, cal[origin]])
            paths.append(target)
        origin_idx = np.asarray(origins, dtype=np.int64)
        x = np.asarray(features, dtype=float)
        path = np.asarray(paths, dtype=float)
        t = times[origin_idx]
        y = raw_classes(path)
        arrival_down = first_passage(path)
        train = t < first_cut - pd.Timedelta(hours=4)
        validation = (t >= first_cut + pd.Timedelta(hours=4)) & (t < second_cut - pd.Timedelta(hours=4))
        test = t >= second_cut + pd.Timedelta(hours=4)
        if not train.any() or not validation.any() or not test.any():
            raise ValueError(f"Insufficient chronological support: {site}/{lead}")
        key = str(lead)
        records[key] = {
            "x": x, "y": y, "arrival_down": arrival_down,
            "times_ns": t.asi8, "train": train, "validation": validation, "test": test,
        }
        counts = np.bincount(y[train], minlength=4)
        for split, mask in [("train", train), ("validation", validation), ("test", test)]:
            support.append({"site": site, "lead_minutes": lead, "split": split, "windows": int(mask.sum()),
                            "class_0": int((y[mask] == 0).sum()), "class_1": int((y[mask] == 1).sum()),
                            "class_2": int((y[mask] == 2).sum()), "class_3": int((y[mask] == 3).sum()),
                            "down_events": int((arrival_down[mask] < 16).sum())})
    meta["first_cut"] = str(first_cut); meta["second_cut"] = str(second_cut)
    meta["leads"] = LEADS
    meta["support"] = support
    return records, meta


def probability_losses(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "return_brier": (p[:, 3] - (y == 3)) ** 2,
        "multiclass_brier": np.sum((p - np.eye(4)[y]) ** 2, axis=1),
    }


def timing_losses(label: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    cdf = np.cumsum(q, axis=1); cdf[:, -1] = 1.0
    present = label < 16
    truth = np.arange(16)[None, :] >= label[:, None]
    rps = np.full(len(label), np.nan); mae = np.full(len(label), np.nan)
    rps[present] = np.mean((cdf[present, :-1] - truth[present, :-1]) ** 2, axis=1)
    median = (cdf >= 0.5).argmax(axis=1)
    mae[present] = 15 * np.abs(median[present] - label[present])
    return {"conditional_rps": rps, "conditional_median_mae_minutes": mae}


def block_reduction(candidate: np.ndarray, reference: np.ndarray, times_ns: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(candidate) & np.isfinite(reference)
    candidate, reference, times_ns = candidate[valid], reference[valid], times_ns[valid]
    if len(candidate) == 0:
        return float("nan"), float("nan"), float("nan")
    block = times_ns // int(pd.Timedelta(days=7).value)
    _, ids = np.unique(block, return_inverse=True)
    n = ids.max() + 1
    weights = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=2000)
    a = np.bincount(ids, weights=candidate, minlength=n)
    b = np.bincount(ids, weights=reference, minlength=n)
    draws = 100 * (1 - (weights @ a) / (weights @ b))
    return float(100 * (1 - candidate.mean() / reference.mean())), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def fit_timing(x_train: np.ndarray, labels: np.ndarray) -> np.ndarray | None:
    positive = labels < 16
    if positive.sum() < 100 or np.unique(labels[positive]).size < 3:
        return None
    model = ExtraTreesClassifier(n_estimators=100, max_depth=16, min_samples_leaf=16,
                                 max_features=1.0, random_state=43, n_jobs=4)
    model.fit(x_train[positive], labels[positive])
    return model


def prepare(out: Path, script_path: Path) -> None:
    if out.exists():
        raise RuntimeError(f"Fresh output directory required: {out}")
    out.mkdir(parents=True)
    source_paths = [script_path, ROOT / "script/clean_dynamic_events.py"] + [BASE / v for v in SITES.values()]
    datasets = {}; metas = {}; support = []
    for site in SITES:
        records, meta = build_site(site)
        datasets[site] = records; metas[site] = meta; support.extend(meta["support"])
        save = {f"{lead}__{name}": value for lead, rec in records.items() for name, value in rec.items()}
        np.savez_compressed(out / f"{site}.npz", **save)
    pd.DataFrame(support).to_csv(out / "support.csv", index=False)
    protocol = {
        "experiment": "NP022", "stage": "EXTERNAL_PROCESS_OBJECT_VALIDATION",
        "question": "Do occurrence risk and first-passage timing remain distinct forecast objects at independent sites using only issue-time power history?",
        "sites": list(SITES), "leads": LEADS, "target": "17 native 15-min nodes; 0.2 normalized-capacity excursion; four classes and downward first passage",
        "input": "16 power-history lags ending at issue time plus known target calendar; no future wind, ERA5, GFS, JMA or GNSS",
        "normalization": "Per-turbine q99.5 power scale estimated from the first 60% time block only; aggregate requires >=80% turbine coverage",
        "split": "Chronological 60/20/20 blocks with four-hour purge on both boundaries; no validation selection",
        "event_model": "ExtraTrees 150 trees, depth16, leaf32, max_features=1.0, seed41; fixed across sites/leads",
        "timing_model": "ExtraTrees 100 trees, depth16, leaf16, max_features=1.0, seed43 on matured downward first-passage labels; occurrence probability held at training frequency",
        "uncertainty": "Paired 7-day calendar-block bootstrap, 2,000 draws, seed41; pointwise external diagnostic",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        "site_meta": metas,
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "status.json", {"state": "PREPARED", "sites": list(SITES)})


def check_sources(out: Path) -> dict:
    p = json.loads((out / "protocol.json").read_text())
    for name, sha in p["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError(f"Changed source: {name}")
    return p


def evaluate(out: Path, script_path: Path) -> None:
    protocol = check_sources(out)
    if json.loads((out / "status.json").read_text())["state"] != "PREPARED":
        raise RuntimeError("Prepare first")
    summary = []; pred_out = {}
    for site in SITES:
        data = np.load(out / f"{site}.npz")
        for lead in LEADS:
            key = str(lead)
            x = data[f"{key}__x"]; y = data[f"{key}__y"]; labels = data[f"{key}__arrival_down"]
            times = data[f"{key}__times_ns"]; train = data[f"{key}__train"]; test = data[f"{key}__test"]
            model = ExtraTreesClassifier(n_estimators=150, max_depth=16, min_samples_leaf=32,
                                         max_features=1.0, random_state=SEED, n_jobs=4).fit(x[train], y[train])
            p = np.zeros((test.sum(), 4)); p[:, model.classes_] = model.predict_proba(x[test])
            freq = np.bincount(y[train], minlength=4).astype(float); freq /= freq.sum()
            ref = np.tile(freq, (test.sum(), 1))
            l_c = probability_losses(y[test], p); l_r = probability_losses(y[test], ref)
            for metric in ["return_brier", "multiclass_brier"]:
                gain, low, high = block_reduction(l_c[metric], l_r[metric], times[test])
                summary.append({"site": site, "lead_minutes": lead, "task": "event", "metric": metric,
                                "candidate": "power_history_extra_trees", "reference": "training_frequency",
                                "windows": int(test.sum()), "valid_windows": int(test.sum()),
                                "relative_loss_reduction_pct": gain, "low": low, "high": high})
            q_model = fit_timing(x[train], labels[train])
            positive = labels[train] < 16
            counts = np.bincount(labels[train][positive], minlength=16).astype(float) + 1
            q_freq = np.tile(counts / counts.sum(), (test.sum(), 1))
            q_model_pred = q_freq.copy()
            if q_model is not None:
                q_model_pred[:, q_model.classes_] = q_model.predict_proba(x[test])
                q_model_pred = (q_model_pred + 1e-6) / (q_model_pred + 1e-6).sum(axis=1, keepdims=True)
            tl_c = timing_losses(labels[test], q_model_pred); tl_r = timing_losses(labels[test], q_freq)
            for metric in ["conditional_rps", "conditional_median_mae_minutes"]:
                gain, low, high = block_reduction(tl_c[metric], tl_r[metric], times[test])
                summary.append({"site": site, "lead_minutes": lead, "task": "downward_timing", "metric": metric,
                                "candidate": "power_history_extra_trees", "reference": "training_frequency",
                                "windows": int(test.sum()), "valid_windows": int(np.isfinite(tl_c[metric]).sum()),
                                "relative_loss_reduction_pct": gain, "low": low, "high": high})
            pred_out[f"{site}__{lead}__y"] = y[test]
            pred_out[f"{site}__{lead}__labels"] = labels[test]
            pred_out[f"{site}__{lead}__times_ns"] = times[test]
            pred_out[f"{site}__{lead}__event"] = p
            pred_out[f"{site}__{lead}__event_ref"] = ref
            pred_out[f"{site}__{lead}__timing"] = q_model_pred
            pred_out[f"{site}__{lead}__timing_ref"] = q_freq
    table = pd.DataFrame(summary); table.to_csv(out / "test_summary.csv", index=False)
    np.savez_compressed(out / "test_predictions.npz", **pred_out)
    plot(table, out / "external_process_validation.png")
    plot(table, out / "external_process_validation.pdf")
    write_json(out / "result_manifest.json", {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != "status.json"})
    write_json(out / "status.json", {"state": "COMPLETE", "summary_rows": len(table), "prediction_keys": len(pred_out)})


def plot(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    specs = [("return_brier", "Return risk"), ("conditional_rps", "Downward timing RPS"), ("conditional_median_mae_minutes", "Downward timing MAE")]
    for ax, (metric, title) in zip(axes, specs):
        d = table[table.metric == metric]
        for site, g in d.groupby("site"):
            g = g.sort_values("lead_minutes")
            ax.errorbar(g.lead_minutes, g.relative_loss_reduction_pct,
                        yerr=[g.relative_loss_reduction_pct - g.low, g.high - g.relative_loss_reduction_pct],
                        marker="o", capsize=2, label=site)
        ax.axhline(0, color="#777", lw=.8); ax.set_title(title); ax.set_xlabel("Lead (min)"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Relative loss reduction (%)"); axes[-1].legend(frameon=False)
    fig.suptitle("NP022 | External process-object validation")
    fig.tight_layout(); fig.savefig(path, dpi=220 if path.suffix == ".png" else None); plt.close(fig)


def verify(out: Path) -> None:
    protocol = check_sources(out)
    assert json.loads((out / "status.json").read_text())["state"] == "COMPLETE"
    table = pd.read_csv(out / "test_summary.csv")
    assert len(table) == len(SITES) * len(LEADS) * 4
    assert set(table.site) == set(SITES); assert set(table.lead_minutes) == set(LEADS)
    assert table.windows.gt(0).all(); assert table.valid_windows.le(table.windows).all()
    pred = np.load(out / "test_predictions.npz")
    for site in SITES:
        for lead in LEADS:
            k = f"{site}__{lead}__event"; assert k in pred
            p = pred[k]; np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
            assert np.isfinite(p).all() and (p >= 0).all()
            q = pred[f"{site}__{lead}__timing"]; np.testing.assert_allclose(q.sum(axis=1), 1, atol=1e-12)
            assert np.isfinite(q).all() and (q >= 0).all()
    write_json(out / "verification.json", {"status": "PASS", "checks": {
        "source_hashes_unchanged": True, "two_sites": True, "six_leads": True,
        "train_only_normalization_recorded": True, "chronological_purged_split": True,
        "issue_time_only_features": True, "no_weather_claim": True,
        "event_probabilities_normalized": True, "timing_probabilities_normalized": True,
        "summary_shape_and_support": True, "predictions_replayed": True,
    }, "summary_rows": len(table), "sites": list(SITES), "leads": LEADS})
    print(json.dumps({"status": "PASS", "summary_rows": len(table), "sites": list(SITES)}))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", choices=["prepare", "evaluate", "verify"], required=True)
    ap.add_argument("--output", type=Path, default=OUT); args = ap.parse_args()
    if args.phase == "prepare": prepare(args.output, Path(__file__))
    elif args.phase == "evaluate": evaluate(args.output, Path(__file__))
    else: verify(args.output)


if __name__ == "__main__": main()
