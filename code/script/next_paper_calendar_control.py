"""Calendar-only controls for the current Renewable Energy manuscript.

Scientific question: do the NP025/NP028 information increments exceed a model
that receives only the known target-calendar variables?  The control uses the
same chronological windows, tree budgets, issue-time labels and block bootstrap
as the frozen information-arm experiments.  It is a control for information
attribution, not a replacement for the target-aligned route.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
LEADS = [15, 60, 120, 240, 480, 720]
SEED = 41


def probability_losses(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "return_brier": (p[:, 3] - (y == 3)) ** 2,
        "multiclass_brier": np.sum((p - np.eye(4)[y]) ** 2, axis=1),
    }


def timing_losses(label: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    cdf = np.cumsum(q, axis=1)
    cdf[:, -1] = 1.0
    present = label < 16
    truth = np.arange(16)[None, :] >= label[:, None]
    rps = np.full(len(label), np.nan)
    mae = np.full(len(label), np.nan)
    rps[present] = np.mean((cdf[present, :-1] - truth[present, :-1]) ** 2, axis=1)
    median = (cdf >= 0.5).argmax(axis=1)
    mae[present] = 15 * np.abs(median[present] - label[present])
    return {"conditional_rps": rps, "conditional_median_mae_minutes": mae}


def block_reduction(candidate: np.ndarray, reference: np.ndarray, times_ns: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(candidate) & np.isfinite(reference)
    candidate, reference, times_ns = candidate[valid], reference[valid], times_ns[valid]
    block = times_ns // int(pd.Timedelta(days=7).value)
    _, ids = np.unique(block, return_inverse=True)
    n = ids.max() + 1
    weights = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=2000)
    a = np.bincount(ids, weights=candidate, minlength=n)
    b = np.bincount(ids, weights=reference, minlength=n)
    draws = 100 * (1 - (weights @ a) / (weights @ b))
    return float(100 * (1 - candidate.mean() / reference.mean())), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def fit_timing(x_train: np.ndarray, labels: np.ndarray) -> ExtraTreesClassifier | None:
    positive = labels < 16
    if positive.sum() < 100 or np.unique(labels[positive]).size < 3:
        return None
    model = ExtraTreesClassifier(n_estimators=100, max_depth=16, min_samples_leaf=16,
                                 max_features=1.0, random_state=43, n_jobs=4)
    return model.fit(x_train[positive], labels[positive])


def load_npz(tag: str) -> tuple[np.lib.npyio.NpzFile, Path]:
    path = ROOT / "outputs/next_paper" / tag / "dataset.npz"
    return np.load(path, allow_pickle=False), path


def evaluate(tag: str) -> pd.DataFrame:
    data, _ = load_npz(tag)
    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    for lead in LEADS:
        key = str(lead)
        # Both source datasets append four known calendar values to every arm.
        # The control keeps those four columns and removes all power/weather data.
        x = data[f"{key}__power_x"][:, -4:]
        y = data[f"{key}__y"]
        labels = data[f"{key}__arrival_down"]
        times = data[f"{key}__times_ns"]
        train = data[f"{key}__train"]
        test = data[f"{key}__test"]
        model = ExtraTreesClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_leaf=32,
            max_features=1.0,
            random_state=SEED,
            n_jobs=4,
        ).fit(x[train], y[train])
        p = np.zeros((int(test.sum()), 4))
        p[:, model.classes_] = model.predict_proba(x[test])
        freq = np.bincount(y[train], minlength=4).astype(float)
        freq /= freq.sum()
        ref = np.tile(freq, (int(test.sum()), 1))
        event = probability_losses(y[test], p)
        event_ref = probability_losses(y[test], ref)
        for metric in ["return_brier", "multiclass_brier"]:
            gain, low, high = block_reduction(event[metric], event_ref[metric], times[test])
            rows.append({
                "source": tag,
                "lead_minutes": lead,
                "task": "event",
                "metric": metric,
                "candidate": "calendar_only",
                "reference": "frequency",
                "windows": int(test.sum()),
                "relative_loss_reduction_pct": gain,
                "low": low,
                "high": high,
                "absolute_candidate": float(np.nanmean(event[metric])),
                "absolute_reference": float(np.nanmean(event_ref[metric])),
            })
        timing_model = fit_timing(x[train], labels[train])
        positive = labels[train] < 16
        counts = np.bincount(labels[train][positive], minlength=16).astype(float) + 1
        qref = np.tile(counts / counts.sum(), (int(test.sum()), 1))
        q = qref.copy()
        if timing_model is not None:
            q[:, timing_model.classes_] = timing_model.predict_proba(x[test])
            q = (q + 1e-6) / (q + 1e-6).sum(axis=1, keepdims=True)
        timing = timing_losses(labels[test], q)
        timing_ref = timing_losses(labels[test], qref)
        for metric in ["conditional_rps", "conditional_median_mae_minutes"]:
            gain, low, high = block_reduction(timing[metric], timing_ref[metric], times[test])
            rows.append({
                "source": tag,
                "lead_minutes": lead,
                "task": "downward_timing",
                "metric": metric,
                "candidate": "calendar_only",
                "reference": "frequency",
                "windows": int(test.sum()),
                "relative_loss_reduction_pct": gain,
                "low": low,
                "high": high,
                "absolute_candidate": float(np.nanmean(timing[metric])),
                "absolute_reference": float(np.nanmean(timing_ref[metric])),
            })
        predictions[f"{key}__event"] = p
        predictions[f"{key}__timing"] = q
        predictions[f"{key}__y"] = y[test]
        predictions[f"{key}__arr"] = labels[test]
        predictions[f"{key}__times"] = times[test]
    return pd.DataFrame(rows), predictions


def main() -> None:
    out = ROOT / "outputs/next_paper/np031_calendar_control"
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    manifest = {"experiment": "NP031", "question": "Do information-arm gains exceed a calendar-only control?", "sources": {}}
    for tag in ["np025", "np028"]:
        rows, pred = evaluate(tag)
        all_rows.append(rows)
        np.savez_compressed(out / f"{tag}_predictions.npz", **pred)
        manifest["sources"][tag] = {
            "dataset": str((ROOT / "outputs/next_paper" / tag / "dataset.npz").relative_to(ROOT)),
            "features": "last four known target-calendar columns from the frozen source dataset",
        }
    table = pd.concat(all_rows, ignore_index=True)
    table.to_csv(out / "calendar_control_summary.csv", index=False)
    (out / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "status.json").write_text(json.dumps({"state": "COMPLETE", "rows": len(table)}) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(table), "output": str(out)}))


if __name__ == "__main__":
    main()
