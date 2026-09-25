"""NP037: threshold sensitivity for the target-conditional information rule.

Rebuild the four-state and downward first-passage labels at 0.15, 0.20 and
0.25 normalized excursion thresholds on the same Suining issue-time cohorts.
The weather/power/joint feature contracts and chronological masks are retained
from NP025 and NP028.  This is a robustness experiment for the process object,
not a new model selection stage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
LEADS = [15, 240, 720]
THRESHOLDS = [0.15, 0.20, 0.25]


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.sum((p - np.eye(4)[y]) ** 2, axis=1)))


def timing_rps(labels: np.ndarray, q: np.ndarray) -> float:
    keep = labels < 16
    labels = labels[keep]
    q = q[keep]
    cdf = np.cumsum(q, axis=1)
    cdf[:, -1] = 1.0
    truth = np.arange(16)[None, :] >= labels[:, None]
    return float(np.mean((cdf[:, :-1] - truth[:, :-1]) ** 2))


def reconstruct_aggregate() -> pd.Series:
    protocol = json.loads((ROOT / "outputs/next_paper/np025/protocol.json").read_text())
    scales = protocol["meta"]["scales"]
    frame = pd.read_csv(
        ROOT / "data/event_clean/site=suining.csv.gz",
        usecols=["turbine_id", "timestamp", "power_kw", "usable_power"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["power_norm"] = frame["power_kw"] / frame["turbine_id"].map(scales)
    frame.loc[~frame["usable_power"].astype(bool), "power_norm"] = np.nan
    turbine = frame.pivot_table(
        index="timestamp", columns="turbine_id", values="power_norm", aggfunc="mean"
    )
    valid = turbine.notna().sum(axis=1) >= 0.8 * len(scales)
    return turbine.mean(axis=1).where(valid).sort_index()


def labels_for_targets(aggregate: pd.Series, times: pd.DatetimeIndex, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    target = np.column_stack(
        [aggregate.reindex(times + pd.Timedelta(minutes=15 * j)).to_numpy() for j in range(17)]
    )
    valid = ~np.isnan(target).any(axis=1)
    running_min = np.minimum.accumulate(target, axis=1)
    running_max = np.maximum.accumulate(target, axis=1)
    up = np.max(target - running_min, axis=1) >= threshold
    down_path = running_max - target
    down = np.max(down_path, axis=1) >= threshold
    state = np.where(up & down, 3, np.where(down, 2, np.where(up, 1, 0))).astype(np.int8)
    # The first-passage object starts after T.  The current node contributes
    # to the running maximum but is not an arrival bin; node 1 is bin 0
    # (15 min), matching the frozen NP025/NP028 label contract.
    crossing = down_path[:, 1:] >= threshold
    first = np.argmax(crossing, axis=1).astype(np.int16)
    arrival = np.where(down, first, 16).astype(np.int16)
    state[~valid] = 0
    arrival[~valid] = 16
    return state, arrival


def fit_event(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    model = ExtraTreesClassifier(
        n_estimators=150, max_depth=16, min_samples_leaf=32,
        max_features=1.0, random_state=41, n_jobs=4,
    ).fit(x[train], y[train])
    p = np.zeros((int(test.sum()), 4))
    p[:, model.classes_] = model.predict_proba(x[test])
    return p


def fit_timing(x: np.ndarray, arrival: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    positive = arrival[train] < 16
    model = ExtraTreesClassifier(
        n_estimators=100, max_depth=16, min_samples_leaf=16,
        max_features=1.0, random_state=43, n_jobs=4,
    ).fit(x[train][positive], arrival[train][positive])
    q = np.zeros((int(test.sum()), 16))
    q[:, model.classes_] = model.predict_proba(x[test])
    return (q + 1e-6) / (q + 1e-6).sum(axis=1, keepdims=True)


def main() -> None:
    aggregate = reconstruct_aggregate()
    out = ROOT / "outputs/next_paper/np037_threshold_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    match_rows: list[dict] = []
    for package in ["np025", "np028"]:
        base = np.load(ROOT / f"outputs/next_paper/{package}/dataset.npz")
        for lead in LEADS:
            key = str(lead)
            times = pd.to_datetime(base[f"{key}__times_ns"], utc=True)
            train = base[f"{key}__train"]
            test = base[f"{key}__test"]
            arms = {
                "power": base[f"{key}__power_x"],
                "weather": base[f"{key}__weather_x"],
                "joint": base[f"{key}__joint_x"],
            }
            for threshold in THRESHOLDS:
                y, arrival = labels_for_targets(aggregate, times, threshold)
                if threshold == 0.20:
                    match_rows.append({
                        "package": package, "lead_minutes": lead,
                        "state_match": float(np.mean(y == base[f"{key}__y"])),
                        "arrival_match": float(np.mean(arrival == base[f"{key}__arrival_down"])),
                    })
                freq = np.bincount(y[train], minlength=4).astype(float)
                freq /= freq.sum()
                reference = brier(y[test], np.tile(freq, (int(test.sum()), 1)))
                for arm, x in arms.items():
                    p = fit_event(x, y, train, test)
                    q = fit_timing(x, arrival, train, test)
                    rows.append({
                        "package": package, "lead_minutes": lead, "threshold": threshold,
                        "arm": arm, "task": "state", "absolute_score": brier(y[test], p),
                        "relative_vs_frequency_pct": 100 * (reference - brier(y[test], p)) / reference,
                        "windows": int(test.sum()), "positive_events": int((arrival[test] < 16).sum()),
                    })
                    timing = timing_rps(arrival[test], q)
                    rows.append({
                        "package": package, "lead_minutes": lead, "threshold": threshold,
                        "arm": arm, "task": "downward_timing", "absolute_score": timing,
                        "relative_vs_frequency_pct": np.nan,
                        "windows": int(test.sum()), "positive_events": int((arrival[test] < 16).sum()),
                    })
    pd.DataFrame(rows).to_csv(out / "threshold_sensitivity.csv", index=False)
    pd.DataFrame(match_rows).to_csv(out / "label_reconstruction_check.csv", index=False)
    protocol = {
        "experiment": "NP037",
        "question": "Does the information-to-process pattern persist when the excursion threshold changes?",
        "packages": ["np025", "np028"], "leads": LEADS, "thresholds": THRESHOLDS,
        "arms": ["power", "weather", "joint"],
        "budgets": "event ExtraTrees 150 trees depth16 leaf32 seed41; timing 100 trees depth16 leaf16 seed43",
        "same_issue_time_cohorts": True, "status": "COMPLETE",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "verification.json").write_text(json.dumps({
        "status": "PASS", "rows": len(rows), "label_checks": match_rows,
        "source": "np025/np028 frozen masks and data/event_clean/site=suining.csv.gz",
    }, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "rows": len(rows), "output": str(out)}))


if __name__ == "__main__":
    main()
