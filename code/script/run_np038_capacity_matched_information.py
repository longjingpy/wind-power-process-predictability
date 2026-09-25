"""Run the corrected NP038 capacity-matched information experiment.

The experiment is deliberately small and fixed before test scoring. It uses
the frozen NP025/NP028 datasets, six issue leads, one event forest budget and
one first-passage timing forest budget. Weather summaries and projections are
compact 20-column arms; the projection is the only arm transformation that
needs a scale correction, so its weather scaler is fitted on the training rows
for each package/lead and then applied to all rows.

This file is the portable entry point. It does not select a model on test
data. Use ``--phase run`` followed by the independent verifier in
``verify_np038_capacity_matched_information.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/next_paper/np038_capacity_matched_information"
LEADS = [15, 60, 120, 240, 480, 720]
ROBUST_LEADS = [15, 240, 720]
PRIMARY_EVENT_SEED = 41
PRIMARY_TIMING_SEED = 43
ROBUST_EVENT_SEEDS = [41, 42, 43]
ROBUST_TIMING_SEEDS = [43, 44, 45]
ROBUST_PAIRS = list(zip(ROBUST_EVENT_SEEDS, ROBUST_TIMING_SEEDS))
PROJECTION_SEED = 20250925
BOOTSTRAP_SEED = 41
BOOTSTRAP_REPS = 2000
DAY_NS = int(pd.Timedelta(days=7).value)
WEATHER_VARIABLES = ["u100_ms", "v100_ms", "temperature", "surface_pressure"]
CALENDAR_NAMES = [
    "calendar_sin_hour",
    "calendar_cos_hour",
    "calendar_sin_dayofyear",
    "calendar_cos_dayofyear",
]
ARM_ORDER = [
    "power20",
    "weather_raw72",
    "joint_raw88",
    "weather_summary20",
    "weather_projection20",
    "noise20",
    "power_weather_summary40",
    "power_weather_projection40",
    "power_noise40",
]
EXPECTED_DIMS = {
    "power20": 20,
    "weather_raw72": 72,
    "joint_raw88": 88,
    "weather_summary20": 20,
    "weather_projection20": 20,
    "noise20": 20,
    "power_weather_summary40": 40,
    "power_weather_projection40": 40,
    "power_noise40": 40,
}
COMPARISONS = [
    ("joint_vs_power", "joint_raw88", "power20"),
    ("weather_summary20_vs_noise20", "weather_summary20", "noise20"),
    ("weather_projection20_vs_noise20", "weather_projection20", "noise20"),
    ("power_weather_summary40_vs_power_noise40", "power_weather_summary40", "power_noise40"),
    ("power_weather_projection40_vs_power_noise40", "power_weather_projection40", "power_noise40"),
]
METRIC_TASKS = [
    ("state", "return_brier"),
    ("state", "multiclass_brier"),
    ("downward_timing", "conditional_rps"),
    ("downward_timing", "conditional_median_mae_minutes"),
]


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def fixed_projection(seed: int = PROJECTION_SEED) -> np.ndarray:
    """Return a canonical 68 by 16 QR matrix for the fixed projection."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(68, 16))
    q, r = np.linalg.qr(base, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def stable_noise(times_ns: np.ndarray, width: int = 16) -> np.ndarray:
    """Make timestamp-keyed noise that is identical across leads/packages."""
    out = np.empty((len(times_ns), width), dtype=float)
    for i, value in enumerate(np.asarray(times_ns, dtype=np.int64)):
        key = hashlib.sha256(str(int(value)).encode("ascii")).digest()[:8]
        seed = int.from_bytes(key, "little")
        out[i] = np.random.default_rng(seed).normal(size=width)
    return out


def power_feature_names() -> list[str]:
    return [f"power_history_minus_{15 * (15 - i)}_minutes" for i in range(16)] + CALENDAR_NAMES


def weather_feature_names() -> list[str]:
    return CALENDAR_NAMES + [
        f"{variable}_target_node_{node:02d}"
        for node in range(17)
        for variable in WEATHER_VARIABLES
    ]


def feature_name_map() -> dict[str, list[str]]:
    power = power_feature_names()
    weather = weather_feature_names()
    summary = CALENDAR_NAMES + [
        f"weather_{stat}_{variable}"
        for stat in ("mean", "std", "first_node", "last_node")
        for variable in WEATHER_VARIABLES
    ]
    projection = CALENDAR_NAMES + [f"weather_projection_component_{i:02d}" for i in range(16)]
    noise = CALENDAR_NAMES + [f"timestamp_noise_component_{i:02d}" for i in range(16)]
    return {
        "power20": power,
        "weather_raw72": weather,
        # Frozen joint_x is power history + 68 weather signal columns.
        "joint_raw88": power + weather[4:],
        "weather_summary20": summary,
        "weather_projection20": projection,
        "noise20": noise,
        # The common four calendar columns are intentionally retained twice.
        "power_weather_summary40": power + summary,
        "power_weather_projection40": power + projection,
        "power_noise40": power + noise,
    }


def feature_arms(
    weather_x: np.ndarray,
    power_x: np.ndarray,
    joint_x: np.ndarray,
    times_ns: np.ndarray,
    train: np.ndarray,
    projection: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray, np.ndarray]:
    """Build all fixed arms and return arms, map, training scaler, noise."""
    if power_x.ndim != 2 or power_x.shape[1] != 20:
        raise ValueError(f"power_x must be 20 columns, got {power_x.shape}")
    if weather_x.ndim != 2 or weather_x.shape[1] != 72:
        raise ValueError(f"weather_x must be 72 columns, got {weather_x.shape}")
    if joint_x.ndim != 2 or joint_x.shape[1] != 88:
        raise ValueError(f"frozen joint_x must be 88 columns, got {joint_x.shape}")
    if len(weather_x) != len(power_x) or len(power_x) != len(joint_x) or len(times_ns) != len(power_x):
        raise ValueError("feature rows and timestamps must have the same length")
    if projection.shape != (68, 16):
        raise ValueError(f"projection must be 68x16, got {projection.shape}")
    train = np.asarray(train, dtype=bool)
    if not train.any():
        raise ValueError("training mask is empty")

    calendar = weather_x[:, :4]
    weather_signal = weather_x[:, 4:]
    train_values = weather_signal[train]
    train_mean = train_values.mean(axis=0)
    train_std = train_values.std(axis=0)
    train_std = np.where(train_std > 0, train_std, 1.0)
    scaled_signal = (weather_signal - train_mean) / train_std
    weather_nodes = weather_signal.reshape(len(weather_signal), 17, 4)
    summary_values = np.column_stack(
        [
            weather_nodes.mean(axis=1),
            weather_nodes.std(axis=1),
            weather_nodes[:, 0, :],
            weather_nodes[:, -1, :],
        ]
    )
    summary20 = np.c_[calendar, summary_values]
    projection20 = np.c_[calendar, scaled_signal @ projection]
    noise20 = np.c_[calendar, stable_noise(times_ns, 16)]
    arms = {
        "power20": np.asarray(power_x, dtype=float),
        "weather_raw72": np.asarray(weather_x, dtype=float),
        # Critical correction: use the frozen 88-column array directly.
        "joint_raw88": np.asarray(joint_x, dtype=float),
        "weather_summary20": summary20,
        "weather_projection20": projection20,
        "noise20": noise20,
        "power_weather_summary40": np.c_[power_x, summary20],
        "power_weather_projection40": np.c_[power_x, projection20],
        "power_noise40": np.c_[power_x, noise20],
    }
    names = feature_name_map()
    feature_map: dict[str, Any] = {}
    for arm in ARM_ORDER:
        x = arms[arm]
        if x.shape[1] != EXPECTED_DIMS[arm] or len(names[arm]) != EXPECTED_DIMS[arm]:
            raise ValueError(f"feature map mismatch for {arm}: {x.shape} / {len(names[arm])}")
        feature_map[arm] = {
            "columns": int(x.shape[1]),
            "features": names[arm],
            "sha256": hash_array(x),
        }
    return arms, feature_map, np.c_[train_mean, train_std], noise20


def measured_tree_stats(model: ExtraTreesClassifier, x_train: np.ndarray) -> dict[str, Any]:
    depths = np.asarray([est.tree_.max_depth for est in model.estimators_], dtype=int)
    leaves = np.asarray([est.tree_.n_leaves for est in model.estimators_], dtype=int)
    used_features: set[int] = set()
    occupied = 0
    for est in model.estimators_:
        split_features = est.tree_.feature
        used_features.update(int(v) for v in split_features[split_features >= 0])
        occupied += int(np.unique(est.apply(x_train)).size)
    return {
        "n_trees": int(len(model.estimators_)),
        "configured_max_depth": 16,
        "configured_min_leaf": int(model.min_samples_leaf),
        "measured_max_depth": int(depths.max()),
        "measured_mean_depth": float(depths.mean()),
        "measured_min_depth": int(depths.min()),
        "measured_total_leaves": int(leaves.sum()),
        "measured_mean_leaves": float(leaves.mean()),
        "measured_max_leaves": int(leaves.max()),
        "used_feature_count": int(len(used_features)),
        "occupied_train_leaves": int(occupied),
        "training_rows": int(len(x_train)),
    }


def fit_event(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    model = ExtraTreesClassifier(
        n_estimators=150, max_depth=16, min_samples_leaf=32,
        max_features=1.0, random_state=seed, n_jobs=4,
    ).fit(x[train], y[train])
    p = np.zeros((int(test.sum()), 4), dtype=float)
    p[:, model.classes_] = model.predict_proba(x[test])
    return p, measured_tree_stats(model, x[train])


def fit_timing(
    x: np.ndarray, arrival: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    positive = np.asarray(train, dtype=bool) & (arrival < 16)
    if positive.sum() == 0 or np.unique(arrival[positive]).size < 2:
        raise ValueError("insufficient positive first-passage labels for timing model")
    model = ExtraTreesClassifier(
        n_estimators=100, max_depth=16, min_samples_leaf=16,
        max_features=1.0, random_state=seed, n_jobs=4,
    ).fit(x[positive], arrival[positive])
    q = np.zeros((int(test.sum()), 16), dtype=float)
    q[:, model.classes_] = model.predict_proba(x[test])
    q = (q + 1e-6) / (q + 1e-6).sum(axis=1, keepdims=True)
    return q, measured_tree_stats(model, x[positive])


def probability_loss_arrays(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "return_brier": (p[:, 3] - (y == 3)) ** 2,
        "multiclass_brier": np.sum((p - np.eye(4)[y]) ** 2, axis=1),
    }


def timing_loss_arrays(arrival: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    cdf = np.cumsum(q, axis=1)
    cdf[:, -1] = 1.0
    present = arrival < 16
    truth = np.arange(16)[None, :] >= arrival[:, None]
    rps = np.full(len(arrival), np.nan, dtype=float)
    mae = np.full(len(arrival), np.nan, dtype=float)
    rps[present] = np.mean((cdf[present, :-1] - truth[present, :-1]) ** 2, axis=1)
    median = (cdf >= 0.5).argmax(axis=1)
    mae[present] = 15.0 * np.abs(median[present] - arrival[present])
    return {"conditional_rps": rps, "conditional_median_mae_minutes": mae}


def frequency_event(y_train: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=4).astype(float)
    freq = counts / counts.sum()
    return np.tile(freq, (n, 1))


def frequency_timing(arrival_train: np.ndarray, n: int) -> np.ndarray:
    labels = arrival_train[arrival_train < 16]
    counts = np.bincount(labels, minlength=16).astype(float) + 1.0
    return np.tile(counts / counts.sum(), (n, 1))


def paired_block_intervals(
    candidate: np.ndarray, reference: np.ndarray, times_ns: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, float | int]:
    if valid is None:
        valid = np.isfinite(candidate) & np.isfinite(reference)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(candidate) & np.isfinite(reference)
    candidate = np.asarray(candidate, dtype=float)[valid]
    reference = np.asarray(reference, dtype=float)[valid]
    times_ns = np.asarray(times_ns, dtype=np.int64)[valid]
    if len(candidate) == 0:
        return {"valid_windows": 0, "blocks": 0, "candidate_score": float("nan"), "reference_score": float("nan"), "difference": float("nan"), "reduction_pct": float("nan"), "difference_low": float("nan"), "difference_high": float("nan"), "reduction_low": float("nan"), "reduction_high": float("nan")}
    _, inverse = np.unique(times_ns // DAY_NS, return_inverse=True)
    n_blocks = int(inverse.max() + 1)
    counts = np.bincount(inverse, minlength=n_blocks).astype(float)
    csum = np.bincount(inverse, weights=candidate, minlength=n_blocks)
    rsum = np.bincount(inverse, weights=reference, minlength=n_blocks)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=BOOTSTRAP_REPS)
    den = draws @ counts
    cmeans = (draws @ csum) / den
    rmeans = (draws @ rsum) / den
    differences = cmeans - rmeans
    reductions = 100.0 * (1.0 - cmeans / rmeans)
    c_score = float(candidate.mean())
    r_score = float(reference.mean())
    return {
        "valid_windows": int(len(candidate)), "blocks": n_blocks,
        "candidate_score": c_score, "reference_score": r_score,
        "difference": float(c_score - r_score),
        "reduction_pct": float(100.0 * (1.0 - c_score / r_score)),
        "difference_low": float(np.quantile(differences, 0.025)),
        "difference_high": float(np.quantile(differences, 0.975)),
        "reduction_low": float(np.quantile(reductions, 0.025)),
        "reduction_high": float(np.quantile(reductions, 0.975)),
    }


def run_arm(
    x: np.ndarray, y: np.ndarray, arrival: np.ndarray, train: np.ndarray,
    test: np.ndarray, event_seed: int, timing_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    p, event_stats = fit_event(x, y, train, test, event_seed)
    q, timing_stats = fit_timing(x, arrival, train, test, timing_seed)
    losses = probability_loss_arrays(y[test], p) | timing_loss_arrays(arrival[test], q)
    return p, q, losses, event_stats, timing_stats


def run_package_lead(
    package: str, lead: int, data: np.lib.npyio.NpzFile, projection: np.ndarray,
    event_seed: int, timing_seed: int, selected_arms: list[str] | None = None,
) -> dict[str, Any]:
    key = str(lead)
    y = data[f"{key}__y"]
    arrival = data[f"{key}__arrival_down"]
    times_ns = data[f"{key}__times_ns"]
    train = data[f"{key}__train"].astype(bool)
    test = data[f"{key}__test"].astype(bool)
    arms, fmap, scaler, noise20 = feature_arms(
        data[f"{key}__weather_x"], data[f"{key}__power_x"], data[f"{key}__joint_x"],
        times_ns, train, projection,
    )
    chosen = ARM_ORDER if selected_arms is None else selected_arms
    rows: list[dict[str, Any]] = []
    complexity: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    losses_registry: dict[str, dict[str, np.ndarray]] = {}
    event_reference = frequency_event(y[train], int(test.sum()))
    timing_reference = frequency_timing(arrival[train], int(test.sum()))
    ref_losses = probability_loss_arrays(y[test], event_reference) | timing_loss_arrays(arrival[test], timing_reference)
    run_type = "primary" if (event_seed, timing_seed) == (PRIMARY_EVENT_SEED, PRIMARY_TIMING_SEED) else "robust"
    for arm in chosen:
        p, q, losses, event_stats, timing_stats = run_arm(arms[arm], y, arrival, train, test, event_seed, timing_seed)
        predictions[arm] = {"event": p, "timing": q}
        losses_registry[arm] = losses
        for task, metric in METRIC_TASKS:
            values = losses[metric]
            score = float(np.nanmean(values))
            ref_score = float(np.nanmean(ref_losses[metric]))
            rows.append({
                "package": package, "lead_minutes": lead, "run_type": run_type,
                "event_seed": event_seed, "timing_seed": timing_seed,
                "seed_label": f"event{event_seed}_timing{timing_seed}",
                "arm": arm, "task": task, "metric": metric,
                "absolute_score": score,
                "relative_vs_frequency_pct": float(100.0 * (1.0 - score / ref_score)) if ref_score else float("nan"),
                "frequency_score": ref_score, "windows": int(test.sum()),
                "valid_windows": int(np.isfinite(values).sum()),
                "positive_events": int((arrival[test] < 16).sum()),
            })
        for task, stats in [("state", event_stats), ("downward_timing", timing_stats)]:
            complexity.append({
                "package": package, "lead_minutes": lead, "run_type": run_type,
                "event_seed": event_seed, "timing_seed": timing_seed,
                "seed_label": f"event{event_seed}_timing{timing_seed}",
                "arm": arm, "task": task, **stats,
            })
    return {
        "rows": rows, "complexity": complexity, "predictions": predictions,
        "losses": losses_registry, "reference_losses": ref_losses,
        "feature_map": fmap, "scaler": scaler, "noise20": noise20,
        "y": y, "arrival": arrival, "times_ns": times_ns,
        "train": train, "test": test, "event_seed": event_seed, "timing_seed": timing_seed,
    }


def split_codes(train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> np.ndarray:
    out = np.full(len(train), -1, dtype=np.int8)
    out[np.asarray(train, dtype=bool)] = 0
    out[np.asarray(validation, dtype=bool)] = 1
    out[np.asarray(test, dtype=bool)] = 2
    return out


def add_prediction_metadata(store: dict[str, np.ndarray], package: str, data: np.lib.npyio.NpzFile, lead: int, run_prefix: str = "") -> None:
    key = str(lead)
    base = f"{run_prefix}{package}__{lead}"
    train = data[f"{key}__train"].astype(bool)
    validation = data[f"{key}__validation"].astype(bool)
    test = data[f"{key}__test"].astype(bool)
    y, arrival, times = data[f"{key}__y"], data[f"{key}__arrival_down"], data[f"{key}__times_ns"]
    store[f"{base}__labels_y"] = y
    store[f"{base}__labels_arrival_down"] = arrival
    store[f"{base}__times_ns"] = times
    store[f"{base}__split_code"] = split_codes(train, validation, test)
    store[f"{base}__test_index"] = np.flatnonzero(test).astype(np.int64)
    store[f"{base}__test_y"] = y[test]
    store[f"{base}__test_arrival_down"] = arrival[test]
    store[f"{base}__test_times_ns"] = times[test]
    for optional in ("issue_times_ns", "target_end_times_ns"):
        source_key = f"{key}__{optional}"
        if source_key in data.files:
            store[f"{base}__{optional}"] = data[source_key]


def save_predictions(store: dict[str, np.ndarray], result: dict[str, Any], package: str, lead: int, run_prefix: str = "") -> None:
    base = f"{run_prefix}{package}__{lead}"
    for arm, preds in result["predictions"].items():
        store[f"{base}__{arm}__event"] = preds["event"]
        store[f"{base}__{arm}__timing"] = preds["timing"]


def build_comparisons(registry: dict[tuple[str, int, int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (package, lead, event_seed, timing_seed), result in registry.items():
        run_type = "primary" if (event_seed, timing_seed) == (PRIMARY_EVENT_SEED, PRIMARY_TIMING_SEED) else "robust"
        for comparison, candidate, reference in COMPARISONS:
            if candidate not in result["losses"] or reference not in result["losses"]:
                continue
            for task, metric in METRIC_TASKS:
                valid = None
                if task == "downward_timing":
                    valid = result["arrival"][result["test"]] < 16
                interval = paired_block_intervals(
                    result["losses"][candidate][metric], result["losses"][reference][metric],
                    result["times_ns"][result["test"]], valid=valid,
                )
                rows.append({
                    "package": package, "lead_minutes": lead, "run_type": run_type,
                    "event_seed": event_seed, "timing_seed": timing_seed,
                    "seed_label": f"event{event_seed}_timing{timing_seed}",
                    "comparison": comparison, "candidate": candidate, "reference": reference,
                    "task": task, "metric": metric, **interval,
                })
    return rows


def source_manifest() -> dict[str, Any]:
    sources = {
        "script/run_np038_capacity_matched_information.py": ROOT / "script/run_np038_capacity_matched_information.py",
        "outputs/next_paper/np025/dataset.npz": ROOT / "outputs/next_paper/np025/dataset.npz",
        "outputs/next_paper/np025/protocol.json": ROOT / "outputs/next_paper/np025/protocol.json",
        "outputs/next_paper/np025/result_manifest.json": ROOT / "outputs/next_paper/np025/result_manifest.json",
        "outputs/next_paper/np028/dataset.npz": ROOT / "outputs/next_paper/np028/dataset.npz",
        "outputs/next_paper/np028/protocol.json": ROOT / "outputs/next_paper/np028/protocol.json",
        "outputs/next_paper/np028/result_manifest.json": ROOT / "outputs/next_paper/np028/result_manifest.json",
    }
    return {"sources": {name: digest(path) for name, path in sources.items()}, "source_paths": list(sources), "frozen_packages": ["np025", "np028"]}


def result_manifest(out: Path) -> dict[str, str]:
    return {path.name: digest(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != "result_manifest.json"}


def status_markdown(primary: pd.DataFrame, comparisons: pd.DataFrame, robust: pd.DataFrame) -> str:
    lines = [
        "# NP038 corrected run status", "",
        "本结果采用冻结的 NP025/NP028 数据集。联合臂直接读取 88 列 `joint_x`；天气投影先按每个 package/lead 的训练行拟合均值和标准差，再使用 seed=20250925 的固定 QR 矩阵。该修正是对 preliminary run 的回顾性更正，不构成预注册时间戳。", "",
        f"主结果：{len(primary)} 行 arm×task×metric；稳健结果：{len(robust)} 行（含 seed 41/43 主配对的留档副本）。所有测试评分均使用冻结测试分割，未做测试调参。", "",
        "## 需要保留的材料性负结果", "",
    ]
    if len(comparisons):
        neg = comparisons[(comparisons.run_type.eq("primary")) & (comparisons.reduction_pct < 0)]
        if len(neg):
            strict = neg[neg.reduction_high < 0].sort_values("reduction_pct").head(12)
            borderline = neg[(neg.reduction_low < 0) & (neg.reduction_high >= 0)].sort_values("reduction_pct").head(6)
            selected = pd.concat([strict, borderline]).drop_duplicates(
                subset=["package", "lead_minutes", "comparison", "metric"]
            )
            for _, row in selected.iterrows():
                lines.append(f"- {row.package} {int(row.lead_minutes)} min {row.comparison} / {row.metric}：候选绝对分数 {row.candidate_score:.6g}，参照 {row.reference_score:.6g}，相对变化 {row.reduction_pct:.2f}%（七日区间 {row.reduction_low:.2f}% 至 {row.reduction_high:.2f}%）。")
            lines.append(f"- 主比较共有 {len(neg)} 个候选分数高于参照的指标，其中 {int((neg.reduction_high < 0).sum())} 个七日区间完全为负；其余结果保存在 comparisons.csv。")
        else:
            lines.append("- 主比较中没有候选绝对分数高于参照的组合；仍应以完整比较表和区间为准。")
    lines.extend([
        "", "## 可移植入口", "", "```bash",
        "cd /mnt/d/projects/WindPowerForcast",
        "/home/ljpy/projects/amd-rocm-pytorch-wsl/.venv/bin/python script/run_np038_capacity_matched_information.py --phase run",
        "/home/ljpy/projects/amd-rocm-pytorch-wsl/.venv/bin/python script/verify_np038_capacity_matched_information.py",
        "```", "",
        "`capacity_matched_summary.csv` 保存 seed 41/43 主结果；`seed_sensitivity.csv` 保存 15/240/720 分钟三组种子；`comparisons.csv` 保存 J 对 P、等维 weather 对 noise、40 列 weather 对 40 列 noise 的绝对分数、配对差值和 2,000 次七日区间。",
    ])
    return "\n".join(lines) + "\n"


def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    projection = fixed_projection()
    np.save(out / "projection_matrix.npy", projection)
    source_info = source_manifest()
    write_json(out / "source_manifest.json", source_info)
    write_json(out / "projection_matrix.json", {"seed": PROJECTION_SEED, "shape": list(projection.shape), "sha256": hash_array(projection), "standardization": "per package/lead training-only weather signal mean/std before projection"})

    primary_rows: list[dict[str, Any]] = []
    robust_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    registry: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    primary_predictions: dict[str, np.ndarray] = {}
    robust_predictions: dict[str, np.ndarray] = {}
    scaler_store: dict[str, np.ndarray] = {}
    feature_manifest: dict[str, Any] = {
        "arm_order": ARM_ORDER, "expected_dimensions": EXPECTED_DIMS,
        "feature_names": feature_name_map(), "projection_seed": PROJECTION_SEED,
        "projection_sha256": hash_array(projection), "weather_variables": WEATHER_VARIABLES,
        "package_lead": {},
    }
    for package in ("np025", "np028"):
        data = np.load(ROOT / f"outputs/next_paper/{package}/dataset.npz")
        for lead in LEADS:
            primary = run_package_lead(package, lead, data, projection, PRIMARY_EVENT_SEED, PRIMARY_TIMING_SEED)
            registry[(package, lead, PRIMARY_EVENT_SEED, PRIMARY_TIMING_SEED)] = primary
            primary_rows.extend(primary["rows"]); complexity_rows.extend(primary["complexity"])
            add_prediction_metadata(primary_predictions, package, data, lead)
            save_predictions(primary_predictions, primary, package, lead)
            scaler_store[f"{package}__{lead}__mean"] = primary["scaler"][:, 0]
            scaler_store[f"{package}__{lead}__std"] = primary["scaler"][:, 1]
            feature_manifest["package_lead"][f"{package}__{lead}"] = {
                "train_rows": int(primary["train"].sum()), "test_rows": int(primary["test"].sum()),
                "weather_scaler_sha256": hash_array(primary["scaler"]),
                "noise_sha256": hash_array(primary["noise20"]), "arms": primary["feature_map"],
            }
            if lead in ROBUST_LEADS:
                for event_seed, timing_seed in ROBUST_PAIRS:
                    robust = primary if (event_seed, timing_seed) == (PRIMARY_EVENT_SEED, PRIMARY_TIMING_SEED) else run_package_lead(package, lead, data, projection, event_seed, timing_seed)
                    registry[(package, lead, event_seed, timing_seed)] = robust
                    robust_rows.extend(robust["rows"])
                    if robust is not primary:
                        complexity_rows.extend(robust["complexity"])
                    prefix = f"event{event_seed}_timing{timing_seed}__"
                    add_prediction_metadata(robust_predictions, package, data, lead, run_prefix=prefix)
                    save_predictions(robust_predictions, robust, package, lead, run_prefix=prefix)

    np.savez_compressed(out / "train_scaler.npz", **scaler_store)
    write_json(out / "feature_manifest.json", feature_manifest)
    pd.DataFrame(primary_rows).to_csv(out / "capacity_matched_summary.csv", index=False)
    pd.DataFrame(robust_rows).to_csv(out / "seed_sensitivity.csv", index=False)
    pd.DataFrame(complexity_rows).to_csv(out / "model_complexity.csv", index=False)
    comparison_rows = build_comparisons(registry)
    pd.DataFrame(comparison_rows).to_csv(out / "comparisons.csv", index=False)
    np.savez_compressed(out / "primary_predictions.npz", **primary_predictions)
    np.savez_compressed(out / "robust_predictions.npz", **robust_predictions)

    protocol = {
        "experiment": "NP038", "stage": "CORRECTED_CAPACITY_MATCHED_INFORMATION",
        "question": "Does the information-to-process pattern persist after matching compact weather representations and no-information controls by feature dimension?",
        "packages": ["np025", "np028"], "leads": LEADS, "robust_leads": ROBUST_LEADS,
        "arms": ARM_ORDER, "dimensions": EXPECTED_DIMS,
        "event_model": "ExtraTrees 150 trees, max_depth=16, min_samples_leaf=32, max_features=1.0; primary seed 41",
        "timing_model": "ExtraTrees 100 trees, max_depth=16, min_samples_leaf=16, max_features=1.0; primary seed 43",
        "robust_seeds": {"event": ROBUST_EVENT_SEEDS, "timing": ROBUST_TIMING_SEEDS, "pairs": ROBUST_PAIRS},
        "projection": "fixed QR projection of 68 standardized weather columns to 16 columns; QR seed 20250925; scaler fitted on training rows only per package/lead",
        "noise": "16 independent standard-normal columns keyed by SHA256(timestamp_ns); the same timestamp produces the same row across leads and packages",
        "combined_dimensions": "power20 concatenated with each weather20; the four calendar columns are retained twice by construction",
        "split": "Frozen chronological train/validation/test masks from NP025/NP028; validation is retained for provenance and no test tuning is performed",
        "uncertainty": "Paired 7-day calendar-block bootstrap, 2,000 draws, seed 41; timing metrics use arrival_down < 16 rows",
        "comparisons": [{"name": name, "candidate": candidate, "reference": reference} for name, candidate, reference in COMPARISONS],
        "correction": {
            "archived_preliminary_output": "temp/np038_capacity_matched_information_defective_20260925",
            "joint_raw88": "uses frozen dataset joint_x exactly 88 columns; preliminary concatenation produced 92",
            "projection_scaling": "training-only per package/lead mean/std before fixed QR projection",
            "retrospective": True, "pre_registration_timestamp_created": False,
        },
        "source_sha256": source_info["sources"], "status": "COMPLETE_PENDING_INDEPENDENT_VERIFICATION",
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "verification.json", {"status": "PENDING_INDEPENDENT_VERIFIER", "message": "Run script/verify_np038_capacity_matched_information.py after this run."})
    (out / "status_zh.md").write_text(
        status_markdown(pd.DataFrame(primary_rows), pd.DataFrame(comparison_rows), pd.DataFrame(robust_rows)),
        encoding="utf-8",
    )
    write_json(out / "result_manifest.json", result_manifest(out))
    print(json.dumps({"status": "RUN_COMPLETE", "primary_rows": len(primary_rows), "robustness_rows": len(robust_rows), "comparison_rows": len(comparison_rows)}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run corrected NP038 capacity-matched information experiment")
    parser.add_argument("--phase", choices=["run"], default="run")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
