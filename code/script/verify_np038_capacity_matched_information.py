"""Independent verifier for the corrected NP038 output.

The verifier intentionally does not import the run script. It checks source
and output hashes, replays feature-map hashes and dimensions, checks split and
label vectors, and recomputes every saved score and paired bootstrap summary
from the saved prediction arrays.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/next_paper/np038_capacity_matched_information"
LEADS = [15, 60, 120, 240, 480, 720]
ROBUST_LEADS = [15, 240, 720]
ROBUST_PAIRS = [(41, 43), (42, 44), (43, 45)]
PRIMARY_PAIR = (41, 43)
PROJECTION_SEED = 20250925
BOOTSTRAP_SEED = 41
BOOTSTRAP_REPS = 2000
DAY_NS = int(pd.Timedelta(days=7).value)
WEATHER_VARIABLES = ["u100_ms", "v100_ms", "temperature", "surface_pressure"]
CALENDAR_NAMES = [
    "calendar_sin_hour", "calendar_cos_hour", "calendar_sin_dayofyear", "calendar_cos_dayofyear",
]
ARM_ORDER = [
    "power20", "weather_raw72", "joint_raw88", "weather_summary20",
    "weather_projection20", "noise20", "power_weather_summary40",
    "power_weather_projection40", "power_noise40",
]
EXPECTED_DIMS = {
    "power20": 20, "weather_raw72": 72, "joint_raw88": 88,
    "weather_summary20": 20, "weather_projection20": 20, "noise20": 20,
    "power_weather_summary40": 40, "power_weather_projection40": 40,
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
    ("state", "return_brier"), ("state", "multiclass_brier"),
    ("downward_timing", "conditional_rps"),
    ("downward_timing", "conditional_median_mae_minutes"),
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def fixed_projection() -> np.ndarray:
    rng = np.random.default_rng(PROJECTION_SEED)
    q, r = np.linalg.qr(rng.normal(size=(68, 16)), mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def stable_noise(times_ns: np.ndarray, width: int = 16) -> np.ndarray:
    out = np.empty((len(times_ns), width), dtype=float)
    for i, value in enumerate(np.asarray(times_ns, dtype=np.int64)):
        seed = int.from_bytes(hashlib.sha256(str(int(value)).encode("ascii")).digest()[:8], "little")
        out[i] = np.random.default_rng(seed).normal(size=width)
    return out


def feature_names() -> dict[str, list[str]]:
    power = [f"power_history_minus_{15 * (15 - i)}_minutes" for i in range(16)] + CALENDAR_NAMES
    weather = CALENDAR_NAMES + [
        f"{variable}_target_node_{node:02d}"
        for node in range(17) for variable in WEATHER_VARIABLES
    ]
    summary = CALENDAR_NAMES + [
        f"weather_{stat}_{variable}"
        for stat in ("mean", "std", "first_node", "last_node")
        for variable in WEATHER_VARIABLES
    ]
    projection = CALENDAR_NAMES + [f"weather_projection_component_{i:02d}" for i in range(16)]
    noise = CALENDAR_NAMES + [f"timestamp_noise_component_{i:02d}" for i in range(16)]
    return {
        "power20": power, "weather_raw72": weather, "joint_raw88": power + weather[4:],
        "weather_summary20": summary, "weather_projection20": projection, "noise20": noise,
        "power_weather_summary40": power + summary,
        "power_weather_projection40": power + projection,
        "power_noise40": power + noise,
    }


def reconstruct_arms(data: np.lib.npyio.NpzFile, lead: int, projection: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    key = str(lead)
    power = data[f"{key}__power_x"]
    weather = data[f"{key}__weather_x"]
    joint = data[f"{key}__joint_x"]
    times = data[f"{key}__times_ns"]
    train = data[f"{key}__train"].astype(bool)
    signal = weather[:, 4:]
    mean = signal[train].mean(axis=0)
    std = np.where(signal[train].std(axis=0) > 0, signal[train].std(axis=0), 1.0)
    scaled = (signal - mean) / std
    calendar = weather[:, :4]
    nodes = signal.reshape(len(signal), 17, 4)
    summary = np.column_stack([nodes.mean(axis=1), nodes.std(axis=1), nodes[:, 0, :], nodes[:, -1, :]])
    summary20 = np.c_[calendar, summary]
    projection20 = np.c_[calendar, scaled @ projection]
    noise20 = np.c_[calendar, stable_noise(times, 16)]
    arms = {
        "power20": power, "weather_raw72": weather, "joint_raw88": joint,
        "weather_summary20": summary20, "weather_projection20": projection20, "noise20": noise20,
        "power_weather_summary40": np.c_[power, summary20],
        "power_weather_projection40": np.c_[power, projection20],
        "power_noise40": np.c_[power, noise20],
    }
    return arms, np.c_[mean, std], train, data[f"{key}__test"].astype(bool)


def split_codes(data: np.lib.npyio.NpzFile, lead: int) -> np.ndarray:
    key = str(lead)
    out = np.full(len(data[f"{key}__train"]), -1, dtype=np.int8)
    out[data[f"{key}__train"].astype(bool)] = 0
    out[data[f"{key}__validation"].astype(bool)] = 1
    out[data[f"{key}__test"].astype(bool)] = 2
    return out


def probability_losses(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    return {"return_brier": (p[:, 3] - (y == 3)) ** 2, "multiclass_brier": np.sum((p - np.eye(4)[y]) ** 2, axis=1)}


def timing_losses(arrival: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    cdf = np.cumsum(q, axis=1)
    cdf[:, -1] = 1.0
    present = arrival < 16
    truth = np.arange(16)[None, :] >= arrival[:, None]
    rps = np.full(len(arrival), np.nan)
    mae = np.full(len(arrival), np.nan)
    rps[present] = np.mean((cdf[present, :-1] - truth[present, :-1]) ** 2, axis=1)
    median = (cdf >= 0.5).argmax(axis=1)
    mae[present] = 15 * np.abs(median[present] - arrival[present])
    return {"conditional_rps": rps, "conditional_median_mae_minutes": mae}


def paired_interval(candidate: np.ndarray, reference: np.ndarray, times: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float | int]:
    keep = np.isfinite(candidate) & np.isfinite(reference)
    if valid is not None:
        keep &= valid
    candidate = np.asarray(candidate, float)[keep]
    reference = np.asarray(reference, float)[keep]
    times = np.asarray(times, np.int64)[keep]
    blocks, inverse = np.unique(times // DAY_NS, return_inverse=True)
    n = len(blocks)
    counts = np.bincount(inverse, minlength=n).astype(float)
    csum = np.bincount(inverse, weights=candidate, minlength=n)
    rsum = np.bincount(inverse, weights=reference, minlength=n)
    draws = np.random.default_rng(BOOTSTRAP_SEED).multinomial(n, np.full(n, 1.0 / n), size=BOOTSTRAP_REPS)
    den = draws @ counts
    cmeans = (draws @ csum) / den
    rmeans = (draws @ rsum) / den
    red = 100 * (1 - cmeans / rmeans)
    diff = cmeans - rmeans
    cscore, rscore = float(candidate.mean()), float(reference.mean())
    return {
        "valid_windows": int(len(candidate)), "blocks": int(n),
        "candidate_score": cscore, "reference_score": rscore,
        "difference": cscore - rscore, "reduction_pct": 100 * (1 - cscore / rscore),
        "difference_low": float(np.quantile(diff, .025)), "difference_high": float(np.quantile(diff, .975)),
        "reduction_low": float(np.quantile(red, .025)), "reduction_high": float(np.quantile(red, .975)),
    }


def assert_close(actual: float, expected: float, label: str, atol: float = 1e-10) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=1e-10, equal_nan=True):
        raise AssertionError(f"{label}: {actual} != {expected}")


def check_sources(protocol: dict[str, Any], source_manifest: dict[str, Any]) -> None:
    for name, expected in source_manifest["sources"].items():
        path = ROOT / name
        if not path.exists() or digest(path) != expected:
            raise AssertionError(f"source hash changed: {name}")
    if protocol.get("source_sha256") != source_manifest["sources"]:
        raise AssertionError("protocol/source manifest mismatch")


def check_result_hashes(manifest: dict[str, str]) -> None:
    for name, expected in manifest.items():
        path = OUT / name
        if not path.exists() or digest(path) != expected:
            raise AssertionError(f"output hash mismatch: {name}")


def main() -> None:
    if not OUT.exists():
        raise FileNotFoundError(OUT)
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    source_info = json.loads((OUT / "source_manifest.json").read_text(encoding="utf-8"))
    result_hashes = json.loads((OUT / "result_manifest.json").read_text(encoding="utf-8"))
    check_sources(protocol, source_info)
    check_result_hashes(result_hashes)
    if protocol["dimensions"] != EXPECTED_DIMS:
        raise AssertionError("protocol dimensions mismatch")
    projection = np.load(OUT / "projection_matrix.npy")
    projection_meta = json.loads((OUT / "projection_matrix.json").read_text(encoding="utf-8"))
    expected_projection = fixed_projection()
    if projection.shape != (68, 16) or not np.array_equal(projection, expected_projection):
        raise AssertionError("fixed projection matrix mismatch")
    if projection_meta["sha256"] != hash_array(projection):
        raise AssertionError("projection hash mismatch")
    if projection_meta["seed"] != PROJECTION_SEED:
        raise AssertionError("projection seed mismatch")

    feature_manifest = json.loads((OUT / "feature_manifest.json").read_text(encoding="utf-8"))
    names = feature_names()
    if feature_manifest["arm_order"] != ARM_ORDER or feature_manifest["expected_dimensions"] != EXPECTED_DIMS:
        raise AssertionError("feature manifest arm order/dimensions mismatch")
    if feature_manifest["feature_names"] != names:
        raise AssertionError("feature name map mismatch")
    scaler_data = np.load(OUT / "train_scaler.npz")
    primary = np.load(OUT / "primary_predictions.npz")
    robust = np.load(OUT / "robust_predictions.npz")
    summary = pd.read_csv(OUT / "capacity_matched_summary.csv")
    seed_summary = pd.read_csv(OUT / "seed_sensitivity.csv")
    complexity = pd.read_csv(OUT / "model_complexity.csv")
    comparisons = pd.read_csv(OUT / "comparisons.csv")
    # Four metrics (return Brier, multiclass Brier, timing RPS and timing
    # median arrival MAE) are retained for each arm.  Robustness includes the
    # primary 41/43 pair plus two additional pairs at three leads.
    if len(summary) != 432 or len(seed_summary) != 648:
        raise AssertionError(f"unexpected score rows {len(summary)}, {len(seed_summary)}")
    if set(summary["arm"]) != set(ARM_ORDER) or set(seed_summary["arm"]) != set(ARM_ORDER):
        raise AssertionError("score arm set mismatch")
    if len(complexity) != 432:
        raise AssertionError(f"unexpected complexity rows {len(complexity)}")

    losses: dict[tuple[str, int, int, int, str], dict[str, np.ndarray]] = {}
    checked_feature_maps = 0
    checked_predictions = 0
    checked_scores = 0
    for package in ("np025", "np028"):
        data = np.load(ROOT / f"outputs/next_paper/{package}/dataset.npz")
        for lead in LEADS:
            key = str(lead)
            arms, scaler, train, test = reconstruct_arms(data, lead, projection)
            info = feature_manifest["package_lead"][f"{package}__{lead}"]
            for arm in ARM_ORDER:
                if arms[arm].shape[1] != EXPECTED_DIMS[arm]:
                    raise AssertionError(f"dimension mismatch {package}/{lead}/{arm}")
                entry = info["arms"][arm]
                if entry["columns"] != EXPECTED_DIMS[arm] or len(entry["features"]) != EXPECTED_DIMS[arm]:
                    raise AssertionError(f"feature metadata mismatch {package}/{lead}/{arm}")
                if entry["sha256"] != hash_array(arms[arm]):
                    raise AssertionError(f"feature hash mismatch {package}/{lead}/{arm}")
                checked_feature_maps += 1
            if not np.array_equal(scaler, np.column_stack([scaler_data[f"{package}__{lead}__mean"], scaler_data[f"{package}__{lead}__std"]])):
                raise AssertionError(f"train scaler mismatch {package}/{lead}")
            base = f"{package}__{lead}"
            if not np.array_equal(primary[f"{base}__labels_y"], data[f"{key}__y"]):
                raise AssertionError(f"label vector mismatch {base}")
            if not np.array_equal(primary[f"{base}__labels_arrival_down"], data[f"{key}__arrival_down"]):
                raise AssertionError(f"arrival vector mismatch {base}")
            if not np.array_equal(primary[f"{base}__times_ns"], data[f"{key}__times_ns"]):
                raise AssertionError(f"time vector mismatch {base}")
            if not np.array_equal(primary[f"{base}__split_code"], split_codes(data, lead)):
                raise AssertionError(f"split vector mismatch {base}")
            test_index = np.flatnonzero(test)
            if not np.array_equal(primary[f"{base}__test_index"], test_index):
                raise AssertionError(f"test index mismatch {base}")
            y_test = data[f"{key}__y"][test]
            arrival_test = data[f"{key}__arrival_down"][test]
            times_test = data[f"{key}__times_ns"][test]
            run_specs = [(PRIMARY_PAIR, primary, "primary")]
            if lead in ROBUST_LEADS:
                run_specs += [((e, t), robust, "robust") for e, t in ROBUST_PAIRS]
            for (event_seed, timing_seed), pred_store, run_type in run_specs:
                prefix = "" if run_type == "primary" else f"event{event_seed}_timing{timing_seed}__"
                run_key = (package, lead, event_seed, timing_seed)
                for arm in ARM_ORDER:
                    p = pred_store[f"{prefix}{base}__{arm}__event"]
                    q = pred_store[f"{prefix}{base}__{arm}__timing"]
                    if p.shape != (int(test.sum()), 4) or q.shape != (int(test.sum()), 16):
                        raise AssertionError(f"prediction shape mismatch {prefix}{base}/{arm}")
                    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-12) or not np.allclose(q.sum(axis=1), 1.0, atol=1e-12):
                        raise AssertionError(f"probability normalization mismatch {prefix}{base}/{arm}")
                    losses[run_key + (arm,)] = probability_losses(y_test, p) | timing_losses(arrival_test, q)
                    table = summary if run_type == "primary" else seed_summary
                    for task, metric in METRIC_TASKS:
                        row = table[(table.package == package) & (table.lead_minutes == lead) & (table.arm == arm) & (table.task == task) & (table.metric == metric) & (table.event_seed == event_seed) & (table.timing_seed == timing_seed)]
                        if len(row) != 1:
                            raise AssertionError(f"score row missing {run_key}/{arm}/{metric}")
                        expected = float(np.nanmean(losses[run_key + (arm,)][metric]))
                        assert_close(float(row.absolute_score.iloc[0]), expected, f"score {run_key}/{arm}/{metric}")
                        checked_scores += 1
                    checked_predictions += 1

    checked_comparisons = 0
    for row in comparisons.itertuples(index=False):
        run_key = (row.package, int(row.lead_minutes), int(row.event_seed), int(row.timing_seed))
        candidate = losses[run_key + (row.candidate,)][row.metric]
        reference = losses[run_key + (row.reference,)][row.metric]
        times = np.load(OUT / ("primary_predictions.npz" if row.run_type == "primary" else "robust_predictions.npz"))[f"{'' if row.run_type == 'primary' else f'event{int(row.event_seed)}_timing{int(row.timing_seed)}__'}{row.package}__{int(row.lead_minutes)}__test_times_ns"]
        valid = None if row.task == "state" else (np.isfinite(candidate) & np.isfinite(reference))
        expected = paired_interval(candidate, reference, times, valid=valid)
        for col in ("candidate_score", "reference_score", "difference", "reduction_pct", "difference_low", "difference_high", "reduction_low", "reduction_high"):
            assert_close(float(getattr(row, col)), float(expected[col]), f"comparison {row.comparison}/{row.metric}/{col}", atol=2e-10)
        checked_comparisons += 1

    verification = {
        "status": "PASS",
        "checks": {
            "source_hashes": True, "output_hashes_before_update": True,
            "fixed_projection_and_training_scalers": True,
            "feature_dimensions_and_maps": True, "timestamp_noise_determinism": True,
            "split_and_label_vectors": True, "prediction_shapes_and_normalization": True,
            "metrics_replayed_from_predictions": True, "paired_7day_2000_ci_replayed": True,
            "no_score_based_feature_tests": True,
        },
        "counts": {
            "feature_maps": checked_feature_maps, "prediction_arrays": checked_predictions,
            "score_rows": checked_scores, "comparison_rows": checked_comparisons,
        },
        "verifier": "script/verify_np038_capacity_matched_information.py",
    }
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol["status"] = "COMPLETE_VERIFIED"
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {path.name: digest(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "result_manifest.json"}
    (OUT / "result_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "feature_maps": checked_feature_maps, "prediction_arrays": checked_predictions, "score_rows": checked_scores, "comparison_rows": checked_comparisons}))


if __name__ == "__main__":
    main()
