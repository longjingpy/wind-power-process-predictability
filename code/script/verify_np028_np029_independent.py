"""Independent numerical and source audit of NP028/NP029 frozen artifacts.

This verifier reads raw weather responses, cleaned Suining SCADA, frozen
features/predictions, and reported scores. It reconstructs the weather table,
power targets, model predictions, paired losses, and calendar-block intervals
without calling either experiment's scoring or verification functions.

Gneiting and Raftery (2007), doi:10.1198/016214506000001437, motivates
proper probability scores. Künsch (1989), doi:10.1214/aos/1176347265,
motivates block resampling of overlapping forecast windows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external_weather/suining_gfs_previous_day1_202401_202408"
NP028 = ROOT / "outputs/next_paper/np028"
NP029 = ROOT / "outputs/next_paper/np029"
LEADS = (15, 60, 120, 240, 480, 720)
ARMS = ("power", "weather", "joint")
METRICS = ("return_brier", "multiclass_brier", "conditional_rps", "conditional_median_mae_minutes")
N_BOOT = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weather_source_audit() -> dict:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    combined = pd.read_csv(DATA / "suining_gfs_previous_day1_hourly.csv")
    parts = []
    for entry in manifest["months"]:
        month = entry["start_date"][:7].replace("-", "")
        raw = DATA / f"response_{month}.json"
        assert sha256(raw) == entry["response_sha256"]
        payload = json.loads(raw.read_text(encoding="utf-8"))
        part = pd.read_csv(DATA / f"weather_{month}.csv")
        assert len(part) == entry["rows"] == len(payload["hourly"]["time"])
        for variable in manifest["requested_variables"]:
            np.testing.assert_allclose(part[variable].to_numpy(float),
                                       np.asarray(payload["hourly"][variable], dtype=float),
                                       rtol=0, atol=1e-12, equal_nan=True)
        parts.append(part)
    rebuilt = pd.concat(parts, ignore_index=True).sort_values("time").drop_duplicates("time")
    assert list(combined["time"]) == list(rebuilt["time"])
    for variable in [*manifest["requested_variables"], "u100_ms", "v100_ms"]:
        np.testing.assert_allclose(combined[variable], rebuilt[variable],
                                   rtol=0, atol=1e-10, equal_nan=True)
    speed = combined["wind_speed_100m_previous_day1"].to_numpy(float) / 3.6
    direction = np.deg2rad(combined["wind_direction_100m_previous_day1"].to_numpy(float))
    np.testing.assert_allclose(combined["u100_ms"], -speed * np.sin(direction), rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(combined["v100_ms"], -speed * np.cos(direction), rtol=0, atol=1e-12, equal_nan=True)
    assert manifest["issue_offset_hours"] == 24
    assert len(combined) == manifest["combined_rows"] == 5856
    return {"monthly_response_hashes_checked": len(parts), "combined_weather_rows": len(combined),
            "wind_missing_fraction": float(combined["u100_ms"].isna().mean())}


def independent_losses(labels: np.ndarray, arrival: np.ndarray, event: np.ndarray, timing: np.ndarray) -> dict[str, np.ndarray]:
    event_truth = np.eye(4)[labels]
    losses = {
        "return_brier": (event[:, 3] - (labels == 3)) ** 2,
        "multiclass_brier": np.square(event - event_truth).sum(axis=1),
    }
    cdf = timing.cumsum(axis=1)
    outcome_cdf = np.arange(16)[None, :] >= arrival[:, None]
    present = arrival < 16
    rps = np.full(len(arrival), np.nan)
    mae = np.full(len(arrival), np.nan)
    rps[present] = np.square(cdf[present, :-1] - outcome_cdf[present, :-1]).mean(axis=1)
    median = (cdf >= .5).argmax(axis=1)
    mae[present] = 15 * np.abs(median[present] - arrival[present])
    losses["conditional_rps"] = rps
    losses["conditional_median_mae_minutes"] = mae
    return losses


def reduction_interval(candidate: np.ndarray, reference: np.ndarray, time_ns: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(candidate) & np.isfinite(reference)
    a = candidate[valid]
    b = reference[valid]
    block = time_ns[valid] // pd.Timedelta(days=7).value
    _, group = np.unique(block, return_inverse=True)
    n = int(group.max()) + 1
    draws = np.random.default_rng(41).multinomial(n, np.full(n, 1 / n), size=N_BOOT)
    a_sum = np.bincount(group, weights=a, minlength=n)
    b_sum = np.bincount(group, weights=b, minlength=n)
    gains = 100 * (1 - (draws @ a_sum) / (draws @ b_sum))
    return 100 * (1 - a.mean() / b.mean()), *np.quantile(gains, [.025, .975])


def raw_power_audit(dataset: np.lib.npyio.NpzFile, protocol: dict) -> dict:
    raw = pd.read_csv(ROOT / "data/event_clean/site=suining.csv.gz", parse_dates=["timestamp"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw[raw["usable_power"].astype(bool)].copy()
    raw["turbine_id"] = raw["turbine_id"].astype(str)
    wide = raw.pivot_table(index="timestamp", columns="turbine_id", values="power_kw", aggfunc="mean")
    grid = pd.date_range(wide.index.min().floor("15min"), wide.index.max().floor("15min"), freq="15min", tz="UTC")
    wide = wide.reindex(grid)
    scales = protocol["meta"]["scales"]
    for turbine, scale in scales.items():
        wide[turbine] = wide[turbine] / scale
    aggregate = wide.mean(axis=1, skipna=True).where(wide.notna().sum(axis=1) >= 12).to_numpy(float)
    start_ns = grid[0].value
    step_ns = pd.Timedelta(minutes=15).value
    for lead in LEADS:
        key = str(lead)
        issue = dataset[f"{key}__issue_times_ns"]
        origin = dataset[f"{key}__times_ns"]
        end = dataset[f"{key}__target_end_times_ns"]
        assert np.all(origin - issue == pd.Timedelta(minutes=lead).value)
        assert np.all(end - origin == pd.Timedelta(hours=4).value)
        assert np.all(end - pd.Timedelta(hours=24).value <= issue)
        test = dataset[f"{key}__test"]
        idx = ((origin[test] - start_ns) // step_ns).astype(int)
        issue_idx = ((issue[test] - start_ns) // step_ns).astype(int)
        paths = np.column_stack([aggregate[idx + j] for j in range(17)])
        history = np.column_stack([aggregate[issue_idx - 15 + j] for j in range(16)])
        np.testing.assert_allclose(history, dataset[f"{key}__power_x"][test, :16], rtol=0, atol=1e-12)
        up = np.max(paths - np.minimum.accumulate(paths, axis=1), axis=1) >= .2 - 1e-12
        down_excur = np.maximum.accumulate(paths, axis=1) - paths
        down = np.max(down_excur, axis=1) >= .2 - 1e-12
        y = up.astype(np.int8) + 2 * down.astype(np.int8)
        hits = down_excur[:, 1:] >= .2 - 1e-12
        arrival = np.where(hits.any(axis=1), hits.argmax(axis=1), 16)
        np.testing.assert_array_equal(y, dataset[f"{key}__y"][test])
        np.testing.assert_array_equal(arrival, dataset[f"{key}__arrival_down"][test])
    return {"raw_scada_target_and_history_leads_checked": len(LEADS)}


def replay_and_score() -> tuple[dict, dict]:
    protocol = json.loads((NP028 / "protocol.json").read_text(encoding="utf-8"))
    dataset = np.load(NP028 / "dataset.npz")
    frozen = np.load(NP028 / "test_predictions.npz")
    reported = pd.read_csv(NP028 / "test_summary.csv")
    assert len(reported) == 144
    raw_check = raw_power_audit(dataset, protocol)
    max_prediction_difference = 0.0
    score_rows = 0
    replayed_fits = 0
    for lead in LEADS:
        key = str(lead)
        y = dataset[f"{key}__y"]
        arrival = dataset[f"{key}__arrival_down"]
        train = dataset[f"{key}__train"]
        test = dataset[f"{key}__test"]
        times = dataset[f"{key}__times_ns"][test]
        np.testing.assert_array_equal(y[test], frozen[f"{key}__y"])
        np.testing.assert_array_equal(arrival[test], frozen[f"{key}__arr"])
        np.testing.assert_array_equal(times, frozen[f"{key}__times"])
        event_frequency = np.bincount(y[train], minlength=4).astype(float)
        event_frequency /= event_frequency.sum()
        event_ref = np.broadcast_to(event_frequency, (test.sum(), 4))
        positive = arrival[train] < 16
        timing_counts = np.bincount(arrival[train][positive], minlength=16).astype(float) + 1
        timing_ref = np.broadcast_to(timing_counts / timing_counts.sum(), (test.sum(), 16))
        baseline = independent_losses(y[test], arrival[test], event_ref, timing_ref)
        arm_losses = {}
        for arm in ARMS:
            x = dataset[f"{key}__{arm}_x"]
            event_model = ExtraTreesClassifier(n_estimators=150, max_depth=16, min_samples_leaf=32,
                                               max_features=1.0, random_state=41, n_jobs=4)
            event_model.fit(x[train], y[train])
            event_pred = np.zeros((test.sum(), 4))
            event_pred[:, event_model.classes_] = event_model.predict_proba(x[test])
            replayed_fits += 1
            timing_model = ExtraTreesClassifier(n_estimators=100, max_depth=16, min_samples_leaf=16,
                                                max_features=1.0, random_state=43, n_jobs=4)
            timing_model.fit(x[train][positive], arrival[train][positive])
            timing_pred = np.array(timing_ref, copy=True)
            timing_pred[:, timing_model.classes_] = timing_model.predict_proba(x[test])
            timing_pred = (timing_pred + 1e-6) / (timing_pred + 1e-6).sum(axis=1, keepdims=True)
            replayed_fits += 1
            for kind, prediction in [("event", event_pred), ("timing", timing_pred)]:
                frozen_prediction = frozen[f"{key}__{arm}__{kind}"]
                difference = float(np.max(np.abs(prediction - frozen_prediction)))
                max_prediction_difference = max(max_prediction_difference, difference)
                np.testing.assert_allclose(prediction, frozen_prediction, rtol=0, atol=1e-12)
            arm_losses[arm] = independent_losses(y[test], arrival[test], event_pred, timing_pred)
            for metric in METRICS:
                task = "event" if metric in METRICS[:2] else "downward_timing"
                row = reported[(reported.lead_minutes == lead) & (reported.arm == arm) &
                               (reported.metric == metric) & (reported.reference == "frequency")]
                assert len(row) == 1
                actual = reduction_interval(arm_losses[arm][metric], baseline[metric], times)
                np.testing.assert_allclose(actual, row[["relative_loss_reduction_pct", "low", "high"]].iloc[0].to_numpy(float), rtol=0, atol=1e-10)
                assert row.iloc[0].task == task
                score_rows += 1
        comparisons = [("weather", "power", "weather_increment"),
                       ("joint", "power", "joint_increment"),
                       ("joint", "weather", "power_increment_on_weather")]
        for candidate, reference, label in comparisons:
            for metric in METRICS:
                row = reported[(reported.lead_minutes == lead) & (reported.arm == label) &
                               (reported.metric == metric) & (reported.candidate == candidate) &
                               (reported.reference == reference)]
                assert len(row) == 1
                actual = reduction_interval(arm_losses[candidate][metric], arm_losses[reference][metric], times)
                np.testing.assert_allclose(actual, row[["relative_loss_reduction_pct", "low", "high"]].iloc[0].to_numpy(float), rtol=0, atol=1e-10)
                score_rows += 1
    assert score_rows == len(reported) == 144
    return raw_check, {"model_fits_replayed": replayed_fits, "score_rows_recomputed": score_rows,
                       "maximum_prediction_difference": max_prediction_difference}


def route_audit() -> dict:
    source = np.load(NP028 / "test_predictions.npz")
    route = np.load(NP029 / "route_predictions.npz")
    table = pd.read_csv(NP029 / "routing_summary.csv")
    identity = pd.read_csv(NP029 / "state_identity.csv")
    assert len(table) == len(identity) == 12
    rows = 0
    max_state_difference = 0.0
    for lead in LEADS:
        key = str(lead)
        event_difference = float(np.max(np.abs(route[f"{key}__event"] - source[f"{key}__joint__event"])))
        max_state_difference = max(max_state_difference, event_difference)
        np.testing.assert_allclose(route[f"{key}__timing"], source[f"{key}__power__timing"], rtol=0, atol=0)
        np.testing.assert_array_equal(route[f"{key}__arr"], source[f"{key}__arr"])
        np.testing.assert_array_equal(route[f"{key}__times"], source[f"{key}__times"])
        assert (identity[identity.lead_minutes == lead].max_absolute_probability_difference == event_difference).all()
        labels = source[f"{key}__y"]
        arrival = source[f"{key}__arr"]
        times = source[f"{key}__times"]
        candidate_loss = independent_losses(labels, arrival, route[f"{key}__event"], route[f"{key}__timing"])
        reference_loss = independent_losses(labels, arrival, source[f"{key}__joint__event"], source[f"{key}__joint__timing"])
        for metric in METRICS[2:]:
            row = table[(table.lead_minutes == lead) & (table.metric == metric)]
            assert len(row) == 1
            actual = reduction_interval(candidate_loss[metric], reference_loss[metric], times)
            np.testing.assert_allclose(actual, row[["relative_loss_reduction_pct", "low", "high"]].iloc[0].to_numpy(float), rtol=0, atol=1e-10)
            rows += 1
    assert max_state_difference == 0.0
    return {"routing_rows_recomputed": rows, "state_probability_max_difference": max_state_difference}


def main() -> None:
    source = weather_source_audit()
    raw, scores = replay_and_score()
    route = route_audit()
    report = {"status": "PASS", "weather": source, "power": raw, "np028": scores, "np029": route}
    (NP028 / "independent_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (NP029 / "independent_verification.json").write_text(json.dumps({"status": "PASS", **route}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
