"""Independent replay of NP039/NP040 rows, intervals, identity and hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEADS = [15, 60, 120, 240, 480, 720]
PACKAGES = {"np025": "np026", "np028": "np029"}
REFERENCES = ["direct_joint", "constructed_joint"]
METRICS = ["joint_first_passage_RPS", "conditional_rps", "conditional_median_mae_minutes"]
TIMING_BINS = 16


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def independent_rps(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    truth = np.eye(p.shape[1], dtype=float)[y]
    return np.square(np.cumsum(p - truth, axis=1)).sum(axis=1)


def independent_timing_losses(arrival: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    a = np.asarray(arrival, dtype=int)
    probs = np.asarray(q, dtype=float)
    sums = probs.sum(axis=1)
    valid_q = np.isfinite(probs).all(axis=1) & (probs >= 0).all(axis=1) & (sums > 0)
    probs = probs / np.where(sums[:, None] > 0, sums[:, None], 1.0)
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    present = valid_q & (a >= 0) & (a < TIMING_BINS)
    truth = np.arange(TIMING_BINS)[None, :] >= a[:, None]
    rps = np.full(len(a), np.nan)
    mae = np.full(len(a), np.nan)
    rps[present] = ((cdf[present, :-1] - truth[present, :-1]) ** 2).mean(axis=1)
    median = (cdf >= 0.5).argmax(axis=1)
    mae[present] = 15.0 * np.abs(median[present] - a[present])
    return {"conditional_rps": rps, "conditional_median_mae_minutes": mae}


def independent_gain_draws(candidate: np.ndarray, reference: np.ndarray, ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    c = np.asarray(candidate, dtype=float)
    r = np.asarray(reference, dtype=float)
    valid = np.isfinite(c) & np.isfinite(r)
    n_blocks = weights.shape[1]
    c_sum = np.bincount(ids[valid], weights=c[valid], minlength=n_blocks)
    r_sum = np.bincount(ids[valid], weights=r[valid], minlength=n_blocks)
    c_draw = weights @ c_sum
    r_draw = weights @ r_sum
    point = 100.0 * (1.0 - c[valid].mean() / r[valid].mean())
    result = np.full(len(weights), point)
    good = r_draw > 0
    result[good] = 100.0 * (1.0 - c_draw[good] / r_draw[good])
    assert np.isfinite(result).all()
    return result


def check_source_hashes(protocol: dict) -> None:
    for name, expected in protocol["source_sha256"].items():
        actual = digest(ROOT / name)
        assert actual == expected, (name, actual, expected)


def main() -> None:
    route_dir = ROOT / "outputs/next_paper/np039_route_direct_joint_bootstrap"
    global_dir = ROOT / "outputs/next_paper/np040_route_global_summary"
    route_protocol = json.loads((route_dir / "protocol.json").read_text())
    check_source_hashes(route_protocol)
    for name, expected in route_protocol["artifact_sha256"].items():
        assert digest(route_dir / name) == expected, name

    intervals = pd.read_csv(route_dir / "route_direct_intervals.csv")
    predictions = np.load(route_dir / "predictions.npz")
    blocks = np.load(route_dir / "bootstrap_weights.npz")
    draws = np.load(route_dir / "bootstrap_draws.npz")
    assert len(intervals) == len(PACKAGES) * len(LEADS) * len(REFERENCES) * len(METRICS)
    assert set(intervals.metric) == set(METRICS)
    assert np.isfinite(intervals[["route_score", "reference_score", "relative_loss_reduction_pct", "low", "high"]].to_numpy()).all()

    identity = pd.read_csv(route_dir / "state_identity.csv")
    assert len(identity) == len(PACKAGES) * len(LEADS)
    assert identity.state_identity_max_abs_diff.max() <= 1e-15
    assert identity.label_identity.all() and identity.time_identity.all()

    # Recompute every NP039 row from saved probabilities and the same package weights.
    for package, route_package in PACKAGES.items():
        ids = blocks[f"{package}__block_ids"]
        weights = blocks[f"{package}__weights"]
        assert weights.shape[0] == 2000
        for lead in LEADS:
            prefix = f"{package}__{lead}__"
            arr = predictions[prefix + "arrival_down"]
            y = arr
            route_p = predictions[prefix + "route"]
            constructed_p = predictions[prefix + "constructed"]
            direct_p = predictions[prefix + "direct"]
            route_t = predictions[prefix + "route_timing"]
            joint_t = predictions[prefix + "constructed_timing"]
            direct_t = predictions[prefix + "direct_timing"]
            route_timing_replay = independent_timing_losses(arr, route_t)
            joint_timing_replay = independent_timing_losses(arr, joint_t)
            direct_timing_replay = independent_timing_losses(arr, direct_t)
            assert np.allclose(predictions[prefix + "route_conditional_rps"], route_timing_replay["conditional_rps"], equal_nan=True)
            assert np.allclose(predictions[prefix + "route_conditional_median_mae_minutes"], route_timing_replay["conditional_median_mae_minutes"], equal_nan=True)
            assert np.allclose(predictions[prefix + "constructed_conditional_rps"], joint_timing_replay["conditional_rps"], equal_nan=True)
            assert np.allclose(predictions[prefix + "constructed_conditional_median_mae_minutes"], joint_timing_replay["conditional_median_mae_minutes"], equal_nan=True)
            assert np.allclose(predictions[prefix + "direct_conditional_rps"], direct_timing_replay["conditional_rps"], equal_nan=True)
            assert np.allclose(predictions[prefix + "direct_conditional_median_mae_minutes"], direct_timing_replay["conditional_median_mae_minutes"], equal_nan=True)
            assert len(y) == len(ids) == len(predictions[prefix + "times_ns"])
            metric_losses = {
                "joint_first_passage_RPS": {
                    "direct_joint": (independent_rps(route_p, y), independent_rps(direct_p, y)),
                    "constructed_joint": (independent_rps(route_p, y), independent_rps(constructed_p, y)),
                },
                "conditional_rps": {
                    "direct_joint": (independent_timing_losses(arr, route_t)["conditional_rps"], independent_timing_losses(arr, direct_t)["conditional_rps"]),
                    "constructed_joint": (independent_timing_losses(arr, route_t)["conditional_rps"], independent_timing_losses(arr, joint_t)["conditional_rps"]),
                },
                "conditional_median_mae_minutes": {
                    "direct_joint": (independent_timing_losses(arr, route_t)["conditional_median_mae_minutes"], independent_timing_losses(arr, direct_t)["conditional_median_mae_minutes"]),
                    "constructed_joint": (independent_timing_losses(arr, route_t)["conditional_median_mae_minutes"], independent_timing_losses(arr, joint_t)["conditional_median_mae_minutes"]),
                },
            }
            for reference in REFERENCES:
                for metric in METRICS:
                    candidate, reference_loss = metric_losses[metric][reference]
                    valid = np.isfinite(candidate) & np.isfinite(reference_loss)
                    replay_draws = independent_gain_draws(candidate, reference_loss, ids, weights)
                    row = intervals[(intervals.package == package) & (intervals.lead_minutes == lead) & (intervals.reference == reference) & (intervals.metric == metric)].iloc[0]
                    assert int(row.valid_windows) == int(valid.sum())
                    assert np.isclose(row.route_score, np.nanmean(candidate), atol=1e-12)
                    assert np.isclose(row.reference_score, np.nanmean(reference_loss), atol=1e-12)
                    expected_point = 100.0 * (1.0 - np.nanmean(candidate) / np.nanmean(reference_loss))
                    assert np.isclose(row.relative_loss_reduction_pct, expected_point, atol=1e-12)
                    assert np.isclose(row.low, np.quantile(replay_draws, 0.025), atol=1e-12)
                    assert np.isclose(row.high, np.quantile(replay_draws, 0.975), atol=1e-12)
                    key = f"{package}__{reference}__{metric}__{lead}"
                    assert np.allclose(draws[key], replay_draws, atol=1e-12)

            # Route state, labels and times retain identity with the frozen source package.
            source = np.load(ROOT / f"outputs/next_paper/{route_package}/route_predictions.npz")
            assert np.array_equal(predictions[prefix + "route_event"], source[f"{lead}__event"])
            assert np.array_equal(predictions[prefix + "arrival_down"], source[f"{lead}__arr"])
            assert np.array_equal(predictions[prefix + "times_ns"], source[f"{lead}__times"])

    # NP036 is an inherited point-score receipt; all 24 joint rows must match.
    np036 = pd.read_csv(ROOT / "outputs/next_paper/np036_joint_timing_baseline/joint_timing_baseline.csv")
    for _, old in np036.iterrows():
        reference = "constructed_joint" if old.get("reference") == "constructed_joint" else "direct_joint"
        row = intervals[(intervals.package == old.package) & (intervals.lead_minutes == old.lead_minutes) & (intervals.reference == reference) & (intervals.metric == "joint_first_passage_RPS")].iloc[0]
        assert np.isclose(row.route_score, old.route_score, atol=1e-12)
        assert np.isclose(row.reference_score, old.direct_joint_score, atol=1e-12)
        assert np.isclose(row.relative_loss_reduction_pct, old.route_gain_vs_direct_pct, atol=1e-12)

    # Recompute NP040 as mean-of-ratios per draw and reject the old CI-average columns.
    global_summary = pd.read_csv(global_dir / "global_summary.csv")
    assert len(global_summary) == len(PACKAGES) * len(REFERENCES) * len(METRICS)
    assert "mean_low_of_pointwise_ci" not in global_summary.columns
    assert "mean_high_of_pointwise_ci" not in global_summary.columns
    for _, row in global_summary.iterrows():
        arrays = [draws[f"{row.package}__{row.reference}__{row.metric}__{lead}"] for lead in LEADS]
        global_draws = np.mean(np.vstack(arrays), axis=0)
        point = float(global_draws.mean())
        assert np.isclose(row.equal_weight_mean_gain_pct, point, atol=1e-12)
        assert np.isclose(row.global_low, np.quantile(global_draws, 0.025), atol=1e-12)
        assert np.isclose(row.global_high, np.quantile(global_draws, 0.975), atol=1e-12)
        assert int(row.n_leads) == 6

    print(json.dumps({"status": "PASS", "route_rows": len(intervals), "global_rows": len(global_summary), "np036_rows_matched": len(np036)}))


if __name__ == "__main__":
    main()
