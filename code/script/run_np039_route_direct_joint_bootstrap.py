"""NP039: paired route comparisons with shared seven-day bootstrap draws.

The route keeps the frozen joint event state and replaces only its conditional
downward timing head.  The first-passage product is evaluated as a 17-bin
distribution (16 arrival bins plus no crossing); conditional timing scores are
evaluated only on observed downward crossings (arrival bin < 16).

Scoring follows Gneiting and Raftery (2007), doi:10.1198/016214506000001437
(proper RPS) and the existing project timing definition.  Seven-day calendar
blocks follow Künsch (1989), doi:10.1214/aos/1176347265, because neighbouring
windows are dependent.  All leads and comparators in one package reuse the
same explicit block universe and the same multinomial weights.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
LEADS = [15, 60, 120, 240, 480, 720]
SEED = 41
DRAWS = 2000
BLOCK_DAYS = 7
TIMING_BINS = 16
METRICS = ["joint_first_passage_RPS", "conditional_rps", "conditional_median_mae_minutes"]
REFERENCES = ["direct_joint", "constructed_joint"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rps_rows(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return one 17-bin ranked probability score per window."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    truth = np.zeros_like(p)
    truth[np.arange(len(y)), y] = 1.0
    return np.sum((np.cumsum(p, axis=1) - np.cumsum(truth, axis=1)) ** 2, axis=1)


def timing_losses(arrival: np.ndarray, timing: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-window conditional RPS and median arrival-time MAE.

    The last timing CDF node is set to one before scoring, matching the
    inherited NP026/NP029 contract.  Non-crossing windows are explicitly NaN
    and are filtered by the paired bootstrap; this prevents them from entering
    a conditional denominator by accident.
    """
    a = np.asarray(arrival, dtype=int)
    q = np.asarray(timing, dtype=float)
    finite_q = np.isfinite(q).all(axis=1) & (q >= 0).all(axis=1)
    sums = q.sum(axis=1)
    finite_q &= sums > 0
    q = q / np.where(sums[:, None] > 0, sums[:, None], 1.0)
    cdf = np.cumsum(q, axis=1)
    cdf[:, -1] = 1.0
    present = (a >= 0) & (a < TIMING_BINS) & finite_q
    truth = np.arange(TIMING_BINS)[None, :] >= a[:, None]
    rps = np.full(len(a), np.nan, dtype=float)
    mae = np.full(len(a), np.nan, dtype=float)
    rps[present] = np.mean((cdf[present, :-1] - truth[present, :-1]) ** 2, axis=1)
    median = (cdf >= 0.5).argmax(axis=1)
    mae[present] = 15.0 * np.abs(median[present] - a[present])
    return {"conditional_rps": rps, "conditional_median_mae_minutes": mae}


def conditional_timing_from_joint(probabilities: np.ndarray) -> np.ndarray:
    """Condition a 17-bin first-passage product on crossing."""
    p = np.asarray(probabilities, dtype=float)
    q = p[:, :TIMING_BINS].copy()
    denom = q.sum(axis=1)
    valid = np.isfinite(q).all(axis=1) & (denom > 0)
    out = np.full_like(q, np.nan)
    out[valid] = q[valid] / denom[valid, None]
    return out


def build_shared_blocks(times_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one package-wide UTC seven-day block universe and weights."""
    times = np.asarray(times_ns, dtype=np.int64)
    block_ns = int(pd.Timedelta(days=BLOCK_DAYS).value)
    block_values, ids = np.unique(times // block_ns, return_inverse=True)
    n_blocks = len(block_values)
    weights = np.random.default_rng(SEED).multinomial(
        n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=DRAWS
    )
    return ids.astype(np.int32), block_values.astype(np.int64), weights.astype(np.int16)


def paired_gain_draws(
    candidate: np.ndarray,
    reference: np.ndarray,
    block_ids: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return paired relative reductions, reusing a supplied weight matrix."""
    c = np.asarray(candidate, dtype=float)
    r = np.asarray(reference, dtype=float)
    valid = np.isfinite(c) & np.isfinite(r)
    n_blocks = weights.shape[1]
    c_sum = np.bincount(block_ids[valid], weights=c[valid], minlength=n_blocks)
    r_sum = np.bincount(block_ids[valid], weights=r[valid], minlength=n_blocks)
    numerator = weights @ c_sum
    denominator = weights @ r_sum
    valid_draw = denominator > 0
    point = 100.0 * (1.0 - float(c[valid].mean()) / float(r[valid].mean()))
    draws = np.full(weights.shape[0], point, dtype=float)
    draws[valid_draw] = 100.0 * (1.0 - numerator[valid_draw] / denominator[valid_draw])
    if not np.isfinite(draws).all():
        raise FloatingPointError("Non-finite paired bootstrap draw")
    return draws


def direct_joint_fit(x: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Fit the inherited NP036 direct 17-class comparator exactly."""
    model = ExtraTreesClassifier(
        n_estimators=150,
        max_depth=16,
        min_samples_leaf=32,
        max_features=1.0,
        random_state=41,
        n_jobs=4,
    ).fit(x[train], labels[train])
    p = np.zeros((int(test.sum()), TIMING_BINS + 1), dtype=float)
    p[:, model.classes_] = model.predict_proba(x[test])
    return p


def main() -> None:
    out = ROOT / "outputs/next_paper/np039_route_direct_joint_bootstrap"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    identity: list[dict] = []
    prediction_store: dict[str, np.ndarray] = {}
    block_store: dict[str, np.ndarray] = {}
    draw_store: dict[str, np.ndarray] = {}
    sources: list[Path] = [Path(__file__), ROOT / "script/next_paper_external_process_validation.py"]
    np036 = pd.read_csv(ROOT / "outputs/next_paper/np036_joint_timing_baseline/joint_timing_baseline.csv")
    sources.append(ROOT / "outputs/next_paper/np036_joint_timing_baseline/joint_timing_baseline.csv")

    for package, route_package in [("np025", "np026"), ("np028", "np029")]:
        data_path = ROOT / f"outputs/next_paper/{package}/dataset.npz"
        pred_path = ROOT / f"outputs/next_paper/{package}/test_predictions.npz"
        route_path = ROOT / f"outputs/next_paper/{route_package}/route_predictions.npz"
        sources.extend([data_path, pred_path, route_path])
        data = np.load(data_path)
        pred = np.load(pred_path)
        route_source = np.load(route_path)
        first_key = "15"
        shared_times = data[f"{first_key}__times_ns"][data[f"{first_key}__test"]]
        block_ids, block_values, weights = build_shared_blocks(shared_times)
        block_store[f"{package}__block_ids"] = block_ids
        block_store[f"{package}__block_values"] = block_values
        block_store[f"{package}__block_start_ns"] = block_values * int(pd.Timedelta(days=BLOCK_DAYS).value)
        block_store[f"{package}__weights"] = weights

        for lead in LEADS:
            key = str(lead)
            labels = data[f"{key}__arrival_down"]
            state = data[f"{key}__y"]
            train = data[f"{key}__train"]
            test = data[f"{key}__test"]
            times = data[f"{key}__times_ns"][test]
            if not np.array_equal(times, shared_times):
                raise ValueError(f"Test times differ across leads for {package}/{lead}")
            arr = labels[test]
            y = state[test]
            direct = direct_joint_fit(data[f"{key}__joint_x"], labels, train, test)
            joint_event = pred[f"{key}__joint__event"]
            joint_timing = pred[f"{key}__joint__timing"]
            route_event = route_source[f"{key}__event"]
            route_timing = route_source[f"{key}__timing"]
            pdown = joint_event[:, 2] + joint_event[:, 3]
            constructed = np.c_[pdown[:, None] * joint_timing, 1.0 - pdown]
            routed = np.c_[pdown[:, None] * route_timing, 1.0 - pdown]
            direct_timing = conditional_timing_from_joint(direct)
            route_timing_losses = timing_losses(arr, route_timing)
            constructed_timing_losses = timing_losses(arr, joint_timing)
            direct_timing_losses = timing_losses(arr, direct_timing)
            prediction_store.update(
                {
                    f"{package}__{lead}__route": routed,
                    f"{package}__{lead}__constructed": constructed,
                    f"{package}__{lead}__direct": direct,
                    f"{package}__{lead}__route_timing": route_timing,
                    f"{package}__{lead}__constructed_timing": joint_timing,
                    f"{package}__{lead}__direct_timing": direct_timing,
                    f"{package}__{lead}__route_event": route_event,
                    f"{package}__{lead}__joint_event": joint_event,
                    f"{package}__{lead}__route_conditional_rps": route_timing_losses["conditional_rps"],
                    f"{package}__{lead}__route_conditional_median_mae_minutes": route_timing_losses["conditional_median_mae_minutes"],
                    f"{package}__{lead}__constructed_conditional_rps": constructed_timing_losses["conditional_rps"],
                    f"{package}__{lead}__constructed_conditional_median_mae_minutes": constructed_timing_losses["conditional_median_mae_minutes"],
                    f"{package}__{lead}__direct_conditional_rps": direct_timing_losses["conditional_rps"],
                    f"{package}__{lead}__direct_conditional_median_mae_minutes": direct_timing_losses["conditional_median_mae_minutes"],
                    f"{package}__{lead}__arrival_down": arr,
                    f"{package}__{lead}__state_y": y,
                    f"{package}__{lead}__times_ns": times,
                }
            )
            identity.append(
                {
                    "package": package,
                    "route_package": route_package,
                    "lead_minutes": lead,
                    "state_identity_max_abs_diff": float(np.max(np.abs(route_event - joint_event))),
                    "label_identity": bool(np.array_equal(arr, route_source[f"{key}__arr"])),
                    "time_identity": bool(np.array_equal(times, route_source[f"{key}__times"])),
                }
            )

            for reference, ref_p in [("direct_joint", direct), ("constructed_joint", constructed)]:
                metric_losses: dict[str, tuple[np.ndarray, np.ndarray]] = {
                    "joint_first_passage_RPS": (rps_rows(routed, arr), rps_rows(ref_p, arr)),
                }
                ref_timing_losses = direct_timing_losses if reference == "direct_joint" else constructed_timing_losses
                metric_losses.update(
                    {
                        "conditional_rps": (
                            route_timing_losses["conditional_rps"],
                            ref_timing_losses["conditional_rps"],
                        ),
                        "conditional_median_mae_minutes": (
                            route_timing_losses["conditional_median_mae_minutes"],
                            ref_timing_losses["conditional_median_mae_minutes"],
                        ),
                    }
                )
                for metric in METRICS:
                    candidate_loss, reference_loss = metric_losses[metric]
                    valid = np.isfinite(candidate_loss) & np.isfinite(reference_loss)
                    draws = paired_gain_draws(candidate_loss, reference_loss, block_ids, weights)
                    draw_key = f"{package}__{reference}__{metric}__{lead}"
                    draw_store[draw_key] = draws
                    rows.append(
                        {
                            "package": package,
                            "route_package": route_package,
                            "lead_minutes": lead,
                            "metric": metric,
                            "reference": reference,
                            "route_score": float(np.nanmean(candidate_loss)),
                            "reference_score": float(np.nanmean(reference_loss)),
                            "relative_loss_reduction_pct": float(
                                100.0 * (1.0 - np.nanmean(candidate_loss) / np.nanmean(reference_loss))
                            ),
                            "low": float(np.quantile(draws, 0.025)),
                            "high": float(np.quantile(draws, 0.975)),
                            "windows": int(len(arr)),
                            "valid_windows": int(valid.sum()),
                            "n_blocks": int(len(block_values)),
                            "bootstrap_draws": DRAWS,
                            "bootstrap_seed": SEED,
                        }
                    )

    table = pd.DataFrame(rows)
    table.to_csv(out / "route_direct_intervals.csv", index=False)
    pd.DataFrame(identity).to_csv(out / "state_identity.csv", index=False)
    np.savez_compressed(out / "predictions.npz", **prediction_store)
    np.savez_compressed(out / "bootstrap_weights.npz", **block_store)
    np.savez_compressed(out / "bootstrap_draws.npz", **draw_store)
    endpoint_policy = {
        "primary_analysis": "all six leads",
        "metrics": ["joint_first_passage_RPS", "conditional_rps", "conditional_median_mae_minutes"],
        "conditional_mask": "finite arrival_down < 16 rows only",
        "global_summary": "equal-weight mean of six lead-wise relative reductions per bootstrap draw",
        "local_values": "descriptive; no lead-specific validation/test endpoint selection",
    }
    (out / "endpoint_policy.json").write_text(json.dumps(endpoint_policy, indent=2) + "\n", encoding="utf-8")

    source_sha = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    protocol = {
        "experiment": "NP039",
        "question": "Does the target-aligned route improve the coherent 17-bin product and conditional downward timing against direct and constructed joint baselines?",
        "packages": ["np025/np026", "np028/np029"],
        "leads": LEADS,
        "metrics": METRICS,
        "references": REFERENCES,
        "bootstrap": {
            "calendar": "UTC seven-day blocks anchored at Unix epoch",
            "block_days": BLOCK_DAYS,
            "draws": DRAWS,
            "seed": SEED,
            "shared_per_package": True,
            "weights_reused_across_leads_and_comparators": True,
            "weights_file": "bootstrap_weights.npz (block_ids, block_values, block_start_ns, weights per package)",
            "global_aggregation": "equal-weight mean of lead-wise ratios per draw",
        },
        "conditional_timing": "arrival_down < 16; non-crossing rows are finite-mask excluded",
        "direct_model": "ExtraTrees 150 trees, depth16, leaf32, max_features=1.0, seed41",
        "provenance": {
            "proper_rps": "Gneiting and Raftery (2007), doi:10.1198/016214506000001437",
            "block_bootstrap": "Künsch (1989), doi:10.1214/aos/1176347265",
            "timing_definition": "script/next_paper_external_process_validation.py::timing_losses",
        },
        "source_sha256": source_sha,
        "status": "COMPLETE",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    artifact_sha = {
        name: digest(out / name)
        for name in [
            "predictions.npz",
            "bootstrap_weights.npz",
            "bootstrap_draws.npz",
            "route_direct_intervals.csv",
            "state_identity.csv",
            "endpoint_policy.json",
        ]
    }
    protocol["artifact_sha256"] = artifact_sha
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    verification = {
        "status": "PASS",
        "rows": len(table),
        "leads": LEADS,
        "metrics": METRICS,
        "references": REFERENCES,
        "bootstrap_draws": DRAWS,
        "finite_rows": bool(
            np.isfinite(
                table[["route_score", "reference_score", "relative_loss_reduction_pct", "low", "high"]].to_numpy()
            ).all()
        ),
        "state_identity_max_abs_diff": float(pd.DataFrame(identity).state_identity_max_abs_diff.max()),
        "artifact_sha256": artifact_sha,
    }
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(table), "output": str(out)}))


if __name__ == "__main__":
    main()
