"""NP041: six-lead forecast-to-alert decision sensitivity.

NP008 full rolling probabilities are frozen.  A threshold is selected only on
the validation predictions for each lead and cost ratio, then applied once to
the test cohort.  The output also retains no-alert, always-alert, training
frequency and validation-frequency constant policies so a C=10 result is
interpretable against explicit constant baselines.  NP023 is copied as a
two-lead receipt; no intermediate NP023 lead is fabricated.

The cost matrix is a study-defined sensitivity screen (false-positive cost 1,
false-negative cost C).  Künsch (1989), doi:10.1214/aos/1176347265, motivates
the paired seven-day block uncertainty for dependent target windows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/next_paper"
LEADS = [15, 60, 120, 240, 480, 720]
COST_RATIOS = [2.0, 5.0, 10.0]
THRESHOLDS = np.linspace(0.0, 1.0, 201)
DRAWS = 2000
SEED = 41
BLOCK_DAYS = 7
FORECAST_METHOD = "np008_full_rolling"
CONSTANT_METHODS = ["no_alert", "always_alert", "train_frequency", "validation_frequency", "validation_constant"]
POLICY_METHODS = [FORECAST_METHOD, *CONSTANT_METHODS]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decision_cost(alert: np.ndarray, event: np.ndarray, ratio: float) -> np.ndarray:
    """False-negative cost is ratio; false-positive cost is one."""
    a = np.asarray(alert, dtype=bool)
    e = np.asarray(event, dtype=bool)
    return np.where(a & ~e, 1.0, np.where(~a & e, float(ratio), 0.0)).astype(float)


def select_threshold(probability: np.ndarray, event: np.ndarray, ratio: float) -> tuple[float, float]:
    """Select the lowest threshold among tied validation losses."""
    p = np.asarray(probability, dtype=float)
    y = np.asarray(event, dtype=bool)
    losses = np.asarray([decision_cost(p >= t, y, ratio).mean() for t in THRESHOLDS])
    best = np.flatnonzero(np.isclose(losses, losses.min(), rtol=0.0, atol=1e-15))[0]
    return float(THRESHOLDS[best]), float(losses[best])


def choose_validation_constant(event: np.ndarray, ratio: float) -> tuple[str, float]:
    """Choose no-alert or always-alert using validation only."""
    y = np.asarray(event, dtype=bool)
    no_cost = float(decision_cost(np.zeros(len(y), dtype=bool), y, ratio).mean())
    always_cost = float(decision_cost(np.ones(len(y), dtype=bool), y, ratio).mean())
    if no_cost <= always_cost:
        return "no_alert", no_cost
    return "always_alert", always_cost


def build_shared_blocks(times_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block_ns = int(pd.Timedelta(days=BLOCK_DAYS).value)
    values, ids = np.unique(np.asarray(times_ns, dtype=np.int64) // block_ns, return_inverse=True)
    weights = np.random.default_rng(SEED).multinomial(len(values), np.full(len(values), 1.0 / len(values)), size=DRAWS)
    return ids.astype(np.int32), values.astype(np.int64), weights.astype(np.int16)


def paired_relative_draws(candidate: np.ndarray, reference: np.ndarray, ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    c = np.asarray(candidate, dtype=float)
    r = np.asarray(reference, dtype=float)
    n_blocks = weights.shape[1]
    c_sum = np.bincount(ids, weights=c, minlength=n_blocks)
    r_sum = np.bincount(ids, weights=r, minlength=n_blocks)
    c_draw = weights @ c_sum
    r_draw = weights @ r_sum
    point = 100.0 * (1.0 - c.mean() / r.mean())
    draws = np.full(len(weights), point, dtype=float)
    valid = r_draw > 0
    draws[valid] = 100.0 * (1.0 - c_draw[valid] / r_draw[valid])
    if not np.isfinite(draws).all():
        raise FloatingPointError("Non-finite decision bootstrap draw")
    return draws


def main() -> None:
    out = BASE / "np041_all_lead_decision_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    data_path = BASE / "np008/dataset.npz"
    test_path = BASE / "np008/test_predictions.npz"
    val_path = BASE / "np008/validation_predictions.npz"
    train_path = BASE / "np007/dataset.npz"
    data = np.load(data_path)
    test_predictions = np.load(test_path)
    validation_predictions = np.load(val_path)
    train_data = np.load(train_path)
    validation_mask = data["validation"]
    test_mask = data["test"]
    validation_event = data["outcome"][validation_mask] == 3
    test_event = data["outcome"][test_mask] == 3
    test_times = data["times_ns"][test_mask]
    if not np.array_equal(test_times, data["times_ns"][test_mask]):
        raise ValueError("Inconsistent NP008 test times")
    block_ids, block_values, weights = build_shared_blocks(test_times)
    np.savez_compressed(
        out / "bootstrap_weights.npz",
        block_ids=block_ids,
        block_values=block_values,
        weights=weights,
    )

    train_frequency = float(train_data["training_frequency"][3])
    validation_frequency = float(validation_event.mean())
    policy_rows: list[dict] = []
    selection_rows: list[dict] = []
    decision_rows: list[dict] = []
    paired_rows: list[dict] = []
    loss_store: dict[str, np.ndarray] = {}
    baseline_receipt: dict[str, dict] = {}

    for lead in LEADS:
        p_val = np.asarray(validation_predictions[f"{lead}__full__rolling"][:, 3], dtype=float)
        p_test = np.asarray(test_predictions[f"{lead}__full__rolling"][:, 3], dtype=float)
        for ratio in COST_RATIOS:
            threshold, validation_cost = select_threshold(p_val, validation_event, ratio)
            constant_action, constant_validation_cost = choose_validation_constant(validation_event, ratio)
            baseline_receipt[f"{lead}__{ratio:g}"] = {
                "method": constant_action,
                "validation_cost": constant_validation_cost,
            }
            policy_specs = {
                FORECAST_METHOD: {
                    "p_val": p_val,
                    "p_test": p_test,
                    "threshold": threshold,
                    "threshold_source": "validation_only",
                    "validation_cost": validation_cost,
                    "constant_action": "thresholded_forecast",
                },
                "train_frequency": {
                    "p_val": np.full(len(validation_event), train_frequency),
                    "p_test": np.full(len(test_event), train_frequency),
                    "threshold_source": "validation_only",
                    "constant_action": "thresholded_train_frequency",
                },
                "validation_frequency": {
                    "p_val": np.full(len(validation_event), validation_frequency),
                    "p_test": np.full(len(test_event), validation_frequency),
                    "threshold_source": "validation_only",
                    "constant_action": "thresholded_validation_frequency",
                },
            }
            for method in ["train_frequency", "validation_frequency"]:
                spec = policy_specs[method]
                t, vc = select_threshold(spec["p_val"], validation_event, ratio)
                spec["threshold"] = t
                spec["validation_cost"] = vc
            policy_specs["no_alert"] = {
                "p_val": np.zeros(len(validation_event)),
                "p_test": np.zeros(len(test_event)),
                "threshold": -1.0,
                "threshold_source": "deterministic_policy",
                "constant_action": "no_alert",
                "validation_cost": float(decision_cost(np.zeros(len(validation_event), dtype=bool), validation_event, ratio).mean()),
            }
            policy_specs["always_alert"] = {
                "p_val": np.ones(len(validation_event)),
                "p_test": np.ones(len(test_event)),
                "threshold": -1.0,
                "threshold_source": "deterministic_policy",
                "constant_action": "always_alert",
                "validation_cost": float(decision_cost(np.ones(len(validation_event), dtype=bool), validation_event, ratio).mean()),
            }
            policy_specs["validation_constant"] = {
                "p_val": np.zeros(len(validation_event)) if constant_action == "no_alert" else np.ones(len(validation_event)),
                "p_test": np.zeros(len(test_event)) if constant_action == "no_alert" else np.ones(len(test_event)),
                "threshold": -1.0,
                "threshold_source": "validation_constant_selection",
                "constant_action": constant_action,
                "validation_cost": constant_validation_cost,
            }
            for method, spec in policy_specs.items():
                v_alert = spec["p_val"] >= spec["threshold"] if spec["threshold"] >= 0 else np.full(len(validation_event), spec["constant_action"] == "always_alert", dtype=bool)
                t_alert = spec["p_test"] >= spec["threshold"] if spec["threshold"] >= 0 else np.full(len(test_event), spec["constant_action"] == "always_alert", dtype=bool)
                v_loss = decision_cost(v_alert, validation_event, ratio)
                t_loss = decision_cost(t_alert, test_event, ratio)
                rtag = f"r{int(ratio)}"
                loss_store[f"validation__{lead}__{rtag}__{method}"] = v_loss
                loss_store[f"test__{lead}__{rtag}__{method}"] = t_loss
                receipt = {
                    "lead_minutes": lead,
                    "cost_ratio": ratio,
                    "method": method,
                    "threshold": float(spec["threshold"]),
                    "threshold_source": spec["threshold_source"],
                    "validation_cost": float(v_loss.mean()),
                    "validation_alert_rate": float(v_alert.mean()),
                    "constant_action": spec["constant_action"],
                    "train_frequency": train_frequency,
                    "validation_frequency": validation_frequency,
                    "test_not_used_for_selection": True,
                }
                selection_rows.append(receipt)
                for split, loss, alert in [("validation", v_loss, v_alert), ("test", t_loss, t_alert)]:
                    policy_rows.append(
                        {
                            "split": split,
                            "lead_minutes": lead,
                            "cost_ratio": ratio,
                            "method": method,
                            "metric": "actual_cost",
                            "value": float(loss.mean()),
                            "windows": int(len(loss)),
                            "alert_rate": float(alert.mean()),
                            "threshold": float(spec["threshold"]),
                            "threshold_source": spec["threshold_source"],
                            "validation_cost": float(v_loss.mean()),
                            "constant_action": spec["constant_action"],
                        }
                    )
                    policy_rows.append(
                        {
                            "split": split,
                            "lead_minutes": lead,
                            "cost_ratio": ratio,
                            "method": method,
                            "metric": "alert_rate",
                            "value": float(alert.mean()),
                            "windows": int(len(loss)),
                            "alert_rate": float(alert.mean()),
                            "threshold": float(spec["threshold"]),
                            "threshold_source": spec["threshold_source"],
                            "validation_cost": float(v_loss.mean()),
                            "constant_action": spec["constant_action"],
                        }
                    )

            candidate_loss = loss_store[f"test__{lead}__{rtag}__{FORECAST_METHOD}"]
            baseline_loss = loss_store[f"test__{lead}__{rtag}__validation_constant"]
            no_alert_loss = loss_store[f"test__{lead}__{rtag}__no_alert"]
            draws = paired_relative_draws(candidate_loss, baseline_loss, block_ids, weights)
            point = float(100.0 * (1.0 - candidate_loss.mean() / baseline_loss.mean()))
            decision_rows.append(
                {
                    "lead_minutes": lead,
                    "cost_ratio": ratio,
                    "method": FORECAST_METHOD,
                    "metric": "cost_reduction_vs_validation_constant_pct",
                    "value": point,
                    "low": float(np.quantile(draws, 0.025)),
                    "high": float(np.quantile(draws, 0.975)),
                    "forecast_cost": float(candidate_loss.mean()),
                    "baseline_method": "validation_constant",
                    "baseline_cost": float(baseline_loss.mean()),
                    "no_alert_cost": float(no_alert_loss.mean()),
                    "cost_reduction_vs_no_alert_pct": float(100.0 * (1.0 - candidate_loss.mean() / no_alert_loss.mean())),
                    "threshold": float(threshold),
                    "validation_cost": float(validation_cost),
                    "alert_rate": float((p_test >= threshold).mean()),
                    "windows": int(len(candidate_loss)),
                    "bootstrap_draws": DRAWS,
                    "bootstrap_seed": SEED,
                }
            )
            paired_rows.append(
                {
                    "lead_minutes": lead,
                    "cost_ratio": ratio,
                    "candidate": FORECAST_METHOD,
                    "reference": "validation_constant",
                    "relative_cost_reduction_pct": point,
                    "low": float(np.quantile(draws, 0.025)),
                    "high": float(np.quantile(draws, 0.975)),
                }
            )

    decision = pd.DataFrame(decision_rows)
    policy = pd.DataFrame(policy_rows)
    receipt = pd.DataFrame(selection_rows)
    pairs = pd.DataFrame(paired_rows)
    decision.to_csv(out / "decision_summary.csv", index=False)
    policy.to_csv(out / "policy_summary.csv", index=False)
    receipt.to_csv(out / "selection_receipt.csv", index=False)
    pairs.to_csv(out / "paired_intervals.csv", index=False)
    np.savez_compressed(out / "losses.npz", **loss_store)
    np023_pairs = pd.read_csv(BASE / "np023/paired_decision_intervals.csv")
    if set(np023_pairs.lead_minutes) != {15, 720}:
        raise ValueError("NP023 comparison must retain its two original leads")
    np023_pairs.to_csv(out / "np023_comparison.csv", index=False)
    (out / "validation_constant_receipt.json").write_text(json.dumps(baseline_receipt, indent=2) + "\n", encoding="utf-8")

    tracked = [Path(__file__), data_path, test_path, val_path, train_path, BASE / "np023/paired_decision_intervals.csv"]
    protocol = {
        "experiment": "NP041",
        "question": "Does frozen NP008 full rolling state probability reduce alert cost relative to a validation-selected constant policy across all six issue leads?",
        "source": "NP008 validation/test predictions and NP008 labels; training frequency from NP007 frozen dataset",
        "leads": LEADS,
        "cost_ratios": COST_RATIOS,
        "methods": POLICY_METHODS,
        "threshold_grid": "0.00 to 1.00 by 0.005; validation-only selection, one test application",
        "constant_policies": {
            "no_alert": "never issue an alert",
            "always_alert": "issue an alert for every window",
            "train_frequency": train_frequency,
            "validation_frequency": validation_frequency,
            "validation_constant": "lower validation cost of no_alert and always_alert; ties choose no_alert",
        },
        "uncertainty": "shared UTC seven-day blocks, 2,000 draws, seed41; frozen predictions",
        "np023_scope": "comparison receipt retains only NP023 leads 15 and 720 minutes",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in tracked},
        "status": "COMPLETE",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    artifact_sha = {name: digest(out / name) for name in ["decision_summary.csv", "policy_summary.csv", "selection_receipt.csv", "paired_intervals.csv", "losses.npz", "bootstrap_weights.npz", "np023_comparison.csv"]}
    protocol["artifact_sha256"] = artifact_sha
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    verification = {
        "status": "PASS",
        "decision_rows": len(decision),
        "policy_rows": len(policy),
        "selection_rows": len(receipt),
        "paired_rows": len(pairs),
        "six_leads": sorted(decision.lead_minutes.unique().tolist()) == LEADS,
        "np023_leads_preserved": sorted(np023_pairs.lead_minutes.unique().tolist()) == [15, 720],
        "finite_decision_values": bool(np.isfinite(decision[["value", "low", "high", "forecast_cost", "baseline_cost"]].to_numpy()).all()),
        "validation_only_thresholds": bool(receipt.test_not_used_for_selection.all()),
        "no_alert_and_always_alert_retained": set(CONSTANT_METHODS[:2]).issubset(set(policy.method)),
        "saved_loss_arrays": len(loss_store),
        "artifact_sha256": artifact_sha,
    }
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "decision_rows": len(decision), "policy_rows": len(policy), "output": str(out)}))


if __name__ == "__main__":
    main()
