"""Independent checks for the NP041 decision interface and NP023 scope."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/next_paper/np041_all_lead_decision_sensitivity"
LEADS = [15, 60, 120, 240, 480, 720]
RATIOS = [2.0, 5.0, 10.0]
POLICIES = ["np008_full_rolling", "no_alert", "always_alert", "train_frequency", "validation_frequency", "validation_constant"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cost(alert: np.ndarray, event: np.ndarray, ratio: float) -> np.ndarray:
    a = np.asarray(alert, dtype=bool)
    e = np.asarray(event, dtype=bool)
    return np.where(a & ~e, 1.0, np.where(~a & e, ratio, 0.0))


def main() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    for name, expected in protocol["source_sha256"].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in protocol["artifact_sha256"].items():
        assert digest(OUT / name) == expected, name

    decision = pd.read_csv(OUT / "decision_summary.csv")
    policy = pd.read_csv(OUT / "policy_summary.csv")
    receipt = pd.read_csv(OUT / "selection_receipt.csv")
    pairs = pd.read_csv(OUT / "paired_intervals.csv")
    np023 = pd.read_csv(OUT / "np023_comparison.csv")
    losses = np.load(OUT / "losses.npz")
    boot = np.load(OUT / "bootstrap_weights.npz")

    assert len(decision) == 18
    assert sorted(decision.lead_minutes.unique().tolist()) == LEADS
    assert sorted(decision.cost_ratio.unique().tolist()) == RATIOS
    assert decision.method.eq("np008_full_rolling").all()
    assert np.isfinite(decision[["value", "low", "high", "forecast_cost", "baseline_cost", "no_alert_cost", "cost_reduction_vs_no_alert_pct"]].to_numpy()).all()
    assert (decision.high >= decision.low).all()
    assert len(policy) == 6 * 3 * 6 * 2 * 2
    assert set(policy.method) == set(POLICIES)
    assert set(policy.metric) == {"actual_cost", "alert_rate"}
    assert np.isfinite(policy[["value", "alert_rate", "threshold", "validation_cost"]].to_numpy()).all()
    assert len(receipt) == 6 * 3 * 6
    assert receipt.test_not_used_for_selection.all()
    assert len(pairs) == 18
    assert len(np023) == 24
    assert sorted(np023.lead_minutes.unique().tolist()) == [15, 720]
    assert len(losses.files) == 6 * 3 * 6 * 2
    assert boot["weights"].shape[0] == 2000
    assert boot["block_ids"].shape[0] == 6496

    # Replay the six-lead forecast and validation-constant cost reductions.
    data = np.load(ROOT / "outputs/next_paper/np008/dataset.npz")
    test_pred = np.load(ROOT / "outputs/next_paper/np008/test_predictions.npz")
    val_pred = np.load(ROOT / "outputs/next_paper/np008/validation_predictions.npz")
    yv = data["outcome"][data["validation"]] == 3
    yt = data["outcome"][data["test"]] == 3
    ids = boot["block_ids"]
    weights = boot["weights"]
    for _, row in decision.iterrows():
        lead = int(row.lead_minutes)
        ratio = float(row.cost_ratio)
        threshold = float(row.threshold)
        pval = val_pred[f"{lead}__full__rolling"][:, 3]
        ptest = test_pred[f"{lead}__full__rolling"][:, 3]
        # Receipt is validation-only; locate the same threshold without reading test losses.
        v_losses = np.asarray([cost(pval >= t, yv, ratio).mean() for t in np.linspace(0, 1, 201)])
        best = np.flatnonzero(np.isclose(v_losses, v_losses.min(), atol=1e-15, rtol=0))[0]
        assert np.isclose(threshold, np.linspace(0, 1, 201)[best])
        action = json.loads((OUT / "validation_constant_receipt.json").read_text())[f"{lead}__{ratio:g}"]["method"]
        alert = ptest >= threshold
        baseline_alert = np.zeros(len(yt), dtype=bool) if action == "no_alert" else np.ones(len(yt), dtype=bool)
        candidate = cost(alert, yt, ratio)
        baseline = cost(baseline_alert, yt, ratio)
        expected = 100.0 * (1.0 - candidate.mean() / baseline.mean())
        assert np.isclose(row.value, expected, atol=1e-12)
        assert np.isclose(row.forecast_cost, candidate.mean(), atol=1e-12)
        assert np.isclose(row.baseline_cost, baseline.mean(), atol=1e-12)
        no_alert = cost(np.zeros(len(yt), dtype=bool), yt, ratio)
        assert np.isclose(row.no_alert_cost, no_alert.mean(), atol=1e-12)
        assert np.isclose(row.cost_reduction_vs_no_alert_pct, 100.0 * (1.0 - candidate.mean() / no_alert.mean()), atol=1e-12)
        csum = np.bincount(ids, weights=candidate, minlength=weights.shape[1])
        bsum = np.bincount(ids, weights=baseline, minlength=weights.shape[1])
        draws = 100.0 * (1.0 - (weights @ csum) / (weights @ bsum))
        assert np.isclose(row.low, np.quantile(draws, 0.025), atol=1e-12)
        assert np.isclose(row.high, np.quantile(draws, 0.975), atol=1e-12)

    c10 = policy[(policy.split == "test") & (policy.cost_ratio == 10.0) & (policy.lead_minutes == 720) & (policy.metric == "actual_cost")].set_index("method").value
    assert "no_alert" in c10.index and "always_alert" in c10.index
    assert np.isfinite(c10["no_alert"]) and np.isfinite(c10["always_alert"])
    print(json.dumps({"status": "PASS", "decision_rows": len(decision), "policy_rows": len(policy), "np023_leads": [15, 720]}))


if __name__ == "__main__":
    main()
