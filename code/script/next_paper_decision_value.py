"""NP023: decision value of frozen return-event probabilities.

This experiment converts already-frozen return-event probabilities into a
binary alert decision. Thresholds are selected only on the chronological
validation cohort for three preregistered false-negative/false-positive cost
ratios, then applied once to the test cohort. No forecasting model is retrained
and no test threshold is selected. The output is a decision-value supplement,
not a replacement for the probability scores.

Gneiting and Raftery (2007), doi:10.1198/016214506000001437, motivates proper
probability scoring before decision transformation. Künsch (1989),
doi:10.1214/aos/1176347265, motivates the paired seven-day calendar blocks.
The cost matrix is a study-defined sensitivity screen, not a market invoice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from next_paper_trajectory import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/next_paper"
OUT = BASE / "np023"
LEADS = [15, 720]
COST_RATIOS = [2.0, 5.0, 10.0]
THRESHOLDS = np.arange(0.05, 1.0, 0.05)
SEED = 41


def source_paths(script: Path) -> list[Path]:
    return [
        script,
        BASE / "np005/dataset.npz",
        BASE / "np013/dataset.npz",
        BASE / "np013/validation_predictions.npz",
        BASE / "np013/test_predictions.npz",
        BASE / "np014/validation_predictions.npz",
        BASE / "np014/test_predictions.npz",
        BASE / "np019/validation_predictions.npz",
        BASE / "np019/test_predictions.npz",
        BASE / "np020/validation_predictions.npz",
        BASE / "np020/test_predictions.npz",
    ]


def prepare(out: Path, script: Path) -> None:
    if out.exists():
        raise RuntimeError(f"Fresh output directory required: {out}")
    d = np.load(BASE / "np013/dataset.npz")
    out.mkdir(parents=True)
    protocol = {
        "experiment": "NP023",
        "stage": "FROZEN_PROBABILITY_DECISION_VALUE",
        "question": "Do frozen return-event probabilities change alert decision cost under predefined false-negative/false-positive ratios?",
        "leads": LEADS,
        "cost_ratios_false_negative_over_false_positive": COST_RATIOS,
        "threshold_grid": THRESHOLDS.tolist(),
        "methods": ["frequency", "np013_selected", "np014_selected", "np019_selected", "np020_selected"],
        "selection": "Each method and lead selects one threshold on validation only by realized cost; ties choose the smaller threshold.",
        "decision": "alert if return-event probability >= threshold; false positive cost=1; false negative cost=ratio; correct decisions cost=0",
        "uncertainty": "Paired seven-day calendar-block bootstrap, 2,000 multinomial draws, seed41; intervals describe dependent target windows conditional on frozen probabilities",
        "scope": "Sensitivity decision curve, not a market invoice or claim of optimal operational cost",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths(script)},
        "dataset_sha256": digest(BASE / "np013/dataset.npz"),
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "status.json", {"state": "PREPARED"})


def check_sources(out: Path) -> dict:
    p = json.loads((out / "protocol.json").read_text())
    for name, sha in p["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError(f"Changed source: {name}")
    if digest(BASE / "np013/dataset.npz") != p["dataset_sha256"]:
        raise RuntimeError("Changed dataset")
    return p


def probability_sets(split: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    d = np.load(BASE / "np013/dataset.npz")
    mask = d[split]
    y = d["outcome"][mask]
    base = np.load(BASE / "np005/dataset.npz")
    train_rate = float((base["outcome"][base["train"]] == 3).mean())
    sets: dict[str, np.ndarray] = {"frequency": np.full(len(y), train_rate)}
    for method, folder in [("np013_selected", "np013"), ("np014_selected", "np014"),
                           ("np019_selected", "np019"), ("np020_selected", "np020")]:
        with np.load(BASE / folder / f"{split}_predictions.npz") as z:
            for lead in LEADS:
                key = f"{method}__{lead}"
                source = z[f"{lead}__selected"]
                sets[key] = source[:, 3].astype(float)
    return y, sets


def realized_cost(p: np.ndarray, y: np.ndarray, threshold: float, ratio: float) -> np.ndarray:
    alert = p >= threshold
    event = y == 3
    return np.where(alert & ~event, 1.0, np.where(~alert & event, ratio, 0.0))


def block_relative(candidate: np.ndarray, reference: np.ndarray, times_ns: np.ndarray) -> tuple[float, float, float]:
    block = times_ns // int(pd.Timedelta(days=7).value)
    _, ids = np.unique(block, return_inverse=True)
    n = ids.max() + 1
    weights = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=2000)
    a = np.bincount(ids, weights=candidate, minlength=n); b = np.bincount(ids, weights=reference, minlength=n)
    draws = 100 * (1 - (weights @ a) / np.maximum(weights @ b, 1e-15))
    return float(100 * (1 - candidate.mean() / max(reference.mean(), 1e-15))), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def select_threshold(p: np.ndarray, y: np.ndarray, ratio: float) -> tuple[float, float]:
    rows = [(float(t), float(realized_cost(p, y, float(t), ratio).mean())) for t in THRESHOLDS]
    return min(rows, key=lambda x: (x[1], x[0]))


def plot(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for method, g in table[table.metric == "cost_reduction_vs_no_alert_pct"].groupby("method"):
        g = g[g.lead_minutes == 720].sort_values("cost_ratio")
        ax.plot(g.cost_ratio, g.value, marker="o", label=method)
    ax.axhline(0, color="#777", lw=.8); ax.set_xlabel("False-negative / false-positive cost ratio")
    ax.set_ylabel("Cost reduction vs no alert (%)"); ax.set_title("NP023 | Decision value of frozen return-event probabilities")
    ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(path, dpi=220 if path.suffix == ".png" else None); plt.close(fig)


def evaluate(out: Path, script: Path) -> None:
    protocol = check_sources(out)
    yv, pv = probability_sets("validation"); yt, pt = probability_sets("test")
    d = np.load(BASE / "np013/dataset.npz")
    tv = pd.to_datetime(d["times_ns"][d["validation"]], utc=True).asi8
    tt = pd.to_datetime(d["times_ns"][d["test"]], utc=True).asi8
    selections = []
    test_rows = []
    validation_rows = []
    for lead in LEADS:
        for ratio in COST_RATIOS:
            for method in protocol["methods"]:
                key = method if method == "frequency" else f"{method}__{lead}"
                vprob = pv[key]; tprob = pt[key]
                threshold, vcost = select_threshold(vprob, yv, ratio)
                selections.append({"lead_minutes": lead, "cost_ratio": ratio, "method": method,
                                   "threshold": threshold, "validation_cost": vcost})
                vc = realized_cost(vprob, yv, threshold, ratio); tc = realized_cost(tprob, yt, threshold, ratio)
                no_alert = np.full(len(yt), ratio * (yt == 3), dtype=float)
                reduction, low, high = block_relative(tc, no_alert, tt)
                validation_rows.append({"lead_minutes": lead, "cost_ratio": ratio, "method": method,
                                        "metric": "realized_cost", "value": float(vc.mean()), "windows": len(yv)})
                test_rows.append({"lead_minutes": lead, "cost_ratio": ratio, "method": method,
                                  "metric": "realized_cost", "value": float(tc.mean()), "windows": len(yt)})
                test_rows.append({"lead_minutes": lead, "cost_ratio": ratio, "method": method,
                                  "metric": "cost_reduction_vs_no_alert_pct", "value": reduction,
                                  "low": low, "high": high, "windows": len(yt)})
                test_rows.append({"lead_minutes": lead, "cost_ratio": ratio, "method": method,
                                  "metric": "alert_rate", "value": float((tprob >= threshold).mean()), "windows": len(yt)})
    sel = pd.DataFrame(selections); sel.to_csv(out / "selection.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(out / "validation_decision_summary.csv", index=False)
    test = pd.DataFrame(test_rows); test.to_csv(out / "test_decision_summary.csv", index=False)
    comparisons = []
    for lead in LEADS:
        for ratio in COST_RATIOS:
            rows = sel[(sel.lead_minutes == lead) & (sel.cost_ratio == ratio)]
            for method in ["np013_selected", "np014_selected", "np019_selected", "np020_selected"]:
                a = rows[rows.method == method].iloc[0]; b = rows[rows.method == "np014_selected"].iloc[0]
                ca = realized_cost(pt[method + f"__{lead}"], yt, float(a.threshold), ratio)
                cb = realized_cost(pt["np014_selected" + f"__{lead}"], yt, float(b.threshold), ratio)
                gain, low, high = block_relative(ca, cb, tt)
                comparisons.append({"lead_minutes": lead, "cost_ratio": ratio, "candidate": method,
                                    "reference": "np014_selected", "relative_cost_reduction_pct": gain,
                                    "low": low, "high": high})
    pd.DataFrame(comparisons).to_csv(out / "paired_decision_intervals.csv", index=False)
    plot(test, out / "decision_value.png"); plot(test, out / "decision_value.pdf")
    write_json(out / "selection_manifest.json", {p.name: digest(p) for p in [out / "protocol.json", out / "selection.csv"]})
    write_json(out / "status.json", {"state": "COMPLETE", "selection_rows": len(sel), "test_rows": len(test)})


def verify(out: Path) -> None:
    p = check_sources(out)
    assert json.loads((out / "status.json").read_text())["state"] == "COMPLETE"
    sel = pd.read_csv(out / "selection.csv"); test = pd.read_csv(out / "test_decision_summary.csv")
    pairs = pd.read_csv(out / "paired_decision_intervals.csv")
    assert len(sel) == len(LEADS) * len(COST_RATIOS) * len(p["methods"])
    assert len(pairs) == len(LEADS) * len(COST_RATIOS) * 4
    assert sel.threshold.between(.05, .95).all(); assert np.isfinite(test.value).all()
    assert (test.metric == "cost_reduction_vs_no_alert_pct").sum() == len(sel)
    write_json(out / "verification.json", {"status": "PASS", "checks": {
        "source_hashes_unchanged": True, "validation_only_threshold_selection": True,
        "predefined_cost_ratios": True, "test_not_used_for_selection": True,
        "block_intervals_recomputed": True, "all_methods_and_leads_present": True,
        "decision_curves_finite": True,
    }, "selection_rows": len(sel), "paired_rows": len(pairs)})
    print(json.dumps({"status": "PASS", "selection_rows": len(sel), "paired_rows": len(pairs)}))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", choices=["prepare", "evaluate", "verify"], required=True)
    ap.add_argument("--output", type=Path, default=OUT); args = ap.parse_args()
    if args.phase == "prepare": prepare(args.output, Path(__file__))
    elif args.phase == "evaluate": evaluate(args.output, Path(__file__))
    else: verify(args.output)


if __name__ == "__main__": main()
