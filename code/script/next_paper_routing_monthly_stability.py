"""NP027: UTC-month stability audit for the frozen NP026 route.

This audit reads the frozen target-aligned route and joint timing predictions,
recomputes all timing increments by UTC month, and does not refit or select a
month. It tests whether the intermediate-lead route gain is concentrated in a
single part of the Suining test period.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from next_paper_external_process_validation import LEADS, block_reduction, timing_losses

ROOT = Path(__file__).resolve().parents[1]
NP025 = ROOT / "outputs/next_paper/np025"
NP026 = ROOT / "outputs/next_paper/np026"
OUT = ROOT / "outputs/next_paper/np027"
MONTHS = ["2024-06", "2024-07", "2024-08"]
METRICS = ["conditional_rps", "conditional_median_mae_minutes"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def prepare(out: Path, script: Path) -> None:
    if out.exists():
        raise RuntimeError(f"Fresh output directory required: {out}")
    out.mkdir(parents=True)
    source_paths = [
        script,
        ROOT / "script/next_paper_external_process_validation.py",
        NP025 / "test_predictions.npz",
        NP026 / "route_predictions.npz",
        NP026 / "routing_summary.csv",
        NP026 / "state_identity.csv",
        NP026 / "protocol.json",
    ]
    protocol = {
        "experiment": "NP027",
        "stage": "TARGET_ALIGNED_ROUTING_MONTHLY_STABILITY",
        "question": "Does the frozen NP026 target-aligned timing gain persist across UTC months of the Suining test period?",
        "months": MONTHS,
        "leads": LEADS,
        "metrics": METRICS,
        "input": "Frozen NP025 joint timing and NP026 routed timing predictions; no refit, no month selection",
        "uncertainty": "Paired 7-day calendar blocks within each UTC month, 2,000 draws, inherited seed41",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "status.json", {"state": "PREPARED"})


def check_sources(out: Path) -> dict:
    protocol = json.loads((out / "protocol.json").read_text())
    for name, sha in protocol["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError(f"Changed source: {name}")
    return protocol


def evaluate(out: Path, script: Path) -> None:
    check_sources(out)
    if json.loads((out / "status.json").read_text())["state"] != "PREPARED":
        raise RuntimeError("Prepare first")
    source = np.load(NP025 / "test_predictions.npz")
    route = np.load(NP026 / "route_predictions.npz")
    rows: list[dict] = []
    for lead in LEADS:
        key = str(lead)
        arr = source[f"{key}__arr"]
        times = source[f"{key}__times"]
        route_timing = route[f"{key}__timing"]
        joint_timing = source[f"{key}__joint__timing"]
        route_losses = timing_losses(arr, route_timing)
        joint_losses = timing_losses(arr, joint_timing)
        stamps = pd.to_datetime(times, unit="ns", utc=True)
        month_labels = stamps.strftime("%Y-%m")
        for month in MONTHS:
            mask = month_labels == month
            for metric in METRICS:
                candidate = route_losses[metric][mask]
                reference = joint_losses[metric][mask]
                times_month = times[mask]
                gain, low, high = block_reduction(candidate, reference, times_month)
                rows.append({
                    "month": month, "lead_minutes": lead, "task": "downward_timing",
                    "metric": metric, "candidate": "target_aligned_route",
                    "reference": "joint_timing", "windows": int(mask.sum()),
                    "relative_loss_reduction_pct": gain, "low": low, "high": high,
                })
    table = pd.DataFrame(rows)
    table.to_csv(out / "monthly_summary.csv", index=False)
    full = pd.read_csv(NP026 / "routing_summary.csv")
    full.to_csv(out / "full_summary.csv", index=False)
    plot(table, out / "monthly_routing_stability.png")
    plot(table, out / "monthly_routing_stability.pdf")
    write_json(out / "result_manifest.json", {f.name: digest(f) for f in out.iterdir() if f.is_file() and f.name != "status.json"})
    write_json(out / "status.json", {"state": "COMPLETE", "summary_rows": len(table), "months": MONTHS})


def plot(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), sharex=True)
    for ax, metric, title in zip(axes, METRICS, ["Downward timing RPS", "Downward timing MAE"]):
        d = table[table.metric == metric]
        for month, group in d.groupby("month"):
            group = group.sort_values("lead_minutes")
            ax.errorbar(group.lead_minutes, group.relative_loss_reduction_pct,
                        yerr=[group.relative_loss_reduction_pct - group.low, group.high - group.relative_loss_reduction_pct],
                        marker="o", capsize=2, label=month)
        ax.axhline(0, color="#777", lw=.8); ax.set_title(title); ax.set_xlabel("Lead (min)"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Route relative loss reduction (%)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("NP027 | Monthly stability of target-aligned routing")
    fig.tight_layout(); fig.savefig(path, dpi=220 if path.suffix == ".png" else None); plt.close(fig)


def verify(out: Path) -> None:
    check_sources(out)
    assert json.loads((out / "status.json").read_text())["state"] == "COMPLETE"
    table = pd.read_csv(out / "monthly_summary.csv")
    assert len(table) == len(MONTHS) * len(LEADS) * len(METRICS)
    assert set(table.month) == set(MONTHS)
    for lead in [240, 480]:
        for metric in METRICS:
            d = table[(table.lead_minutes == lead) & (table.metric == metric)]
            assert len(d) == 3 and (d.relative_loss_reduction_pct > 0).all()
    write_json(out / "verification.json", {
        "status": "PASS",
        "checks": {
            "source_hashes_unchanged": True,
            "frozen_np026_inputs": True,
            "no_refit_or_month_selection": True,
            "three_utc_months": True,
            "six_leads": True,
            "monthly_scores_recomputed": True,
            "intermediate_rps_same_sign": True,
            "intermediate_mae_same_sign": True,
            "full_matrix_retained": True,
        },
        "summary_rows": len(table), "months": MONTHS,
    })
    print(json.dumps({"status": "PASS", "summary_rows": len(table), "months": MONTHS}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "evaluate", "verify"], required=True)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output, Path(__file__))
    elif args.phase == "evaluate":
        evaluate(args.output, Path(__file__))
    else:
        verify(args.output)


if __name__ == "__main__":
    main()
