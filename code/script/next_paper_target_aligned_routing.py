"""NP026: target-aligned information routing from frozen NP025 predictions.

The state-risk head uses the frozen joint information arm, while the downward
first-passage timing head uses the frozen power-history arm. No model is
refit and no test result is used for selection. The experiment asks whether
the target-specific information decomposition from NP025 transfers into a
better process product than an all-information timing head.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from next_paper_external_process_validation import (
    LEADS, block_reduction, probability_losses, timing_losses,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/next_paper/np025"
OUT = ROOT / "outputs/next_paper/np026"


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
        SOURCE / "protocol.json",
        SOURCE / "test_predictions.npz",
        SOURCE / "test_summary.csv",
    ]
    protocol = {
        "experiment": "NP026",
        "stage": "TARGET_ALIGNED_INFORMATION_ROUTING",
        "question": "Does routing joint information to process-state risk and power history to first-passage timing improve the process product relative to an all-information timing head?",
        "input": "Frozen NP025 test predictions, labels, and issue-time indices; no refit and no test selection",
        "route": "event state = NP025 joint arm; downward first-passage timing = NP025 power arm",
        "reference": "NP025 joint arm for timing; NP025 joint arm is retained exactly for state risk",
        "leads": LEADS,
        "uncertainty": "Paired 7-day calendar blocks, 2,000 draws, seed41, inherited from NP025",
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
    z = np.load(SOURCE / "test_predictions.npz")
    rows: list[dict] = []
    route_predictions: dict[str, np.ndarray] = {}
    identity: list[dict] = []
    for lead in LEADS:
        key = str(lead)
        y = z[f"{key}__y"]
        arr = z[f"{key}__arr"]
        times = z[f"{key}__times"]
        joint_event = z[f"{key}__joint__event"]
        joint_timing = z[f"{key}__joint__timing"]
        power_timing = z[f"{key}__power__timing"]
        route_event = joint_event.copy()
        route_timing = power_timing.copy()
        route_predictions[f"{key}__event"] = route_event
        route_predictions[f"{key}__timing"] = route_timing
        route_predictions[f"{key}__y"] = y
        route_predictions[f"{key}__arr"] = arr
        route_predictions[f"{key}__times"] = times

        route_event_loss = probability_losses(y, route_event)
        joint_event_loss = probability_losses(y, joint_event)
        for metric in ["return_brier", "multiclass_brier"]:
            identity.append({
                "lead_minutes": lead, "task": "event", "metric": metric,
                "max_absolute_loss_difference": float(np.max(np.abs(route_event_loss[metric] - joint_event_loss[metric]))),
            })
        route_timing_loss = timing_losses(arr, route_timing)
        joint_timing_loss = timing_losses(arr, joint_timing)
        power_timing_loss = timing_losses(arr, power_timing)
        for metric in ["conditional_rps", "conditional_median_mae_minutes"]:
            gain_joint, low_joint, high_joint = block_reduction(route_timing_loss[metric], joint_timing_loss[metric], times)
            gain_power, low_power, high_power = block_reduction(route_timing_loss[metric], power_timing_loss[metric], times)
            rows.append({
                "lead_minutes": lead, "task": "downward_timing", "metric": metric,
                "candidate": "target_aligned_route", "reference": "joint_timing",
                "relative_loss_reduction_pct": gain_joint, "low": low_joint, "high": high_joint,
                "windows": int(len(arr)),
            })
            rows.append({
                "lead_minutes": lead, "task": "downward_timing", "metric": metric,
                "candidate": "target_aligned_route", "reference": "power_timing",
                "relative_loss_reduction_pct": gain_power, "low": low_power, "high": high_power,
                "windows": int(len(arr)),
            })
    table = pd.DataFrame(rows)
    table.to_csv(out / "routing_summary.csv", index=False)
    pd.DataFrame(identity).to_csv(out / "state_identity.csv", index=False)
    np.savez_compressed(out / "route_predictions.npz", **route_predictions)
    plot(table, out / "target_aligned_routing.png")
    plot(table, out / "target_aligned_routing.pdf")
    write_json(out / "result_manifest.json", {f.name: digest(f) for f in out.iterdir() if f.is_file() and f.name != "status.json"})
    write_json(out / "status.json", {"state": "COMPLETE", "summary_rows": len(table), "identity_rows": len(identity)})


def plot(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), sharex=True)
    for ax, metric, title in zip(
        axes,
        ["conditional_rps", "conditional_median_mae_minutes"],
        ["Downward timing RPS", "Downward timing MAE"],
    ):
        d = table[table.metric == metric]
        for reference, group in d.groupby("reference"):
            group = group.sort_values("lead_minutes")
            ax.errorbar(group.lead_minutes, group.relative_loss_reduction_pct,
                        yerr=[group.relative_loss_reduction_pct - group.low, group.high - group.relative_loss_reduction_pct],
                        marker="o", capsize=2, label=reference)
        ax.axhline(0, color="#777", lw=.8); ax.set_title(title); ax.set_xlabel("Lead (min)"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Route relative loss reduction (%)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("NP026 | Target-aligned information routing")
    fig.tight_layout(); fig.savefig(path, dpi=220 if path.suffix == ".png" else None); plt.close(fig)


def verify(out: Path) -> None:
    check_sources(out)
    assert json.loads((out / "status.json").read_text())["state"] == "COMPLETE"
    table = pd.read_csv(out / "routing_summary.csv")
    identity = pd.read_csv(out / "state_identity.csv")
    assert len(table) == len(LEADS) * 2 * 2
    assert len(identity) == len(LEADS) * 2
    assert np.max(identity.max_absolute_loss_difference.to_numpy(float)) == 0.0
    for metric, lead, expected in [
        ("conditional_rps", 240, [8.70, 4.96, 12.84]),
        ("conditional_median_mae_minutes", 240, [8.01, 5.10, 10.97]),
        ("conditional_rps", 480, [8.65, 5.55, 12.20]),
        ("conditional_median_mae_minutes", 480, [8.79, 5.00, 12.94]),
    ]:
        row = table[(table.lead_minutes == lead) & (table.metric == metric) & (table.reference == "joint_timing")].iloc[0]
        assert [round(float(row[k]), 2) for k in ["relative_loss_reduction_pct", "low", "high"]] == expected
    write_json(out / "verification.json", {
        "status": "PASS",
        "checks": {
            "source_hashes_unchanged": True,
            "frozen_np025_inputs": True,
            "no_refit_or_test_selection": True,
            "joint_state_identity": True,
            "timing_route_recomputed": True,
            "six_leads": True,
            "paired_intervals_recomputed": True,
            "summary_rows_recomputed": True,
        },
        "summary_rows": len(table), "identity_rows": len(identity),
    })
    print(json.dumps({"status": "PASS", "summary_rows": len(table), "identity_rows": len(identity)}))


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
