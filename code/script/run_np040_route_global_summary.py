"""NP040: global route gains from one shared-draw six-lead aggregation.

For each package, comparator and metric, NP040 takes the six lead-wise paired
bootstrap ratios saved by NP039 and computes an equal-weight mean *within each
draw*.  The reported interval is then the 2.5/97.5 percentiles of those global
draws.  This preserves the paired dependence and avoids averaging pointwise
confidence limits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEADS = [15, 60, 120, 240, 480, 720]
REFERENCES = ["direct_joint", "constructed_joint"]
METRICS = ["joint_first_passage_RPS", "conditional_rps", "conditional_median_mae_minutes"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    route_dir = ROOT / "outputs/next_paper/np039_route_direct_joint_bootstrap"
    out = ROOT / "outputs/next_paper/np040_route_global_summary"
    out.mkdir(parents=True, exist_ok=True)
    intervals = pd.read_csv(route_dir / "route_direct_intervals.csv")
    draws = np.load(route_dir / "bootstrap_draws.npz")
    rows: list[dict] = []
    for (package, reference, metric), group in intervals.groupby(
        ["package", "reference", "metric"], sort=True
    ):
        group = group.sort_values("lead_minutes")
        observed_leads = group.lead_minutes.astype(int).tolist()
        if observed_leads != LEADS:
            raise ValueError(f"Incomplete lead set for {package}/{reference}/{metric}: {observed_leads}")
        lead_draws = []
        for lead in LEADS:
            key = f"{package}__{reference}__{metric}__{lead}"
            if key not in draws:
                raise KeyError(key)
            lead_draws.append(np.asarray(draws[key], dtype=float))
        global_draws = np.mean(np.vstack(lead_draws), axis=0)
        if not np.isfinite(global_draws).all():
            raise FloatingPointError(f"Non-finite global draws for {package}/{reference}/{metric}")
        rows.append(
            {
                "package": package,
                "reference": reference,
                "metric": metric,
                "leads": ",".join(str(x) for x in LEADS),
                "n_leads": len(LEADS),
                "equal_weight_mean_gain_pct": float(global_draws.mean()),
                "global_low": float(np.quantile(global_draws, 0.025)),
                "global_high": float(np.quantile(global_draws, 0.975)),
                "min_lead_gain_pct": float(group.relative_loss_reduction_pct.min()),
                "max_lead_gain_pct": float(group.relative_loss_reduction_pct.max()),
                "mean_route_score": float(group.route_score.mean()),
                "mean_reference_score": float(group.reference_score.mean()),
                "min_valid_windows": int(group.valid_windows.min()),
                "windows_per_lead": int(group.windows.iloc[0]),
                "bootstrap_draws": int(len(global_draws)),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["package", "reference", "metric"]).reset_index(drop=True)
    summary.to_csv(out / "global_summary.csv", index=False)

    protocol = {
        "experiment": "NP040",
        "question": "What is the equal-weight all-lead route value for 17-bin, conditional RPS and conditional median MAE?",
        "input": "NP039 route_direct_intervals.csv and bootstrap_draws.npz",
        "leads": LEADS,
        "metrics": METRICS,
        "references": REFERENCES,
        "global_metric": "mean of six lead-wise relative reductions per shared bootstrap draw",
        "interval": "2.5/97.5 percentiles of the global draw distribution",
        "removed_columns": ["mean_low_of_pointwise_ci", "mean_high_of_pointwise_ci"],
        "source_sha256": {
            "outputs/next_paper/np039_route_direct_joint_bootstrap/route_direct_intervals.csv": digest(route_dir / "route_direct_intervals.csv"),
            "outputs/next_paper/np039_route_direct_joint_bootstrap/bootstrap_draws.npz": digest(route_dir / "bootstrap_draws.npz"),
            "outputs/next_paper/np039_route_direct_joint_bootstrap/bootstrap_weights.npz": digest(route_dir / "bootstrap_weights.npz"),
        },
        "status": "COMPLETE",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    verification = {
        "status": "PASS",
        "rows": len(summary),
        "six_leads": bool((summary.n_leads == 6).all()),
        "metrics": sorted(summary.metric.unique().tolist()),
        "finite_global_draws": True,
        "no_pointwise_ci_average_columns": not any(c.startswith("mean_low_of_pointwise") or c.startswith("mean_high_of_pointwise") for c in summary.columns),
        "source_sha256": protocol["source_sha256"],
    }
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(summary), "output": str(out)}))


if __name__ == "__main__":
    main()
