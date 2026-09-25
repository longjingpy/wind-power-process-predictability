"""Verify the NP037 threshold reconstruction and positive sensitivity rows."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/next_paper/np037_threshold_sensitivity"
labels = pd.read_csv(OUT / "label_reconstruction_check.csv")
assert len(labels) == 6
assert (labels["state_match"] == 1.0).all()
assert (labels["arrival_match"] == 1.0).all()

table = pd.read_csv(OUT / "threshold_sensitivity.csv")
assert len(table) == 108
state = table[
    (table["task"] == "state")
    & (table["arm"] == "weather")
    & (table["threshold"].isin([0.15, 0.25]))
]
assert len(state) == 12
assert (state["relative_vs_frequency_pct"] > 0).all()
verification = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
assert verification["status"] == "PASS"
print({"status": "PASS", "rows": len(table), "label_checks": len(labels), "positive_off_design_state_rows": len(state)})
