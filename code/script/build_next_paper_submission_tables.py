"""Build traceable Markdown tables for the next-paper submission package."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/d/projects/WindPowerForcast")
OUT = Path("/mnt/c/Users/admin/Desktop/next-paper/next_paper_submission_tables.md")


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name)


def select(df: pd.DataFrame, **conditions) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in conditions.items():
        mask &= df[key] == value
    rows = df[mask]
    if len(rows) != 1:
        raise ValueError((conditions, len(rows)))
    return rows.iloc[0]


def interval(row: pd.Series) -> str:
    return f"{row['relative_loss_reduction_pct']:.2f}% ({row['low']:.2f}, {row['high']:.2f})"


def main() -> None:
    np005 = load("outputs/next_paper/np005/paired_intervals.csv")
    np008 = load("outputs/next_paper/np008/paired_intervals.csv")
    np009 = load("outputs/next_paper/np009/paired_intervals.csv")
    np018 = load("outputs/next_paper/np018/paired_intervals.csv")
    np020 = load("outputs/next_paper/np020/paired_intervals.csv")
    np025 = load("outputs/next_paper/np025/test_summary.csv")
    np026 = load("outputs/next_paper/np026/routing_summary.csv")
    np027 = load("outputs/next_paper/np027/monthly_summary.csv")
    np028 = load("outputs/next_paper/np028/test_summary.csv")
    np029 = load("outputs/next_paper/np029/routing_summary.csv")
    np030 = load("outputs/next_paper/np030/monthly_summary.csv")

    lines = [
        "# Submission tables (source-driven draft)",
        "",
        "Generated 2026-09-24 from verified CSV artifacts. Percentages are relative loss reductions; intervals are paired seven-day calendar-block intervals unless stated otherwise.",
        "",
        "## Table 1. Data and protocol",
        "",
        "| Item | Value | Source |",
        "|---|---|---|",
        "| Main site | Pizhou administrative reference: 34.3403°N, 118.0068°E; onsite wind reference 140 m | `site_metadata.json` |",
        "| Target | 17 native 15-min power nodes over [T, T+4 h]; excursion threshold 0.2 pu | `np005/protocol.json` |",
        "| Issue leads | 15, 60, 120, 240, 480, 720 min | `np005/protocol.json` |",
        "| Main test cohort | 6,496 target windows | `np005/protocol.json` |",
        "| External decomposition | Suining; power-only, weather-only, and joint arms; 17 nodes; 60/20/20 chronological split with four-hour purge; 6,996 test windows per lead | `np025/protocol.json` |",
        "| Strict weather clock | GFS `previous_day1` fields fixed 24 h before valid time; request January–August 2024; complete weather support from 16 February; 3,772 test windows per lead | `np028/protocol.json` |",
        "| Target-aligned route | Joint state head plus power-history timing head; frozen NP025 predictions, no refit | `np026/protocol.json` |",
        "",
        "## Table 2. Process skill across issue lead",
        "",
        "| Process object | Lead | Relative loss reduction | Source |",
        "|---|---:|---:|---|",
    ]
    for lead in [15, 720]:
        r = select(np008, candidate=f"{lead}__full__rolling", reference=f"{lead}__weather__rolling", metric="multiclass_brier", state="all", block_days=7)
        lines.append(f"| Four-class state: full vs weather | {lead} min | {interval(r)} | `np008/paired_intervals.csv` |")
    r = select(np005, candidate="power_weather:720:joint", reference="frequency:0:reference", metric="return_brier", state="all", block_days=7)
    lines.append(f"| Return-event risk | 720 min | {interval(r)} | `np005/paired_intervals.csv` |")
    for metric, label in [("conditional_rps", "Downward first-passage RPS"), ("conditional_median_mae_minutes", "Downward first-passage median MAE")]:
        r = select(np009, candidate="15__down__context", reference="15__down__frequency", metric=metric, state="all", block_days=7)
        lines.append(f"| {label} | 15 min | {interval(r)} | `np009/paired_intervals.csv` |")
    lines += [
        "",
        "## Table 3. Information roles by process target",
        "",
        "| Information comparison | Target | Lead | Relative loss reduction | Source |",
        "|---|---|---:|---:|---|",
    ]
    for lead in [15, 720]:
        r = select(np025, arm="weather", task="event", metric="multiclass_brier", lead_minutes=lead)
        lines.append(f"| Weather-only vs frequency | Four-class state | {lead} min | {interval(r)} | `np025/test_summary.csv` |")
    r = select(np025, arm="joint_increment", task="event", metric="multiclass_brier", lead_minutes=720)
    lines.append(f"| Weather added to power | Four-class state | 720 min | {interval(r)} | `np025/test_summary.csv` |")
    for lead in [15, 60]:
        for metric, label in [("conditional_rps", "RPS"), ("conditional_median_mae_minutes", "Median MAE")]:
            r = select(np025, arm="power_increment_on_weather", task="downward_timing", metric=metric, lead_minutes=lead)
            lines.append(f"| Power history added to weather | Downward timing {label} | {lead} min | {interval(r)} | `np025/test_summary.csv` |")
    r = select(np028, arm="weather_increment", task="event", metric="multiclass_brier", lead_minutes=720)
    lines.append(f"| Strict previous-day1 weather added to power | Four-class state | 720 min | {interval(r)} | `np028/test_summary.csv` |")
    r = select(np028, arm="joint_increment", task="event", metric="multiclass_brier", lead_minutes=720)
    lines.append(f"| Strict previous-day1 joint arm added to power | Four-class state | 720 min | {interval(r)} | `np028/test_summary.csv` |")
    for metric, label in [("conditional_rps", "RPS"), ("conditional_median_mae_minutes", "Median MAE")]:
        r = select(np028, arm="power_increment_on_weather", task="downward_timing", metric=metric, lead_minutes=15)
        lines.append(f"| Power history added to strict previous-day1 weather | Downward timing {label} | 15 min | {interval(r)} | `np028/test_summary.csv` |")
    # Routing belongs to the climax table, not to the information-role table.
    lines += [
        "",
        "## Table 4. Target-aligned routing and strict-clock checks",
        "",
        "| Comparison | Metric | Lead | Relative loss reduction | Source |",
        "|---|---|---:|---:|---|",
    ]
    for lead in [240, 480]:
        for metric, label in [("conditional_rps", "Downward timing RPS"), ("conditional_median_mae_minutes", "Downward timing median MAE")]:
            r = select(np026, lead_minutes=lead, metric=metric, reference="joint_timing")
            lines.append(f"| Target-aligned route vs joint timing | {label} | {lead} min | {interval(r)} | `np026/routing_summary.csv` |")
    lines.append("| State head identity | Maximum absolute probability difference = 0 | all leads | 0.00 | `np026/state_identity.csv` |")
    for lead in [60, 120]:
        r = select(np029, lead_minutes=lead, metric="conditional_rps", reference="joint_timing")
        lines.append(f"| Strict-clock routing vs joint timing | Downward timing RPS | {lead} min | {interval(r)} | `np029/routing_summary.csv` |")
    lines += [
        "",
        "## Supplementary stability tables",
        "",
        "| Comparison | Metric | Lead | Monthly point estimates | Source |",
        "|---|---|---:|---|---|",
    ]
    for lead in [240, 480]:
        for metric, label in [("conditional_rps", "Timing RPS"), ("conditional_median_mae_minutes", "Timing median MAE")]:
            rows = np027[(np027.lead_minutes == lead) & (np027.metric == metric)]
            point = ", ".join(f"{r.month}: {r.relative_loss_reduction_pct:.2f}%" for _, r in rows.iterrows())
            lines.append(f"| NP027 routing stability | {label} | {lead} min | {point} | `np027/monthly_summary.csv` |")
    for lead in [15, 60, 120]:
        for metric in ["conditional_rps", "conditional_median_mae_minutes"]:
            rows = np030[np030.lead_minutes == lead]
            point = ", ".join(f"{r.month}: {r.relative_loss_reduction_pct:.2f}%" for _, r in rows[rows.metric == metric].iterrows())
            label = "Timing RPS" if metric == "conditional_rps" else "Timing median MAE"
            lines.append(f"| NP030 strict-clock stability | {label} | {lead} min | {point} | `np030/monthly_summary.csv` |")
    lines += [
        "",
        "The evidence-backed draft and Claim Ledger remain the authoritative sources for rounded claims, interpretation, and detailed limitations. This table file is a submission-layout aid.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
