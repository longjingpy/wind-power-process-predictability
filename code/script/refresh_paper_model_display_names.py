"""
Refresh paper-facing model display names in stage-00 assets and shared figure outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CAPACITY_KW = 87.45 * 1000.0
LABEL_MAP = {
    "Spatial Model": "Fusion Model",
    "Best": "Temporal-Only Model",
    "LSTM": "Lightweight LSTM Baseline",
    "Transformer": "Lightweight Transformer Baseline",
    "Persistence": "Persistence",
    "Lag1 (AR1)": "ARX Wind-Speed Baseline",
    "Direct Multi-Lag Linear Baseline": "ARX Wind-Speed Baseline",
}
COLOR_MAP = {
    "Fusion Model": "#005fce",
    "Temporal-Only Model": "#4f46e5",
    "Lightweight LSTM Baseline": "#0ea5e9",
    "Lightweight Transformer Baseline": "#10b981",
    "Persistence": "#e67e22",
    "ARX Wind-Speed Baseline": "#2e8b57",
}
STYLE_MAP = {
    "Fusion Model": ("#005fce", "-", "o"),
    "Temporal-Only Model": ("#4f46e5", "-", "o"),
    "Lightweight LSTM Baseline": ("#0ea5e9", "-", "o"),
    "Lightweight Transformer Baseline": ("#10b981", "-", "o"),
    "Persistence": ("#e67e22", "--", "s"),
    "ARX Wind-Speed Baseline": ("#2e8b57", "--", "^"),
}


TARGET_DIRS = [
    ROOT / "outputs" / "paper_materials" / "stage_00_main_results",
    ROOT / "outputs" / "report" / "current_6model_common_replot_named" / "analysis" / "figures" / "presentation",
]


def _rename_label(value: str) -> str:
    return LABEL_MAP.get(str(value), str(value))


def _plot_overall(overall_df: pd.DataFrame, out_path: Path, note_text: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    colors = [COLOR_MAP[label] for label in overall_df["label"]]
    bars = ax.bar(overall_df["label"], overall_df["nmae_pct"], color=colors, width=0.72)
    ymax = float(overall_df["nmae_pct"].max()) * 1.18
    ax.set_ylim(0, ymax)
    ax.set_title("6-Model Overall nMAE on Common Intersection", fontsize=22, weight="bold")
    ax.set_ylabel("nMAE (%)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, overall_df["nmae_pct"].tolist()):
        ax.text(bar.get_x() + bar.get_width() / 2, value + ymax * 0.015, f"{value:.1f}", ha="center", va="bottom", fontsize=11)
    ax.text(
        1.02, 0.98, note_text,
        transform=ax.transAxes, va="top", ha="left", fontsize=11,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def _plot_leadwise(compare_df: pd.DataFrame, out_path: Path, note_text: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    for label in compare_df["label"].drop_duplicates():
        frame = compare_df[compare_df["label"] == label].sort_values("step_index")
        color, linestyle, marker = STYLE_MAP[label]
        ax.plot(
            frame["lead_min_effective"],
            frame["nmae_pct"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.4,
            markersize=6,
            label=label,
        )
    ax.set_title("6-Model Lead-wise nMAE on Common Intersection", fontsize=22, weight="bold")
    ax.set_xlabel("Lead Time (min)")
    ax.set_ylabel("nMAE (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=11, ncol=2)
    ax.text(
        1.02, 0.98, note_text,
        transform=ax.transAxes, va="top", ha="left", fontsize=11,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def _refresh_result_dir(base_dir: Path) -> None:
    tables_dir = base_dir / "tables" if (base_dir / "tables").exists() else base_dir
    figures_dir = base_dir / "figures" if (base_dir / "figures").exists() else base_dir

    overall_csv = tables_dir / "overall_compare_6model_common_nmae.csv"
    leadwise_csv = tables_dir / "leadwise_compare_6model_common_nmae.csv"
    summary_json = tables_dir / "summary_6model_common.json"
    overall_fig = figures_dir / "04_overall_summary_6model_common_nmae.png"
    leadwise_fig = figures_dir / "02_leadwise_nmae_compare_6model_common.png"
    if not (overall_csv.exists() and leadwise_csv.exists() and summary_json.exists()):
        return

    overall_df = pd.read_csv(overall_csv)
    leadwise_df = pd.read_csv(leadwise_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    overall_df["label"] = overall_df["label"].map(_rename_label)
    leadwise_df["label"] = leadwise_df["label"].map(_rename_label)
    if "lead0_best" in summary and isinstance(summary["lead0_best"], dict):
        summary["lead0_best"]["label"] = _rename_label(summary["lead0_best"]["label"])
    if "lead0_full" in summary and isinstance(summary["lead0_full"], list):
        for row in summary["lead0_full"]:
            row["label"] = _rename_label(row["label"])

    overall_df.to_csv(overall_csv, index=False, encoding="utf-8")
    leadwise_df.to_csv(leadwise_csv, index=False, encoding="utf-8")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    note = (
        f"Common intersection only\n"
        f"origins={int(summary['common_origin_count'])}, rows={int(summary['common_row_count'])}\n"
        f"15-min best={summary['lead0_best']['label']} ({float(summary['lead0_best']['nmae_pct']):.2f}%)"
    )
    _plot_overall(overall_df, overall_fig, note)
    _plot_leadwise(leadwise_df, leadwise_fig, note)


def _refresh_stage00_summary() -> None:
    summary_path = ROOT / "outputs" / "paper_materials" / "stage_00_main_results" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    labels = payload.get("model_contract", {}).get("labels", [])
    payload["model_contract"]["labels"] = [_rename_label(x) for x in labels]
    if "best_overall_label" in payload.get("key_metrics", {}):
        payload["key_metrics"]["best_overall_label"] = _rename_label(payload["key_metrics"]["best_overall_label"])
    if "best_15min_label" in payload.get("key_metrics", {}):
        payload["key_metrics"]["best_15min_label"] = _rename_label(payload["key_metrics"]["best_15min_label"])
    if "paper_claim" in payload:
        best_label = payload["key_metrics"]["best_overall_label"]
        lead_label = payload["key_metrics"]["best_15min_label"]
        payload["paper_claim"] = (
            f"On the shared evaluation subset, {best_label} is the strongest overall model "
            f"({float(payload['key_metrics']['best_overall_nmae_pct']):.2f}% nMAE), while "
            f"{lead_label} is best at the 15-min lead."
        )
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    for target_dir in TARGET_DIRS:
        _refresh_result_dir(target_dir)
    _refresh_stage00_summary()
    print("Refreshed paper-facing model labels.")


if __name__ == "__main__":
    main()
