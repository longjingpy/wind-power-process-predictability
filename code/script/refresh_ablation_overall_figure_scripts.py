from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPERFIGURE = ROOT / "paperfigure"

TARGETS = [
    ("fig09_prior_source_ablation", "prior_source_overall_nmae.csv", "Prior-Source Ablation"),
    ("fig10_confidence_gating_ablation", "gating_overall_nmae.csv", "Confidence-Gating Ablation"),
    ("fig11_training_recipe_ablation", "training_recipe_overall_nmae.csv", "Training-Recipe Ablation"),
    ("fig12_data_contract_ablation", "data_contract_overall_nmae.csv", "Data-Contract Ablation (gapfill-on vs no-gapfill)"),
    ("fig13_gnss_signal_isolation", "gnss_signal_isolation_overall_nmae.csv", "GNSS Signal-Isolation Ablation"),
]

TEMPLATE = '''from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
OUT = HERE / "figure_rebuilt.png"
CSV_NAME = "{csv_name}"
TITLE = "{title}"


def main() -> None:
    overall_df = pd.read_csv(TABLES / CSV_NAME)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labels = overall_df["label"].astype(str).tolist()
    values = overall_df["nmae_pct"].tolist()
    bars = ax.bar(labels, values, width=0.72, color=plt.cm.tab10.colors[: len(labels)])
    ymax = float(max(values)) * 1.18
    ax.set_ylim(0, ymax)
    ax.set_ylabel("nMAE (%)")
    ax.set_title(TITLE, fontsize=15, weight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.015,
            f"{{value:.2f}}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(OUT, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {{OUT}}")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    for folder, csv_name, title in TARGETS:
        path = PAPERFIGURE / folder / "plot_figure.py"
        path.write_text(TEMPLATE.format(csv_name=csv_name, title=title), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
