"""Build Figure 8 from the two-lead NP023 receipt and six-lead NP041 output."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/next_paper"
OUT = DATA / "manuscript_figures/nature_v1"
TEXT = "#18212B"
BLUE = "#2166AC"
ORANGE = "#D55E00"
PURPLE = "#7B3294"
GREEN = "#1B9E77"
GREY = "#5B6470"
RATIOS = [2.0, 5.0, 10.0]
NP023_LEADS = [15, 720]
NP041_LEADS = [15, 60, 120, 240, 480, 720]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    for name in ["arial.ttf", "arialbd.ttf"]:
        path = Path("/mnt/c/Windows/Fonts") / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    np023_test = pd.read_csv(DATA / "np023/test_decision_summary.csv")
    np023_pairs = pd.read_csv(DATA / "np023/paired_decision_intervals.csv")
    np023_data = np.load(DATA / "np013/dataset.npz")
    np041 = pd.read_csv(DATA / "np041_all_lead_decision_sensitivity/decision_summary.csv")

    # Panel a: NP023 has only 15 and 720 min.  Derive an explicit no-alert
    # absolute cost from the frozen test event rate; do not relabel frequency.
    event_test = np023_data["outcome"][np023_data["test"]] == 3
    absolute_rows: list[dict] = []
    for lead in NP023_LEADS:
        ratio = 5.0
        absolute_rows.append({"lead_minutes": lead, "method": "No alert", "value": float(ratio * event_test.mean())})
        for method, label in [("np014_selected", "NP014"), ("np019_selected", "NP019 = NP020")]:
            row = np023_test[(np023_test.lead_minutes == lead) & (np023_test.cost_ratio == ratio) & (np023_test.method == method) & (np023_test.metric == "realized_cost")]
            if len(row) != 1:
                raise ValueError(f"Missing NP023 absolute cost: {lead}/{method}")
            absolute_rows.append({"lead_minutes": lead, "method": label, "value": float(row.value.iloc[0])})
    absolute = pd.DataFrame(absolute_rows)
    assert np.isfinite(absolute.value).all()

    # Panel b: paired NP023 gains against NP014, retaining two real leads and
    # all three ratios.  NP019 and NP020 are identical in the frozen receipt.
    pair_rows: list[dict] = []
    for candidate, label in [("np013_selected", "NP013"), ("np019_selected", "NP019 = NP020")]:
        for ratio in RATIOS:
            for lead in NP023_LEADS:
                row = np023_pairs[(np023_pairs.lead_minutes == lead) & (np023_pairs.cost_ratio == ratio) & (np023_pairs.candidate == candidate) & (np023_pairs.reference == "np014_selected")]
                if len(row) != 1:
                    raise ValueError(f"Missing NP023 paired interval: {lead}/{ratio}/{candidate}")
                pair_rows.append({"lead_minutes": lead, "cost_ratio": ratio, "method": label, "value": float(row.relative_cost_reduction_pct.iloc[0]), "low": float(row.low.iloc[0]), "high": float(row.high.iloc[0])})
        duplicate = np023_pairs[(np023_pairs.candidate == "np020_selected") & (np023_pairs.reference == "np014_selected")]
        assert len(duplicate) == 6
    pairs = pd.DataFrame(pair_rows)
    assert np.isfinite(pairs[["value", "low", "high"]].to_numpy()).all()

    # Panel c: all six NP041 leads against its validation-selected constant.
    heat = np041[(np041.method == "np008_full_rolling") & (np041.metric == "cost_reduction_vs_validation_constant_pct")].pivot(index="cost_ratio", columns="lead_minutes", values="value").reindex(index=RATIOS, columns=NP041_LEADS)
    if heat.shape != (3, 6) or not np.isfinite(heat.to_numpy()).all():
        raise ValueError("NP041 six-lead decision surface is incomplete or non-finite")

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.35), constrained_layout=True, gridspec_kw={"width_ratios": [1.0, 1.35, 1.3]})

    # a
    methods_a = ["No alert", "NP014", "NP019 = NP020"]
    colors_a = [GREY, BLUE, PURPLE]
    xpos = np.arange(len(NP023_LEADS))
    width = 0.22
    for i, (method, color) in enumerate(zip(methods_a, colors_a)):
        vals = absolute[absolute.method == method].set_index("lead_minutes").reindex(NP023_LEADS).value.to_numpy()
        axes[0].bar(xpos + (i - 1) * width, vals, width=width, color=color, label=method)
    axes[0].set_xticks(xpos, ["15", "720"])
    axes[0].set_xlabel("Issue lead (min)")
    axes[0].set_ylabel("Test cost (C = 5)")
    axes[0].set_title("NP023 absolute decision cost", loc="left")
    axes[0].legend(frameon=False, fontsize=6.4, loc="upper left")

    # b
    colors_ratio = {2.0: BLUE, 5.0: ORANGE, 10.0: PURPLE}
    offsets = {"NP013": -0.18, "NP019 = NP020": 0.18}
    ratio_offsets = {2.0: -0.06, 5.0: 0.0, 10.0: 0.06}
    for method in ["NP013", "NP019 = NP020"]:
        for ratio in RATIOS:
            subset = pairs[(pairs.method == method) & (pairs.cost_ratio == ratio)].set_index("lead_minutes").reindex(NP023_LEADS)
            x = xpos + offsets[method] + ratio_offsets[ratio]
            y = subset.value.to_numpy()
            lower = y - subset.low.to_numpy()
            upper = subset.high.to_numpy() - y
            axes[1].errorbar(x, y, yerr=np.vstack([lower, upper]), fmt="o", ms=3.5, capsize=2, lw=1.0, color=colors_ratio[ratio], label=f"C={ratio:g}" if method == "NP013" else None)
    axes[1].axhline(0, color=GREY, lw=0.8, ls="--")
    axes[1].set_xticks(xpos, ["15", "720"])
    axes[1].set_xlabel("Issue lead (min)")
    axes[1].set_ylabel("Gain vs NP014 (%)")
    axes[1].set_title("NP023 paired gain vs NP014", loc="left")
    axes[1].text(0.02, 0.04, "points: NP013 / NP019 = NP020", transform=axes[1].transAxes, fontsize=6.5, color=TEXT)
    axes[1].legend(frameon=False, fontsize=6.2, ncol=3, loc="upper right")

    # c
    image = axes[2].imshow(heat.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=min(-20.0, float(heat.to_numpy().min())), vmax=max(20.0, float(heat.to_numpy().max())))
    axes[2].set_xticks(np.arange(len(NP041_LEADS)), [str(x) for x in NP041_LEADS])
    axes[2].set_yticks(np.arange(len(RATIOS)), [f"{x:g}" for x in RATIOS])
    axes[2].set_xlabel("Issue lead (min)")
    axes[2].set_ylabel("Cost ratio C")
    axes[2].set_title("NP041 full rolling vs validation constant", loc="left")
    for i in range(len(RATIOS)):
        for j in range(len(NP041_LEADS)):
            value = float(heat.iloc[i, j])
            axes[2].text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6.5, color=TEXT)
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04, label="Cost reduction (%)")

    for i, ax in enumerate(axes):
        ax.text(-0.18, 1.08, "abc"[i], transform=ax.transAxes, fontsize=11, fontweight="bold", color=TEXT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y" if i < 2 else "x", color="#E5E9EF", lw=0.6)
        ax.set_axisbelow(True)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 450})]:
        fig.savefig(OUT / f"fig8_decision_value.{ext}", bbox_inches="tight", pad_inches=0.08, **kwargs)
    plt.close(fig)

    sources = [
        "np023/test_decision_summary.csv",
        "np023/paired_decision_intervals.csv",
        "np013/dataset.npz",
        "np041_all_lead_decision_sensitivity/decision_summary.csv",
    ]
    manifest = {
        "figure": "fig8_decision_value",
        "panels": {
            "a": "NP023 C=5 absolute test cost for no alert, NP014 and NP019=NP020 at 15/720 min",
            "b": "NP023 paired gains against NP014 for NP013 and NP019=NP020 at the two registered leads and all three ratios",
            "c": "NP041 full rolling cost reduction against validation-selected constant policy at six leads and three ratios",
        },
        "source_paths": sources,
        "source_sha256": {s: digest(DATA / s) for s in sources},
        "style": "Arial/TrueType; NP023 two-lead receipt and NP041 six-lead extension",
    }
    manifest["sha256"] = {f"fig8_decision_value.{ext}": digest(OUT / f"fig8_decision_value.{ext}") for ext in ["pdf", "svg", "png"]}
    (OUT / "fig8_decision_value_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT / "fig8_decision_value.pdf"), "panel_a_rows": len(absolute), "panel_b_rows": len(pairs), "panel_c_shape": list(heat.shape)}))


if __name__ == "__main__":
    main()
