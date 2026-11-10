"""Fig. 3c and the SI category bars (PDF and PNG)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .common import CATEGORIES  # noqa: E402

FS_TEXT, FS_SMALL = 7, 5.8
COL4 = {"vertical": "#018571", "temporal": "#80cdc1", "reuse": "#dfc27d", "simpler": "#a6611a"}
COL6 = {**COL4, "reuse_notes": "#f3e3bd", "reuse_segment": "#dfc27d", "reuse_melody": "#b8954a"}
SHORT = {"vertical": "Vert.\nshift", "temporal": "Temp.\nshift", "reuse": "Reuse",
         "reuse_notes": "Prev.\nnote", "reuse_segment": "Same\nmelody",
         "reuse_melody": "Earlier\nmelody", "simpler": "Simpler\nprog."}


def bars(cats, tab, name, width, out):
    fig, ax = plt.subplots(figsize=(width, 1.9))
    x = np.arange(len(cats))
    ax.bar(x, tab.loc[cats, "mean"], yerr=tab.loc[cats, "sem"], color=[COL6[c] for c in cats],
           edgecolor="black", lw=0.5, capsize=2, error_kw={"lw": 0.7})
    ax.set_xticks(x, [SHORT[c] for c in cats], fontsize=FS_SMALL)
    reuse = [k for k, c in enumerate(cats) if c.startswith("reuse_")]
    if len(reuse) > 1:  # group label and bracket under the three reuse bars
        lo, hi = min(reuse) - 0.4, max(reuse) + 0.4
        ax.annotate("", xy=(lo, -0.30), xytext=(hi, -0.30), xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
        ax.text((lo + hi) / 2, -0.34, "Reuse from", ha="center", va="top", fontsize=FS_SMALL,
                transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_ylabel("Unique proportion", fontsize=FS_TEXT)
    ax.tick_params(axis="y", labelsize=FS_SMALL)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
    print(out / f"{name}.png")


def panel_c(tab, out):
    """Four-category bar plot for Fig. 3c."""
    cats = ["vertical", "temporal", "reuse", "simpler"]
    fig, ax = plt.subplots(figsize=(3.3, 3.44))
    x = np.arange(len(cats))
    ax.bar(x, tab.loc[cats, "mean"], yerr=tab.loc[cats, "sem"], color=[COL6[c] for c in cats],
           edgecolor="black", lw=0.8, capsize=3, error_kw={"lw": 1.0}, width=0.8)
    ax.set_xticks(x, ["Vert. Shift", "Temp. Shift", "Reuse", "Simpler Prog."],
                  fontsize=15, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Unique Proportion", fontsize=17)
    ax.set_yticks([0.0, 0.05, 0.10])
    ax.tick_params(axis="y", labelsize=15)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_categories_main_panel.pdf")
    fig.savefig(out / "fig_categories_main_panel.png", dpi=300)
    plt.close(fig)
    print(out / "fig_categories_main_panel.png")


def run(output, min_len=3, font="DejaVu Sans"):
    plt.rcParams.update({"font.family": font, "pdf.fonttype": 42, "ps.fonttype": 42})
    out = output / "figures"
    out.mkdir(parents=True, exist_ok=True)
    prop = pd.read_csv(output / f"proportions_min{min_len}.csv")
    four = prop[prop.split == "pooled"].set_index("category")
    six = prop[prop.split == "six"].set_index("category")
    bars(CATEGORIES, six, "fig_categories_si", 3.0, out)
    panel_c(four, out)
