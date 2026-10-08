# -*- coding: utf-8 -*-
"""plot_threshold_sensitivity.py —— Figure 3：阈值敏感性（消融）。

数据源: results/e5_efficiency.csv（baseline 引擎各 min-sup 配置中位耗时）
输出:   figures/fig_threshold_sensitivity.svg / .png（PNG 供 png_to_emf.ps1）
风格:   fig_style.py（墨/青/琥珀，Arial，与 Figure 1 一致）
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (AMBER, INK, SUB, TEAL, TEAL_DARK, apply_style,
                       clean_spines)

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "e5_efficiency.csv")
OUT = os.path.join(ROOT, "figures", "fig_threshold_sensitivity")

df = pd.read_csv(RES, encoding="utf-8-sig")
base = df[(df["engine"] == "baseline") & (df["med_total_s"] > 0)]

STYLE = {
    "hai": ("HAI", AMBER, "o"),
    "skab_valve1": ("SKAB valve-1", TEAL, "s"),
    "ai4i": ("AI4I 2020", TEAL_DARK, "^"),
    "cmapss": ("C-MAPSS", INK, "D"),
    "cwru": ("CWRU", SUB, "v"),
}

apply_style()
fig, ax = plt.subplots(figsize=(6.9, 3.6))

for key, (lbl, color, mk) in STYLE.items():
    sub = base[base["dataset"] == key].sort_values("thr")
    if sub.empty:
        continue
    ax.plot(sub["thr"], sub["med_total_s"], marker=mk, ms=5, lw=1.6,
            color=color, label=lbl, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.6)

# HAI 的极端不稳定标注：阈值 0.1→0.2（2×），耗时 18.3s→0.038s（≈480×）
ax.annotate("2× threshold → ≈480× time\n(18.3 s → 0.038 s)",
            xy=(0.1, 18.33), xytext=(0.017, 3.2),
            fontsize=8.5, color=AMBER,
            arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.1))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Minimum-support threshold (min-sup)")
ax.set_ylabel("Median mining time (s, baseline)")
ax.set_xlim(0.008, 0.6)
ax.set_ylim(5e-5, 60)
ax.grid(True, which="both")
ax.legend(loc="upper right", ncol=2)
clean_spines(ax)
fig.tight_layout()
fig.savefig(OUT + ".svg", format="svg")
fig.savefig(OUT + ".png", dpi=220)
print("saved", OUT + ".png")
