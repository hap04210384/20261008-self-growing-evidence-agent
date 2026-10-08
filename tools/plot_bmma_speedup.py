# -*- coding: utf-8 -*-
"""plot_bmma_speedup.py —— Figure 4：BMMA 加速消融（baseline vs BMMA）。

数据源: results/e5_efficiency.csv（11 个数据集/阈值配置，5 次重复中位）
输出:   figures/fig_bmma_speedup.svg / .png（PNG 供 png_to_emf.ps1）
风格:   fig_style.py（墨/青/琥珀，Arial，与 Figure 1 一致）
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (AMBER, HAIR, INK, SUB, TEAL_DARK, apply_style,
                       clean_spines)

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "e5_efficiency.csv")
OUT = os.path.join(ROOT, "figures", "fig_bmma_speedup")

df = pd.read_csv(RES, encoding="utf-8-sig")
piv = df.pivot_table(index=["dataset", "thr"], columns="engine",
                     values="med_total_s", aggfunc="first").reset_index()
# 只保留 baseline 与 BMMA 均有数据的配置（11 个）
piv = piv.dropna(subset=["baseline", "bmma"])
piv = piv.sort_values("baseline", ascending=False).reset_index(drop=True)

DS_LABEL = {
    "hai": "HAI",
    "skab_valve1": "SKAB v1",
    "ai4i": "AI4I",
    "cmapss": "C-MAPSS",
    "cwru": "CWRU",
}
labels = [f"{DS_LABEL[r['dataset']]}\n{round(r['thr'], 2)}" for _, r in piv.iterrows()]
heavy = piv["baseline"] >= 0.01  # 重配置：baseline 中位耗时 ≥ 10 ms
speedup = piv["baseline"] / piv["bmma"]
med_heavy = speedup[heavy].median()

apply_style()
fig, ax = plt.subplots(figsize=(6.9, 3.6))

x = np.arange(len(piv))
w = 0.38
b1 = ax.bar(x - w / 2, piv["baseline"], w, color=HAIR, edgecolor=SUB,
            linewidth=0.6, label="baseline")
b2 = ax.bar(x + w / 2, piv["bmma"], w, color=AMBER, edgecolor="none",
            label="BMMA (tensor cores)")

# 加速比标注：≥1 青绿，<1 灰
for xi, sp in zip(x, speedup):
    c = TEAL_DARK if sp >= 1 else SUB
    ax.annotate(f"×{sp:.1f}", xy=(xi, max(piv.loc[xi, 'baseline'],
                                          piv.loc[xi, 'bmma']) * 1.25),
                ha="center", fontsize=7.5, color=c)

# 重配置底色带
for xi, h in zip(x, heavy):
    if h:
        ax.axvspan(xi - 0.5, xi + 0.5, color=AMBER, alpha=0.07, zorder=0)

ax.set_yscale("log")
ax.set_ylabel("Median mining time (s)")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(5e-5, 200)
ax.grid(True, axis="y")
ax.legend(loc="upper right")
clean_spines(ax)

# 图内说明：重配置中位加速
first_heavy = int(np.argmax(heavy.values))
ax.annotate(f"heavy configs\n(baseline ≥ 10 ms):\nmedian ×{med_heavy:.1f}",
            xy=(first_heavy, 0.6), xytext=(first_heavy + 0.8, 1.1),
            fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=SUB, lw=1.0))

fig.tight_layout()
fig.savefig(OUT + ".svg", format="svg")
fig.savefig(OUT + ".png", dpi=220)
print("saved", OUT + ".png")
print("heavy median speedup:", med_heavy)
