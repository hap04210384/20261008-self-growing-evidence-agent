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
from fig_style import (AMBER, HAIR, INK, PANEL, SUB, TEAL_DARK, apply_style,
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

apply_style()
fig, ax = plt.subplots(figsize=(6.9, 3.9))

x = np.arange(len(piv))
w = 0.38

# 数据集分组：交替底色 + 组间分隔线
groups = []
start = 0
for i in range(1, len(piv) + 1):
    if i == len(piv) or piv.loc[i, "dataset"] != piv.loc[start, "dataset"]:
        groups.append((start, i - 1))
        start = i
for gi, (a, b) in enumerate(groups):
    if gi % 2 == 1:
        ax.axvspan(a - 0.5, b + 0.5, color=PANEL, alpha=0.6, zorder=0)
    if b + 1 < len(piv):
        ax.axvline(b + 0.5, color=HAIR, lw=0.8, zorder=1)

heavy = piv["baseline"] >= 0.01  # 重配置：baseline 中位耗时 ≥ 10 ms
speedup = piv["baseline"] / piv["bmma"]
med_heavy = speedup[heavy].median()

# 重配置琥珀色高亮
for xi, h in zip(x, heavy):
    if h:
        ax.axvspan(xi - 0.5, xi + 0.5, color=AMBER, alpha=0.08, zorder=0)

b1 = ax.bar(x - w / 2, piv["baseline"], w, color=HAIR, edgecolor=SUB,
            linewidth=0.6, label="baseline")
b2 = ax.bar(x + w / 2, piv["bmma"], w, color=AMBER, edgecolor="none",
            label="BMMA (tensor cores)")

# 加速比标注：≥1 绿，<1 灰
for xi, sp in zip(x, speedup):
    c = TEAL_DARK if sp >= 1 else SUB
    ax.annotate(f"×{sp:.1f}", xy=(xi, max(piv.loc[xi, 'baseline'],
                                          piv.loc[xi, 'bmma']) * 2.1),
                ha="center", fontsize=7.5, color=c, fontweight="bold")

# 重配置柱内标注中位耗时（秒）：灰柱内用深色、红柱内用白字（竖排避免溢出邻柱）
for xi, h in zip(x, heavy):
    if h:
        ax.annotate(f"{piv.loc[xi, 'baseline']:.2g} s",
                    xy=(xi - w / 2, piv.loc[xi, 'baseline'] * 0.18),
                    ha="center", va="center", fontsize=6.8, color=INK,
                    rotation=90)
        ax.annotate(f"{piv.loc[xi, 'bmma']:.2g} s",
                    xy=(xi + w / 2, piv.loc[xi, 'bmma'] * 0.18),
                    ha="center", va="center", fontsize=6.8, color="white",
                    fontweight="bold", rotation=90)

# 10 ms 重配置分界参考线
ax.axhline(0.01, color=INK, lw=0.8, ls=(0, (4, 3)), alpha=0.5)
ax.text(len(piv) - 0.55, 0.0115, "10 ms — heavy-config cutoff",
        fontsize=7.6, color=INK, ha="right")

ax.set_yscale("log")
ax.set_ylabel("Median mining time (s)")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(5e-5, 300)
ax.grid(True, axis="y")
ax.legend(loc="upper right")
clean_spines(ax)

# HAI@0.10：位图内存压力反转收益
i_heavy0 = int(piv.index[piv["dataset"] == "hai"][0])
ax.annotate("995 MFIs — bitmap memory\npressure reverses the gain (0.6×)",
            xy=(i_heavy0 + w / 2, piv.loc[i_heavy0, "bmma"] * 1.15),
            xytext=(i_heavy0 + 0.9, 90), fontsize=8.2, color=AMBER,
            arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.0,
                            connectionstyle="arc3,rad=0.2"))

# 轻配置：固定内核启动开销主导
ax.annotate("light configs: fixed kernel-setup\noverhead dominates (0.3–1.0×)",
            xy=(len(piv) - 1.5, 3e-4), xytext=(len(piv) - 4.6, 0.004),
            fontsize=8.2, color=SUB,
            arrowprops=dict(arrowstyle="->", color=SUB, lw=1.0,
                            connectionstyle="arc3,rad=-0.2"))

# 重配置中位加速说明
first_heavy = int(np.argmax(heavy.values))
ax.annotate(f"heavy configs (baseline ≥ 10 ms):\nmedian ×{med_heavy:.1f}",
            xy=(first_heavy + w, piv.loc[first_heavy, "bmma"] * 1.2),
            xytext=(first_heavy + 1.4, 2.2), fontsize=8.4, color=INK,
            arrowprops=dict(arrowstyle="->", color=SUB, lw=1.0))

fig.tight_layout(rect=[0, 0.052, 1, 1])
fig.text(0.065, 0.012,
         "Bars: median of five runs per configuration; bold ×N: BMMA speedup vs. baseline of the same "
         "configuration (green = faster,\ngray = slower). Seconds annotated on heavy configurations. "
         "Highlighted bands: baseline ≥ 10 ms.",
         fontsize=7.8, color=SUB, ha="left", va="bottom", linespacing=1.4)
fig.savefig(OUT + ".svg", format="svg")
fig.savefig(OUT + ".png", dpi=220)
print("saved", OUT + ".png")
print("heavy median speedup:", med_heavy)
