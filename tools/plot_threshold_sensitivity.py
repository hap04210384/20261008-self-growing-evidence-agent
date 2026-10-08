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
from fig_style import (AMBER, INK, ORANGE, PANEL, PURPLE, SUB, TEAL,
                       TEAL_DARK, add_footnote, apply_style, clean_spines)

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "e5_runs.csv")
OUT = os.path.join(ROOT, "figures", "fig_threshold_sensitivity")

runs = pd.read_csv(RES)
base = (runs[runs["engine"] == "baseline"]
        .groupby(["dataset", "thr"])["secs"].median()
        .reset_index().rename(columns={"secs": "med_total_s"}))

STYLE = {
    "hai": ("HAI", ORANGE, "o"),
    "skab_valve1": ("SKAB valve-1", TEAL, "s"),
    "ai4i": ("AI4I 2020", PURPLE, "^"),
    "cmapss": ("C-MAPSS", INK, "D"),
    "cwru": ("CWRU", SUB, "v"),
}

apply_style()
fig, ax = plt.subplots(figsize=(6.9, 4.0))

# 低阈值危险区
ax.axvspan(0.008, 0.05, color=ORANGE, alpha=0.07, zorder=0)

for key, (lbl, color, mk) in STYLE.items():
    sub = base[base["dataset"] == key].sort_values("thr")
    if sub.empty:
        continue
    ax.plot(sub["thr"], sub["med_total_s"], marker=mk, ms=5, lw=1.6,
            color=color, label=lbl, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.6)

# 每条数据集线段首点旁标注「阈值倍数 → 耗时倍数」（由真实数据计算）
for key, (lbl, color, mk) in STYLE.items():
    sub = base[base["dataset"] == key].sort_values("thr")
    if len(sub) < 2 or key == "hai":   # HAI 已有专项标注
        continue
    t_ratio = sub.iloc[-1]["thr"] / sub.iloc[0]["thr"]
    r_ratio = sub.iloc[-1]["med_total_s"] / sub.iloc[0]["med_total_s"]
    x0, y0 = sub.iloc[0]["thr"], sub.iloc[0]["med_total_s"]
    if max(sub["med_total_s"]) / min(sub["med_total_s"]) < 1.5:
        txt = "runtime-insensitive\n(trivial load)"
        off, ha = (-4, -18), "right"
    else:
        txt = f"{t_ratio:.2g}× thr → {r_ratio:.2g}× time"
        off, ha = (2, 8), "left"
    ax.annotate(txt, xy=(x0, y0), xytext=off, textcoords="offset points",
                ha=ha, fontsize=7.6, color=color, linespacing=1.25)

# HAI 的极端不稳定标注：阈值 0.1→0.2（2×），耗时 18.8s→0.039s（≈490×）
ax.annotate("2× threshold → ≈490× time\n(18.8 s → 0.039 s)",
            xy=(0.1, 18.82), xytext=(0.016, 3.2),
            fontsize=8.5, color=ORANGE, linespacing=1.3,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1,
                            connectionstyle="arc3,rad=0.18"))

# 1 s 在线更新预算参考线
ax.axhline(1.0, color=INK, lw=0.8, ls=(0, (4, 3)), alpha=0.5)
ax.text(0.58, 1.18, "1 s — online-update budget", fontsize=8.3, color=INK,
        ha="right")

# ours 卖点框（左下，危险区旁）
ax.text(0.011, 0.62,
        "AnyFIM: threshold-free —\nno min-sup to set (Sec. 3.3)",
        fontsize=8.6, color=TEAL_DARK, va="top", linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.5", fc=PANEL, ec=TEAL, lw=0.9))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Minimum-support threshold (min-sup)")
ax.set_ylabel("Median mining time (s, baseline)")
ax.set_xlim(0.008, 0.6)
ax.set_ylim(5e-5, 60)
ax.grid(True, which="both")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=5,
          columnspacing=1.4, handlelength=1.6)
clean_spines(ax)
# 底部通栏脚注：左右与图区拉通，按实际行数预留高度（多余留白自动回收）
add_footnote(fig, [ax],
             "Shaded band: low-threshold regime — runtime swings of orders of "
             "magnitude within a single 2× threshold step; with no plateau, no "
             "fixed threshold is a safe default across industrial deployments. "
             "Annotations give the threshold-ratio → runtime-ratio measured for "
             "each dataset.",
             fontsize=7.8, color=SUB, linespacing=1.4)
fig.savefig(OUT + ".svg", format="svg")
fig.savefig(OUT + ".png", dpi=220)
print("saved", OUT + ".png")
