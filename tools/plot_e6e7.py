# -*- coding: utf-8 -*-
"""plot_e6e7.py —— Figure 6：离散化消融 + 窗口参数敏感性（双面板）。

数据源: results/e6e7_ablation/summary.json（exp_e6e7_ablation.py 产物）
风格:   tools/fig_style.py（墨/青/琥珀，与全篇一致）
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from fig_style import BLUE, INK, SUB, ORANGE, apply_style, clean_spines

import matplotlib.pyplot as plt
import numpy as np

SUM = json.load(open(os.path.join(ROOT, "results", "e6e7_ablation",
                                  "summary.json"), encoding="utf-8"))
OUT = os.path.join(ROOT, "figures", "fig_discretizer_window")

DISC = [("p2_online", "P² online\n(ours)"),
        ("equal_freq", "equal-frequency\n(offline oracle)"),
        ("equal_width", "equal-width\n(fixed)")]
WINS = ["w15", "w30", "w60"]

apply_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.2))

# ---------- (a) 离散化消融 ----------
x = np.arange(len(DISC))
w = 0.36
for off, grp, color, lbl in ((-w / 2, "valve1", BLUE, "SKAB valve-1"),
                             (w / 2, "valve2", ORANGE, "SKAB valve-2")):
    cov = [SUM["E6_discretizer"][grp][k]["fault_coverage_topk"] for k, _ in DISC]
    ax1.bar(x + off, cov, w, color=color, label=lbl)
    for xi, c in enumerate(cov):
        ax1.annotate(f"{c:.2f}", xy=(xi + off, c + 0.02), ha="center",
                     fontsize=8, color=INK, fontweight="bold")
# 库规模（valve1 / valve2 MFI 数）写进 x 轴刻度标签
mfi1 = [SUM["E6_discretizer"]["valve1"][k]["n_mfi"] for k, _ in DISC]
mfi2 = [SUM["E6_discretizer"]["valve2"][k]["n_mfi"] for k, _ in DISC]
ax1.set_xticks(x)
ax1.set_xticklabels(
    [f"{lbl}\n{m1} / {m2} MFIs" for (_, lbl), m1, m2 in zip(DISC, mfi1, mfi2)],
    fontsize=8.5)
ax1.set_ylabel("Fault coverage (top-5 patterns)")
ax1.set_ylim(0, 0.92)
ax1.set_yticks([0.0, 0.25, 0.5, 0.75])
ax1.grid(True, axis="y")
ax1.legend(loc="upper right", fontsize=8.5)
clean_spines(ax1)
ax1.set_title("(a) discretizer ablation (equal pattern budget)",
              fontsize=9.5, loc="left")

# ---------- (b) 窗口敏感性 ----------
x2 = np.arange(len(WINS))
for off, grp, color, lbl in ((-w / 2, "valve1", BLUE, "SKAB valve-1"),
                             (w / 2, "valve2", ORANGE, "SKAB valve-2")):
    cov = [SUM["E7_window"][grp][k]["fault_coverage_topk"] for k in WINS]
    ax2.bar(x2 + off, cov, w, color=color, label=lbl)
    for xi, c in enumerate(cov):
        ax2.annotate(f"{c:.2f}", xy=(xi + off, c + 0.02), ha="center",
                     fontsize=8, color=INK, fontweight="bold")
ax2.axvline(1, color=SUB, lw=0.8, ls=":")
ax2.annotate("default w = 30", xy=(1, 0.87), fontsize=8, color=SUB,
             ha="center",
             bbox=dict(fc="#FFFFFF", ec="none", pad=1.2))
ax2.set_xticks(x2)
ax2.set_xticklabels(["w = 15", "w = 30", "w = 60"], fontsize=8.5)
ax2.set_ylabel("Fault coverage (top-5 patterns)")
ax2.set_ylim(0, 0.92)
ax2.set_yticks([0.0, 0.25, 0.5, 0.75])
ax2.grid(True, axis="y")
clean_spines(ax2)
ax2.set_title("(b) window size sensitivity (hop = w/2)",
              fontsize=9.5, loc="left")

fig.tight_layout()
fig.savefig(OUT + ".svg", format="svg")
fig.savefig(OUT + ".png", dpi=220)
print("saved", OUT + ".png")
