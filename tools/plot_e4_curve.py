# -*- coding: utf-8 -*-
"""plot_e4_curve.py —— Figure 2：AnyFIM 随时模式的故障覆盖率曲线（真实引擎数据）。

数据源: results/anytime_v2/e4_coverage_<grp>.csv（exp_e4_anytime.py 产物）
输出:   figures/fig_anytime_engine.svg / .png（PNG 供 png_to_emf.ps1 转 EMF）
风格:   fig_style.py（墨/青/琥珀，Arial，与 Figure 1 一致）
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import AMBER, INK, TEAL, apply_style, clean_spines

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "anytime_v2")
OUT = os.path.join(ROOT, "figures", "fig_anytime_engine")

apply_style()

fig, ax = plt.subplots(figsize=(6.9, 3.4))  # 17.5cm x 8.6cm

for group, color, off in (("valve1", TEAL, (-8, 9)), ("valve2", AMBER, (-64, 9))):
    df = pd.read_csv(os.path.join(RES, f"e4_coverage_{group}.csv"))
    lbl = "SKAB valve-1" if group == "valve1" else "SKAB valve-2"
    ax.plot(df["stage"], df["coverage"], marker="o", ms=4, lw=1.6,
            color=color, label=lbl, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.6)
    # 端点数值标注（首次到达 1.0 的阶段），错开避免重叠
    hit = df[df["coverage"] >= 1.0].iloc[0]
    ax.annotate(f"{hit['coverage']:.1f} @ $k$={int(hit['stage'])}",
                xy=(hit["stage"], hit["coverage"]),
                xytext=off, textcoords="offset points",
                ha="right", fontsize=8.5, color=color)

ax.set_xlabel("Anytime stage $k$ (sensor states activated)")
ax.set_ylabel("Fault-pattern coverage")
ax.set_xlim(0.5, 22.5)
ax.set_ylim(-0.05, 1.12)
ax.set_xticks([1, 4, 7, 10, 13, 16, 19, 22])
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.grid(True)
ax.legend(loc="center left")
clean_spines(ax)
fig.tight_layout()
fig.savefig(OUT + ".svg", format="svg")
fig.savefig(OUT + ".png", dpi=220)
print("saved", OUT + ".png")
