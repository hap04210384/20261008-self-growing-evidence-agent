# -*- coding: utf-8 -*-
"""绘制 anytime 模式覆盖率曲线（anytime_v1 数据，备用图，风格与全篇统一）。

输出 figures/fig_anytime_spike.svg/.png。
风格: fig_style.py（墨/青/琥珀，Arial，与 Figure 1 一致）
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import AMBER, TEAL, apply_style, clean_spines

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "anytime_v1")
OUT = os.path.join(ROOT, "figures", "fig_anytime_spike")

apply_style()

fig, ax = plt.subplots(figsize=(6.9, 3.4))  # 17.5cm x 8.6cm

for group, color in (("valve1", TEAL), ("valve2", AMBER)):
    csv = os.path.join(RES, f"anytime_curve_{group}.csv")
    df = pd.read_csv(csv)
    df.columns = [c.strip("\ufeff") for c in df.columns]
    mcol = "coverage_mean" if "coverage_mean" in df.columns else "mean_cov"
    scol = "coverage_std" if "coverage_std" in df.columns else "std_cov"
    ax.plot(df["frac"], df[mcol], marker="o", ms=4, lw=1.6,
            color=color, label=f"{group} coverage", markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.6)
    ax.fill_between(df["frac"],
                    df[mcol] - df[scol],
                    df[mcol] + df[scol],
                    color=color, alpha=0.15, linewidth=0)

ax.set_xlabel("Fraction of transaction stream processed")
ax.set_ylabel("Fault-pattern coverage")
ax.set_xlim(0.1, 1.0)
ax.set_ylim(-0.05, 1.05)
ax.grid(True)
ax.legend(loc="upper left")
clean_spines(ax)
fig.tight_layout()
fig.savefig(OUT + ".svg", format="svg")
fig.savefig(OUT + ".png", dpi=220)
print("saved", OUT + ".png")
