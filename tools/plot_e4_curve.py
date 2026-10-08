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
from fig_style import (AMBER, BLUE, INK, PANEL, SUB, TEAL, TEAL_DARK,
                       apply_style, clean_spines)

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "anytime_v2")
OUT = os.path.join(ROOT, "figures", "fig_anytime_engine")

apply_style()

fig, ax = plt.subplots(figsize=(6.9, 3.9))

# 正常工况零覆盖区（两个 run 在 k<=13 均为 0）
ax.axvspan(0.5, 13.5, color=SUB, alpha=0.07, zorder=0)
ax.text(9.2, 0.87, "normal-operation states only — coverage 0",
        ha="center", fontsize=8.4, color=SUB)

dfs = {}
for group, color, off in (("valve1", TEAL, (-8, 10)), ("valve2", AMBER, (-66, 10))):
    df = pd.read_csv(os.path.join(RES, f"e4_coverage_{group}.csv"))
    dfs[group] = df
    lbl = "SKAB valve-1" if group == "valve1" else "SKAB valve-2"
    ax.plot(df["stage"], df["coverage"], marker="o", ms=4, lw=1.6,
            color=color, label=lbl, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.6)
    ax.fill_between(df["stage"], df["coverage"], color=color, alpha=0.08, lw=0)
    # 端点数值标注（首次到达 1.0 的阶段），错开避免重叠
    hit = df[df["coverage"] >= 1.0].iloc[0]
    ax.annotate(f"{hit['coverage']:.1f} @ $k$={int(hit['stage'])}",
                xy=(hit["stage"], hit["coverage"]),
                xytext=off, textcoords="offset points",
                ha="right", fontsize=8.5, color=color)

# 完全覆盖参考线
ax.axhline(1.0, color=INK, lw=0.8, ls=(0, (4, 3)), alpha=0.5)
ax.text(11.5, 1.14, "complete fault-pattern family", ha="center",
        fontsize=8.4, color=INK)

# valve-2 拐点：故障特征状态开始进入
v2 = dfs["valve2"]
lift = int(v2[v2["coverage"] > 0].iloc[0]["stage"])
ax.annotate(f"fault-characteristic states\nenter the stream ($k$={lift})",
            xy=(lift, 0.015), xytext=(lift - 4.8, 0.40),
            fontsize=8.3, color=AMBER, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.0,
                            connectionstyle="arc3,rad=-0.25"))

# anytime 语义框（左下零覆盖区内的空白带）
ax.text(1.0, 0.05,
        "AnyFIM anytime semantics:\nthe library is correct at ANY\ninterruption stage $k$",
        fontsize=8.3, color=TEAL_DARK, va="bottom", ha="left", linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.5", fc=PANEL, ec=TEAL, lw=0.9))

ax.set_xlabel("Anytime stage $k$ (sensor states activated)")
ax.set_ylabel("Fault-pattern coverage")
ax.set_xlim(0.5, 22.5)
ax.set_ylim(-0.05, 1.19)
ax.set_xticks([1, 4, 7, 10, 13, 16, 19, 22])
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.grid(True)
ax.legend(loc="upper left")
clean_spines(ax)
# 底部给三行脚注预留 ~10.5% 高度，保证脚注与横坐标标签之间有明显间隙
fig.tight_layout(rect=[0, 0.105, 1, 1])
fig.text(0.065, 0.012,
         "Guarantees: coverage is monotone non-decreasing by construction — earlier prefixes remain valid\n"
         "after resumption; 44/45 stages bit-identical to exhaustive MFI mining at the per-stage support\n"
         "threshold (single exception: a tied-support stage — a direct illustration of anytime semantics).",
         fontsize=7.8, color=SUB, ha="left", va="bottom", linespacing=1.4)
fig.savefig(OUT + ".svg", format="svg")
fig.savefig(OUT + ".png", dpi=220)
print("saved", OUT + ".png")
