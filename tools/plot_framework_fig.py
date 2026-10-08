# -*- coding: utf-8 -*-
"""绘制 Figure 1：自生长证据框架总览图。

布局：顶行 源 -> 1 在线离散化 -> 2 窗口事务化 -> 3 anytime 免阈值挖掘
      -> 4 证据约束 LLM 归因；模式库在阶段 3 正下方（单调生长），
      诊断输出在阶段 4 正下方。
产出 figures/fig_framework.svg / .png（PNG 供 png_to_emf.ps1 转 EMF）。
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_SVG = os.path.join(ROOT, "figures", "fig_framework.svg")
OUT_PNG = os.path.join(ROOT, "figures", "fig_framework.png")

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(10.2, 5.6))
ax.set_xlim(0, 102)
ax.set_ylim(0, 58)
ax.axis("off")


def box(x, y, w, h, title, lines, fc, title_fs=9.0, fs=7.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.5,rounding_size=1.1",
                                fc=fc, ec="0.25", lw=1.1))
    ty = y + h - 3.0
    ax.text(x + w / 2, ty, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, ty - 3.2 - i * 2.8, ln, ha="center", va="center",
                fontsize=fs)


def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0, lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=lw, color="0.2",
                                 shrinkA=1, shrinkB=1))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label,
                ha="center", va="center", fontsize=7.3, style="italic",
                color="0.15")


C_SRC, C_STAGE, C_MINE = "#F2F2F2", "#EAF1FA", "#FDEBD0"
C_LIB, C_OUT = "#E8F6EC", "#F6E8F0"

# ---- 顶行：源 + 四阶段 ----
box(2, 34, 15, 17, "Industrial asset",
    ["multivariate", "sensor stream"], C_SRC)
xs = np.linspace(3.4, 15.6, 200)
for yy, ph in [(38.0, 0.0), (36.3, 1.3), (34.6, 2.6)]:
    ax.plot(xs, yy + 0.5 * np.sin(2.2 * xs + ph) + 0.22 * np.sin(5.1 * xs + ph),
            color="0.35", lw=0.7)
ax.text(9.5, 32.3, "pumps · turbines · bearings · ICS",
        ha="center", fontsize=6.6, color="0.35")

box(20, 34, 17, 17, "1  Online discretization",
    ["P² running quantiles", "min-width guard", "3 states / channel"], C_STAGE)
box(40, 34, 17, 17, "2  Window transactionization",
    ["median = operating", "point per window"], C_STAGE)
box(60, 34, 19, 17, "3  Anytime threshold-free",
    ["MFI mining", "items by desc. support", "BMMA tensor-core counting"], C_MINE)
ax.text(69.5, 32.3, "interruptible & resumable at any moment",
        ha="center", fontsize=7.0, style="italic", color="0.25")
box(81, 34, 19, 17, "4  LLM fault attribution",
    ["cite patterns only;", "every claim bound to", "a counted support"], C_STAGE)

for x1, x2 in [(17.4, 19.6), (37.4, 39.6), (57.4, 59.6), (79.4, 81.6)]:
    arrow(x1, 42.5, x2, 42.5)

# ---- 底行：模式库 + 诊断输出 ----
box(54, 8, 26, 15, "Self-growing pattern library",
    ["MFIs + support / confidence / first-seen",
     "monotone: resume extends, never recomputes",
     "grows online with the stream"], C_LIB, title_fs=8.8, fs=7.5)
box(83, 8, 17, 15, "Auditable diagnosis",
    ["valve partially closed", "← Pressure=1 ∧ Flow=0",
     "(43–93% of anomaly wins.)"], C_OUT, fs=7.4)

arrow(69, 33.6, 69, 23.6, label="MFI stream", lx=6.2, ly=0)
arrow(78.2, 23.6, 81.4, 35.8, label="top-k patterns", lx=-5.6, ly=1.2)
arrow(91, 33.6, 91, 23.6, label="cited evidence", lx=6.0, ly=0)

# ---- 时间向标注 ----
ax.add_patch(FancyArrowPatch((20, 55.2), (100, 55.2), arrowstyle="-|>",
                             mutation_scale=12, lw=1.0, color="0.45"))
ax.text(60, 56.6, "sensor stream  t  →", ha="center", fontsize=8.0,
        color="0.35")

fig.tight_layout(pad=0.3)
fig.savefig(OUT_SVG, format="svg")
fig.savefig(OUT_PNG, dpi=300)
print("saved", OUT_SVG, "and", OUT_PNG)
