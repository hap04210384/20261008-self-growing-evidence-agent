# -*- coding: utf-8 -*-
"""fig_style.py —— 全篇插图统一视觉语言（与 Figure 1 框架图一致）。

配色：墨 ink / 青 teal（引擎①）/ 琥珀 amber（引擎②·引用）/ 纸底 paper。
字体：Arial。所有绘图脚本 import 本模块后调用 apply_style()。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#20242B"
SUB = "#5A6270"
HAIR = "#D8DEE6"
PAPER = "#FFFFFF"
PANEL = "#EDF1F5"
TEAL = "#1E8449"
TEAL_DARK = "#145A32"
AMBER = "#C0392B"
AMBER_PALE = "#F5D5D0"
DARK = "#242932"


def apply_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": SUB,
        "ytick.color": SUB,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "grid.color": HAIR,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.9,
    })


def clean_spines(ax, keep=("left", "bottom")):
    """只保留指定边框，其余隐藏；刻度朝外。"""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(direction="out", length=3.5, width=0.8)
