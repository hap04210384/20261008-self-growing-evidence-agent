# -*- coding: utf-8 -*-
"""fig_style.py —— 全篇插图统一视觉语言（与 Figure 1 框架图一致）。

配色：墨 ink / 绿 green（主色·引擎①）/ 红 red（辅色·引擎②）/ 纯白底 paper。
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
BLUE = "#2F6690"     # 钢蓝（与 AMBER 红错开，图2/图6 使用）
ORANGE = "#C0662A"   # 赭橙（图3 HAI / 图6 Qwen）
PURPLE = "#6E4A8E"   # 灰紫（图3 AI4I）
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


def add_footnote(fig, axes, text, fontsize=7.8, color=None, bottom=0.012,
                 gap=0.022, linespacing=1.4, top=None):
    """底部通栏脚注：独立窄 axes 承载自动换行文本，左右与主图区拉通。

    两遍排版：先按图区宽度测出换行后的实际行高，再用 tight_layout 给
    脚注预留恰好高度（省下的底部留白自动回收），最后把脚注贴到图底。
    """
    if color is None:
        color = SUB
    top_arg = 1 if top is None else top
    axes = list(axes)
    fig.tight_layout(rect=[0, 0.02, 1, top_arg])
    pos0 = min(a.get_position().x0 for a in axes)
    pos1 = max(a.get_position().x1 for a in axes)

    def _measure():
        tmp = fig.add_axes([pos0, 0, pos1 - pos0, 1])
        tmp.axis("off")
        t = tmp.text(0, 0, text, fontsize=fontsize, color=color, ha="left",
                     va="bottom", wrap=True, linespacing=linespacing)
        fig.canvas.draw()
        bb = t.get_window_extent(fig.canvas.get_renderer())
        h = bb.height / (fig.dpi * fig.get_size_inches()[1])
        fig.delaxes(tmp)
        return h

    h = _measure()
    fig.tight_layout(rect=[0, bottom + h + gap, 1, top_arg])
    pos0 = min(a.get_position().x0 for a in axes)
    pos1 = max(a.get_position().x1 for a in axes)
    h = _measure()          # 宽度不变，重测仅为稳妥
    fax = fig.add_axes([pos0, bottom, pos1 - pos0, h])
    fax.axis("off")
    fax.text(0, 0, text, fontsize=fontsize, color=color, ha="left",
             va="bottom", wrap=True, linespacing=linespacing)
    return fax
