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


def _measure_text_pt(fig, s, fontsize, linespacing=1.4):
    """用渲染器实测字符串宽度/高度（pt）：临时挂一个透明文本到 figure 上。"""
    t = fig.text(0, 0, s, fontsize=fontsize, alpha=0, linespacing=linespacing)
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi * 72.0, bb.height / fig.dpi * 72.0


def _greedy_wrap(fig, text, fontsize, width_pt, linespacing=1.4):
    """按实测字宽做贪心换行，行宽尽量贴近 width_pt（排满到右边界）。"""
    lines, cur = [], ""
    for word in text.split():
        cand = word if not cur else cur + " " + word
        w, _ = _measure_text_pt(fig, cand, fontsize, linespacing)
        if w <= width_pt or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def add_footnote(fig, axes, text, fontsize=7.8, color=None, bottom=0.012,
                 gap=0.022, linespacing=1.4, top=None):
    """底部通栏脚注：按图区真实宽度实测换行，左右与主图区拉通。

    不用 matplotlib wrap=True（其换行边界比坐标区窄一截，右侧留大空），
    改为渲染器实测字宽的贪心换行；再按实际行高用 tight_layout 给脚注
    预留恰好高度，省下的底部留白自动回收。
    """
    if color is None:
        color = SUB
    top_arg = 1 if top is None else top
    axes = list(axes)
    fig_w_in, fig_h_in = fig.get_size_inches()
    fig.tight_layout(rect=[0, 0.02, 1, top_arg])

    def _layout():
        pos0 = min(a.get_position().x0 for a in axes)
        pos1 = max(a.get_position().x1 for a in axes)
        width_pt = (pos1 - pos0) * fig_w_in * 72.0
        wrapped = _greedy_wrap(fig, text, fontsize, width_pt, linespacing)
        _, h_pt = _measure_text_pt(fig, wrapped, fontsize, linespacing)
        return pos0, pos1, wrapped, h_pt / 72.0 / fig_h_in

    fig.canvas.draw()
    pos0, pos1, wrapped, h = _layout()
    fig.tight_layout(rect=[0, bottom + h + gap, 1, top_arg])
    fig.canvas.draw()
    pos0, pos1, wrapped, h = _layout()   # 高度变化后图区宽度不变，重算稳妥
    fax = fig.add_axes([pos0, bottom, pos1 - pos0, h])
    fax.axis("off")
    fax.text(0, 0, wrapped, fontsize=fontsize, color=color, ha="left",
             va="bottom", linespacing=linespacing)
    return fax
