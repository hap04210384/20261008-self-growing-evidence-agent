# -*- coding: utf-8 -*-
"""summarize_llm_budget.py —— 证据预算消融的统计汇总 + Figure 7（纯本地，不耗 API）。

输入：results/llm_diag/budget.jsonl（k ∈ {1,2,3,5,8}）
      results/llm_diag/calls.jsonl（with_lib=false 即 k=0，复用）
判据：与 llm_diag_stats.py 一致——仅统计 JSON 可解析且 diagnosis 合法
      的回复；不可解析回复不计入分母。

产出：results/llm_diag/budget_stats.json
      figures/png/fig7_llm_budget/fig7_llm_budget.png（高分辨率，供转 EMF）
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from exp_llm_diagnosis import parse_json  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "tools"))
from fig_style import (apply_style, clean_spines, INK, SUB, HAIR, PAPER,
                       TEAL, TEAL_DARK, AMBER, BLUE, ORANGE,
                       add_footnote)  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BUDGET = os.path.join(ROOT, "results", "llm_diag", "budget.jsonl")
CALLS = os.path.join(ROOT, "results", "llm_diag", "calls.jsonl")
OUTJSON = os.path.join(ROOT, "results", "llm_diag", "budget_stats.json")
FIGDIR = os.path.join(ROOT, "figures", "png", "fig7_llm_budget")
os.makedirs(FIGDIR, exist_ok=True)

MODEL_ORDER = ["deepseek-chat", "qwen-plus", "glm-4-air"]
MODEL_LABEL = {"deepseek-chat": "DeepSeek-V3", "qwen-plus": "Qwen-Plus",
               "glm-4-air": "GLM-4-Air"}
MODEL_COLOR = {"deepseek-chat": BLUE, "qwen-plus": ORANGE,
               "glm-4-air": TEAL_DARK}
MODEL_MARK = {"deepseek-chat": "o", "qwen-plus": "s", "glm-4-air": "^"}
MODEL_DODGE = {"deepseek-chat": -0.16, "qwen-plus": 0.0, "glm-4-air": 0.16}
KS = (0, 1, 2, 3, 5, 8)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b, c):
    if b + c == 0:
        return (0.0, 1.0)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2))
    return (chi2, p)


# ---------- 载入与判分 ----------
recs = []
for line in open(BUDGET, encoding="utf-8"):
    recs.append(json.loads(line))
for line in open(CALLS, encoding="utf-8"):
    r = json.loads(line)
    if r["with_lib"] is False:
        r["k"] = 0
        recs.append(r)

correct = {}
unparsed = 0
for r in recs:
    j = parse_json(r["reply"])
    if j is None:
        unparsed += 1
        continue
    diag = str(j.get("diagnosis", "")).strip().lower()
    correct[(r["model"], r["k"], r["case_key"])] = \
        int(diag in ("normal", "faulty") and
            (diag == "faulty") == (r["truth"] == "faulty"))

# ---------- 聚合 (model, k) ----------
agg = {}
for m in MODEL_ORDER:
    for k in KS:
        ks = ns = 0
        for (mm, kk, ck), v in correct.items():
            if mm == m and kk == k:
                ks += v
                ns += 1
        lo, hi = wilson(ks, ns)
        agg[(m, k)] = {"correct": ks, "n": ns,
                       "acc": round(ks / max(ns, 1), 4),
                       "wilson95": [round(lo, 4), round(hi, 4)]}

# ---------- McNemar（按 case_key 配对）----------
CONTRASTS = [(0, 3), (0, 5), (3, 5), (5, 8)]
mc = {}
for m in MODEL_ORDER:
    mc[m] = {}
    keys = {ck for (mm, kk, ck) in correct if mm == m}
    for k1, k2 in CONTRASTS:
        b = c = paired = 0
        for ck in keys:
            a = correct.get((m, k1, ck))
            bq = correct.get((m, k2, ck))
            if a is None or bq is None:
                continue
            paired += 1
            if bq == 1 and a == 0:
                b += 1
            elif bq == 0 and a == 1:
                c += 1
        chi2, p = mcnemar(b, c)
        mc[m][f"k{k1}_vs_k{k2}"] = {
            "n_paired": paired, "hi_only_correct": b,
            "lo_only_correct": c, "chi2": round(chi2, 3),
            "p": round(p, 5)}

stats = {"n_budget_calls": sum(1 for r in recs if r["k"] != 0),
         "n_reused_k0": sum(1 for r in recs if r["k"] == 0),
         "unparsed_excluded": unparsed,
         "ks": list(KS), "per_model_k": {}, "mcnemar": mc}
for m in MODEL_ORDER:
    stats["per_model_k"][m] = {str(k): agg[(m, k)] for k in KS}
json.dump(stats, open(OUTJSON, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# ---------- Figure 7 ----------
# 峰值 vs k=0 的 McNemar p（Qwen 峰在 k=8，需补算 k0_vs_k8）
PEAK_K = {"deepseek-chat": 5, "qwen-plus": 8, "glm-4-air": 3}
peak_p = {}
for m in MODEL_ORDER:
    kp = PEAK_K[m]
    keys = {ck for (mm, kk, ck) in correct if mm == m}
    b = c = 0
    for ck in keys:
        a = correct.get((m, 0, ck))
        bq = correct.get((m, kp, ck))
        if a is None or bq is None:
            continue
        if bq == 1 and a == 0:
            b += 1
        elif bq == 0 and a == 1:
            c += 1
    peak_p[m] = mcnemar(b, c)[1]

apply_style()
fig, ax = plt.subplots(figsize=(7.2, 4.9))
DATA = {}
for m in MODEL_ORDER:
    xs, ys, lo_e, hi_e = [], [], [], []
    for k in KS:
        d = agg[(m, k)]
        xs.append(k + MODEL_DODGE[m])
        ys.append(d["acc"])
        lo_e.append(d["acc"] - d["wilson95"][0])
        hi_e.append(d["wilson95"][1] - d["acc"])
    yerr = [lo_e, hi_e]
    DATA[m] = (xs, ys)
    ax.errorbar(xs, ys, yerr=yerr, color=MODEL_COLOR[m],
                marker=MODEL_MARK[m], ms=6.5, lw=1.6,
                elinewidth=1.0, capsize=2.5, capthick=0.9,
                label=MODEL_LABEL[m], zorder=3)

# 小预算区间底纹：k = 3–5 已含大部分可提取信号（标签放带内底部空旷区）
ax.axvspan(2.62, 5.38, color=SUB, alpha=0.075, zorder=0)
ax.annotate("small budget suffices:\nk = 3\u20135 captures most of the signal",
            (4.0, 0.428), ha="center", va="bottom", fontsize=8.2,
            color=INK, zorder=4)

# 峰值大圈高亮 + 峰值旁直接标注（避免顶部汇总框与曲线/圈相交）
for m in MODEL_ORDER:
    i = KS.index(PEAK_K[m])
    x, y = DATA[m][0][i], DATA[m][1][i]
    acc = agg[(m, PEAK_K[m])]["acc"]
    gain = acc - agg[(m, 0)]["acc"]
    ax.plot([x], [y], marker="o", ms=13.5, mfc="none",
            mec=MODEL_COLOR[m], mew=1.5, zorder=4)
    if m == "deepseek-chat":
        ax.annotate(f"peak {acc:.2f} (\u0394 +{gain:.2f})\nMcNemar p = {peak_p[m]:.4f}",
                    (x, y), textcoords="offset points", xytext=(2, 15),
                    ha="center", fontsize=8, color=MODEL_COLOR[m], zorder=5)
    elif m == "qwen-plus":
        ax.annotate(f"peak {acc:.2f} (\u0394 +{gain:.2f}), p = {peak_p[m]:.3f}",
                    (x, y), textcoords="offset points", xytext=(-12, -4),
                    ha="right", fontsize=8, color=MODEL_COLOR[m], zorder=5)
    else:
        ax.annotate(f"peak {acc:.2f} (\u0394 +{gain:.2f})\np = {peak_p[m]:.3f}",
                    (x, y), textcoords="offset points", xytext=(10, 4),
                    ha="left", fontsize=8, color=MODEL_COLOR[m], zorder=5)

# GLM-4-Air 过峰后退化提示
ax.annotate("GLM-4-Air degrades\npast its peak", (8.5, 0.705),
            ha="right", va="top", fontsize=8, color=TEAL_DARK,
            style="italic", zorder=4)

# 端点 + 关键峰标注（中部拥挤点不标，避免互相压盖）
def lab(m, i, txt, dy):
    x, y = DATA[m][0][i], DATA[m][1][i]
    ax.annotate(txt, (x, y), textcoords="offset points",
                xytext=(0, dy), ha="center", fontsize=8, color=SUB)
for m in MODEL_ORDER:
    # k=0 三个点挤在一起：改为左右错位标注
    if m == "deepseek-chat":
        ax.annotate(f"{DATA[m][1][0]:.2f}", (DATA[m][0][0], DATA[m][1][0]),
                    textcoords="offset points", xytext=(-9, -3),
                    ha="right", fontsize=8, color=SUB)
    elif m == "qwen-plus":
        ax.annotate(f"{DATA[m][1][0]:.2f}", (DATA[m][0][0], DATA[m][1][0]),
                    textcoords="offset points", xytext=(9, -3),
                    ha="left", fontsize=8, color=SUB)
    else:
        ax.annotate(f"{DATA[m][1][0]:.2f}", (DATA[m][0][0], DATA[m][1][0]),
                    textcoords="offset points", xytext=(6, 8),
                    ha="left", fontsize=8, color=SUB)
    lab(m, -1, f"{DATA[m][1][-1]:.2f}", 10 if m != "qwen-plus" else -16)
lab("glm-4-air", 3, f"{DATA['glm-4-air'][1][3]:.2f}", -20)           # k=3 峰

ax.axhline(0.5, color=SUB, lw=0.9, ls=(0, (4, 3)), zorder=1)
ax.annotate("chance = 0.50", (-0.55, 0.485), va="top", fontsize=8,
            color=SUB, ha="left")
ax.set_xticks(list(KS))
ax.set_xticklabels(["0\n(no library)", "1", "2", "3", "5", "8\n(full)"])
ax.set_xlabel("Number of attached library patterns  k")
ax.set_ylabel("Diagnosis accuracy on 198 windows")
ax.set_ylim(0.40, 1.00)
ax.set_xlim(-0.7, 9.0)
ax.grid(axis="y")
clean_spines(ax)
ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.985), ncol=1,
          handlelength=1.8)
# 底部通栏脚注：左右与图区拉通，按实际行数预留高度（多余留白自动回收）
add_footnote(fig, [ax],
             "198 balanced SKAB windows per model · frozen prompts, decoding "
             "temperature 0 · whiskers: Wilson 95% CI · dashed line: chance "
             "(0.5) · p: paired McNemar, peak vs. k = 0",
             fontsize=7.4, color=SUB, linespacing=1.5)
fig.savefig(os.path.join(FIGDIR, "fig7_llm_budget.png"), dpi=300)
print("saved", os.path.join(FIGDIR, "fig7_llm_budget.png"))

# ---------- 摘要 ----------
for m in MODEL_ORDER:
    lab = MODEL_LABEL[m]
    line = "  ".join(f"k={k}:{agg[(m, k)]['acc']:.2f}" for k in KS)
    print(f"{lab:12s} {line}")
    for k1, k2 in CONTRASTS:
        d = mc[m][f"k{k1}_vs_k{k2}"]
        print(f"    k{k1}->k{k2}: b={d['hi_only_correct']} "
              f"c={d['lo_only_correct']} p={d['p']}")
print("saved", OUTJSON)
