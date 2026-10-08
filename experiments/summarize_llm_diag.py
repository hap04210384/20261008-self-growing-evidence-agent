# -*- coding: utf-8 -*-
"""summarize_llm_diag.py —— 汇总 LLM 诊断消融结果 -> summary.json + Figure 5。

Figure 5 双面板：(a) 诊断准确率（有/无证据 × 3 模型）；(b) 引文忠实度。
风格: tools/fig_style.py（墨/青/琥珀，与全篇一致）
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT := os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "tools"))
from fig_style import AMBER, INK, PANEL, SUB, TEAL, TEAL_DARK, apply_style, clean_spines
from exp_llm_diagnosis import GROUPS, parse_json, load_group

import matplotlib.pyplot as plt
import numpy as np

OUTDIR = os.path.join(ROOT, "results", "llm_diag")
CALLS = os.path.join(OUTDIR, "calls.jsonl")
SUMMARY = os.path.join(OUTDIR, "summary.json")
FIG = os.path.join(ROOT, "figures", "fig_llm_diagnosis")

MODEL_ORDER = ["deepseek-chat", "qwen-plus", "glm-4-air"]
MODEL_LABEL = {"deepseek-chat": "DeepSeek-V3", "qwen-plus": "Qwen-Plus",
               "glm-4-air": "GLM-4-Air"}

recs = [json.loads(l) for l in open(CALLS, encoding="utf-8")]
libs = {g: {r["pattern"] for r in load_group(g)[2]} for g in GROUPS}

acc = defaultdict(lambda: [0, 0])
faith = defaultdict(lambda: [0, 0])
hallu = defaultdict(lambda: [0, 0])
unparsed = 0
for r in recs:
    j = parse_json(r["reply"])
    if j is None:
        unparsed += 1
        continue
    diag = str(j.get("diagnosis", "")).strip().lower()
    correct = diag in ("normal", "faulty") and \
        (diag == "faulty") == (r["truth"] == "faulty")
    k = (r["model"], r["with_lib"])
    acc[k][1] += 1
    acc[k][0] += int(correct)
    cites = j.get("cited_patterns") or []
    if r["with_lib"]:
        for c in cites:
            faith[r["model"]][1] += 1
            faith[r["model"]][0] += int(c.strip() in libs[r["group"]])
    else:
        hallu[r["model"]][1] += 1
        hallu[r["model"]][0] += int(len(cites) > 0)

summary = {
    "n_calls": len(recs), "unparsed": unparsed,
    "models": {},
}
for m in MODEL_ORDER:
    a0, a1 = acc[(m, False)], acc[(m, True)]
    f, h = faith[m], hallu[m]
    summary["models"][m] = {
        "acc_without_library": round(a0[0] / max(a0[1], 1), 4),
        "acc_with_library": round(a1[0] / max(a1[1], 1), 4),
        "n_without_library": a0[1], "n_with_library": a1[1],
        "citation_faithfulness": round(f[0] / max(f[1], 1), 4),
        "citations_total": f[1],
        "condA_hallucinated_citations": h[0], "condA_n": h[1],
    }
json.dump(summary, open(SUMMARY, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))

# ---------- Figure 5 ----------
apply_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 4.0),
                               gridspec_kw={"width_ratios": [1.55, 1]})

x = np.arange(len(MODEL_ORDER))
w = 0.36
a_nolib = [summary["models"][m]["acc_without_library"] for m in MODEL_ORDER]
a_lib = [summary["models"][m]["acc_with_library"] for m in MODEL_ORDER]

b1 = ax1.bar(x - w / 2, a_nolib, w, color=SUB, label="without pattern library")
b2 = ax1.bar(x + w / 2, a_lib, w, color=TEAL,
             label="with mined pattern library")
for xi, v in zip(x, a_nolib):
    ax1.annotate(f"{v:.2f}", xy=(xi - w / 2, v + 0.02), ha="center",
                 fontsize=8.5, color=SUB)
for xi, v in zip(x, a_lib):
    ax1.annotate(f"{v:.2f}", xy=(xi + w / 2, v + 0.02), ha="center",
                 fontsize=8.5, color=TEAL_DARK, fontweight="bold")
# 提升幅度写在青色柱内（白字），避免与图例争抢顶部空间
for xi, (a, b) in enumerate(zip(a_nolib, a_lib)):
    ax1.annotate(f"+{(b - a) * 100:.0f} pp", xy=(xi + w / 2, b - 0.07),
                 ha="center", fontsize=8.5, color="white", fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER])
ax1.set_xlim(-0.95, 2.55)
ax1.set_ylabel("Window diagnosis accuracy")
ax1.set_ylim(0, 1.24)
ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax1.axhline(0.5, color=SUB, lw=0.8, ls=":")  # chance level
ax1.text(-0.88, 0.512, "chance\n(0.5)", fontsize=7.2, color=SUB, ha="left",
         va="bottom", linespacing=1.25)
ax1.grid(True, axis="y")
ax1.legend(loc="upper left", fontsize=8.5, handlelength=1.5)
clean_spines(ax1)

# 右上说明框：增益来自证据质量而非提示词长度
ax1.text(2.45, 1.17, "largest gain on the strongest\nmodel — evidence quality,\n"
         "not prompt size, drives it",
         fontsize=7.8, color=TEAL_DARK, ha="right", va="top", linespacing=1.35,
         bbox=dict(boxstyle="round,pad=0.45", fc=PANEL, ec=TEAL, lw=0.9))

faith_v = [summary["models"][m]["citation_faithfulness"] for m in MODEL_ORDER]
bars = ax2.bar(x, faith_v, 0.5, color=AMBER)
for xi, v in zip(x, faith_v):
    ax2.annotate(f"{v * 100:.1f}%", xy=(xi, v + 0.012), ha="center",
                 fontsize=8.5, color=INK, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], fontsize=7.2)
ax2.set_xlim(-0.95, 2.55)
ax2.set_ylabel("Citation faithfulness")
ax2.set_ylim(0.55, 1.16)
ax2.axhline(1.0, color=INK, lw=0.8, ls=(0, (4, 3)), alpha=0.5)
ax2.text(-0.88, 0.985, "perfect\nfaithfulness", fontsize=7.2, color=INK,
         ha="left", va="center", linespacing=1.25)
ax2.grid(True, axis="y")
clean_spines(ax2)

# GLM-4-Air 的 0.7% 缺口标注：文字放右上方空白区（文字已点名 GLM-4-Air，无需箭头）
glm_faith = summary["models"]["glm-4-air"]["citation_faithfulness"]
glm_ncite = summary["models"]["glm-4-air"]["citations_total"]
ax2.annotate(f"GLM-4-Air: ≈{round((1 - glm_faith) * glm_ncite)} of {glm_ncite} "
             "citations\nnot verbatim in the library",
             xy=(2.52, 1.15), fontsize=7.4, color=AMBER,
             ha="right", va="top", linespacing=1.35)

ax2.set_title("(b) every claim auditable", fontsize=9.5, loc="left")
ax1.set_title("(a) evidence improves diagnosis", fontsize=9.5, loc="left")

fig.tight_layout(rect=[0, 0.11, 1, 0.95])
fig.text(0.065, 0.014,
         "60 balanced windows per condition (normal/faulty; ten per class from SKAB valve-1, valve-2, "
         "and other-fault runs); frozen\nprompts, decoding temperature 0. +pp = percentage-point gain "
         "of the same model with vs. without the mined library. Without the library,\nGLM-4-Air "
         "fabricated citations in 32/60 windows; with the library, the single unverifiable citation "
         "is annotated in panel (b).",
         fontsize=7.3, color=SUB, ha="left", va="bottom", linespacing=1.35)
fig.savefig(FIG + ".svg", format="svg")
fig.savefig(FIG + ".png", dpi=220)
print("saved", FIG + ".png")
