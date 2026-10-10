# -*- coding: utf-8 -*-
"""summarize_hai_diag.py —— HAI 简化诊断协议统计（纯本地）。

两个手臂（insample / timesplit）× 条件（nolib / mined / random）× 模型：
- 准确率 + Wilson 95% CI（窗口极少，区间会很宽——如实报告）；
- 引用忠实度：mined 条件的引用对挖掘库、random 条件的引用对随机库逐字核验；
- 无库条件幻觉率；
- mined vs nolib 的配对 McNemar（按 case_key 配对，连续性校正）。

产出：results/hai_diag/summary.json
"""
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from exp_llm_diagnosis import parse_json    # noqa: E402

CALLS = os.path.join(ROOT, "results", "hai_diag", "calls.jsonl")
LIBS = os.path.join(ROOT, "results", "hai_diag", "libraries.json")
OUT = os.path.join(ROOT, "results", "hai_diag", "summary.json")

MODELS = ["deepseek-chat", "qwen-plus", "glm-4-air",
          "glm-4-flash", "qwen-turbo"]
LABEL = {"deepseek-chat": "DeepSeek-V3", "qwen-plus": "Qwen-Plus",
         "glm-4-air": "GLM-4-Air", "glm-4-flash": "GLM-4-Flash",
         "qwen-turbo": "Qwen-Turbo"}


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
    return (chi2, math.erfc(math.sqrt(chi2 / 2)))


recs = [json.loads(l) for l in open(CALLS, encoding="utf-8")]
libs = json.load(open(LIBS, encoding="utf-8"))
ARMS = [a for a in ("insample", "timesplit") if a in libs]
libset = {}
for arm in ARMS:
    libset[(arm, "mined")] = {r["pattern"] for r in libs[arm]["rows"]}
    libset[(arm, "random")] = {r["pattern"] for r in libs[arm]["random"]}

correct = {}
faith = defaultdict(lambda: [0, 0])
hallu = defaultdict(lambda: [0, 0])
unparsed = 0
for r in recs:
    j = parse_json(r["reply"])
    if j is None:
        unparsed += 1
        continue
    diag = str(j.get("diagnosis", "")).strip().lower()
    correct[(r["model"], r["case_key"])] = \
        int(diag in ("normal", "faulty") and
            (diag == "faulty") == (r["truth"] == "faulty"))
    cites = j.get("cited_patterns") or []
    mode = r["case_key"].split("|")[1]
    if mode == "nolib":
        hallu[r["model"]][1] += 1
        hallu[r["model"]][0] += int(len(cites) > 0)
    else:
        ls = libset[(r["arm"], mode)]
        for c in cites:
            faith[(r["model"], mode)][1] += 1
            faith[(r["model"], mode)][0] += int(c.strip() in ls)

summary = {"n_calls": len(recs), "unparsed": unparsed,
           "lib_stats": {a: libs[a]["stats"] for a in ARMS},
           "arms": {}}
for arm in ARMS:
    arm_out = {}
    keys = {r["case_key"] for r in recs if r["arm"] == arm}
    for m in MODELS:
        per = {}
        for mode in ("nolib", "mined", "random"):
            ks = [correct[(m, k)] for k in keys
                  if k.split("|")[1] == mode
                  and (m, k) in correct]
            n = len(ks)
            k = sum(ks)
            lo, hi = wilson(k, n)
            per[mode] = {"correct": k, "n": n, "acc": round(k / max(n, 1), 4),
                         "wilson95": [round(lo, 4), round(hi, 4)]}
        b = c = both = 0
        for ck in keys:
            if not ck.endswith("|mined"):
                continue
            base = ck[:-6]
            w = correct.get((m, base + "|mined"))
            wo = correct.get((m, base + "|nolib"))
            if w is None or wo is None:
                continue
            both += 1
            b += (w == 1 and wo == 0)
            c += (w == 0 and wo == 1)
        chi2, p = mcnemar(b, c)
        f, h = faith[(m, "mined")], hallu[m]
        arm_out[m] = {
            "per_condition": per,
            "mcnemar_mined_vs_nolib": {"n_paired": both, "b": b, "c": c,
                                       "chi2": round(chi2, 3),
                                       "p": round(p, 5)},
            "citation_faithfulness_mined":
                round(f[0] / max(f[1], 1), 4), "citations_mined": f[1],
            "faithfulness_random":
                round(faith[(m, "random")][0] / max(faith[(m, "random")][1], 1), 4),
            "cond_nolib_hallu": h[0], "cond_nolib_n": h[1],
        }
    summary["arms"][arm] = arm_out

json.dump(summary, open(OUT, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
for arm in ARMS:
    print(f"== {arm} ==")
    for m in MODELS:
        s = summary["arms"][arm][m]
        pc = s["per_condition"]
        print(f"{LABEL[m]:12s} "
              f"nolib={pc['nolib']['acc']:.2f} mined={pc['mined']['acc']:.2f} "
              f"rand={pc['random']['acc']:.2f} "
              f"faith={s['citation_faithfulness_mined']:.2f} "
              f"p(mined vs none)={s['mcnemar_mined_vs_nolib']['p']}")
print("saved", OUT)
