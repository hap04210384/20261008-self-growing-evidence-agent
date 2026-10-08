# -*- coding: utf-8 -*-
"""llm_diag_stats.py —— LLM 诊断消融的统计补强（不消耗 API，纯本地）。

基于 results/llm_diag/calls.jsonl 计算：
1. 每个 (model, with_lib, group) 的准确率与 Wilson 95% 置信区间；
2. 每个 model 有/无证据库的 McNemar 配对检验（按 case_key 配对，连续性校正）；
3. 按故障类（valve1 / valve2 / other）的分解准确率。

产出: results/llm_diag/stats.json（并在 stdout 打印摘要）
"""
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CALLS = os.path.join(ROOT, "results", "llm_diag", "calls.jsonl")
OUT = os.path.join(ROOT, "results", "llm_diag", "stats.json")

MODEL_ORDER = ["deepseek-chat", "qwen-plus", "glm-4-air"]
MODEL_LABEL = {"deepseek-chat": "DeepSeek-V3", "qwen-plus": "Qwen-Plus",
               "glm-4-air": "GLM-4-Air"}
GROUP_LABEL = {"valve1": "valve-1", "valve2": "valve-2", "other": "other-fault"}

sys_path_fix = os.path.join(HERE)
import sys
sys.path.insert(0, sys_path_fix)
from exp_llm_diagnosis import parse_json  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b, c):
    """连续性校正的 McNemar 卡方；返回 (chi2, p)。"""
    if b + c == 0:
        return (0.0, 1.0)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2))  # chi2 df=1 的生存函数
    return (chi2, p)


recs = [json.loads(l) for l in open(CALLS, encoding="utf-8")]

# 每条记录的正确性（与 summarize_llm_diag.py 同一判据）
correct = {}
unparsed = 0
for r in recs:
    j = parse_json(r["reply"])
    if j is None:
        unparsed += 1
        continue
    diag = str(j.get("diagnosis", "")).strip().lower()
    correct[(r["model"], r["with_lib"], r["case_key"])] = \
        int(diag in ("normal", "faulty") and
            (diag == "faulty") == (r["truth"] == "faulty"))

stats = {"n_calls": len(recs), "unparsed": unparsed, "models": {}}

# 按 (model, with_lib, group) 分解 + Wilson 区间
groups = sorted({r["group"] for r in recs})
agg = defaultdict(lambda: [0, 0])
for r in recs:
    k = correct.get((r["model"], r["with_lib"], r["case_key"]))
    if k is None:
        continue
    gk = (r["model"], r["with_lib"], r["group"])
    agg[gk][0] += k
    agg[gk][1] += 1

for m in MODEL_ORDER:
    mstat = {"per_condition": {}, "per_group": {}, "mcnemar": {}}
    for lib in (False, True):
        ks = sum(agg[(m, lib, g)][0] for g in groups)
        ns = sum(agg[(m, lib, g)][1] for g in groups)
        lo, hi = wilson(ks, ns)
        mstat["per_condition"][str(lib)] = {
            "correct": ks, "n": ns, "acc": round(ks / max(ns, 1), 4),
            "wilson95": [round(lo, 4), round(hi, 4)]}
        for g in groups:
            k, n = agg[(m, lib, g)]
            glo, ghi = wilson(k, n)
            mstat["per_group"].setdefault(GROUP_LABEL[g], {})[str(lib)] = {
                "correct": k, "n": n, "acc": round(k / max(n, 1), 4),
                "wilson95": [round(glo, 4), round(ghi, 4)]}
    # McNemar：按 case_key 配对（同模型 有库 vs 无库）
    keys = {r["case_key"] for r in recs if r["model"] == m}
    b = c = both = 0
    for ck in keys:
        w = correct.get((m, True, ck))
        wo = correct.get((m, False, ck))
        if w is None or wo is None:
            continue
        both += 1
        if w == 1 and wo == 0:
            b += 1
        elif w == 0 and wo == 1:
            c += 1
    chi2, p = mcnemar(b, c)
    mstat["mcnemar"] = {"n_paired": both, "lib_only_correct": b,
                        "nolib_only_correct": c,
                        "chi2": round(chi2, 3), "p": round(p, 5)}
    stats["models"][m] = mstat

json.dump(stats, open(OUT, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

for m in MODEL_ORDER:
    s = stats["models"][m]
    lab = MODEL_LABEL[m]
    for lib in (False, True):
        d = s["per_condition"][str(lib)]
        print(f"{lab:12s} lib={lib!s:5s} acc={d['acc']:.3f} "
              f"CI[{d['wilson95'][0]:.3f},{d['wilson95'][1]:.3f}] n={d['n']}")
    for g, gd in s["per_group"].items():
        a0, a1 = gd["False"]["acc"], gd["True"]["acc"]
        print(f"  {lab:12s} {g:11s} w/o={a0:.2f}  with={a1:.2f}")
    mc = s["mcnemar"]
    print(f"  McNemar: b={mc['lib_only_correct']} c={mc['nolib_only_correct']} "
          f"chi2={mc['chi2']} p={mc['p']}")
print("saved", OUT)
