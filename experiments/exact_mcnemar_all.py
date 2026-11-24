# -*- coding: utf-8 -*-
"""exact_mcnemar_all.py —— 第二轮修回：全部 McNemar p 值改精确双边重算。

规范方法（写入手稿 3.8）：
  精确双边 McNemar 检验：在 H0 下 B ~ Binomial(b+c, 0.5)，
  p = 2 * min(P(X<=b), P(X>=b))，上限 1.0。
  （替代此前的连续性校正渐近卡方；LORO 汇总脚本的公式在 b=c 时不 cap，
  本脚本为唯一规范实现。）

覆盖手稿全部 p 值位点：
  S1 主诊断 198 窗：库 vs 无库（3 模型）
  S2 随机证据对照：random vs 无库（3 模型）；minsup(Apriori) vs 我们的库（3 模型）
  S3 小模型：glm-4-flash / qwen-turbo 库 vs 无库
  S4 HAI：DeepSeek 库 vs 无库
  S5 LORO：loro vs nolib（3 模型）、loro vs insample（DS、Qwen）
  S6 预算消融：全部 k 对配对 + 各模型峰值 p

产物：results/exact_mcnemar_revision.json（新 p 值权威来源）
"""
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.join(ROOT, "reproducibility", "20261008-self-growing-evidence-agent")
sys.path.insert(0, HERE)

from exp_llm_diagnosis import parse_json  # noqa: E402

OUT = os.path.join(ROOT, "results", "exact_mcnemar_revision.json")


def exact_mcnemar(b, c):
    """精确双边 McNemar：2*min(P(X<=b), P(X>=b))，X~Bin(b+c,1/2)，cap 1。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k) = sum_{i=0}^k C(n,i) / 2^n
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def corrected_mcnemar(b, c):
    """旧方法（连续性校正渐近），用于对照输出。"""
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return math.erfc(math.sqrt(chi2 / 2))


def correctness(calls_path, cond_key=None, strip_suffix=False):
    """{(model, cond, case_key): 0/1}；cond_key 指定用哪个字段作条件。
    strip_suffix=True 时去掉 case_key 末尾 "|<cond>"（llm_baseline 等格式）。
    with_lib 字段可能是字符串 "True"/"False"（llm_small），按字符串判。"""
    correct = {}
    for line in open(calls_path, encoding="utf-8"):
        r = json.loads(line)
        j = parse_json(r["reply"])
        if j is None:
            continue
        diag = str(j.get("diagnosis", "")).strip().lower()
        ok = int(diag in ("normal", "faulty") and
                 (diag == "faulty") == (r["truth"] == "faulty"))
        if cond_key:
            cond = r[cond_key]
        else:
            cond = "lib" if str(r.get("with_lib")) == "True" else "nolib"
        ck = r["case_key"]
        if strip_suffix and "|" in ck:
            ck = ck.rsplit("|", 1)[0]
        correct[(r["model"], cond, ck)] = ok
    return correct


def pair(correct, model, cond_a, cond_b):
    keys = {k[2] for k in correct if k[0] == model}
    b = c = 0
    for ck in keys:
        a = correct.get((model, cond_a, ck))
        bb = correct.get((model, cond_b, ck))
        if a is None or bb is None:
            continue
        if a == 1 and bb == 0:
            b += 1
        elif a == 0 and bb == 1:
            c += 1
    return b, c


res = {}

# ---------------- S1 主诊断 198 窗 ----------------
main = correctness(os.path.join(ROOT, "results", "llm_diag", "calls.jsonl"))
res["S1_main_lib_vs_nolib"] = {}
for m in ("deepseek-chat", "qwen-plus", "glm-4-air"):
    b, c = pair(main, m, "lib", "nolib")
    res["S1_main_lib_vs_nolib"][m] = {"b": b, "c": c,
                                      "p_exact": exact_mcnemar(b, c),
                                      "p_old": corrected_mcnemar(b, c)}

# ---------------- S2 随机 / minsup 对照 ----------------
base = correctness(os.path.join(REPO, "results", "llm_baseline", "calls.jsonl"),
                   cond_key="cond", strip_suffix=True)
# 与主实验合并以配对（no-lib 与 lib 条件来自 calls.jsonl）
merged = dict(main)
merged.update(base)
res["S2_random_vs_nolib"] = {}
res["S2_minsup_vs_lib"] = {}
for m in ("deepseek-chat", "qwen-plus", "glm-4-air"):
    b, c = pair(merged, m, "random", "nolib")
    res["S2_random_vs_nolib"][m] = {"b": b, "c": c,
                                    "p_exact": exact_mcnemar(b, c),
                                    "p_old": corrected_mcnemar(b, c)}
    b, c = pair(merged, m, "minsup", "lib")
    res["S2_minsup_vs_lib"][m] = {"b": b, "c": c,
                                  "p_exact": exact_mcnemar(b, c),
                                  "p_old": corrected_mcnemar(b, c)}

# ---------------- S3 小模型 ----------------
small = correctness(os.path.join(REPO, "results", "llm_small", "calls.jsonl"),
                    strip_suffix=True)
res["S3_small_lib_vs_nolib"] = {}
for m in ("glm-4-flash", "qwen-turbo"):
    b, c = pair(small, m, "lib", "nolib")
    res["S3_small_lib_vs_nolib"][m] = {"b": b, "c": c,
                                       "p_exact": exact_mcnemar(b, c),
                                       "p_old": corrected_mcnemar(b, c)}

# ---------------- S4 HAI ----------------
hai = correctness(os.path.join(REPO, "results", "hai_diag", "calls.jsonl"),
                  cond_key="mode", strip_suffix=True)
res["S4_hai_lib_vs_nolib"] = {}
for m in ("deepseek-chat", "qwen-plus", "glm-4-air"):
    b, c = pair(hai, m, "mined", "nolib")
    res["S4_hai_lib_vs_nolib"][m] = {"b": b, "c": c,
                                     "p_exact": exact_mcnemar(b, c),
                                     "p_old": corrected_mcnemar(b, c)}

# ---------------- S5 LORO ----------------
loro = correctness(os.path.join(REPO, "results", "loro", "calls.jsonl"),
                   cond_key="mode", strip_suffix=True)
res["S5_loro_vs_nolib"] = {}
res["S5_loro_vs_insample"] = {}
for m in ("deepseek-chat", "qwen-plus", "glm-4-air"):
    b, c = pair(loro, m, "loro", "nolib")
    res["S5_loro_vs_nolib"][m] = {"b": b, "c": c,
                                  "p_exact": exact_mcnemar(b, c),
                                  "p_old": corrected_mcnemar(b, c)}
    b, c = pair(loro, m, "loro", "insample")
    res["S5_loro_vs_insample"][m] = {"b": b, "c": c,
                                     "p_exact": exact_mcnemar(b, c),
                                     "p_old": corrected_mcnemar(b, c)}

# ---------------- S6 预算消融 ----------------
budget_path = os.path.join(ROOT, "results", "llm_diag", "budget.jsonl")
budget_correct = {}
for line in open(budget_path, encoding="utf-8"):
    r = json.loads(line)
    j = parse_json(r["reply"])
    if j is None:
        continue
    diag = str(j.get("diagnosis", "")).strip().lower()
    ok = int(diag in ("normal", "faulty") and
             (diag == "faulty") == (r["truth"] == "faulty"))
    budget_correct[(r["model"], r["k"], r["case_key"])] = ok
# k=0 复用主实验 no-lib（注意：只取 nolib 条件，lib 条件不得混入）
for (m, cond, ck), ok in list(main.items()):
    if cond == "nolib":
        budget_correct[(m, 0, ck)] = ok

res["S6_budget_pairs"] = {}
KS = (1, 2, 3, 5, 8)
for m in ("deepseek-chat", "qwen-plus", "glm-4-air"):
    keys = {k[2] for k in budget_correct if k[0] == m}
    for k in KS:
        b = c = 0
        for ck in keys:
            hi = budget_correct.get((m, k, ck))
            lo = budget_correct.get((m, 0, ck))
            if hi is None or lo is None:
                continue
            if hi == 1 and lo == 0:
                b += 1
            elif hi == 0 and lo == 1:
                c += 1
        res["S6_budget_pairs"][f"{m}_k{k}_vs_k0"] = {
            "b": b, "c": c, "p_exact": exact_mcnemar(b, c),
            "p_old": corrected_mcnemar(b, c)}
    # k=5 vs k=8（深修正文引用的"p = 0.83"）
    b = c = 0
    for ck in keys:
        a5 = budget_correct.get((m, 5, ck))
        a8 = budget_correct.get((m, 8, ck))
        if a5 is None or a8 is None:
            continue
        if a5 == 1 and a8 == 0:
            b += 1
        elif a5 == 0 and a8 == 1:
            c += 1
    res["S6_budget_pairs"][f"{m}_k5_vs_k8"] = {
        "b": b, "c": c, "p_exact": exact_mcnemar(b, c),
        "p_old": corrected_mcnemar(b, c)}


# ---- Holm 校正（3.8 节声明的两个检验族）----
def holm(ps):
    """Holm step-down，返回与输入同序的校正后 p 值。"""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * ps[i])
        prev = max(prev, val)
        adj[i] = prev
    return adj

# 族1：4.7 节 LORO 六个检验（3 模型 × {loro_vs_nolib, loro_vs_insample}）
f1_keys = [(m, d) for m in ("deepseek-chat", "qwen-plus", "glm-4-air")
           for d in ("S5_loro_vs_nolib", "S5_loro_vs_insample")]
f1_ps = [res[d][m]["p_exact"] for m, d in f1_keys]
f1_adj = holm(f1_ps)
res["holm_families"] = {"loro_six": {}, "budget_peak": {}}
for (m, d), a in zip(f1_keys, f1_adj):
    res["holm_families"]["loro_six"][f"{m}|{d}"] = a

# 族2：4.8 节峰值检验（3 模型的 peak vs k0）
f2_keys = [("deepseek-chat", "k5"), ("qwen-plus", "k8"), ("glm-4-air", "k3")]
f2_keys = [(m, k) for m, k in f2_keys]
f2_ps = [res["S6_budget_pairs"][f"{m}_{k}_vs_k0"]["p_exact"] for m, k in f2_keys]
f2_adj = holm(f2_ps)
for (m, k), a in zip(f2_keys, f2_adj):
    res["holm_families"]["budget_peak"][f"{m}|peak_{k}_vs_k0"] = a

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)

# 打印对照表
for site, d in res.items():
    print(f"== {site}")
    for m, v in d.items():
        if not isinstance(v, dict) or "b" not in v:
            continue
        print(f"  {m:28s} b={v['b']:3d} c={v['c']:3d} "
              f"p_exact={v['p_exact']:.5f}  p_old={v['p_old']:.5f}")
print("saved", OUT)
