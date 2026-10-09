# -*- coding: utf-8 -*-
"""summarize_loro.py —— LORO 无泄漏评估统计汇总（手稿 4.5 节数据源）。

指标（每个 model × group × split × mode）：
  accuracy        诊断正确率（66 窗：33 正常 + 33 故障）
  fpr             假阳性率（正常窗被判 faulty 的比例）
  faithfulness    引用忠实度：cited_patterns 全部真实存在于对应模式库
                  （nolib 条件无库可引，cited 非空即记不忠实）

跨 split 汇总：均值 ± 标准差（seed 方差，审稿主要问题 7 的一部分）。
显著性：每个模型对 nolib 的 McNemar 检验（loro / insample 分别）；
        3 个模型间 Holm 校正（对 loro 相对 nolib 的提升幅度）。
所有数进 results/loro/summary.json；关键表打印到 stdout。
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "results", "loro")
CALLS = os.path.join(OUTDIR, "calls.jsonl")
LIBS = os.path.join(OUTDIR, "libraries.json")
SUMMARY = os.path.join(OUTDIR, "summary.json")

MODELS = ("deepseek-chat", "qwen-plus", "glm-4-air")
GROUPS = ("valve1", "valve2", "other")
SPLITS = (0, 1, 2)
MODES = ("nolib", "loro", "insample")


def parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(c - h, 0.0), min(c + h, 1.0))


def mcnemar(b, c):
    """精确二项检验（两分类不一致对），返回 (p 值, 方向)。
    b = nolib 对 / 新条件错；c = nolib 错 / 新条件对。"""
    if b + c == 0:
        return (1.0, "=")
    k = min(b, c)
    n = b + c
    p = 2.0 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    return (min(p, 1.0), ">" if c > b else ("<" if b > c else "="))


def holm(pvals):
    """Holm 逐步校正，返回校正后 p 值列表（与输入同序）。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, val)
        adj[i] = prev
    return adj


def main():
    libs = json.load(open(LIBS, encoding="utf-8"))

    lib_sets = {}
    for key, meta in libs.items():
        for mode_src, tag in (("loro", "loro"), ("in", "insample")):
            lib_sets[(key, tag)] = {r["pattern"] for r in meta[mode_src]}

    # ---- 逐调用解析 ----
    per_call = {}
    for line in open(CALLS, encoding="utf-8"):
        r = json.loads(line)
        if str(r["reply"]).startswith("__ERROR__"):
            continue
        d = parse_json(r["reply"]) or {}
        diag_pred = d.get("diagnosis", "?")
        cited = d.get("cited_patterns", []) or []
        g, s, idx, mode = r["case_key"].split("|")
        truth = r["truth"]
        correct = (diag_pred == truth)
        fp = (diag_pred == "faulty" and truth == "normal")
        key3 = (r["model"], g, int(s))
        lib = lib_sets.get((f"{g}|{s}", mode), set()) if mode != "nolib" \
            else set()
        if mode == "nolib":
            faithful = (len(cited) == 0)
        else:
            faithful = all(c in lib for c in cited)
        per_call[(r["model"], r["case_key"])] = {
            "correct": correct, "fp": fp, "faithful": faithful,
            "mode": mode, "group": g, "split": int(s), "model": r["model"],
        }

    # ---- cell 级聚合 ----
    cells = {}
    for model in MODELS:
        for g in GROUPS:
            for s in SPLITS:
                for mode in MODES:
                    recs = [v for (m, k), v in per_call.items()
                            if v["mode"] == mode and v["group"] == g
                            and v["split"] == s and m == model]
                    if not recs:
                        continue
                    n = len(recs)
                    acc = sum(r["correct"] for r in recs) / n
                    cells[(model, g, s, mode)] = {
                        "n": n, "n_correct": sum(r["correct"] for r in recs),
                        "acc": acc, "fpr": None,
                        "faithful": sum(r["faithful"] for r in recs) / n,
                    }

    # FPR：按正常窗（truth == normal）单独统计
    fpr_cells = {}
    for line in open(CALLS, encoding="utf-8"):
        r = json.loads(line)
        if str(r["reply"]).startswith("__ERROR__"):
            continue
        d = parse_json(r["reply"]) or {}
        g, s, idx, mode = r["case_key"].split("|")
        if r["truth"] != "normal":
            continue
        pred = d.get("diagnosis", "?")
        k = (r["model"], g, int(s), mode)
        a = fpr_cells.setdefault(k, [0, 0])
        a[0] += (pred == "faulty")
        a[1] += 1
    for k, (fp, n) in fpr_cells.items():
        cells[k]["fpr"] = fp / n

    # ---- (model, mode) 跨 split 汇总：mean ± std ----
    agg = {}
    for model in MODELS:
        for mode in MODES:
            accs, fprs, fths, ns = [], [], [], []
            for g in GROUPS:
                for s in SPLITS:
                    c = cells.get((model, g, s, mode))
                    if c:
                        accs.append(c["acc"])
                        fprs.append(c["fpr"])
                        fths.append(c["faithful"])
                        ns.append(c["n"])
            if not accs:
                continue
            mean = sum(accs) / len(accs)
            sd = (sum((x - mean) ** 2 for x in accs) / (len(accs) - 1)) ** 0.5 \
                if len(accs) > 1 else 0.0
            wf = wilson(mean, sum(ns))
            agg[(model, mode)] = {
                "acc_mean": mean, "acc_std": sd,
                "acc_wilson_lo": wf[1], "acc_wilson_hi": wf[2],
                "fpr_mean": sum(fprs) / len(fprs),
                "faith_mean": sum(fths) / len(fths),
                "n_windows": sum(ns), "n_cells": len(accs),
            }

    # ---- McNemar：每个模型，loro / insample 分别 vs nolib ----
    mcn = {}
    for model in MODELS:
        for g in GROUPS:
            for s in SPLITS:
                pairs = {}
                for (m, k), v in per_call.items():
                    if (m, v["group"], v["split"]) != (model, g, s):
                        continue
                    pairs.setdefault(k.split("|")[2], {})[v["mode"]] = v
                for mode in ("loro", "insample"):
                    b = c = 0
                    for idx, pm in pairs.items():
                        if "nolib" not in pm or mode not in pm:
                            continue
                        if pm["nolib"]["correct"] and not pm[mode]["correct"]:
                            b += 1
                        elif not pm["nolib"]["correct"] and pm[mode]["correct"]:
                            c += 1
                    a = mcn.setdefault((model, mode), [0, 0])
                    a[0] += b
                    a[1] += c
    mcn_res = {}
    for (model, mode), (b, c) in mcn.items():
        p, direc = mcnemar(b, c)
        mcn_res[(model, mode)] = {"b": b, "c": c, "p": p, "dir": direc}

    # ---- Holm：3 模型 × {loro, insample} 共 6 个检验 ----
    keys6 = [(m, md) for m in MODELS for md in ("loro", "insample")]
    ps = [mcn_res[k]["p"] for k in keys6]
    adj = holm(ps)
    for k, a in zip(keys6, adj):
        mcn_res[k]["p_holm"] = a

    # ---- 输出 ----
    out = {
        "cells": {f"{m}|{g}|{s}|{md}": v
                  for (m, g, s, md), v in cells.items()},
        "agg": {f"{m}|{md}": v for (m, md), v in agg.items()},
        "mcnemar": {f"{m}|{md}": v for (m, md), v in mcn_res.items()},
    }
    json.dump(out, open(SUMMARY, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"{'model':<14}{'mode':<10}{'acc':>16}{'fpr':>8}{'faith':>8}")
    for model in MODELS:
        for mode in MODES:
            a = agg.get((model, mode))
            if a:
                print(f"{model:<14}{mode:<10}"
                      f"{a['acc_mean']:.3f}±{a['acc_std']:.3f}   "
                      f"{a['fpr_mean']:.3f}  {a['faith_mean']:.3f}")
    print()
    for (model, mode), v in mcn_res.items():
        print(f"McNemar {model} {mode} vs nolib: b={v['b']} c={v['c']} "
              f"p={v['p']:.4f} Holm={v.get('p_holm', 0):.4f} ({v['dir']})")
    print("->", SUMMARY)


if __name__ == "__main__":
    sys.exit(main())
