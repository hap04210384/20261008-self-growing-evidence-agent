# -*- coding: utf-8 -*-
"""summarize_llm_small.py —— GLM-4-Flash / Qwen-Turbo 小模型消融统计（纯本地）。

与 summarize_llm_diag.py 完全相同的判据：
- 准确率（valid diagnosis 且与真值一致）；
- 引用忠实度：with_lib 条件下引用模式是否逐字存在于该组所附库（top-8）；
- 无库条件幻觉率（condA 中引用非空的比例）；
- 有库 vs 无库的配对 McNemar（按 case_key 配对，连续性校正）+ Wilson 95% CI。

产出：results/llm_small/summary.json
"""
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from exp_llm_diagnosis import parse_json, load_group, GROUPS   # noqa: E402

CALLS = os.path.join(ROOT, "results", "llm_small", "calls.jsonl")
OUT = os.path.join(ROOT, "results", "llm_small", "summary.json")

SMALL = ["glm-4-flash", "qwen-turbo"]
LABEL = {"glm-4-flash": "GLM-4-Flash", "qwen-turbo": "Qwen-Turbo"}


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
libs = {g: {r["pattern"] for r in load_group(g)[2]} for g in GROUPS}

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
    correct[(r["model"], r["with_lib"], r["case_key"])] = \
        int(diag in ("normal", "faulty") and
            (diag == "faulty") == (r["truth"] == "faulty"))
    cites = j.get("cited_patterns") or []
    if r["with_lib"]:
        for c in cites:
            faith[r["model"]][1] += 1
            faith[r["model"]][0] += int(c.strip() in libs[r["group"]])
    else:
        hallu[r["model"]][1] += 1
        hallu[r["model"]][0] += int(len(cites) > 0)

summary = {"n_calls": len(recs), "unparsed": unparsed, "models": {}}
for m in SMALL:
    ks = {False: [0, 0], True: [0, 0]}
    for (mm, lib, ck), v in correct.items():
        if mm == m:
            ks[lib][0] += v
            ks[lib][1] += 1
    b = c = both = 0
    keys = {ck for (mm, lib, ck) in correct if mm == m}
    for ck in keys:
        w = correct.get((m, True, ck))
        wo = correct.get((m, False, ck))
        if w is None or wo is None:
            continue
        both += 1
        b += (w == 1 and wo == 0)
        c += (w == 0 and wo == 1)
    chi2, p = mcnemar(b, c)
    lo0, hi0 = wilson(ks[False][0], ks[False][1])
    lo1, hi1 = wilson(ks[True][0], ks[True][1])
    f, h = faith[m], hallu[m]
    summary["models"][m] = {
        "acc_without_library": round(ks[False][0] / max(ks[False][1], 1), 4),
        "acc_with_library": round(ks[True][0] / max(ks[True][1], 1), 4),
        "n": ks[False][1],
        "wilson95_without": [round(lo0, 4), round(hi0, 4)],
        "wilson95_with": [round(lo1, 4), round(hi1, 4)],
        "mcnemar": {"n_paired": both, "lib_only_correct": b,
                    "nolib_only_correct": c, "chi2": round(chi2, 3),
                    "p": round(p, 5)},
        "citation_faithfulness": round(f[0] / max(f[1], 1), 4),
        "citations_total": f[1],
        "condA_hallucinated_citations": h[0], "condA_n": h[1],
    }

json.dump(summary, open(OUT, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
for m in SMALL:
    s = summary["models"][m]
    print(f"{LABEL[m]:12s} w/o={s['acc_without_library']:.2f} "
          f"CI[{s['wilson95_without'][0]:.2f},{s['wilson95_without'][1]:.2f}] "
          f"with={s['acc_with_library']:.2f} "
          f"CI[{s['wilson95_with'][0]:.2f},{s['wilson95_with'][1]:.2f}] "
          f"faith={s['citation_faithfulness']:.3f} "
          f"hallu={s['condA_hallucinated_citations']}/{s['condA_n']} "
          f"mcnemar p={s['mcnemar']['p']}")
print("saved", OUT)
