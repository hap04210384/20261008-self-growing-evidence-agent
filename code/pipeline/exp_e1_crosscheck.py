# -*- coding: utf-8 -*-
"""exp_e1_crosscheck.py —— E1 AnyFIM 模式的故障语义交叉验证。

对 SKAB 三组（valve1 / valve2 / other）：
1. AnyFIM 实用深度内累积池，按异常窗命中率排序取 top-10 解码模式；
2. 用富集模式集（E4 真值口径）做窗口级覆盖：异常窗被标记比例、正常窗误标比例；
3. 验证 valve 组应复现已知阀件故障签名 Pressure=1 ∧ Flow=0 家族，
   other 组应给出不同模式。

产出: results/e1_crosscheck_<grp>.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from engine_adapter import (decode_itemset, load_items_map, run_anyfim)  # noqa: E402
from exp_e4_anytime import (EXCL_THRESH, MIN_CNT, PRACT_MIN_SUP,           # noqa: E402
                            load_group_txns)

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)


def main():
    report = {}
    for grp in ("valve1", "valve2", "other"):
        fimi = os.path.abspath(os.path.join(DATA, "fimi", f"skab_{grp}.txt"))
        items = load_items_map(fimi + ".items.json")
        inv = {v: k for k, v in items.items()}

        r = run_anyfim(fimi, len(items), timeout=1800)
        assert r["ok"], f"AnyFIM failed on {grp}"

        txns, labels = load_group_txns(grp)
        n, npos = len(txns), sum(labels)
        str_txns = [set(t) for t in txns]

        cnt = __import__("collections").Counter()
        for t in txns:
            for x in set(t):
                cnt[x] += 1
        ranked = sorted(cnt.values(), reverse=True)
        K = max(k for k in r["stages"] if ranked[k - 1] / n >= PRACT_MIN_SUP)

        seen, union = set(), set()
        for k in sorted(r["stages"]):
            if k > K:
                break
            seen |= {x for x, _ in r["stages"][k]}
            union = set(seen)

        def stats(x):
            s = set(inv[i] for i in x)
            c = sum(1 for t in str_txns if s <= t)
            p = sum(1 for t, l in zip(str_txns, labels) if l and s <= t)
            return c, p

        scored = []
        for x in union:
            c, p = stats(x)
            lift = (p / npos) / (c / n) if c else 0.0
            scored.append((x, c, p, lift))
        truth = [(x, c, p, l) for x, c, p, l in scored
                 if c >= MIN_CNT and p / c >= EXCL_THRESH]

        # top-10 按异常窗命中数降序
        top = sorted(truth, key=lambda t: -t[2])[:10]
        top_str = [{"pattern": " & ".join(sorted(decode_itemset(x, inv))),
                    "cnt": c, "pos": p, "lift": round(l, 3)}
                   for x, c, p, l in top]

        # 窗口级覆盖：窗口含任一真值模式即标记
        truth_sets = [set(inv[i] for i in x) for x, *_ in truth]
        flagged_pos = sum(1 for t, l in zip(str_txns, labels)
                          if l and any(s <= t for s in truth_sets))
        flagged_neg = sum(1 for t, l in zip(str_txns, labels)
                          if not l and any(s <= t for s in truth_sets))
        nneg = n - npos

        report[grp] = {
            "windows": n, "anomaly_windows": npos,
            "practical_depth": K, "pool_size": len(union),
            "n_truth_patterns": len(truth),
            "top10": top_str,
            "anomaly_window_coverage": round(flagged_pos / npos, 4),
            "normal_window_false_rate": round(flagged_neg / nneg, 4),
        }
        print(f"{grp}: truth={len(truth)} "
              f"anom_cov={report[grp]['anomaly_window_coverage']:.1%} "
              f"false={report[grp]['normal_window_false_rate']:.1%}")
        for t in top_str[:5]:
            print(f"   {t['pattern']}  (cnt={t['cnt']}, pos={t['pos']}, "
                  f"lift={t['lift']})")

    with open(os.path.join(OUT, "e1_crosscheck.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
