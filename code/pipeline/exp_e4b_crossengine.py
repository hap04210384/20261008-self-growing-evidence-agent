# -*- coding: utf-8 -*-
"""exp_e4b_crossengine.py —— 两引擎跨验证：AnyFIM 每轮池 == TensorFIM@精确阈值。

阈值语义：第 k 轮激活支持度第 k 高的项，阈值 = 该项 count/n（与引擎内部
的浮点除法逐位一致，避免打印截断造成的边界差一）。
"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from engine_adapter import (load_items_map, parse_anyfim_results,  # noqa: E402
                            run_tensorfim)
from exp_e4_anytime import load_group_txns                       # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "anytime_v2")
os.makedirs(OUT, exist_ok=True)


def main():
    for grp in ("valve1", "valve2"):
        fimi = os.path.abspath(os.path.join(DATA, "fimi", f"skab_{grp}.txt"))
        items = load_items_map(fimi + ".items.json")
        inv = {v: k for k, v in items.items()}
        stages, meta = parse_anyfim_results(fimi + f"-{len(items)}-stages=Results.txt")

        txns, _ = load_group_txns(grp)
        n = len(txns)
        cnt = Counter()
        for t in txns:
            for x in set(t):
                cnt[x] += 1
        ranked = sorted(cnt.values(), reverse=True)  # 第 k 高项的计数

        rows, nbad = [], 0
        for k in sorted(stages):
            s_k = ranked[k - 1] / n
            pool = {frozenset(inv[i] for i in x) for x, _ in stages[k]}
            r = run_tensorfim("bmma", fimi, s_k, reps=1, timeout=600)
            tf = {frozenset(inv[i] for i in x) for x in r["mfis"]}
            ok = pool == tf
            if not ok:
                nbad += 1
            rows.append({"stage": k, "s_k": round(s_k, 6),
                         "anyfim": len(pool), "tensorfim": len(tf),
                         "identical": ok})
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT, f"e4_crossengine_{grp}.csv"),
                  index=False, encoding="utf-8-sig")
        bad = df[~df["identical"]]
        print(f"{grp}: identical {len(df) - nbad}/{len(df)} stages")
        if len(bad):
            print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
