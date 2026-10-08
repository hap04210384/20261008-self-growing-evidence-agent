# -*- coding: utf-8 -*-
"""exp_e5_efficiency.py —— E5 TensorFIM 效率对照（baseline vs BMMA）。

5 个工业 FIMI 数据集，每档相对支持度阈值，两引擎各 reps 次取中位数。
阈值选取（预实验安全边界）：C-MAPSS >= 0.3（0.2 会爆 20 万节点）；
SKAB/AI4I/HAI/CWRU 用低中两档。

用法: python exp_e5_efficiency.py [dataset ...]   # 不带参数跑全部
产出: results/e5_efficiency.csv（追加模式安全中断）
"""
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from engine_adapter import run_tensorfim            # noqa: E402

DATA = os.path.join(ROOT, "data", "fimi")
OUT = os.path.join(ROOT, "results")
CSV = os.path.join(OUT, "e5_efficiency.csv")

REPS = 5
# 数据集: 阈值列表（相对支持度）
CONFIGS = {
    "skab_valve1": [0.05, 0.10, 0.20],
    "cmapss":      [0.30, 0.40],
    "ai4i":        [0.01, 0.05],
    "hai":         [0.10, 0.20],
    "cwru":        [0.10, 0.30],
}


def med(xs):
    xs = sorted(xs)
    m = len(xs)
    return xs[m // 2] if m % 2 else (xs[m // 2 - 1] + xs[m // 2]) / 2


def main():
    os.makedirs(OUT, exist_ok=True)
    done = set()
    if os.path.exists(CSV):
        prev = pd.read_csv(CSV)
        done = set(zip(prev["dataset"], prev["thr"], prev["engine"]))
        rows = prev.to_dict("records")
    else:
        rows = []

    targets = sys.argv[1:] or list(CONFIGS)
    for ds in targets:
        fimi = os.path.abspath(os.path.join(DATA, f"{ds}.txt"))
        for thr in CONFIGS[ds]:
            for eng in ("baseline", "bmma"):
                if (ds, thr, eng) in done:
                    continue
                t0 = time.time()
                r = run_tensorfim(eng, fimi, thr, reps=REPS, timeout=600)
                wall = time.time() - t0
                row = {
                    "dataset": ds, "thr": thr, "engine": eng,
                    "ok": r["ok"], "n_mfis": len(r["mfis"]),
                    "med_comp_s": round(med(r["comp_times"]), 6) if r["comp_times"] else None,
                    "med_total_s": round(med(r["tot_times"]), 6) if r["tot_times"] else None,
                    "reps_ok": len(r["comp_times"]),
                }
                rows.append(row)
                print(f"{ds}@{thr} {eng}: ok={r['ok']} mfis={row['n_mfis']} "
                      f"med_comp={row['med_comp_s']}s wall={wall:.1f}s", flush=True)
                pd.DataFrame(rows).to_csv(CSV, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["dataset", "thr"], columns="engine",
                         values="med_comp_s")
    piv["speedup"] = piv["baseline"] / piv["bmma"]
    piv.to_csv(os.path.join(OUT, "e5_speedup.csv"), encoding="utf-8-sig")
    print("\n", piv.to_string())


if __name__ == "__main__":
    main()
