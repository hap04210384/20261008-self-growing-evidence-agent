# -*- coding: utf-8 -*-
"""exp_e5_runs.py —— E5 效率对照的逐次计时（供 IQR / 误差条）。

与 exp_e5_efficiency.py 完全相同的 11 个配置与 reps=5，借助
run_tensorfim 内置的 reps 循环取得每次重复的耗时并逐行落盘。
按 (dataset, thr, engine) 粒度断点续跑；被打断后重跑同一命令即可。

用法: python exp_e5_runs.py [dataset[:thr] ...]   # 不带参数跑全部
产出: results/e5_runs.csv
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from engine_adapter import run_tensorfim            # noqa: E402

DATA = os.path.join(ROOT, "data", "fimi")
OUT = os.path.join(ROOT, "results")
CSV = os.path.join(OUT, "e5_runs.csv")

REPS = 5
CONFIGS = {
    "skab_valve1": [0.05, 0.10, 0.20],
    "cmapss":      [0.30, 0.40],
    "ai4i":        [0.01, 0.05],
    "hai":         [0.10, 0.20],
    "cwru":        [0.10, 0.20],
}


def main():
    os.makedirs(OUT, exist_ok=True)
    done = set()
    if os.path.exists(CSV):
        prev = pd.read_csv(CSV)
        done = set(zip(prev["dataset"], prev["thr"], prev["engine"]))
        f = open(CSV, "a", encoding="utf-8")
    else:
        f = open(CSV, "w", encoding="utf-8")
        f.write("dataset,thr,engine,rep,secs\n")

    args = sys.argv[1:]
    targets = []
    for ds, thrs in CONFIGS.items():
        for thr in thrs:
            if args and not any(a.split(":")[0] == ds and
                                (":" not in a or abs(float(a.split(":")[1])
                                                     - thr) < 1e-9)
                                for a in args):
                continue
            targets.append((ds, thr))

    for ds, thr in targets:
        fimi = os.path.abspath(os.path.join(DATA, f"{ds}.txt"))
        for eng in ("baseline", "bmma"):
            if (ds, thr, eng) in done:
                continue
            res = run_tensorfim(eng, fimi, thr, reps=REPS)
            if not res.get("ok"):
                print(f"SKIP {ds}@{thr} {eng}: not ok", flush=True)
                continue
            for rep, secs in enumerate(res.get("tot_times") or []):
                f.write(f"{ds},{thr},{eng},{rep},{secs:.6f}\n")
            f.flush()
            ts = res.get("tot_times") or []
            print(f"{ds}@{thr} {eng}: {len(ts)} reps, "
                  f"median {sorted(ts)[len(ts)//2]:.4f}s", flush=True)
    f.close()


if __name__ == "__main__":
    main()
