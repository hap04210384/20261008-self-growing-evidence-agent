# -*- coding: utf-8 -*-
"""
anytime_curve.py —— E4 实验雏形：anytime 模式质量曲线（模拟）

设计：SKAB 事务按到达顺序逐步喂入；在每个检查点（10%..100%）用参考
挖掘器对前缀挖 MFI，对比"全量数据挖出的故障模式（真值）"：

  - 覆盖率 = 已发现的真值故障模式 / 全部真值故障模式
  - 精确率 = 已发现模式中属于真值的比例

真值故障模式定义：全量挖掘结果中 lift > lift_thresh 的模式。
预期：覆盖率单调上升（anytime 单调性），早期即有非零覆盖 —— 支撑 SP1/SP4。

ENGINE_HOOK：正式版替换为引擎① AnyFIM（天然随时输出，无需逐检查点重挖）。
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data/SKAB_raw")
    ap.add_argument("--out", default="../../results/anytime_v1")
    ap.add_argument("--group", default="valve1")
    ap.add_argument("--min-sup", type=float, default=0.1)
    ap.add_argument("--lift-thresh", type=float, default=3.0)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=15)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    spec = importlib.util.spec_from_file_location(
        "skab_spike", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "skab_spike.py"))
    spike = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spike)

    data_dir = os.path.join(args.data, "data", args.group)
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    fracs = np.round(np.arange(0.1, 1.01, 0.1), 1)

    cover_rows, prec_rows = [], []
    for f in files:
        df = spike.load_skab_file(os.path.join(data_dir, f))
        anom = df["anomaly"].values
        df = df.drop(columns=[c for c in ["datetime", "anomaly", "changepoint"]
                              if c in df.columns])
        df["__anom__"] = anom
        edges = spike.fit_bins(df, spike.SENSOR_COLS, 3)
        _, txns, txn_anom = spike.window_transactions(df, spike.SENSOR_COLS, edges,
                                                      args.window, args.stride)
        if len(txns) < 20:
            continue
        # 真值：全量挖掘中的故障模式（高 lift）
        mfis_full, _ = spike.mine_mfi(txns, min_sup=args.min_sup)
        pat_df, _ = spike.pattern_fault_analysis(mfis_full, txns, txn_anom)
        truth = set(pat_df[pat_df["lift"] > args.lift_thresh]["pattern"])
        if not truth:
            continue
        for fr in fracs:
            k = max(1, int(round(fr * len(txns))))
            mfis_k, _ = spike.mine_mfi(txns[:k], min_sup=args.min_sup)
            found = {" ∧ ".join(p) for p, _ in mfis_k}
            hit = truth & found
            cover_rows.append({"file": f, "frac": fr,
                               "coverage": len(hit) / len(truth)})
            prec_rows.append({"file": f, "frac": fr,
                              "precision": (len(hit) / len(found)) if found else 0.0})

    cov = pd.DataFrame(cover_rows).groupby("frac")["coverage"].agg(["mean", "std"])
    prc = pd.DataFrame(prec_rows).groupby("frac")["precision"].agg(["mean", "std"])
    tbl = cov.join(prc, lsuffix="_cov", rsuffix="_prec").reset_index()
    tbl.to_csv(os.path.join(args.out, f"anytime_curve_{args.group}.csv"),
               index=False, encoding="utf-8-sig")
    print(tbl.to_string(index=False))
    with open(os.path.join(args.out, f"anytime_meta_{args.group}.json"), "w",
              encoding="utf-8") as fp:
        json.dump({"files_used": len(set(cover_rows and [r["file"] for r in cover_rows])),
                   "min_sup": args.min_sup, "lift_thresh": args.lift_thresh},
                  fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
