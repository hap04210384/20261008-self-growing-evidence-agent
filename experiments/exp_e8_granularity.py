# -*- coding: utf-8 -*-
"""exp_e8_granularity.py —— 离散化状态数消融（E8）。

同一窗口协议（w=30, hop=15）、同一等模式预算（top-5 判别性模式，
lift>=2 且 sup_anom>=0.1），比较状态数 n_bins ∈ {3, 5, 7}：
    - p2_online: P² 分位数在线自适应（本文，含最小箱宽保护）
    - equal_freq: 离线全局分位数分箱（oracle 基线）
指标：fault_coverage_topk（异常窗被 top-5 判别性模式并集覆盖比例）、
      n_mfi、n_discriminative、空箱通道数。

产物：results/e8_granularity/summary.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "pipeline"))
import skab_spike as spike
from online_discretize import OnlineBinner, bin_index as p2_bin_index

OUTDIR = os.path.join(ROOT, "results", "e8_granularity")
os.makedirs(OUTDIR, exist_ok=True)

GROUPS = ("valve1", "valve2")
W_DEFAULT, HOP_DEFAULT = 30, 15
MIN_SUP = 0.05
DISC_LIFT, DISC_SUP = 2.0, 0.10
N_BINS_LIST = (3, 5, 7)
TOPK_PATTERNS = 5


def load_group_raw(grp):
    folder = os.path.join(ROOT, "data", "SKAB_raw", "data", grp)
    frames = []
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".csv"):
            df = spike.load_skab_file(os.path.join(folder, fn))
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"anomaly": "__anom__"})
    df = df.drop(columns=[c for c in ["datetime", "changepoint"]
                          if c in df.columns])
    return df


def txns_p2(df, cols, window, hop, n_bins):
    binners = {c: OnlineBinner(n_bins) for c in cols}
    used = {c: set() for c in cols}
    n = len(df)
    txns, anom = [], []
    for s in range(0, n - window + 1, hop):
        e = s + window
        for t in range(s, e):
            for c in cols:
                binners[c].add(float(df[c].iloc[t]))
        items = []
        for c in cols:
            b = p2_bin_index(float(df[c].iloc[s:e].median()),
                             binners[c].edges())
            used[c].add(b)
            items.append(f"{c}={b}")
        txns.append(items)
        anom.append(int(df["__anom__"].iloc[s:e].max()))
    return txns, np.array(anom), used


def txns_fixed(df, cols, edges, window, hop):
    txns, anom = [], []
    for s in range(0, len(df) - window + 1, hop):
        block = df.iloc[s:s + window]
        txns.append([f"{c}={spike.bin_index(float(block[c].median()), edges[c])}"
                     for c in cols])
        anom.append(int(block["__anom__"].max()))
    return txns, np.array(anom)


def evaluate(txns, anom, n_bins, used=None):
    mfis, _state = spike.mine_mfi(txns, min_sup=MIN_SUP)
    pats, _base = spike.pattern_fault_analysis(mfis, txns, anom)
    disc = pats[(pats["lift"] >= DISC_LIFT) & (pats["sup_anom"] >= DISC_SUP)]
    topk = disc.head(TOPK_PATTERNS)
    anom_idx = np.where(anom == 1)[0]
    covered = set()
    for _, r in topk.iterrows():
        pset = set(r["pattern"].split(" ∧ "))
        for i in anom_idx:
            if pset <= set(txns[i]):
                covered.add(i)
    if used is not None:
        eb = sum(n_bins - len(u) for u in used.values())
    else:
        eb = None
    return {
        "n_windows": len(txns),
        "n_anom_windows": int(len(anom_idx)),
        "anom_base_rate": round(float(np.mean(anom)), 4),
        "n_mfi": len(mfis),
        "n_discriminative": int(len(disc)),
        "fault_coverage_topk": round(len(covered) / max(1, len(anom_idx)), 4),
        "empty_bins": eb,
    }


def main():
    summary = {}
    for grp in GROUPS:
        df = load_group_raw(grp)
        cols = spike.SENSOR_COLS
        res = {}
        for nb in N_BINS_LIST:
            txns, anom, used = txns_p2(df, cols, W_DEFAULT, HOP_DEFAULT, nb)
            r = {"p2_online": evaluate(txns, anom, nb, used=used)}
            ef = spike.fit_bins(df, cols, nb)
            txns, anom = txns_fixed(df, cols, ef, W_DEFAULT, HOP_DEFAULT)
            r["equal_freq"] = evaluate(txns, anom, nb)
            res[f"n_bins={nb}"] = r
            print(f"done {grp} n_bins={nb}")
        summary[grp] = res

    with open(os.path.join(OUTDIR, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
