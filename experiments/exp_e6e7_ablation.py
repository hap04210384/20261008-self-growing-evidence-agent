# -*- coding: utf-8 -*-
"""exp_e6e7_ablation.py —— 离散化消融（E6）+ 窗口参数敏感性（E7）。

E6: 同一窗口协议（w=30, hop=15），三种离散化：
    - p2_online: P² 分位数在线自适应（本文，含最小箱宽保护）
    - equal_width: 全局 min/max 等宽分箱（传统基线）
    - equal_freq:  全局分位数分箱（原型 fit_bins）
    指标：异常窗被判别性模式（lift>=2 且 sup_anom>=0.1）覆盖的比例、
          MFI 数、判别性模式数、空箱（边界塌陷）通道数。
E7: 仅 P² 在线离散化，w ∈ {15, 30, 60}，hop = w/2，同指标。

产物：results/e6e7_ablation/summary.json
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

OUTDIR = os.path.join(ROOT, "results", "e6e7_ablation")
os.makedirs(OUTDIR, exist_ok=True)

GROUPS = ("valve1", "valve2")
W_DEFAULT, HOP_DEFAULT = 30, 15
MIN_SUP = 0.05
DISC_LIFT, DISC_SUP = 2.0, 0.10


def load_group_raw(grp):
    """返回拼接后的 (df, anom 列名 __anom__)，按实验组读取 SKAB 原始文件。"""
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


def window_medians(df, cols, window, hop):
    """滑窗中位数序列 + 每窗异常标志。返回 (meds[list[dict]], anom[np.array])。"""
    meds, anom = [], []
    n = len(df)
    for s in range(0, n - window + 1, hop):
        block = df.iloc[s:s + window]
        meds.append({c: float(block[c].median()) for c in cols})
        anom.append(int(block["__anom__"].max()))
    return meds, np.array(anom)


def txns_p2(df, cols, window, hop):
    """P² 在线：逐样本更新估计器，每窗用当时边界定项。
    返回额外 used_bins: {col: 决策时刻出现过的分箱编号集合}（在线空箱统计）。"""
    binners = {c: OnlineBinner() for c in cols}
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
            b = p2_bin_index(float(df[c].iloc[s:e].median()), binners[c].edges())
            used[c].add(b)
            items.append(f"{c}={b}")
        txns.append(items)
        anom.append(int(df["__anom__"].iloc[s:e].max()))
    edges = {c: binners[c].edges() for c in cols}
    return txns, np.array(anom), edges, used


def txns_fixed(df, cols, edges, window, hop):
    meds, anom = window_medians(df, cols, window, hop)
    txns = [[f"{c}={spike.bin_index(m[c], edges[c])}" for c in cols]
            for m in meds]
    return txns, anom, edges


def empty_bins(edges, meds, cols):
    """每通道：窗口中位数未覆盖到的分箱数（边界塌陷/空箱指示）。"""
    total = 0
    for c in cols:
        used = {spike.bin_index(m[c], edges[c]) for m in meds}
        total += max(0, (len(edges[c]) - 1) - len(used))
    return total


TOPK_PATTERNS = 5  # 等模式预算：所有变体只用 top-5 判别性模式比覆盖率


def evaluate(txns, anom, edges, meds, cols, used=None):
    mfis, _state = spike.mine_mfi(txns, min_sup=MIN_SUP)
    pats, base_rate = spike.pattern_fault_analysis(mfis, txns, anom)
    disc = pats[(pats["lift"] >= DISC_LIFT) & (pats["sup_anom"] >= DISC_SUP)]
    topk = disc.head(TOPK_PATTERNS)
    anom_idx = np.where(anom == 1)[0]
    covered = set()
    for _, r in topk.iterrows():
        pset = set(r["pattern"].split(" ∧ "))
        for i in anom_idx:
            if pset <= set(txns[i]):
                covered.add(i)
    cov = len(covered) / max(1, len(anom_idx))
    if used is not None:  # 在线语义：决策时刻从未用到的分箱
        eb = sum(3 - len(u) for u in used.values())
    else:
        eb = empty_bins(edges, meds, cols) if meds is not None else None
    return {
        "n_windows": len(txns),
        "n_anom_windows": int(len(anom_idx)),
        "anom_base_rate": round(float(np.mean(anom)), 4),
        "n_mfi": len(mfis),
        "n_discriminative": int(len(disc)),
        "fault_coverage_topk": round(float(cov), 4),
        "empty_bins": eb,
    }


def main():
    summary = {"E6_discretizer": {}, "E7_window": {}}
    for grp in GROUPS:
        df = load_group_raw(grp)
        cols = spike.SENSOR_COLS

        # ---- E6: 三种离散化，同窗口协议 ----
        res6 = {}
        txns, anom, edges, used = txns_p2(df, cols, W_DEFAULT, HOP_DEFAULT)
        meds, _ = window_medians(df, cols, W_DEFAULT, HOP_DEFAULT)
        res6["p2_online"] = evaluate(txns, anom, edges, meds, cols, used=used)

        allv = {c: df[c].values for c in cols}
        ew = {c: np.linspace(allv[c].min(), allv[c].max(), 4) for c in cols}
        txns, anom, _ = txns_fixed(df, cols, ew, W_DEFAULT, HOP_DEFAULT)
        res6["equal_width"] = evaluate(txns, anom, ew, meds, cols)

        ef = spike.fit_bins(df, cols, 3)
        txns, anom, _ = txns_fixed(df, cols, ef, W_DEFAULT, HOP_DEFAULT)
        res6["equal_freq"] = evaluate(txns, anom, ef, meds, cols)

        summary["E6_discretizer"][grp] = res6

        # ---- E7: 窗口敏感性（仅 P² 在线）----
        res7 = {}
        for w in (15, 30, 60):
            txns, anom, edges, used = txns_p2(df, cols, w, w // 2)
            meds_w, _ = window_medians(df, cols, w, w // 2)
            res7[f"w{w}"] = evaluate(txns, anom, edges, meds_w, cols, used=used)
        summary["E7_window"][grp] = res7
        print("done", grp)

    with open(os.path.join(OUTDIR, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
