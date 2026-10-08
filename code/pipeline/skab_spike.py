# -*- coding: utf-8 -*-
"""
skab_spike.py —— 最小端到端链路原型（论文项目 spike）

链路：SKAB 传感时序 → 分位数离散化 → 滑动窗事务化 → MFI 挖掘
      → 模式-故障对照分析（验证"挖出的模式能否对上物理注入故障"）

用法：
    python skab_spike.py --data ../data/SKAB_raw --out ../results/spike_v1

说明：本脚本为打通链路的参考实现；正式版由引擎①AnyFIM / 引擎②TensorFIM 替换
      （ENGINE_HOOK 注释处），见 README.md 引擎口径。
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd

# SKAB 传感器列（8 路物理传感器；流量列原始名含空格）
SENSOR_COLS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS",
]


# ---------------------------------------------------------------- 数据加载
def load_skab_file(path):
    df = pd.read_csv(path, sep=";")
    return df


# ---------------------------------------------------------------- 离散化
def fit_bins(df, cols, n_bins=3):
    """在整段文件上估计每路传感器的分位数分箱边界。

    原型简化：用全文件分位数；正式版需在滑动/在线设定下估计
    （见开题文档贡献点2——自适应离散化）。
    """
    edges = {}
    for c in cols:
        qs = np.linspace(0, 1, n_bins + 1)
        e = np.quantile(df[c].values, qs)
        e = np.unique(e)  # 去重，防常量列
        if len(e) < 2:      # 常量列：造一个平凡区间，全部落 bin 0
            e = np.array([e[0] - 0.5, e[0] + 0.5])
        edges[c] = e
    return edges


def bin_index(value, edges):
    """value 落入的分箱编号（0..k-1），最右闭区间。"""
    idx = np.searchsorted(edges, value, side="right") - 1
    return int(min(max(idx, 0), len(edges) - 2))


# ---------------------------------------------------------------- 事务化
def window_transactions(df, cols, edges, window=30, stride=15):
    """滑动窗口事务化：窗口内每路传感器取中位数 → 全局分箱 → 一个状态项，
    窗口内 8 个状态项构成一个 transaction。

    语义：transaction = 该窗口设备的"典型工况"（每路传感器一个代表状态），
    模式 = 跨传感器共现的工况组合。
    """
    tids, txn_items, txn_anom = [], [], []
    n = len(df)
    for s in range(0, n - window + 1, stride):
        e = s + window
        block = df.iloc[s:e]
        items = []
        for c in cols:
            med = block[c].median()
            items.append(f"{c}={bin_index(med, edges[c])}")
        tids.append((s, e))
        txn_items.append(items)
        txn_anom.append(int(block["__anom__"].max()))
    return tids, txn_items, np.array(txn_anom)


# ---------------------------------------------------------------- MFI 挖掘（参考实现）
def mine_mfi(transactions, min_sup=0.05, min_count=None):
    """垂直位图 DFS 挖掘极大频繁项集（精确、输出 MFI）。

    ENGINE_HOOK: 正式版替换为引擎①AnyFIM（免阈值随时，输出按支持度降序
    的模式流），此处 min_sup 仅用于原型闭环。

    min_count: 可选绝对频次阈值（优先于 min_sup），用于与引擎的
    "count >= s*n" 实数语义逐点比对（避免 ceil/round 边界差一）。
    2026-10-07 修复：DFS 分支序保证只能找到"按字典序首项分支"的极大集，
    会多发射被其它分支更大 MFI 包含的非极大项集——返回前做包含过滤，
    只保留不被任何其它结果真包含的项集。
    """
    from collections import defaultdict

    n = len(transactions)
    thresh = max(1, int(min_count)) if min_count is not None \
        else max(1, int(np.ceil(min_sup * n)))
    # 项 -> tidset
    tidsets = defaultdict(set)
    for i, t in enumerate(transactions):
        for item in t:
            tidsets[item].add(i)

    items = sorted([it for it, ts in tidsets.items() if len(ts) >= thresh])
    mfis = []
    state = {"nodes": 0, "cap": 200000}

    def dfs(prefix, cand_items, cur_ts):
        """prefix 的 tidset 交集为 cur_ts；无合法扩展时 prefix 即极大。"""
        state["nodes"] += 1
        if state["nodes"] > state["cap"]:
            raise RuntimeError(
                f"MFI 搜索树超过安全上限 {state['cap']}（数据过密，请提高 min_sup）")
        ext = [(c, cur_ts & tidsets[c]) for c in cand_items]
        ext = [(c, ts) for c, ts in ext if len(ts) >= thresh]
        if not ext:
            if prefix:
                mfis.append((tuple(prefix), len(cur_ts)))
            return
        for i, (c, ts) in enumerate(ext):
            dfs(prefix + [c], [x for x, _ in ext[i + 1:]], ts)

    dfs([], items, set(range(n)))

    # 包含过滤：按长度降序，只保留不被已保留（更长）结果真包含的项集
    kept = []
    for pat, cnt in sorted(mfis, key=lambda pc: -len(pc[0])):
        s = set(pat)
        if any(s < set(k) for k, _ in kept):
            continue
        kept.append((pat, cnt))
    return kept, n


# ---------------------------------------------------------------- 模式-故障对照
def pattern_fault_analysis(mfis, transactions, txn_anom):
    """每个 MFI 在正常窗/异常窗中的富集度（lift），按异常窗支持排序输出。"""
    rows = []
    anom_idx = set(np.where(txn_anom == 1)[0])
    norm_idx = set(np.where(txn_anom == 0)[0])
    n_anom = max(1, len(anom_idx))
    n_norm = max(1, len(norm_idx))
    base_rate = len(anom_idx) / (len(anom_idx) + len(norm_idx) + 1e-9)

    for pattern, _ in mfis:
        pset = set(pattern)
        hit_anom = sum(1 for i in anom_idx if pset <= set(transactions[i]))
        hit_norm = sum(1 for i in norm_idx if pset <= set(transactions[i]))
        sup_anom = hit_anom / n_anom
        sup_norm = hit_norm / n_norm
        lift = (sup_anom + 1e-9) / (sup_norm + 1e-9)
        rows.append({
            "pattern": " ∧ ".join(pattern),
            "len": len(pattern),
            "sup_anom": round(sup_anom, 3),
            "sup_norm": round(sup_norm, 3),
            "lift": round(lift, 2),
        })
    df = pd.DataFrame(rows).sort_values(
        ["lift", "sup_anom"], ascending=False).reset_index(drop=True)
    df["pct_of_anom_covered_rank"] = df.index + 1
    return df, base_rate


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/SKAB_raw")
    ap.add_argument("--out", default="../results/spike_v1")
    ap.add_argument("--min-sup", type=float, default=0.05)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--bins", type=int, default=3)
    ap.add_argument("--group", default="valve1", choices=["valve1", "valve2", "other"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data_dir = os.path.join(args.data, "data", args.group)
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))

    all_patterns = []
    summary = []
    for f in files:
        df = load_skab_file(os.path.join(data_dir, f))
        anom = df["anomaly"].values
        df = df.drop(columns=[c for c in ["datetime", "anomaly", "changepoint"]
                              if c in df.columns])
        df["__anom__"] = anom
        edges = fit_bins(df, SENSOR_COLS, n_bins=args.bins)
        tids, txns, txn_anom = window_transactions(df, SENSOR_COLS, edges,
                                                  args.window, args.stride)
        mfis, n_txn = mine_mfi(txns, min_sup=args.min_sup)
        pat_df, base = pattern_fault_analysis(mfis, txns, txn_anom)
        pat_df["file"] = f
        all_patterns.append(pat_df)
        summary.append({
            "file": f, "rows": len(df), "windows": n_txn,
            "anom_windows": int(txn_anom.sum()),
            "n_mfi": len(mfis), "anom_base_rate": round(base, 3),
            "top_lift": pat_df["lift"].iloc[0] if len(pat_df) else None,
            "top_pattern": pat_df["pattern"].iloc[0] if len(pat_df) else None,
        })
        print(f"[{f}] windows={n_txn} anom={int(txn_anom.sum())} "
              f"MFI={len(mfis)} top_lift={summary[-1]['top_lift']}")

    patterns = pd.concat(all_patterns, ignore_index=True)
    patterns.to_csv(os.path.join(args.out, f"patterns_{args.group}.csv"),
                    index=False, encoding="utf-8-sig")
    with open(os.path.join(args.out, f"summary_{args.group}.json"), "w",
              encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {args.out}/patterns_{args.group}.csv")


if __name__ == "__main__":
    main()
