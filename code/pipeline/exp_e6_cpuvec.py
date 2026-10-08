# -*- coding: utf-8 -*-
"""exp_e6_cpuvec.py —— E6 CPU 向量化基线：NumPy 位打包 AND + 查表 popcount。

与 E5 完全同一批配置（数据集 × 相对支持度阈值），同一计时协议
（REPS 次取中位数），输出 maximal frequent itemsets，并与 E5 中
baseline 引擎的 MFI 数逐位核对（位精确一致才可信）。

内核：每路项一条垂直位图（np.packbits）；每个搜索节点一次性与
全部后续候选项位图做 uint8 位块 AND，再经 256 项 LUT 查表 popcount
得到所有候选支持度 —— 即"CPU 向量化"（SIMD 友好）实现。
项目按支持度降序进入 DFS，已发现的 MFI 用于超集剪枝。

产物：results/e6_cpuvec.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

DATA = os.path.join(ROOT, "data", "fimi")
OUT = os.path.join(ROOT, "results")
CSV = os.path.join(OUT, "e6_cpuvec.csv")

REPS = 5
# 重配置单次运行远超普通配置（Python 节点开销），减少重复次数；
# hai@0.1 附带计时预算，超时记为 ">budget"
REPS_OVERRIDES = {("hai", 0.10): 1}
BUDGET_OVERRIDES = {("hai", 0.10): 240.0}
CONFIGS = {
    "skab_valve1": [0.05, 0.10, 0.20],
    "cmapss":      [0.30, 0.40],
    "ai4i":        [0.01, 0.05],
    "hai":         [0.10, 0.20],
    "cwru":        [0.10, 0.20],
}

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def load_fimi_bitmap(path):
    """返回 (bm, n_txns)：bm[i] 为第 i 个（0-based）项的打包位图。"""
    txn_rows = [line.split() for line in open(path, encoding="utf-8")
                if line.strip()]
    n_txns = len(txn_rows)
    n_items = max((int(x) for row in txn_rows for x in row), default=0)
    cols = np.zeros((n_items, n_txns), dtype=np.uint8)
    for ti, row in enumerate(txn_rows):
        for x in row:
            cols[int(x) - 1, ti] = 1
    return np.packbits(cols, axis=1), n_txns


def mine_mfi_cpuvec(bm, n_txns, rel_thr, budget_s=None):
    """返回 (mfis, comp_s, timed_out)。mfis: list[frozenset[int]]（0-based 项）。
    budget_s: 计时预算，超时抛出 _Timeout（由调用方记录为超时）。"""
    t0 = time.perf_counter()
    state = {"nodes": 0}
    abs_sup = rel_thr * n_txns
    sup1 = POP[bm].sum(axis=1)
    order = np.where(sup1 >= abs_sup)[0]
    order = order[np.argsort(-sup1[order], kind="stable")]
    pos = {it: k for k, it in enumerate(order)}
    sub = bm[order]                      # 按支持度降序重排
    mfi = []

    def is_covered(itemset):
        for m in mfi:
            if itemset <= m:
                return True
        return False

    suffix = [set(int(x) for x in order[k:]) for k in range(len(order) + 1)]

    def dfs(node_bm, node_set, start):
        state["nodes"] += 1
        if budget_s is not None and state["nodes"] % 8192 == 0:
            if time.perf_counter() - t0 > budget_s:
                raise TimeoutError
        # 一次性向量化：当前节点与全部候选项 AND + LUT popcount
        counts = POP[sub & node_bm].sum(axis=1)
        freq = [int(p) for p in np.where(counts >= abs_sup)[0]
                if int(order[p]) not in node_set]
        if not freq:
            # 无频繁超集 → 真极大（对全局频繁项验证过，不止前向）
            if node_set and not is_covered(node_set):
                mfi.append(node_set)
            return
        covering = next((mm for mm in mfi if node_set <= mm), None)
        if covering is not None and (node_set | suffix[start]) <= covering:
            return    # 本分支所有可达项均已被 M 覆盖，整体剪枝（完备）
        for j in freq:
            if j < start:
                continue      # 前序项的超集由更靠前根的发现路径负责
            dfs(sub[j] & node_bm, node_set | {int(order[j])}, j + 1)
        # freq 非空时 node_set 本身非极大，不记录；
        # 若全部前向扩展被剪，说明其超集已被已发现 MFI 覆盖，亦无需记录。

    for k in range(len(order)):
        dfs(sub[k], {int(order[k])}, k + 1)
    comp = time.perf_counter() - t0
    return mfi, comp, False


def med(xs):
    xs = sorted(xs)
    m = len(xs)
    return xs[m // 2] if m % 2 else (xs[m // 2 - 1] + xs[m // 2]) / 2


def main():
    os.makedirs(OUT, exist_ok=True)
    done = set()
    rows = []
    if os.path.exists(CSV):
        prev = pd.read_csv(CSV)
        done = set(zip(prev["dataset"], prev["thr"]))
        rows = prev.to_dict("records")

    e5 = pd.read_csv(os.path.join(OUT, "e5_efficiency.csv"))
    e5_base = {(r["dataset"], round(float(r["thr"]), 6)): int(r["n_mfis"])
               for _, r in e5[e5["engine"] == "baseline"].iterrows()}

    plan = [(ds, thr) for ds, thrs in CONFIGS.items() for thr in thrs]
    # 重配置放最后，先完成轻配置
    plan.sort(key=lambda t: BUDGET_OVERRIDES.get((t[0], round(t[1], 2)), 0.0))
    for ds, thr in plan:
        fimi = os.path.abspath(os.path.join(DATA, f"{ds}.txt"))
        bm, n_txns = load_fimi_bitmap(fimi)
        key = (ds, round(thr, 6))
        n_reps = REPS_OVERRIDES.get((ds, round(thr, 2)), REPS)
        budget = BUDGET_OVERRIDES.get((ds, round(thr, 2)))
        expect = e5_base.get(key)
        times = [r for r in rows
                 if (r["dataset"], round(float(r["thr"]), 6)) == key]
        for rep in range(len(times), n_reps):
            try:
                mfis, comp, _to = mine_mfi_cpuvec(bm, n_txns, thr,
                                                  budget_s=budget)
                ok = (expect == len(mfis))
                rows.append({"dataset": ds, "thr": thr, "rep": rep,
                             "ok": ok, "n_mfis": len(mfis),
                             "expect_baseline_mfis": expect,
                             "comp_s": round(comp, 6)})
                print(f"{ds}@{thr} rep{rep}: mfis={len(mfis)} "
                      f"(baseline {expect}, match={ok}) comp={comp:.3f}s",
                      flush=True)
            except TimeoutError:
                rows.append({"dataset": ds, "thr": thr, "rep": rep,
                             "ok": "timeout", "n_mfis": None,
                             "expect_baseline_mfis": expect,
                             "comp_s": None})
                print(f"{ds}@{thr} rep{rep}: TIMEOUT >{budget}s", flush=True)
            pd.DataFrame(rows).to_csv(CSV, index=False,
                                      encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    agg = df.groupby(["dataset", "thr"]).agg(
        med_comp_s=("comp_s", "median"), reps_ok=("comp_s", "count"),
        ok=("ok", "min")).reset_index()
    agg.to_csv(os.path.join(OUT, "e6_cpuvec_summary.csv"), index=False,
               encoding="utf-8-sig")
    print("\n", agg.to_string())


if __name__ == "__main__":
    main()
