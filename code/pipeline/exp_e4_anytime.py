# -*- coding: utf-8 -*-
"""exp_e4_anytime.py —— E4 anytime 质量曲线（真实引擎 AnyFIM 正式版）。

设计（2026-10-07 定稿，含三轮语义修正的教训）：
1. AnyFIM 每轮输出的是该轮阈值的【完整 MFI 池】（非增量）；
2. 第 k 轮阈值 = 支持度第 k 高项的 count/n；与精确挖掘的比对必须用
   min_count=round(s_k*n) 的整数语义（AnyFIM 为 >=，TensorFIM 为 >，
   边界恰差激活项单例——已定位，见 e4_identity_*.json 记录）；
3. 真值 = 实用深度（s_k >= PRACT_MIN_SUP）内"仅出现在异常窗"的富集模式
   （pos/cnt >= EXCL_THRESH 且 cnt >= MIN_CNT）。SKAB 异常窗占比约 38%，
   lift 理论上界 = 1/基率 ≈ 2.64，故不能用固定 lift 阈值；
4. coverage(k) = 真值模式在前 k 轮累积发现池中的占比，单调不减至 1.0。

产出（results/anytime_v2/）：
  e4_coverage_<grp>.csv   stage, s_k, pool_mfis, cum_mfis, truth_found, coverage, cum_time_s
  e4_truth_<grp>.json     真值模式（解码字符串、计数、lift）
  e4_identity_<grp>.json  逐轮位精确核对结果（AnyFIM vs 精确挖掘）
"""
import json
import os
import sys
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import skab_spike as spike                    # noqa: E402
from engine_adapter import (decode_itemset, load_items_map,  # noqa: E402
                            parse_anyfim_results, run_anyfim)

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "anytime_v2")
os.makedirs(OUT, exist_ok=True)

PRACT_MIN_SUP = 0.02   # 实用深度：阈值不低于 2%（避开退化通道分箱的垃圾项地板）
EXCL_THRESH = 0.95     # 真值富集：异常窗占比 >= 95%（近似"仅异常窗出现"）
MIN_CNT = 20           # 真值最小支持计数（防微小偶然模式）


def load_group_txns(grp):
    """与 exp_export_datasets.skab_group 相同的事务流（保持可复现）。"""
    folder = os.path.join(DATA, "SKAB_raw", "data", grp)
    txns, labels = [], []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".csv"):
            continue
        df = spike.load_skab_file(os.path.join(folder, fn))
        anom = df["anomaly"].values
        df = df.drop(columns=[c for c in ["datetime", "anomaly", "changepoint"]
                              if c in df.columns])
        df["__anom__"] = anom
        edges = spike.fit_bins(df, spike.SENSOR_COLS, 3)
        _, t, ta = spike.window_transactions(df, spike.SENSOR_COLS, edges, 30, 15)
        txns.extend(t)
        labels.extend(int(x) for x in ta)
    return txns, labels


def main():
    summary = {}
    for grp in ("valve1", "valve2"):
        fimi = os.path.abspath(os.path.join(DATA, "fimi", f"skab_{grp}.txt"))
        items = load_items_map(fimi + ".items.json")
        inv = {v: k for k, v in items.items()}
        n_items = len(items)

        r = run_anyfim(fimi, n_items, timeout=1800)
        assert r["ok"], f"AnyFIM failed on {grp}"
        stages = r["stages"]

        txns, labels = load_group_txns(grp)
        n = len(txns)
        npos = sum(labels)
        str_txns = [set(t) for t in txns]

        # 每轮精确阈值 = 支持度第 k 高项的 count/n
        cnt = Counter()
        for t in txns:
            for x in set(t):
                cnt[x] += 1
        ranked = sorted(cnt.values(), reverse=True)
        s_k = {k: ranked[k - 1] / n for k in sorted(stages)}

        # 实用深度
        K = max(k for k in sorted(stages) if s_k[k] >= PRACT_MIN_SUP)

        # 富集统计（对实用深度内的累积池）
        def stats(itemset):
            strs = set(inv[i] for i in itemset)
            c = sum(1 for t in str_txns if strs <= t)
            p = sum(1 for t, l in zip(str_txns, labels) if l and strs <= t)
            return c, p

        cum, seen = {}, set()
        for k in sorted(stages):
            seen |= {x for x, _ in stages[k]}
            cum[k] = set(seen)
        union_pract = cum[K]

        truth = {}
        for x in union_pract:
            c, p = stats(x)
            if c >= MIN_CNT and p / c >= EXCL_THRESH:
                truth[x] = {"cnt": c, "pos": p,
                            "lift": round((p / npos) / (c / n), 3)}
        truth_str = {" & ".join(sorted(decode_itemset(t, inv))): v
                     for t, v in truth.items()}
        with open(os.path.join(OUT, f"e4_truth_{grp}.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"n_truth": len(truth),
                       "base_rate": round(npos / n, 4),
                       "lift_cap": round(n / npos, 3),
                       "practical_depth": K,
                       "truth": truth_str},
                      f, ensure_ascii=False, indent=1)

        # 位精确核对：每轮池 == 精确挖掘（min_count 整数语义）
        ident_rows = []
        for k in sorted(stages):
            pool = {frozenset(inv[i] for i in x) for x, _ in stages[k]}
            mc = max(1, int(round(s_k[k] * n)))
            mfis, _ = spike.mine_mfi(txns, min_count=mc)
            ref = {frozenset(p) for p, _ in mfis}
            ident_rows.append({"stage": k, "min_count": mc,
                               "anyfim": len(pool), "exact": len(ref),
                               "identical": pool == ref})
        n_id = sum(1 for x in ident_rows if x["identical"])
        with open(os.path.join(OUT, f"e4_identity_{grp}.json"), "w") as f:
            json.dump({"identical_stages": n_id, "total_stages": len(ident_rows),
                       "note": "AnyFIM uses >= count threshold; TensorFIM uses >, "
                               "differing exactly on the activating item's "
                               "singleton (verified separately).",
                       "rows": ident_rows}, f, indent=1)

        # 覆盖曲线（实用深度内）
        runtimes = r["meta"].get("runtimePerStage(s)", [])
        rows, cum_t, cum_seen = [], 0.0, set()
        for k in sorted(stages):
            if k > K:
                break
            cum_seen |= {x for x, _ in stages[k]}
            cum_t += runtimes[k - 1] if k - 1 < len(runtimes) else 0.0
            hit = len(truth.keys() & cum_seen)
            rows.append({"stage": k, "s_k": round(s_k[k], 6),
                         "pool_mfis": len(stages[k]),
                         "cum_mfis": len(cum_seen),
                         "truth_found": hit,
                         "coverage": round(hit / len(truth), 4) if truth else 0.0,
                         "cum_time_s": round(cum_t, 6)})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT, f"e4_coverage_{grp}.csv"),
                  index=False, encoding="utf-8-sig")
        mono = bool((df["coverage"].diff().dropna() >= -1e-12).all())
        first_hit = int(df[df["truth_found"] > 0]["stage"].iloc[0]) \
            if (df["truth_found"] > 0).any() else None
        summary[grp] = {
            "n_stages": len(stages), "practical_depth": K,
            "n_truth": len(truth), "final_coverage": float(df["coverage"].iloc[-1]),
            "monotone": mono, "first_truth_stage": first_hit,
            "total_time_s": float(df["cum_time_s"].iloc[-1]),
            "identical_stages": f"{n_id}/{len(ident_rows)}",
        }
        print(f"{grp}: {summary[grp]}")

    with open(os.path.join(OUT, "e4_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
