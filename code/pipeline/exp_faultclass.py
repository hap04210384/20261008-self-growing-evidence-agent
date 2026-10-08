# -*- coding: utf-8 -*-
"""exp_faultclass.py —— E8：逐故障类的模式-物理对照（4.1 节细化实验）。

设计（与 3.8 节评测协议一致）：
1. SKAB 官方 README 给出每个 run 文件的注入物理（valve1=入口阀门关闭、
   valve2=出口阀门关闭、other/1-4=泄漏与添加、other/5-9=转子不平衡、
   other/10-11=水量渐增/突增、other/12=抽干气蚀、other/13=两相流气蚀、
   other/14=高温供水），据此把 run 分成 8 个故障类；
2. 每个类沿用既有管线（每文件 fit_bins + window=30/stride=15 事务化），
   导出的 fimi 用正式引擎 AnyFIM 挖掘（与全文一致，非参考实现）；
3. 判别性排名协议与 3.8 节相同：候选 = 异常窗支持 >= ANOM_SUP_THRESH
   的累积 MFI 池模式；排名键 = anomalous-window lift
   = (p/n_pos) / (c/n)，其中 p=该类异常窗命中数，c=该类全部窗命中数，
   正常窗 = 同类 run 中 anomaly=0 的窗口；
4. 每类取 top-2，报告：解码模式、异常窗覆盖率 p/n_pos、正常窗出现率
   (c-p)/(n-n_pos)、lift。

产出：results/faultclass/summary.json
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import skab_spike as spike                    # noqa: E402
from engine_adapter import run_anyfim, write_fimi  # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "faultclass")
os.makedirs(OUT, exist_ok=True)

ANOM_SUP_THRESH = 0.10   # 异常窗支持度下限（3.8 节协议）
TOP_K = 2                # 每类报 top-2
TOP_UNION = 5            # 协议口径的 top-5 并集覆盖率

# 故障类 -> [(子目录, 文件)]（文件 None = 整个目录）
CLASSES = [
    ("Valve closure (inlet)", [("valve1", None)]),
    ("Valve closure (outlet)", [("valve2", None)]),
    ("Fluid leak/addition", [("other", {"1.csv", "2.csv", "3.csv", "4.csv"})]),
    ("Rotor imbalance", [("other", {"5.csv", "6.csv", "7.csv", "8.csv",
                                     "9.csv"})]),
    ("Water increase", [("other", {"10.csv", "11.csv"})]),
    ("Cavitation (draining)", [("other", {"12.csv"})]),
    ("Two-phase flow", [("other", {"13.csv"})]),
    ("Hot water supply", [("other", {"14.csv"})]),
]


def load_class_txns(spec):
    """与 exp_e4_anytime.load_group_txns 相同的文件级事务流。"""
    txns, labels = [], []
    for sub, files in spec:
        folder = os.path.join(DATA, "SKAB_raw", "data", sub)
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".csv"):
                continue
            if files is not None and fn not in files:
                continue
            df = spike.load_skab_file(os.path.join(folder, fn))
            anom = df["anomaly"].values
            df = df.drop(columns=[c for c in ["datetime", "anomaly",
                                              "changepoint"]
                                  if c in df.columns])
            df["__anom__"] = anom
            edges = spike.fit_bins(df, spike.SENSOR_COLS, 3)
            _, t, ta = spike.window_transactions(df, spike.SENSOR_COLS,
                                                 edges, 30, 15)
            txns.extend(t)
            labels.extend(int(x) for x in ta)
    return txns, labels


def main():
    summary = {}
    for cname, spec in CLASSES:
        txns, labels = load_class_txns(spec)
        n, npos = len(txns), sum(labels)
        str_txns = [set(t) for t in txns]

        # 引擎挖掘（正式 AnyFIM，与全文一致）
        fimi = os.path.abspath(os.path.join(OUT, "fimi.tmp.txt"))
        items = write_fimi(txns, fimi, fimi + ".items.json")
        inv = {v: k for k, v in items.items()}
        r = run_anyfim(fimi, upto_stage=len(items), timeout=1800)
        assert r["ok"], f"AnyFIM failed on {cname}"
        pool = set()
        for k in sorted(r["stages"]):
            pool |= {frozenset(inv[i] for i in x) for x, _ in r["stages"][k]}
        os.remove(fimi)

        # 判别性评分（3.8 节协议）
        scored = []
        p_min = max(2, int(ANOM_SUP_THRESH * npos + 0.999))
        for pat in pool:
            c = sum(1 for t in str_txns if pat <= t)
            p = sum(1 for t, l in zip(str_txns, labels) if l and pat <= t)
            if p < p_min:
                continue
            lift = (p / npos) / (c / n)
            scored.append({"pattern": " & ".join(sorted(pat)),
                           "cnt": c, "pos": p,
                           "cov": round(p / npos, 4),
                           "norm_occ": round((c - p) / (n - npos), 4)
                           if n > npos else 0.0,
                           "lift": round(lift, 2)})
        scored.sort(key=lambda d: (-d["lift"], -d["pos"]))
        # top-5 判别模式的并集覆盖率（3.8 节协议的 fault-pattern coverage）
        top_sets = [frozenset(p.split(" & ")) for p in
                    (d["pattern"] for d in scored[:TOP_UNION])]
        union_cov = (sum(1 for t, l in zip(str_txns, labels)
                         if l and any(s <= t for s in top_sets)) / npos
                     if top_sets else 0.0)
        summary[cname] = {
            "n_windows": n, "n_anomaly": npos,
            "pool_mfis": len(pool), "n_scored": len(scored),
            "top": scored[:TOP_K],
            "top5_union_cov": round(union_cov, 4),
        }
        print(f"{cname}: windows={n} anom={npos} pool={len(pool)} "
              f"top1={scored[0]['pattern'] if scored else None} "
              f"cov={scored[0]['cov'] if scored else None} "
              f"lift={scored[0]['lift'] if scored else None} "
              f"union5={union_cov:.4f}")

    with open(os.path.join(OUT, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("saved", os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
