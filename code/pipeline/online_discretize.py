# -*- coding: utf-8 -*-
"""
online_discretize.py —— 在线（流式）分位数离散化：贡献点 2 的原型

方法：每路传感器用 P² 算法（Jain & Chlamtac, 1985）在线估计
{q1/4, q1/2, q3/4} 三个分位数 → 动态分箱边界；新样本到达即入箱。
与离线全文件分位数的分歧率随流推进应单调下降 —— 支撑"模式库随数据流自生长"叙事。

验证：SKAB valve1 全部文件，逐样本在线更新；每个窗口用"窗口中位数 +
当时边界"在线定项，与离线定项比对，统计逐段一致率。
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd


class P2Quantile:
    """单分位数的 P² 在线估计器（经典五标记点法）。"""

    def __init__(self, q):
        self.q = q
        self.init = []

    def _setup(self, obs):
        obs = sorted(obs)
        self.n = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.qv = obs[:5]
        self.n[1] = 1 + 2 * self.q
        self.n[2] = 1 + 4 * self.q
        self.n[3] = 3 + 2 * self.q

    def add(self, x):
        if len(self.init) < 5:
            self.init.append(x)
            if len(self.init) == 5:
                self._setup(self.init)
            return
        if x < self.qv[0]:
            self.qv[0] = x
            k = 0
        elif x < self.qv[1]:
            k = 0
        elif x < self.qv[2]:
            k = 1
        elif x < self.qv[3]:
            k = 2
        elif x <= self.qv[4]:
            k = 3
        else:
            self.qv[4] = x
            k = 3
        self.n = [n + 1 for n in self.n]
        for i in range(1, 4):
            self.n[i] += (1 if i <= k else 0)
        for i in (1, 2, 3):  # 经典 P² 抛物线调整
            d = self.n[i] - self.n[i - 1]
            span = self.n[i + 1] - self.n[i - 1]
            if d >= 1 and span > 1:
                qp = (d + 1) * (self.qv[i + 1] - self.qv[i - 1]) / span
                if self.qv[i - 1] < self.qv[i] + qp < self.qv[i + 1]:
                    self.qv[i] += qp

    def value(self):
        if len(self.init) < 5:
            return float(np.median(self.init)) if self.init else 0.0
        return float(self.qv[2])


class OnlineBinner:
    """一路传感器：两个 P² 估计器（1/3、2/3 分位）→ 动态边界，3 分箱。"""

    def __init__(self):
        self.est = [P2Quantile(q) for q in (1 / 3, 2 / 3)]

    def add(self, x):
        for e in self.est:
            e.add(x)

    def edges(self, min_gap=None):
        q1, q2 = sorted(e.value() for e in self.est)
        iqr = max(q2 - q1, 1e-9)
        if min_gap is None:
            min_gap = 0.05 * iqr          # 最小箱宽：防退化分布下边界重合
        if q2 - q1 < min_gap:
            mid = (q1 + q2) / 2
            q1, q2 = mid - min_gap / 2, mid + min_gap / 2
        return np.array([q1 - iqr, q1, q2, q2 + iqr])  # 与离线 3 分箱同结构


def bin_index(value, edges):
    idx = int(np.searchsorted(edges, value, side="right") - 1)
    return min(max(idx, 0), len(edges) - 2)


def run_file(df, cols, window, stride):
    """单文件：逐样本在线更新边界；每个窗口用窗口中位数+当时边界定项。"""
    binners = {c: OnlineBinner() for c in cols}
    n = len(df)
    med_cache = {}
    results = []  # (win_idx, agree, n_items)
    win_idx = 0
    for s in range(0, n - window + 1, stride):
        e = s + window
        for t in range(s, e):  # 窗口内样本先更新估计器，再定窗口项
            for c in cols:
                binners[c].add(float(df[c].iloc[t]))
        online_items = {}
        for c in cols:
            med = float(df[c].iloc[s:e].median())
            online_items[c] = bin_index(med, binners[c].edges())
        results.append((win_idx, s, online_items))
        win_idx += 1
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data/SKAB_raw")
    ap.add_argument("--out", default="../../results/online_v1")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--group", default="valve1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    spec = importlib.util.spec_from_file_location(
        "skab_spike", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "skab_spike.py"))
    spike = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spike)

    data_dir = os.path.join(args.data, "data", args.group)
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))

    # 每个窗口在全部文件上的 在线vs离线 一致率，按窗口序号分段统计
    seg_agree = {k: [] for k in range(5)}  # 5 段（0-20%, ..., 80-100% 窗口序号）
    for f in files:
        df = spike.load_skab_file(os.path.join(data_dir, f))
        df = df.drop(columns=[c for c in ["datetime", "anomaly", "changepoint"]
                              if c in df.columns])
        online = run_file(df, spike.SENSOR_COLS, args.window, args.stride)
        edges_off = spike.fit_bins(df, spike.SENSOR_COLS, 3)
        n_win = len(online)
        for win_idx, s, online_items in online:
            seg = min(4, win_idx * 5 // max(1, n_win))
            agree = 0
            for c in spike.SENSOR_COLS:
                med = float(df[c].iloc[s:s + args.window].median())
                off = spike.bin_index(med, edges_off[c])
                if off == online_items[c]:
                    agree += 1
            seg_agree[seg].append(agree / len(spike.SENSOR_COLS))

    report = {f"{i*20}-{(i+1)*20}%窗口": {
        "files": len(files),
        "windows": len(v),
        "item_agreement": round(float(np.mean(v)), 4) if v else None}
        for i, v in seg_agree.items()}
    out = os.path.join(args.out, f"online_vs_offline_{args.group}.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
