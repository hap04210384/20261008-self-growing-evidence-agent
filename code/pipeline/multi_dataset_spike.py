# -*- coding: utf-8 -*-
"""
multi_dataset_spike.py —— 多数据集适配器（spike 阶段）

四个工业数据集统一走同一链路：适配器输出 (传感列DataFrame, 异常标签)
→ fit_bins → window_transactions → mine_mfi → 对照分析。
复用 skab_spike.py 中的核心函数（导入方式见下）。

数据集：
  cmapss  NASA C-MAPSS FD001（航天动力，21 路传感器，退化末段为异常窗）
  ai4i    UCI AI4I 2020（离散制造，逐行事务，无需滑窗）
  cwru    CWRU 轴承（旋转机械，3 振动通道 × 时域特征事务化；按故障类别分组对照）
  hai     HAI 21.03 水电站工控（流程工业，攻击段为异常窗）
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "skab_spike", os.path.join(os.path.dirname(os.path.abspath(__file__)), "skab_spike.py"))
spike = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spike)

fit_bins = spike.fit_bins
window_transactions = spike.window_transactions
mine_mfi = spike.mine_mfi


# ---------------------------------------------------------------- C-MAPSS
def load_cmapss(path, tail_frac=0.15, window=20, stride=10):
    """FD00x 训练集：每台发动机一条 run-to-failure 轨迹。
    异常定义（spike 简化）：位于该台寿命末 tail_frac 段的窗口。"""
    cols = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + \
           [f"s{i}" for i in range(1, 22)]
    df = pd.read_csv(path, sep=" ", header=None).dropna(axis=1, how="all")
    df.columns = cols[: df.shape[1]]
    sens = [c for c in df.columns if c.startswith("s")]
    all_txns, all_anom, units = [], [], []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        cut = g["cycle"].max() * (1 - tail_frac)
        g = g[sens].copy()
        g["__anom__"] = (df.loc[g.index, "cycle"] > cut).astype(int)
        if len(g) < window:
            continue
        _, txns, ta = window_transactions(g, sens,
                                          fit_bins(g, sens, 3), window, stride)
        all_txns.extend(txns)
        all_anom.extend(ta.tolist())
        units.extend([unit] * len(txns))
    return all_txns, np.array(all_anom), units


# ---------------------------------------------------------------- AI4I 2020
def load_ai4i(path):
    """离散制造：每行 = 一个产品 = 一个 transaction（无滑窗）。
    数值特征分位数分箱 + 产品类型直接作项。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    num_cols = ["Air temperature [K]", "Process temperature [K]",
                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    def _e(v):
        e = np.unique(np.quantile(v, np.linspace(0, 1, 4)))
        return e if len(e) >= 2 else np.array([e[0] - 0.5, e[0] + 0.5])
    edges = {c: _e(df[c].values) for c in num_cols}
    txns = []
    for _, row in df.iterrows():
        items = [f"Type={row['Type']}"]
        for c in num_cols:
            items.append(f"{c.split(' [')[0]}={spike.bin_index(row[c], edges[c])}")
        txns.append(items)
    return txns, df["Machine failure"].values.astype(int)


# ---------------------------------------------------------------- CWRU 轴承
CWRU_CLASSES = {  # 文件号 -> (类别, 标签)
    "97": ("normal", 0), "98": ("normal", 0), "99": ("normal", 0),
    "100": ("normal", 0), "105": ("IR007", 1), "118": ("B007", 1),
    "130": ("OR007", 1), "169": ("IR014", 1), "209": ("IR021", 1),
}

def load_cwru(folder, window=4096, stride=2048):
    """轴承振动：每通道滑窗提 3 个时域特征（RMS/峭度/峰峰值），
    特征中位数分箱 → transaction；按故障类别分组算支持度。"""
    import scipy.io as sio
    txns, classes = [], []
    for fid, (cls, _) in sorted(CWRU_CLASSES.items()):
        p = os.path.join(folder, f"{fid}.mat")
        if not os.path.exists(p):
            continue
        try:
            m = sio.loadmat(p)
        except Exception:
            print(f"  [跳过损坏文件 {fid}.mat，稍后修复重跑]")
            continue
        chans = [k for k in m if k.endswith(("_DE_time", "_FE_time", "_BA_time"))]
        if not chans:
            continue
        feats = {}
        for ch in chans:
            x = m[ch].ravel().astype(float)
            for s in range(0, len(x) - window + 1, stride):
                w = x[s:s + window]
                feats.setdefault(f"{ch}_rms", []).append(float(np.sqrt((w ** 2).mean())))
                feats.setdefault(f"{ch}_kurt", []).append(float(pd.Series(w).kurt()))
                feats.setdefault(f"{ch}_pp", []).append(float(w.max() - w.min()))
        if not feats:
            continue
        fdf = pd.DataFrame(feats)
        def _e2(v):
            e = np.unique(np.quantile(v, np.linspace(0, 1, 4)))
            return e if len(e) >= 2 else np.array([e[0] - 0.5, e[0] + 0.5])
        edges = {c: _e2(v) for c, v in fdf.items()}
        for _, row in fdf.iterrows():
            txns.append([f"{c.split('_')[0][:8]}_{c.split('_')[1]}={spike.bin_index(row[c], edges[c])}"
                         for c in fdf.columns])
        classes.extend([cls] * len(fdf))
    return txns, np.array(classes)


# ---------------------------------------------------------------- HAI
def load_hai(path_gz, n_rows=30000, window=120, stride=60):
    """水电站工控 test1：浮点传感器列滑窗中位数分箱；窗内含攻击即异常窗。"""
    df = pd.read_csv(path_gz, compression="gzip", nrows=n_rows)
    drop = {"time"} | {c for c in df.columns if c.startswith("attack")}
    sens = [c for c in df.columns if c not in drop and df[c].dtype != object]
    # 只保留方差较大的连续量测列（去 STATE 开关量），防事务过密
    var = df[sens].var()
    sens = list(var[var > var.quantile(0.4)].index)[:25]
    g = df[sens].copy()
    g["__anom__"] = df["attack"].values.astype(int)
    edges = fit_bins(g, sens, 3)
    _, txns, ta = window_transactions(g, sens, edges, window, stride)
    return txns, ta


# ---------------------------------------------------------------- 对照分析（泛化版）
def support_table(mfis, txns, labels, label_names=None):
    """每个 MFI 在各类别（异常/正常 或 故障类别）中的支持度表。"""
    rows = []
    classes = sorted(set(labels.tolist()))
    idx_by_cls = {c: set(np.where(labels == c)[0]) for c in classes}
    for pattern, cnt in mfis:
        pset = set(pattern)
        row = {"pattern": " ∧ ".join(pattern), "len": len(pattern),
               "count": cnt}
        for c in classes:
            idx = idx_by_cls[c]
            hit = sum(1 for i in idx if pset <= set(txns[i]))
            row[f"sup_{label_names[c] if label_names else c}"] = \
                round(hit / max(1, len(idx)), 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data")
    ap.add_argument("--out", default="../../results/spike_v1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = {}

    # ---- C-MAPSS
    print("=== C-MAPSS FD001 ===")
    txns, anom, units = load_cmapss(os.path.join(args.data, "cmapss/CMAPSSData/train_FD001.txt"))
    mfis, n = mine_mfi(txns, min_sup=0.3)  # 0.2 即搜索爆炸：稠密相关性，阈值敏感（论文动机素材）
    tbl = support_table(mfis, txns, anom)
    tbl = tbl.assign(lift=lambda d: (d["sup_1"] + 1e-3) / (d["sup_0"] + 1e-3))
    tbl = tbl.sort_values("lift", ascending=False)
    tbl.to_csv(os.path.join(args.out, "patterns_cmapss.csv"), index=False, encoding="utf-8-sig")
    report["cmapss"] = {"windows": n, "anom": int(anom.sum()), "mfi": len(mfis),
                        "top": tbl.iloc[0].to_dict() if len(tbl) else None}
    print(f"windows={n} anom={int(anom.sum())} MFI={len(mfis)}")
    if len(tbl):
        print("top:", tbl.iloc[0]["pattern"], "lift=", round(tbl.iloc[0]["lift"], 1))

    # ---- AI4I
    print("\n=== AI4I 2020 ===")
    txns, anom = load_ai4i(os.path.join(args.data, "ai4i/ai4i2020.csv"))
    mfis, n = mine_mfi(txns, min_sup=0.01)
    tbl = support_table(mfis, txns, anom)
    tbl = tbl.assign(lift=lambda d: (d["sup_1"] + 1e-3) / (d["sup_0"] + 1e-3))
    tbl = tbl.sort_values("lift", ascending=False)
    tbl.to_csv(os.path.join(args.out, "patterns_ai4i.csv"), index=False, encoding="utf-8-sig")
    report["ai4i"] = {"rows": n, "fail": int(anom.sum()), "mfi": len(mfis),
                      "top": tbl.iloc[0].to_dict() if len(tbl) else None}
    print(f"rows={n} failure={int(anom.sum())} MFI={len(mfis)}")
    if len(tbl):
        print("top:", tbl.iloc[0]["pattern"], "lift=", round(tbl.iloc[0]["lift"], 1))

    # ---- CWRU
    print("\n=== CWRU 轴承 ===")
    txns, classes = load_cwru(os.path.join(args.data, "cwru"))
    names = {c: c for c in sorted(set(classes.tolist()))}
    mfi_all, n = mine_mfi(txns, min_sup=0.05)
    tbl = support_table(mfi_all, txns, classes, names)
    tbl.to_csv(os.path.join(args.out, "patterns_cwru.csv"), index=False, encoding="utf-8-sig")
    report["cwru"] = {"windows": n, "classes": {c: int((classes == c).sum()) for c in names},
                      "mfi": len(mfi_all)}
    print(f"windows={n} classes={report['cwru']['classes']} MFI={len(mfi_all)}")

    # ---- HAI
    print("\n=== HAI 21.03 test1 ===")
    txns, anom = load_hai(os.path.join(args.data, "hai/test1.csv.gz"))
    mfis, n = mine_mfi(txns, min_sup=0.35)  # HAI 传感器强相关，低阈值即稠密爆炸（同 C-MAPSS 教训）
    tbl = support_table(mfis, txns, anom)
    tbl = tbl.assign(lift=lambda d: (d["sup_1"] + 1e-3) / (d["sup_0"] + 1e-3))
    tbl = tbl.sort_values("lift", ascending=False)
    tbl.to_csv(os.path.join(args.out, "patterns_hai.csv"), index=False, encoding="utf-8-sig")
    report["hai"] = {"windows": n, "attack": int(anom.sum()), "mfi": len(mfis),
                     "top": tbl.iloc[0].to_dict() if len(tbl) else None}
    print(f"windows={n} attack={int(anom.sum())} MFI={len(mfis)}")
    if len(tbl):
        print("top:", tbl.iloc[0]["pattern"], "lift=", round(tbl.iloc[0]["lift"], 1))

    with open(os.path.join(args.out, "multi_dataset_summary.json"), "w",
              encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2, default=str)
    print("\n汇总已写入 multi_dataset_summary.json")


if __name__ == "__main__":
    main()
