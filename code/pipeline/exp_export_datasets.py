# -*- coding: utf-8 -*-
"""exp_export_datasets.py —— 把五个工业数据集导出为 FIMI 整数事务格式，
供 AnyFIM / TensorFIM 引擎直接消费。

产出（data/fimi/）：
  skab_valve1.txt / skab_valve2.txt / skab_other.txt  + .items.json 项映射
  cmapss.txt  ai4i.txt  hai.txt  cwru.txt              + .items.json
同时导出 labels（.labels.json）供覆盖率/lift 复算。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import skab_spike as spike                       # noqa: E402
import multi_dataset_spike as mds                # noqa: E402
from engine_adapter import write_fimi            # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "fimi")
os.makedirs(OUT, exist_ok=True)


def export(name, txns, labels=None):
    fimi = os.path.join(OUT, f"{name}.txt")
    mapping = write_fimi(txns, fimi, fimi + ".items.json")
    if labels is not None:
        with open(fimi + ".labels.json", "w") as f:
            json.dump([str(x) for x in labels], f)
    print(f"{name}: {len(txns)} txns, {len(mapping)} items -> {fimi}")


def skab_group(prefix):
    """SKAB 按实验组导出（valve1 / valve2 / other）。
    目录结构: data/SKAB_raw/data/<group>/<n>.csv"""
    folder = os.path.join(DATA, "SKAB_raw", "data", prefix)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".csv"))
    txns, labels = [], []
    for fn in files:
        df = spike.load_skab_file(os.path.join(folder, fn))
        anom = df["anomaly"].values
        df = df.drop(columns=[c for c in ["datetime", "anomaly", "changepoint"]
                              if c in df.columns])
        df["__anom__"] = anom
        edges = spike.fit_bins(df, spike.SENSOR_COLS, 3)
        _, t, ta = spike.window_transactions(df, spike.SENSOR_COLS, edges, 30, 15)
        txns.extend(t)
        labels.extend(ta.tolist())
    return txns, np.array(labels)


def main():
    for grp in ("valve1", "valve2", "other"):
        t, l = skab_group(grp)
        export("skab_" + grp, t, l)

    t, l, _ = mds.load_cmapss(os.path.join(DATA, "cmapss", "CMAPSSData", "train_FD001.txt"))
    export("cmapss", t, l)

    t, l = mds.load_ai4i(os.path.join(DATA, "ai4i", "ai4i2020.csv"))
    export("ai4i", t, l)

    t, l = mds.load_hai(os.path.join(DATA, "hai", "test1.csv.gz"))
    export("hai", t, l)

    t, l = mds.load_cwru(os.path.join(DATA, "cwru"))
    export("cwru", t, l)


if __name__ == "__main__":
    main()
