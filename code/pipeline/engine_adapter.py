# -*- coding: utf-8 -*-
"""engine_adapter.py —— 论文两大引擎（AnyFIM anytime / TensorFIM）统一适配层。

职责：
1. 事务（["Sensor=state", ...] 字符串列表）<-> FIMI 整数格式互转，项映射存档；
2. 调用 AnytimeMining.exe（AnyFIM anytime，免阈值逐轮激活）并解析分轮输出；
3. 调用 CoParaCG_baseline/bmma.exe（TensorFIM，位精确对照）并解析输出与计时；
4. 公共工具：模式解码（整数项 -> "Sensor=state"）、支持度/置信度/lift 复算。

引擎 exe 路径（本机实测，2026-10-07）：
- ANYFIM_EXE  = code/engine_anyfim/code/AnyFIM-Anytime/AnytimeMining/AnytimeMining.exe
- TENSORFIM_DIR = code/engine_tensorfim/code/engine/run/  (CoParaCG_baseline.exe / CoParaCG_bmma.exe)

注意：AnyFIM 输出文件写在数据集同目录：<dataset>-<stages>-stages=Results.txt；
TensorFIM 输出文件写在数据集同目录：<dataset>-<thr>=Results.txt。
"""
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ANYFIM_EXE = os.path.join(ROOT, "code", "engine_anyfim", "code",
                          "AnyFIM-Anytime", "AnytimeMining", "AnytimeMining.exe")
TENSORFIM_DIR = os.path.join(ROOT, "code", "engine_tensorfim", "code", "engine", "run")

# TensorFIM Results.txt 行格式: "Nth: k { i1 i2 ... } support: s frequency: f"
TF_LINE = re.compile(r"^\d+th:\s+\d+\s+\{\s*([^}]*)\}\s+support:\s+\S+\s+frequency:\s+(\d+)")
TF_COMP = re.compile(r"computation_time\(s\):\s*([\d.eE+-]+)")
TF_TOT = re.compile(r"total_time\(s\):\s*([\d.eE+-]+)")
# AnyFIM 分轮节标题
AF_STAGE = re.compile(r"^=+ (\d+)th stage MFIs Number: (\d+) =+")


# ---------------------------------------------------------------- FIMI 编码
def write_fimi(txns, fimi_path, items_map_path=None):
    """txns: iterable of iterables of 'name=state' 字符串。
    返回 {item_str: int_id}；1-based 整数项。items 映射另存 json（若给路径）。"""
    mapping = {}
    lines = []
    for txn in txns:
        ids = []
        for it in txn:
            if it not in mapping:
                mapping[it] = len(mapping) + 1
            ids.append(mapping[it])
        lines.append(" ".join(map(str, sorted(set(ids)))))
    with open(fimi_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if items_map_path:
        with open(items_map_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=1)
    return mapping


def load_items_map(path):
    return json.load(open(path, encoding="utf-8"))


def decode_itemset(itemset_ids, inv_map):
    def _get(i):
        if i in inv_map:
            return inv_map[i]
        return inv_map[str(i)]
    return frozenset(_get(i) for i in itemset_ids)


# ---------------------------------------------------------------- TensorFIM
def run_tensorfim(engine, fimi_path, rel_thr, reps=1, timeout=1800):
    """engine: 'baseline' | 'bmma'。返回 dict:
    {'ok': bool, 'mfis': {frozenset: freq}, 'comp_times': [..], 'tot_times': [..]}
    输出文件名: <fimi>-<thr>=Results.txt（每轮 rep 前删除重来）。"""
    exe = os.path.join(TENSORFIM_DIR, f"CoParaCG_{engine}.exe")
    rp = f"{fimi_path}-{rel_thr:.6f}=Results.txt"
    comp, tot = [], []
    ok = True
    mfis = {}
    for _ in range(reps):
        if os.path.exists(rp):
            os.remove(rp)
        try:
            p = subprocess.run([exe, fimi_path, str(rel_thr)],
                               capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            ok = False
            continue
        log = p.stdout.decode("utf-8", errors="replace") + \
            p.stderr.decode("utf-8", errors="replace")
        m = TF_COMP.findall(log)
        if m:
            comp.append(float(m[-1]))
        m = TF_TOT.findall(log)
        if m:
            tot.append(float(m[-1]))
        if os.path.exists(rp):
            mfis = parse_tensorfim_results(rp)
        else:
            ok = False
    return {"ok": ok, "mfis": mfis, "comp_times": comp, "tot_times": tot}


def parse_tensorfim_results(path):
    """返回 {frozenset(int_item_ids): frequency}"""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = TF_LINE.match(ln.strip())
            if m:
                out[frozenset(int(x) for x in m.group(1).split())] = int(m.group(2))
    return out


# ---------------------------------------------------------------- AnyFIM
def run_anyfim(fimi_path, upto_stage, timeout=3600):
    """AnyFIM anytime 全量运行。返回 {'ok', 'stages': {k: [(frozenset,freq),...]},
    'meta': {...}, 'results_path': str}。输出: <fimi>-<stages>-stages=Results.txt"""
    rp = f"{fimi_path}-{upto_stage}-stages=Results.txt"
    if os.path.exists(rp):
        os.remove(rp)
    try:
        p = subprocess.run([ANYFIM_EXE, fimi_path, str(upto_stage)],
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "stages": {}, "meta": {}, "results_path": rp}
    if not os.path.exists(rp):
        return {"ok": False, "stages": {}, "meta": {},
                "log": p.stdout.decode("utf-8", errors="replace"),
                "results_path": rp}
    stages, meta = parse_anyfim_results(rp)
    return {"ok": True, "stages": stages, "meta": meta, "results_path": rp}


def parse_anyfim_results(path):
    """解析 AnyFIM Results.txt。
    返回 (stages, meta): stages[k] = [(frozenset(item_ids), frequency), ...]；
    meta 含 runtimePerStage / supportThrePerStage 等。"""
    stages = {}
    meta = {}
    cur = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            m = AF_STAGE.match(ln)
            if m:
                cur = int(m.group(1))
                stages[cur] = []
                continue
            for key, cast in (("runtimePerStage(s):", float),
                              ("supportThrePerStage:", float),
                              ("frequencyThrePerStage:", float),
                              ("MFIsNumPerStage:", int),
                              ("itemsNumPerStage:", int),
                              ("dimensionPerStage:", int),
                              ("preprocessing_time(s):", float)):
                if ln.startswith(key):
                    body = ln[len(key):].strip()
                    meta[key.rstrip(":")] = [cast(x.rstrip(","))
                                             for x in body.split() if x.strip(",")]
            if cur is not None:
                m = TF_LINE.match(ln)
                if m:
                    stages[cur].append(
                        (frozenset(int(x) for x in m.group(1).split()),
                         int(m.group(2))))
    return stages, meta


def cumulative_mfis(stages):
    """按轮合并模式（去重）：返回 {k: set(前 k 轮出现过的全部 itemset)}。
    单调性检查就是比较相邻轮的包含关系。"""
    cum, seen = {}, set()
    for k in sorted(stages):
        for itemset, _ in stages[k]:
            seen.add(itemset)
        cum[k] = set(seen)
    return cum


# ---------------------------------------------------------------- 统计工具
def support_confidence_lift(itemset, txns, labels):
    """txns: 整数项事务列表（与 itemset 同编码）；labels: 0/1。
    返回 (sup, conf(lift 分子侧), lift)。lift = P(X|1)/P(X)。"""
    n = len(txns)
    cnt = sum(1 for t in txns if itemset <= set(t))
    pos = sum(1 for t, l in zip(txns, labels) if l == 1 and itemset <= set(t))
    npos = sum(1 for l in labels if l == 1)
    sup = cnt / n if n else 0.0
    conf = pos / npos if npos else 0.0
    lift = (pos / npos) / (cnt / n) if cnt and npos else 0.0
    return sup, conf, lift
