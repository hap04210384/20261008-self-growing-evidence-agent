# -*- coding: utf-8 -*-
"""exp_llm_hai.py —— 审稿 Major 1(3)：HAI 简化 LLM 诊断协议（SKAB 之外的第二个数据集）。

HAI 是五个基准中除 SKAB 外唯一带窗口级可判定标签的数据集（攻击区间标签，
安全检测语义，非物理故障语义——论文中如实说明）。攻击窗仅 19 个
（data/fimi/hai.txt，w=120/stride=60，与 4.3/4.4 节同一符号化），
故为二元协议（normal / attack≡faulty），全部 19 个攻击窗入测。

两个手臂（每臂三条件：nolib / mined / random，随机证据采样与
exp_llm_baseline.py 同规格：种子化、8 槽、大小 {2,3,4} 均匀、无放回、真实计数）：
  A) in-sample：库在全部 499 窗上挖掘+评分（同 3.6 节参照协议）；
     测试窗 = 19 攻击 + 19 正常（Random(42)）。
  B) time-split（泄露控制）：库仅用前 60% 时间窗（SPLIT=299，±10 保护带防
     窗口重叠泄漏），评分只在库窗上；测试窗 = 后段全部 8 个攻击窗 +
     8 个正常窗（Random("hai|ts")）。支持/提升度对测试标签不可见。
     【2026-10-09 实测搁置】289 窗子集的全阶段 AnyFIM 挖掘 >1h 未完成
     （阶段组合爆炸），论文以 in-sample 手臂报告，split 变体列入未来工作；
     保留代码路径，ARM=insample 时跳过。

模型：3 个主模型 + GLM-4-Flash / Qwen-Turbo（同 exp_llm_small_models）。
调用：(38+16) 窗 × 3 条件 × 5 模型 = 810 次，temperature=0。
产物：results/hai_diag/{calls.jsonl, libraries.json}
"""
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PIPE = os.path.join(ROOT, "code", "pipeline")
sys.path.insert(0, HERE)
sys.path.insert(0, PIPE)

import exp_llm_diagnosis as diag          # noqa: E402  (MODELS/SYS/fmt_lift)
from engine_adapter import (ANYFIM_EXE, parse_anyfim_results,   # noqa: E402
                            write_fimi)

DATA = os.environ.get(
    "FIMI_DIR",
    os.path.join(ROOT, "data", "fimi"))   # 仓库不附带 .txt 事务文件时可指向外部导出目录
OUTDIR = os.path.join(ROOT, "results", "hai_diag")
CALLS = os.path.join(OUTDIR, "calls.jsonl")
LIBS = os.path.join(OUTDIR, "libraries.json")
os.makedirs(OUTDIR, exist_ok=True)

TOPK = 8
LIFT_MIN = 2.0
SUP_ANOM_MIN = 0.10
SPLIT = 299            # 60% of 499
GUARD = 10             # 窗口重叠保护带（w=120/hop=60，单窗跨 2 个位置）
SEED = 42
WORKERS = int(os.environ.get("WORKERS", "8"))

# 家族内更小档（与 exp_llm_small_models.py 相同扩展）
diag.MODELS["glm-4-flash"] = (
    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    diag.MODELS["glm-4-air"][1])
diag.MODELS["qwen-turbo"] = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    diag.MODELS["qwen-plus"][1])
MODELS = ("deepseek-chat", "qwen-plus", "glm-4-air",
          "glm-4-flash", "qwen-turbo")

CTX = ("Context: a hydroelectric industrial-control system (HAI security "
       "testbed) with 25 continuously monitored sensor/actuator channels "
       "recorded at 1 Hz. Each 120-sample window is discretized into states "
       "0/1/2 per channel (0=low, 1=mid, 2=high). A window overlapping an "
       "attack interval is labeled faulty; all other windows are normal.")


def load_hai():
    items = json.load(open(os.path.join(DATA, "hai.txt.items.json"),
                           encoding="utf-8"))
    id2name = {v: k for k, v in items.items()}
    txns = []
    with open(os.path.join(DATA, "hai.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                txns.append(frozenset(id2name[int(t)] for t in line.split()))
    labels = [int(x) for x in json.load(open(
        os.path.join(DATA, "hai.txt.labels.json"), encoding="utf-8"))]
    assert len(txns) == len(labels)
    return txns, labels


# ---------------- 库构建：AnyFIM 挖掘 + 仅用给定窗口评分（评分窗口=库窗口） ----------------
def _results_complete(rp, upto_stage):
    """结果文件写完标志：全部阶段头齐 + 结尾 frequencyThreshold 尾行。
    （AnytimeMining.exe 在部分数据集上写完输出后不自退——2026-10-09 HAI 实测——
    故以文件完整性为准，见 run_anyfim_patient。）"""
    try:
        with open(rp, encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except OSError:
        return False
    if txt.count("stage MFIs Number") < upto_stage:
        return False
    tail = txt[-400:]
    return "frequencyThreshold" in tail and "===" in tail


def run_anyfim_patient(fimi_path, upto_stage, timeout=1800):
    """Popen + 轮询文件完成即杀进程解析；兼容引擎写完输出不自退的挂起 bug。"""
    rp = f"{fimi_path}-{upto_stage}-stages=Results.txt"
    if os.path.exists(rp):
        os.remove(rp)
    p = subprocess.Popen([ANYFIM_EXE, fimi_path, str(upto_stage)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    try:
        while time.time() - t0 < timeout:
            if _results_complete(rp, upto_stage):
                break
            time.sleep(2)
    finally:
        if p.poll() is None:
            p.kill()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
    if not _results_complete(rp, upto_stage):
        return None
    stages, meta = parse_anyfim_results(rp)
    return stages, meta


def mine_library(lib_txns, lib_labels, timeout=1800):
    n, npos = len(lib_txns), sum(lib_labels)
    nneg = n - npos
    fimi = os.path.abspath(os.path.join(
        OUTDIR, f"fimi.{uuid.uuid4().hex[:8]}.txt"))
    items = write_fimi([sorted(t) for t in lib_txns], fimi,
                       fimi + ".items.json")
    inv = {v: k for k, v in items.items()}
    got = run_anyfim_patient(fimi, len(items), timeout=timeout)
    for suffix in ("", ".items.json",
                   f"-{len(items)}-stages=Results.txt"):
        try:
            os.remove(fimi + suffix)
        except OSError:
            pass
    assert got is not None, "AnyFIM failed"
    stages, _meta = got
    pool = set()
    for k in sorted(stages):
        pool |= {frozenset(inv[i] for i in x) for x, _ in stages[k]}

    scored = []
    for pat in pool:
        p = sum(1 for t, l in zip(lib_txns, lib_labels) if l and pat <= t)
        q = sum(1 for t in lib_txns if pat <= t) - p
        sup_anom = p / npos if npos else 0.0
        sup_norm = q / nneg if nneg else 0.0
        if sup_anom < SUP_ANOM_MIN:
            continue
        lift = (sup_anom / sup_norm) if sup_norm > 0 else math.inf
        if lift < LIFT_MIN:
            continue
        scored.append({"pat": pat, "p": p, "sup_anom": sup_anom,
                       "sup_norm": sup_norm, "lift": lift})
    scored.sort(key=lambda d: (-d["lift"], -d["p"]))

    rows = []
    for d in scored[:TOPK]:
        rows.append({"pattern": " ∧ ".join(sorted(d["pat"])),
                     "sup_anom": f"{d['sup_anom']:.3f}",
                     "lift": "inf" if math.isinf(d["lift"])
                             else f"{d['lift']:.1f}"})
    stats = {"n_lib_windows": n, "n_lib_anom": npos, "pool": len(pool),
             "n_pass": len(scored), "n_selected": len(rows)}
    return rows, stats, scored


def random_rows(scored_pool_for_items, txns, labels, seed):
    """与 exp_llm_baseline 同规格：8 槽、大小 {2,3,4} 均匀、无放回、真实计数。"""
    items = sorted({it for t in txns for it in t})
    rng = random.Random(seed)
    n, npos = len(txns), sum(labels)
    nneg = n - npos
    rows = []
    for _ in range(TOPK):
        k = rng.randint(2, 4)
        pat = frozenset(rng.sample(items, k))
        p = sum(1 for t, l in zip(txns, labels) if l and pat <= t)
        q = sum(1 for t in txns if pat <= t) - p
        sup_anom = p / npos if npos else 0.0
        sup_norm = q / nneg if nneg else 0.0
        lift = (sup_anom / sup_norm) if sup_norm > 0 else math.inf
        rows.append({"pattern": " ∧ ".join(sorted(pat)),
                     "sup_anom": f"{sup_anom:.3f}",
                     "lift": "inf" if math.isinf(lift) else f"{lift:.1f}"})
    return rows


def user_prompt(case, with_lib):
    w = " ∧ ".join(case["window"])
    p = CTX + f"\n\nWindow to diagnose:\n{w}"
    if with_lib:
        rows = "\n".join(
            f"- {r['pattern']}  (support in faulty windows: {r['sup_anom']}, "
            f"lift: {diag.fmt_lift(r['lift'])})" for r in case["library"])
        p += ("\n\nMined pattern library of this system (counted from its "
              "own history):\n" + rows)
    p += "\n\nDiagnose this window."
    return p


def build_plan():
    txns, labels = load_hai()
    cached = {}
    if os.path.exists(LIBS):
        try:
            cached = json.load(open(LIBS, encoding="utf-8"))
        except json.JSONDecodeError:
            cached = {}
    plan, libs = [], dict(cached)

    def save_libs():
        with open(LIBS, "w", encoding="utf-8") as f:
            json.dump(libs, f, ensure_ascii=False, indent=1)

    # ---- Arm A: in-sample
    if "insample" in cached:
        rows, stats = cached["insample"]["rows"], cached["insample"]["stats"]
        rand = cached["insample"]["random"]
        print("[insample] loaded cached library", flush=True)
    else:
        rows, stats, _scored = mine_library(txns, labels)
        rand = random_rows(None, txns, labels, f"{SEED}|hai")
        libs["insample"] = {"rows": rows, "random": rand, "stats": stats}
        save_libs()
    rng = random.Random(SEED)
    faulty = [i for i, l in enumerate(labels) if l == 1]
    normal = [i for i, l in enumerate(labels) if l == 0]
    pick = rng.sample(normal, 19) + faulty
    for i in pick:
        for mode, lib in (("nolib", []), ("mined", rows), ("random", rand)):
            plan.append({"arm": "insample", "idx": i,
                         "truth": "faulty" if labels[i] else "normal",
                         "window": sorted(txns[i]), "library": lib,
                         "mode": mode})

    # ---- Arm B: time-split（库窗/测试窗不相交；挖掘更慢，给足 3600s）
    # 2026-10-09 实测：289 窗子集上的 AnyFIM 全阶段挖掘 >1h 未完成（阶段组合爆炸），
    # 论文报告 in-sample 手臂，split 变体列入未来工作；ARM=insample 可跳过本手臂。
    stats_b = None
    if os.environ.get("ARM", "all") != "insample":
        lib_txns, lib_labels = txns[:SPLIT - GUARD], labels[:SPLIT - GUARD]
        if "timesplit" in cached:
            rows_b = cached["timesplit"]["rows"]
            stats_b = cached["timesplit"]["stats"]
            rand_b = cached["timesplit"]["random"]
            print("[timesplit] loaded cached library", flush=True)
        else:
            rows_b, stats_b, _ = mine_library(lib_txns, lib_labels, timeout=3600)
            rand_b = random_rows(None, lib_txns, lib_labels, "hai|ts")
            libs["timesplit"] = {"rows": rows_b, "random": rand_b,
                                 "stats": stats_b, "split": SPLIT,
                                 "guard": GUARD}
            save_libs()
        test_idx = list(range(SPLIT + GUARD, len(txns)))
        t_faulty = [i for i in test_idx if labels[i] == 1]
        t_normal = [i for i in test_idx if labels[i] == 0]
        rng_b = random.Random("hai|ts")
        pick_b = rng_b.sample(t_normal, len(t_faulty)) + t_faulty
        for i in pick_b:
            for mode, lib in (("nolib", []), ("mined", rows_b),
                              ("random", rand_b)):
                plan.append({"arm": "timesplit", "idx": i,
                             "truth": "faulty" if labels[i] else "normal",
                             "window": sorted(txns[i]), "library": lib,
                             "mode": mode})
        print(f"[timesplit] lib={stats_b} test={len(pick_b)} windows "
              f"(attack={len(t_faulty)})", flush=True)
    print(f"[insample] lib={stats} test=38 windows", flush=True)
    return plan, libs


# ---------------- 调用层 ----------------
_lock = threading.Lock()
_counter = {"n": 0}


def ask_fast(model, user):
    url, key = diag.MODELS[model]
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": diag.SYS},
                     {"role": "user", "content": user}],
        "temperature": 0, "max_tokens": 300,
    }).encode("utf-8")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", "Bearer " + key)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                return f"__ERROR__ {type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))


def run():
    plan, libs = build_plan()
    with open(LIBS, "w", encoding="utf-8") as f:
        json.dump(libs, f, ensure_ascii=False, indent=1)

    work = []
    for c in plan:
        key = f"{c['arm']}#{c['idx']}|{c['mode']}"
        for model in MODELS:
            work.append((model, key, c))

    done = set()
    if os.path.exists(CALLS):
        for line in open(CALLS, encoding="utf-8"):
            r = json.loads(line)
            if str(r["reply"]).startswith("__ERROR__"):
                continue
            done.add((r["model"], r["case_key"]))
    todo = [w for w in work if (w[0], w[1]) not in done]
    print(f"total {len(work)}, done {len(done)}, todo {len(todo)}", flush=True)
    if not todo:
        print("ALL DONE")
        return

    fout = open(CALLS, "a", encoding="utf-8")
    t0 = time.time()

    def one(item):
        model, key, c = item
        try:
            reply = ask_fast(model, user_prompt(c, bool(c["library"])))
            rec = {"model": model, "case_key": key, "arm": c["arm"],
                   "idx": c["idx"], "mode": c["mode"], "truth": c["truth"],
                   "reply": reply}
            with _lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                _counter["n"] += 1
                if _counter["n"] % 50 == 0:
                    el = time.time() - t0
                    print(f"[{_counter['n']}] {el:.0f}s "
                          f"({_counter['n']/max(el,1):.2f}/s)", flush=True)
        except Exception as e:
            with _lock:
                print(f"ERR {model} {key}: {e}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, todo))
    print("done. calls ->", CALLS, flush=True)


if __name__ == "__main__":
    run()
