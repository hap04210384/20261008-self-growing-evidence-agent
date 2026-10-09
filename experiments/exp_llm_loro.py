# -*- coding: utf-8 -*-
"""exp_llm_loro.py —— 无泄漏 LLM 诊断评估（审稿主要问题 1 的正面回应）。

协议（frozen，与手稿 3.8 节一致的挖掘/评分口径，仅改变证据来源的窗口集合）：
对每个组 g ∈ {valve1, valve2, other} × 划分种子 s ∈ {0,1,2}：
  1. 该组全部 run 文件用 Random(s) 洗牌，前半 = 库 run，后半 = 测试 run（不相交）；
  2. 仅用库 run 的窗口走原管线（逐 run fit_bins + w=30/hop=15）并用正式引擎
     AnyFIM 挖掘累积 MFI 池；
  3. 模式的 anomalous-window support / normal-window support / lift 只在
     【库 run 窗口】上计算（测试 run 标签在建库与筛选阶段不可见）；
  4. 筛选：lift >= 2（sup_norm=0 视为通过）、sup_anom >= 0.1，按 lift 降序取 top-8
     —— 与手稿 3.6 节诊断协议同规则同预算；
  5. 测试窗口仅从测试 run 抽取：正常 33 + 故障 33（Random(f"{g}|{s}")）；
  6. 诊断两条件：A 无库 / B 附 LORO 库；3 个商业 LLM，temperature=0；
  7. 同批测试窗口 + 用【全部 run】建的对照库（in-sample 参照，量化泄漏贡献）。

产物（results/loro/）：
  libraries.json   每 (group, split) 的库 run 清单与入选 top-8 模式（忠实度核验用）
  calls.jsonl      逐次调用记录（断点续跑：__ERROR__ 记录下轮重试）
  summary.json     由 summarize_loro.py 生成

用法：
  python exp_llm_loro.py            全量（3 组 × 3 种子，含 in-sample 参照）
  python exp_llm_loro.py smoke      通路验证：valve2 单种子、每类 2 窗、单模型
"""
import json
import math
import os
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PIPE = os.path.join(ROOT, "code", "pipeline")
sys.path.insert(0, HERE)
sys.path.insert(0, PIPE)

import exp_llm_diagnosis as diag          # noqa: E402  (MODELS/SYS/CTX/fmt_lift)
import skab_spike as spike                # noqa: E402
from engine_adapter import run_anyfim, write_fimi  # noqa: E402

SKAB = os.path.join(ROOT, "data", "SKAB_raw", "data")
OUTDIR = os.path.join(ROOT, "results", "loro")
CALLS = os.path.join(OUTDIR, "calls.jsonl")
LIBS = os.path.join(OUTDIR, "libraries.json")
os.makedirs(OUTDIR, exist_ok=True)

GROUPS = ("valve1", "valve2", "other")
SPLITS = (0, 1, 2)
N_PER_CLASS = 33          # 与 M4i 的 198 窗协议一致
TOPK = 8                  # 与 3.6 节诊断预算一致
LIFT_MIN = 2.0
SUP_ANOM_MIN = 0.10
WORKERS = int(os.environ.get("WORKERS", "8"))

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
if SMOKE:
    SPLITS = (0,)
    N_PER_CLASS = 2
    GROUPS = ("valve2",)
    ONLY_MODEL = "glm-4-air"
else:
    ONLY_MODEL = os.environ.get("ONLY_MODEL") or None


# ---------------- 数据：逐 run 事务化（与 exp_faultclass 相同管线） ----------------
def group_runs(grp):
    folder = os.path.join(SKAB, grp)
    return [os.path.join(folder, fn) for fn in sorted(os.listdir(folder))
            if fn.endswith(".csv")]


def windows_of(run_files):
    """返回 (txns: [set(str)], labels: [int])，按 run 顺序拼接。"""
    txns, labels = [], []
    for path in run_files:
        df = spike.load_skab_file(path)
        anom = df["anomaly"].values
        df = df.drop(columns=[c for c in ["datetime", "anomaly", "changepoint"]
                              if c in df.columns])
        df["__anom__"] = anom
        edges = spike.fit_bins(df, spike.SENSOR_COLS, 3)
        _, t, ta = spike.window_transactions(df, spike.SENSOR_COLS,
                                             edges, 30, 15)
        txns.extend(set(x) for x in t)
        labels.extend(int(x) for x in ta)
    return txns, labels


# ---------------- 库构建：挖掘 + 仅用库窗口评分 ----------------
def mine_library(lib_txns, lib_labels):
    """AnyFIM 挖掘累积 MFI 池，返回 (rows, stats)。

    rows: 按 3.6 节协议筛选排序后的 top-K {"pattern","sup_anom","lift"}；
    评分严格限定在库窗口（lib_txns/lib_labels）上。
    """
    n, npos = len(lib_txns), sum(lib_labels)
    nneg = n - npos
    fimi = os.path.abspath(os.path.join(OUTDIR, "fimi.tmp.txt"))
    items = write_fimi([sorted(t) for t in lib_txns], fimi,
                       fimi + ".items.json")
    inv = {v: k for k, v in items.items()}
    r = run_anyfim(fimi, upto_stage=len(items), timeout=3600)
    os.remove(fimi)
    assert r["ok"], "AnyFIM failed"
    pool = set()
    for k in sorted(r["stages"]):
        pool |= {frozenset(inv[i] for i in x) for x, _ in r["stages"][k]}

    scored = []
    for pat in pool:
        c = sum(1 for t in lib_txns if pat <= t)
        p = sum(1 for t, l in zip(lib_txns, lib_labels)
                if l and pat <= t)
        q = c - p
        sup_anom = p / npos if npos else 0.0
        sup_norm = q / nneg if nneg else 0.0
        if sup_anom < SUP_ANOM_MIN:
            continue
        lift = (sup_anom / sup_norm) if sup_norm > 0 else math.inf
        if lift < LIFT_MIN:
            continue
        scored.append({"pat": pat, "p": p, "q": q,
                       "sup_anom": sup_anom, "sup_norm": sup_norm,
                       "lift": lift})
    scored.sort(key=lambda d: (-d["lift"], -d["p"]))

    rows = []
    for d in scored[:TOPK]:
        lift_s = ("inf" if math.isinf(d["lift"])
                  else f"{d['lift']:.1f}")
        rows.append({"pattern": " ∧ ".join(sorted(d["pat"])),
                     "sup_anom": f"{d['sup_anom']:.3f}",
                     "lift": lift_s})
    stats = {"n_lib_windows": n, "n_lib_anom": npos, "pool": len(pool),
             "n_pass": len(scored), "n_selected": len(rows)}
    return rows, stats


# ---------------- 主流程 ----------------
def fmt_case(grp, s, idx, truth, window, library):
    return {"group": grp, "split": s, "idx": idx, "truth": truth,
            "window": sorted(window), "library": library}


def build_plan():
    """返回 (cases, libs_meta)。

    cases: 每个测试窗口 × 条件 {nolib, loro, insample}
    libs_meta: (grp, s) -> {"lib_runs","test_runs","loro":rows, "in":rows, stats}
    """
    cases, libs_meta = [], {}
    for grp in GROUPS:
        runs = group_runs(grp)
        for s in SPLITS:
            shuffled = runs[:]
            random.Random(s).shuffle(shuffled)
            half = len(shuffled) // 2
            lib_runs, test_runs = shuffled[:half], shuffled[half:]
            lib_txns, lib_labels = windows_of(lib_runs)
            test_txns, test_labels = windows_of(test_runs)

            loro_rows, loro_stats = mine_library(lib_txns, lib_labels)
            in_rows, in_stats = mine_library(
                *windows_of(runs))          # in-sample 参照：全部 run 建库

            rng = random.Random(f"{grp}|{s}")
            faulty = [i for i, l in enumerate(test_labels) if l == 1]
            normal = [i for i, l in enumerate(test_labels) if l == 0]
            need = min(N_PER_CLASS, len(faulty), len(normal))
            pick = (rng.sample(normal, need) + rng.sample(faulty, need))
            for i in pick:
                truth = "faulty" if test_labels[i] == 1 else "normal"
                base = fmt_case(grp, s, i, truth, test_txns[i], [])
                for mode in ("nolib", "loro", "insample"):
                    c = dict(base)
                    c["library"] = loro_rows if mode == "loro" else \
                        (in_rows if mode == "insample" else [])
                    c["mode"] = mode
                    cases.append(c)
            libs_meta[f"{grp}|{s}"] = {
                "lib_runs": [os.path.basename(x) for x in lib_runs],
                "test_runs": [os.path.basename(x) for x in test_runs],
                "loro": loro_rows, "in": in_rows,
                "loro_stats": loro_stats, "in_stats": in_stats,
            }
            print(f"[{grp}|{s}] lib={len(lib_runs)}runs/{len(lib_txns)}w "
                  f"test={len(test_runs)}runs/{len(test_txns)}w "
                  f"loro_sel={len(loro_rows)} in_sel={len(in_rows)}",
                  flush=True)
    return cases, libs_meta


def user_prompt_mode(case):
    return diag.user_prompt(case, bool(case["library"]))


# ---------------- 调用层（与 exp_llm_scaleup 相同传输协议） ----------------
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
    cases, libs_meta = build_plan()
    with open(LIBS, "w", encoding="utf-8") as f:
        json.dump(libs_meta, f, ensure_ascii=False, indent=1)

    work = []
    for c in cases:
        for model in ([ONLY_MODEL] if ONLY_MODEL else diag.MODELS):
            key = f"{c['group']}|{c['split']}|{c['idx']}|{c['mode']}"
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
            reply = ask_fast(model, user_prompt_mode(c))
            rec = {"model": model, "case_key": key, "group": c["group"],
                   "split": c["split"], "mode": c["mode"], "truth": c["truth"],
                   "reply": reply}
            with _lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                _counter["n"] += 1
                el = time.time() - t0
                if _counter["n"] % 50 == 0:
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
