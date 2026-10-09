# -*- coding: utf-8 -*-
"""exp_llm_baseline.py —— 审稿主要问题 4：传统 min-sup 管线 + 随机证据对照。

在 4.6 节完全相同的 198 个测试窗口（seed=42，每组 33 正常 + 33 故障）上，
把"我们的库"换成两类对照证据，其他全部不变（同一 SYS/CTX/解码设置）：

条件 B1（min-sup 基线）：经典 Apriori 以 min-sup = 0.05 挖掘该组全部窗口的
  频繁项集，按 lift 降序取 top-8 作为证据库 —— 模拟"传统 FIM 管线 + LLM"。
条件 B2（随机对照）：从该组 24 个物品的项空间中均匀随机抽 8 个
  大小 2~4 的项集，支撑/lift 仍按真实窗口计数（只随机"选哪些模式"，
  不伪造数字），检验"有任何库"与"有判别性库"的差别。

  无库条件复用 results/llm_diag/calls.jsonl 的 with_lib=False 记录。

调用：3 模型 × 198 窗口 × 2 条件 = 1,188 次，temperature=0。
产物：results/llm_baseline/calls.jsonl（断点续跑，__ERROR__ 重试）
"""
import itertools
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
sys.path.insert(0, HERE)

import exp_llm_diagnosis as diag          # noqa: E402

diag.N_PER_CLASS = 33   # 与 4.6 节 198 窗口协议一致（scaleup 同款设置）

OUTDIR = os.path.join(ROOT, "results", "llm_baseline")
CALLS = os.path.join(OUTDIR, "calls.jsonl")
LIBS = os.path.join(OUTDIR, "libraries.json")
os.makedirs(OUTDIR, exist_ok=True)

MIN_SUP = 0.05
TOPK = 8
COND_SEED = 20261009
WORKERS = int(os.environ.get("WORKERS", "8"))


# ---------------- Apriori（tidset 交集，项数 <=24，足够快） ----------------
def apriori(txns, min_sup, max_len=8):
    """返回 [frozenset,...]：满足 min_sup 的频繁项集（含 1-项集）。"""
    n = len(txns)
    min_cnt = max(1, math.ceil(min_sup * n))
    tids = {}
    for i, t in enumerate(txns):
        for it in t:
            tids.setdefault(it, set()).add(i)
    freq = {frozenset([it]): s for it, s in tids.items() if len(s) >= min_cnt}
    out = list(freq)
    cur = {k: v for k, v in freq.items() if len(k) == 1}
    k = 2
    while cur and k <= max_len:
        items = sorted({it for s in cur for it in s})
        nxt = {}
        for combo in itertools.combinations(items, k):
            fs = frozenset(combo)
            sub_ok = all(frozenset(c) in cur
                         for c in itertools.combinations(combo, k - 1))
            if not sub_ok:
                continue
            s = None
            for it in combo:
                t = tids[it]
                s = t if s is None else (s & t)
                if len(s) < min_cnt:
                    s = None
                    break
            if s is not None:
                nxt[fs] = s
        out.extend(nxt)
        cur = nxt
        k += 1
    return out


def score_all(txns, labels, pats):
    n, npos = len(txns), sum(labels)
    nneg = n - npos
    rows = []
    for pat in pats:
        p = sum(1 for t, l in zip(txns, labels) if l and pat <= t)
        c = sum(1 for t in txns if pat <= t)
        q = c - p
        sup_anom = p / npos if npos else 0.0
        sup_norm = q / nneg if nneg else 0.0
        lift = sup_anom / sup_norm if sup_norm > 0 else math.inf
        rows.append({"pat": pat, "p": p, "sup_anom": sup_anom,
                     "sup_norm": sup_norm, "lift": lift})
    return rows


def fmt(rows):
    out = []
    for d in rows[:TOPK]:
        lift_s = "inf" if math.isinf(d["lift"]) else f"{d['lift']:.1f}"
        out.append({"pattern": " ∧ ".join(sorted(d["pat"])),
                    "sup_anom": f"{d['sup_anom']:.3f}", "lift": lift_s})
    return out


def build_libraries():
    """每组建 min-sup 库与随机库（字典 group -> {"minsup": rows, "random": rows}）。"""
    libs = {}
    for grp in diag.GROUPS:
        txns, labels, _ = diag.load_group(grp)
        labels = [int(x) for x in labels]
        txns = [set(t) for t in txns]
        items = sorted({it for t in txns for it in t})

        pats = [p for p in apriori(txns, MIN_SUP) if len(p) >= 2]
        scored = score_all(txns, labels, pats)
        scored.sort(key=lambda d: (-d["lift"], -d["p"]))
        minsup_rows = fmt(scored)

        rng = random.Random(f"{COND_SEED}|{grp}")
        rand_rows = []
        for _ in range(TOPK):
            k = rng.randint(2, 4)
            pat = frozenset(rng.sample(items, k))
            d = score_all(txns, labels, [pat])[0]
            rand_rows.append(d)
        rand_rows = fmt(rand_rows)

        libs[grp] = {"minsup": minsup_rows, "random": rand_rows}
        print(f"[{grp}] items={len(items)} minsup_pats={len(pats)} "
              f"minsup_top1={minsup_rows[0]['pattern'][:40]}", flush=True)
    return libs


# ---------------- 调用层（与 exp_llm_loro 相同协议） ----------------
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
    cases = diag.build_cases()          # 与 4.6 节完全相同的 198 窗
    libs = build_libraries()
    with open(LIBS, "w", encoding="utf-8") as f:
        json.dump(libs, f, ensure_ascii=False, indent=1)

    work = []
    for case in cases:
        for cond in ("minsup", "random"):
            c = dict(case)
            c["library"] = libs[case["group"]][cond]
            c["cond"] = cond
            key = f"{case['group']}#{case['idx']}"
            for model in diag.MODELS:
                work.append((model, f"{key}|{cond}", c))

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
            reply = ask_fast(model, diag.user_prompt(c, True))
            rec = {"model": model, "case_key": key, "cond": c["cond"],
                   "group": c["group"], "truth": c["truth"], "reply": reply}
            with _lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                _counter["n"] += 1
                if _counter["n"] % 50 == 0:
                    el = time.time() - t0
                    print(f"[{_counter['n']}] {el:.0f}s", flush=True)
        except Exception as e:
            with _lock:
                print(f"ERR {model} {key}: {e}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, todo))
    print("done. calls ->", CALLS, flush=True)


if __name__ == "__main__":
    run()
