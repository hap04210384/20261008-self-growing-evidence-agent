# -*- coding: utf-8 -*-
"""exp_e9_sensitivity.py —— 审稿主要问题 3/5：证据选择超参（lift, sup_anom）敏感性。

口径与 4.6 节完全一致：同一 198 个测试窗口（seed=42，每组 33+33）、同一池
（results/spike_v1/patterns_{grp}.csv，与 exp_llm_diagnosis.load_group 同源）、
同一排序（sup_anom 降序取 top-8）、同一提示词与解码设置，只改变选择阈值：
  lift     ∈ {1.5, 2.0, 3.0, 4.0}
  sup_anom ∈ {0.05, 0.10, 0.20}
共 12 个配置 × 198 窗 × 3 模型 = 7,128 次调用（temperature=0）。

产物：results/e9_sensitivity/calls.jsonl（断点续跑）+ libraries.json
"""
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import exp_llm_diagnosis as diag          # noqa: E402

diag.N_PER_CLASS = 33   # 198 窗口协议

OUTDIR = os.path.join(ROOT, "results", "e9_sensitivity")
CALLS = os.path.join(OUTDIR, "calls.jsonl")
LIBS = os.path.join(OUTDIR, "libraries.json")
os.makedirs(OUTDIR, exist_ok=True)

LIFTS = (1.5, 2.0, 3.0, 4.0)
SUPS = (0.05, 0.10, 0.20)
TOPK = 8
WORKERS = int(os.environ.get("WORKERS", "8"))
ONLY_MODEL = os.environ.get("ONLY_MODEL") or None   # 服务商限流时按模型分批跑


def pool_of(grp):
    """与 diag.load_group 同源的完整打分池（不去重，保留计数）。"""
    rows = []
    import csv as _csv
    with open(os.path.join(ROOT, "results", "spike_v1",
                           f"patterns_{grp}.csv"), encoding="utf-8-sig") as f:
        for r in _csv.DictReader(f):
            try:
                rows.append({"pattern": r["pattern"],
                             "sup_anom": float(r["sup_anom"]),
                             "sup_norm": float(r["sup_norm"]),
                             "lift": float(r["lift"])})
            except (ValueError, KeyError):
                continue
    return rows


def select(pool, lift_min, sup_min):
    """协议排序：sup_anom 降序，top-8，按 pattern 去重。"""
    seen, out = set(), []
    for r in sorted(pool, key=lambda x: -x["sup_anom"]):
        if r["sup_anom"] < sup_min or r["lift"] < lift_min:
            continue
        if r["pattern"] in seen:
            continue
        seen.add(r["pattern"])
        out.append({"pattern": r["pattern"],
                    "sup_anom": f"{r['sup_anom']:.3f}",
                    "lift": diag.fmt_lift(r["lift"])})
        if len(out) >= TOPK:
            break
    return out


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
    cases = diag.build_cases()
    pools = {g: pool_of(g) for g in diag.GROUPS}
    libs = {}
    for g in diag.GROUPS:
        for L in LIFTS:
            for S in SUPS:
                rows = select(pools[g], L, S)
                libs[f"{g}|L{L}|S{S}"] = rows
    with open(LIBS, "w", encoding="utf-8") as f:
        json.dump(libs, f, ensure_ascii=False, indent=1)

    work = []
    for case in cases:
        for L in LIFTS:
            for S in SUPS:
                c = dict(case)
                c["library"] = libs[f"{case['group']}|L{L}|S{S}"]
                key = f"{case['group']}#{case['idx']}|L{L}|S{S}"
                models = [ONLY_MODEL] if ONLY_MODEL else list(diag.MODELS)
                for model in models:
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
            reply = ask_fast(model, diag.user_prompt(c, True))
            rec = {"model": model, "case_key": key, "group": c["group"],
                   "truth": c["truth"], "reply": reply}
            with _lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                _counter["n"] += 1
                if _counter["n"] % 100 == 0:
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
