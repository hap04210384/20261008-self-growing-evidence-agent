# -*- coding: utf-8 -*-
"""exp_llm_scaleup.py —— 按审稿意见 M4 把 LLM 诊断扩到 198 窗口（并行 + 限时断点续跑）。

设计（frozen）：
- 取样：每组 正常 33 + 故障 33（seed=42 不变），3 组共 198 窗口（每组池足够：
  valve2 故障池 108 >= 33）
- 诊断：3 模型 × 198 窗口 × {无库, 有库} = 1188 次调用
- 预算消融：3 模型 × 198 窗口 × k ∈ {1,2,3,5,8} = 2970 次调用
- 产物：results/llm_diag/calls200.jsonl / budget200.jsonl
  （与 60 窗口版同格式；完成后由人工核对再替换 calls.jsonl / budget.jsonl）
- 并行：8 线程；单次运行限时 BUDGET_SEC（默认 270），到点保存退出，重跑续传。
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

import exp_llm_diagnosis as diag      # noqa: E402
import exp_llm_budget as budget       # noqa: E402

N_PER_CLASS_NEW = 33
BUDGET_SEC = float(os.environ.get("BUDGET_SEC", "270"))
WORKERS = int(os.environ.get("WORKERS", "8"))

OUTDIR = os.path.join(ROOT, "results", "llm_diag")
CALLS200 = os.path.join(OUTDIR, "calls200.jsonl")
BUDGET200 = os.path.join(OUTDIR, "budget200.jsonl")

diag.N_PER_CLASS = N_PER_CLASS_NEW   # 两个模块共用 build_cases，一并生效

_lock = threading.Lock()
_counter = {"n": 0, "done_flag": False}


def ask_fast(model, user):
    """与 diag.ask 相同协议，但传输层超时收紧（20s x 3 次，退避 2/4s），
    避免个别挂死请求拖住整批；失败返回 __ERROR__ 由续跑逻辑重试。"""
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
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                return f"__ERROR__ {type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))


def log(msg):
    with _lock:
        print(msg, flush=True)


def run():
    cases = diag.build_cases()
    print(f"cases: {len(cases)} (per class {N_PER_CLASS_NEW})", flush=True)

    work = []
    for model in diag.MODELS:
        for case in cases:
            key = f"{case['group']}#{case['idx']}"
            work.append((model, key, case, False))
            work.append((model, key, case, True))
    for model in diag.MODELS:
        for case in cases:
            key = f"{case['group']}#{case['idx']}"
            for k in budget.KS:
                work.append((model, key, case, k))

    done = set()
    if os.path.exists(CALLS200):
        for line in open(CALLS200, encoding="utf-8"):
            r = json.loads(line)
            if str(r["reply"]).startswith("__ERROR__"):
                continue   # 失败记录不算完成，下轮重试
            done.add((r["model"], r["case_key"], ("lib", r["with_lib"])))
    if os.path.exists(BUDGET200):
        for line in open(BUDGET200, encoding="utf-8"):
            r = json.loads(line)
            if str(r["reply"]).startswith("__ERROR__"):
                continue
            done.add((r["model"], r["case_key"], ("k", r["k"])))
    todo = []
    for w in work:
        tag = ("lib", w[3]) if isinstance(w[3], bool) else ("k", w[3])
        if (w[0], w[1], tag) not in done:
            todo.append(w)
    print(f"total {len(work)}, done {len(done)}, todo {len(todo)}", flush=True)
    if not todo:
        print("ALL DONE")
        return

    fcalls = open(CALLS200, "a", encoding="utf-8")
    fbudget = open(BUDGET200, "a", encoding="utf-8")
    t0 = time.time()

    def one(item):
        if _counter["done_flag"]:
            return
        model, key, case, tag = item
        try:
            if isinstance(tag, bool):
                reply = ask_fast(model, diag.user_prompt(case, tag))
                rec = {"model": model, "case_key": key, "with_lib": tag,
                       "group": case["group"], "truth": case["truth"],
                       "reply": reply}
                with _lock:
                    fcalls.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fcalls.flush()
            else:
                reply = ask_fast(model, budget.user_prompt_k(case, tag))
                rec = {"model": model, "case_key": key, "k": tag,
                       "group": case["group"], "truth": case["truth"],
                       "reply": reply}
                with _lock:
                    fbudget.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fbudget.flush()
        except Exception as e:  # 单条失败不拖垮整批
            log(f"ERR {model} {key} {tag}: {e}")
            return
        with _lock:
            _counter["n"] += 1
            if _counter["n"] % 50 == 0:
                el = time.time() - t0
                log(f"[{_counter['n']}] elapsed {el:.0f}s "
                    f"({_counter['n']/max(el,1):.1f}/s)")

    if WORKERS <= 1:
        # 顺序模式：实测 0.84s/次，避开线程池与服务商并发限速的相互作用
        n = 0
        for item in todo:
            if time.time() - t0 > BUDGET_SEC:
                break
            model, key, case, tag = item
            if isinstance(tag, bool):
                reply = ask_fast(model, diag.user_prompt(case, tag))
                rec = {"model": model, "case_key": key, "with_lib": tag,
                       "group": case["group"], "truth": case["truth"],
                       "reply": reply}
                fcalls.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fcalls.flush()
            else:
                reply = ask_fast(model, budget.user_prompt_k(case, tag))
                rec = {"model": model, "case_key": key, "k": tag,
                       "group": case["group"], "truth": case["truth"],
                       "reply": reply}
                fbudget.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fbudget.flush()
            n += 1
            if n % 50 == 0:
                print(f"[{n}] elapsed {time.time()-t0:.0f}s", flush=True)
        fcalls.close()
        fbudget.close()
        print(f"chunk finished: {n} new calls this run; rerun to continue",
              flush=True)
        os._exit(0)

    pool = ThreadPoolExecutor(max_workers=WORKERS)
    futs = [pool.submit(one, w) for w in todo]
    while time.time() - t0 < BUDGET_SEC:
        time.sleep(2)
        if all(f.done() for f in futs):
            break
    _counter["done_flag"] = True
    # 等正在执行的收尾（最多再等 30s），随后取消排队任务并硬退出，
    # 避免 hung 连接拖住解释器退出（文件已逐条 flush+close）
    deadline = time.time() + 30
    while time.time() < deadline and not all(f.done() for f in futs):
        time.sleep(1)
    for f in futs:
        f.cancel()
    pool.shutdown(wait=False, cancel_futures=True)

    fcalls.close()
    fbudget.close()
    print(f"chunk finished: {_counter['n']} new calls this run; "
          f"rerun to continue (BUDGET_SEC={BUDGET_SEC})", flush=True)
    os._exit(0)   # 终止可能仍挂在网络读上的工作线程


if __name__ == "__main__":
    run()
