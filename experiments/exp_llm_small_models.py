# -*- coding: utf-8 -*-
"""exp_llm_small_models.py —— 审稿 Major 5：更小档 API 模型的同协议消融。

与 results/llm_diag（Figure 5 数据源）完全相同的 198 个 SKAB 平衡窗口
（seed=42，每组 33 正常 + 33 故障）、同一冻结 prompt/JSON 契约/贪心解码，
仅增加两个家族内更小档模型，形成同家族规模梯度：
  - GLM-4-Flash（智谱，低于 GLM-4-Air 的免费档）
  - Qwen-Turbo（阿里百炼，低于 Qwen-Plus 的最低付费档）

调用：2 模型 × 198 窗口 × 2 条件（无库/有库）= 792 次，temperature=0。
产物：results/llm_small/calls.jsonl（断点续跑，__ERROR__ 重试）
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

diag.N_PER_CLASS = 33   # 与 4.5 节 198 窗口协议一致

# 家族内更小档（不动冻结的 MODELS 原定义，仅在本实验中扩展）
diag.MODELS["glm-4-flash"] = (
    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    diag.MODELS["glm-4-air"][1])
diag.MODELS["qwen-turbo"] = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    diag.MODELS["qwen-plus"][1])

SMALL = ("glm-4-flash", "qwen-turbo")

OUTDIR = os.path.join(ROOT, "results", "llm_small")
CALLS = os.path.join(OUTDIR, "calls.jsonl")
os.makedirs(OUTDIR, exist_ok=True)
WORKERS = int(os.environ.get("WORKERS", "8"))


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
    cases = diag.build_cases()          # 与 4.5 节完全相同的 198 窗
    work = []
    for case in cases:
        key = f"{case['group']}#{case['idx']}"
        for with_lib in (False, True):
            c = dict(case)
            c["with_lib"] = with_lib
            for model in SMALL:
                work.append((model, f"{key}|{'lib' if with_lib else 'nolib'}", c))

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
            reply = ask_fast(model, diag.user_prompt(c, c["with_lib"]))
            rec = {"model": model, "case_key": key,
                   "with_lib": c["with_lib"],
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
