# -*- coding: utf-8 -*-
"""exp_llm_diagnosis.py —— LLM 证据约束诊断消融（Figure 5 数据源）。

设计（frozen，全部写进本文件以便审稿复现）：
- 任务：SKAB 窗口级故障诊断（normal / faulty），三组运行 valve1 / valve2 / other
- 取样：每组 正常 10 + 故障 10 个窗口（seed=42），共 60 窗口
- 条件 A（无证据）：只给窗口离散化状态 + 基础上下文
- 条件 B（有证据）：再加该组模式库 top-8 模式（sup/lift 计数）
- 模型：deepseek-chat / qwen-plus / glm-4-air（temperature=0）
- 指标：诊断准确率；条件 B 引文忠实度（引用模式是否真实存在于模式库）
- 产物：results/llm_diag/calls.jsonl（逐次调用）+ summary.json（汇总）
"""
import json
import os
import random
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "fimi")
OUTDIR = os.path.join(ROOT, "results", "llm_diag")
CALLS = os.path.join(OUTDIR, "calls.jsonl")
SUMMARY = os.path.join(OUTDIR, "summary.json")
os.makedirs(OUTDIR, exist_ok=True)

# ---------- 模型配置（OpenAI 兼容接口）----------
ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
cfg = {}
for line in open(ENV, encoding="utf-8"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()

MODELS = {
    "deepseek-chat": ("https://api.deepseek.com/v1/chat/completions",
                      cfg["DEEPSEEK_API_KEY"]),
    "qwen-plus": ("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                  cfg["DASHSCOPE_API_KEY"]),
    "glm-4-air": ("https://open.bigmodel.cn/api/paas/v4/chat/completions",
                  cfg["ZHIPU_API_KEY"]),
}

GROUPS = ("valve1", "valve2", "other")
N_PER_CLASS = 10
SEED = 42
TOPK = 8

CTX = ("Context: an industrial water-pump testbed with 8 sensors "
       "(Accelerometer1, Accelerometer2, Current, Pressure, Temperature, "
       "Thermocouple, Voltage, Volume Flow Rate). Each 30-sample window is "
       "discretized into states 0/1/2 per sensor (0=low, 1=mid, 2=high).")

SYS = ("You are an industrial fault-diagnosis assistant. Answer with a JSON "
       "object only, no markdown: {\"diagnosis\": \"normal\" or \"faulty\", "
       "\"cited_patterns\": [\"...\"], \"reason\": \"one short sentence\"}. "
       "cited_patterns must be patterns copied verbatim from the provided "
       "library, or an empty list if no library is provided.")


def load_group(grp):
    items = json.load(open(os.path.join(DATA, f"skab_{grp}.txt.items.json"),
                           encoding="utf-8"))
    id2name = {v: k for k, v in items.items()}
    txns = []
    with open(os.path.join(DATA, f"skab_{grp}.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                txns.append([id2name[int(t)] for t in line.split()])
    labels = json.load(open(os.path.join(DATA, f"skab_{grp}.txt.labels.json"),
                            encoding="utf-8"))
    lib = []
    with open(os.path.join(ROOT, "results", "spike_v1",
                           f"patterns_{grp}.csv"), encoding="utf-8-sig") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 5:
                lib.append({"pattern": parts[0], "sup_anom": parts[2],
                            "lift": parts[4]})
    lib.sort(key=lambda r: -float(r["sup_anom"]))
    return txns, labels, lib[:TOPK]


def fmt_lift(lift):
    try:
        v = float(lift)
    except ValueError:
        return lift
    if v > 1e4:
        return "inf (never occurs in normal windows)"
    return f"{v:.1f}"


def build_cases():
    rng = random.Random(SEED)
    cases = []
    for grp in GROUPS:
        txns, labels, lib = load_group(grp)
        faulty = [i for i, l in enumerate(labels) if l == "1"]
        normal = [i for i, l in enumerate(labels) if l == "0"]
        pick = (rng.sample(normal, N_PER_CLASS) +
                rng.sample(faulty, N_PER_CLASS))
        for i in pick:
            cases.append({"group": grp, "idx": i,
                          "truth": "faulty" if labels[i] == "1" else "normal",
                          "window": sorted(txns[i]), "library": lib})
    return cases


def ask(model, user):
    url, key = MODELS[model]
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": user}],
        "temperature": 0, "max_tokens": 300,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + key)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                return f"__ERROR__ {type(e).__name__}: {e}"
            time.sleep(5 * (attempt + 1))


def parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def user_prompt(case, with_lib):
    w = " ∧ ".join(case["window"])
    p = CTX + f"\n\nWindow to diagnose:\n{w}"
    if with_lib:
        rows = "\n".join(
            f"- {r['pattern']}  (support in faulty windows: {r['sup_anom']}, "
            f"lift: {fmt_lift(r['lift'])})" for r in case["library"])
        p += ("\n\nMined pattern library of this pump (counted from its own "
              "history):\n" + rows)
    p += "\n\nDiagnose this window."
    return p


def main():
    only_model = sys.argv[1] if len(sys.argv) > 1 else None
    cases = build_cases()
    lib_sets = {g: {r["pattern"] for r in load_group(g)[2]} for g in GROUPS}
    done = set()
    if os.path.exists(CALLS):
        for line in open(CALLS, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["model"], r["case_key"], r["with_lib"]))
    fout = open(CALLS, "a", encoding="utf-8")
    n = 0
    for model in MODELS:
        if only_model and model != only_model:
            continue
        for ci, case in enumerate(cases):
            for with_lib in (False, True):
                key = f"{case['group']}#{case['idx']}"
                if (model, key, with_lib) in done:
                    continue
                reply = ask(model, user_prompt(case, with_lib))
                rec = {"model": model, "case_key": key, "with_lib": with_lib,
                       "group": case["group"], "truth": case["truth"],
                       "reply": reply}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n += 1
                print(f"[{n}] {model} {key} lib={with_lib} "
                      f"-> {reply[:60]!r}")
    fout.close()
    print("new calls:", n)


if __name__ == "__main__":
    main()
