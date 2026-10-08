# -*- coding: utf-8 -*-
"""exp_llm_budget.py —— 证据预算消融（Figure 7 数据源）。

设计（frozen，全部写进本文件以便审稿复现）：
- 任务：与 exp_llm_diagnosis.py 完全相同的 60 个诊断窗口（seed=42）
- 变量：附着到提示词的模式库条数 k ∈ {1, 2, 3, 5, 8}
  （k=0 即无证据条件，直接复用 results/llm_diag/calls.jsonl 中
   with_lib=false 的 60×3=180 条记录，不重复调用）
- 排序口径：模式按 lift 降序截断 top-k（与手稿 3.5/4.5 的
  "ranked by lift" 口径一致；lift 是 anomalous support 的严格
  单调函数，故与引擎输出顺序一致）
- 模型：deepseek-chat / qwen-plus / glm-4-air（temperature=0）
- 产物：results/llm_diag/budget.jsonl（逐次调用，断点续跑）
- 用法：python exp_llm_budget.py [model]   （缺省跑全部）
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_llm_diagnosis import (MODELS, GROUPS, build_cases, ask, fmt_lift)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "llm_diag", "budget.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

KS = (1, 2, 3, 5, 8)  # k=0 复用 calls.jsonl


def user_prompt_k(case, k):
    """与 exp_llm_diagnosis.user_prompt(case, True) 相同的版式，
    仅模式库截断为 lift 降序 top-k。"""
    w = " ∧ ".join(case["window"])
    p = ("Context: an industrial water-pump testbed with 8 sensors "
         "(Accelerometer1, Accelerometer2, Current, Pressure, Temperature, "
         "Thermocouple, Voltage, Volume Flow Rate). Each 30-sample window is "
         "discretized into states 0/1/2 per sensor (0=low, 1=mid, 2=high)."
         f"\n\nWindow to diagnose:\n{w}")
    rows = "\n".join(
        f"- {r['pattern']}  (support in faulty windows: {r['sup_anom']}, "
        f"lift: {fmt_lift(r['lift'])})" for r in case["library"][:k])
    p += ("\n\nMined pattern library of this pump (counted from its own "
          "history):\n" + rows)
    p += "\n\nDiagnose this window."
    return p


def main():
    only_model = sys.argv[1] if len(sys.argv) > 1 else None
    cases = build_cases()
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["model"], r["case_key"], r["k"]))
    fout = open(OUT, "a", encoding="utf-8")
    n = 0
    for model in MODELS:
        if only_model and model != only_model:
            continue
        for case in cases:
            key = f"{case['group']}#{case['idx']}"
            for k in KS:
                if (model, key, k) in done:
                    continue
                reply = ask(model, user_prompt_k(case, k))
                rec = {"model": model, "case_key": key, "k": k,
                       "group": case["group"], "truth": case["truth"],
                       "reply": reply}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n += 1
                print(f"[{n}] {model} {key} k={k} -> {reply[:60]!r}")
    fout.close()
    print("new calls:", n)


if __name__ == "__main__":
    main()
