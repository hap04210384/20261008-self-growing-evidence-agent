# -*- coding: utf-8 -*-
"""exp_deterministic_matcher.py — deterministic library-matching baseline.

Reviewer 2 #4 / Reviewer 3 ("is the LLM needed at all"):
a purely deterministic classifier replaces the LLM:
window -> contained library patterns -> majority-vote fault attribution.
No model, no API, no randomness.

Protocols (identical windows to the LLM runs, keyed by case_key):
- 60win: keys of results/llm_diag/calls_60win.jsonl
- 198win: keys of results/llm_diag/calls.jsonl

Rule variants:
- strict: any contained top-8 library pattern -> faulty, else normal
- vote:  >=2 contained patterns -> faulty, else normal
Output: results/deterministic_matcher/summary.json
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "fimi")
OUTDIR = os.path.join(ROOT, "results", "deterministic_matcher")
os.makedirs(OUTDIR, exist_ok=True)

GROUPS = ("valve1", "valve2", "other")
TOPK = 8


def load_group(grp):
    items = json.load(open(os.path.join(DATA, f"skab_{grp}.txt.items.json"),
                           encoding="utf-8"))
    id2name = {v: k for k, v in items.items()}
    txns = []
    with open(os.path.join(DATA, f"skab_{grp}.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                txns.append(frozenset(id2name[int(t)] for t in line.split()))
    labels = json.load(open(os.path.join(DATA, f"skab_{grp}.txt.labels.json"),
                            encoding="utf-8"))
    lib = []
    with open(os.path.join(ROOT, "results", "spike_v1",
                           f"patterns_{grp}.csv"), encoding="utf-8-sig") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 5:
                pat = frozenset(x.strip() for x in parts[0].split("∧")
                                if x.strip())
                lib.append({"pattern": pat, "sup_anom": float(parts[2]),
                            "lift": float(parts[4]) if parts[4] else 0.0})
    lib.sort(key=lambda r: -r["sup_anom"])
    return txns, labels, lib[:TOPK]


def keys_of(calls_file):
    keys = set()
    with open(calls_file, encoding="utf-8") as f:
        for line in f:
            keys.add(json.loads(line)["case_key"])
    out = {}
    for k in keys:
        g, i = k.split("#")
        out.setdefault(g, []).append(int(i))
    return out


def run_protocol(name, keymap):
    per_group = {}
    detail = []
    for grp in GROUPS:
        txns, labels, lib = load_group(grp)
        correct = {"strict": 0, "vote": 0}
        n = 0
        nomatch_faulty = 0
        nomatch_normal = 0
        for i in sorted(keymap.get(grp, [])):
            w = txns[i]
            truth = "faulty" if labels[i] == "1" else "normal"
            m = sum(1 for p in lib if p["pattern"] <= w)
            pred_strict = "faulty" if m >= 1 else "normal"
            pred_vote = "faulty" if m >= 2 else "normal"
            correct["strict"] += pred_strict == truth
            correct["vote"] += pred_vote == truth
            n += 1
            if m == 0:
                if truth == "faulty":
                    nomatch_faulty += 1
                else:
                    nomatch_normal += 1
            detail.append({"case_key": f"{grp}#{i}", "group": grp,
                           "truth": truth, "n_match": m,
                           "pred_strict": pred_strict,
                           "pred_vote": pred_vote})
        per_group[grp] = {
            "n": n,
            "acc_strict": round(correct["strict"] / n, 4) if n else None,
            "acc_vote": round(correct["vote"] / n, 4) if n else None,
            "no_match_on_faulty": nomatch_faulty,
            "no_match_on_normal": nomatch_normal,
        }
    tot = sum(g["n"] for g in per_group.values())
    accs = {}
    for rule in ("strict", "vote"):
        accs[rule] = round(sum(g["acc_" + rule] * g["n"]
                               for g in per_group.values()) / tot, 4)
    return {"name": name, "n_windows": tot, "overall": accs,
            "per_group": per_group, "detail": detail}


def main():
    results = []
    for name, calls in (("60win", "calls_60win.jsonl"),
                        ("198win", "calls.jsonl")):
        km = keys_of(os.path.join(ROOT, "results", "llm_diag", calls))
        results.append(run_protocol(name, km))
    summary = {"protocols": results}
    with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    for r in results:
        print(r["name"], "n=%d" % r["n_windows"], r["overall"])
        for g, v in r["per_group"].items():
            print("  ", g, v)
    print("saved", os.path.join(OUTDIR, "summary.json"))


if __name__ == "__main__":
    main()
