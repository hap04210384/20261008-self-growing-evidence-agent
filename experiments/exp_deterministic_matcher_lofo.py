# -*- coding: utf-8 -*-
"""exp_deterministic_matcher_lofo.py — deterministic matcher on LOFO (CWRU).

Pairs with exp_cwru_lofo.py (same libraries, same test windows).
Deterministic rule: contained top-2-per-class patterns vote by class;
  force mode: no match -> "normal" (silent miss on unseen faults)
  abstain mode: no match -> "novel"
LLM reference (exp_cwru_lofo): 100% novel, 0% misattributed.
Output: results/deterministic_matcher/lofo_summary.json
"""
import json
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "pipeline"))

from exp_cwru_lofo import (CLASSES, FAULTS, N_TEST, SEED, load_txns,
                           mine_class_lib)

OUTDIR = os.path.join(ROOT, "results", "deterministic_matcher")


def main():
    txns, labels = load_txns()
    per_held = {}
    for held in FAULTS:
        known = [c for c in CLASSES if c != held]
        lib = mine_class_lib(txns, labels, known)
        pats = [(frozenset(p["pattern"].split(" ∧ ")),
                 max(p["sup"], key=p["sup"].get)) for p in lib]
        rng = random.Random(f"{SEED}|{held}")
        idxs = [i for i, l in enumerate(labels) if l == held]
        test = rng.sample(idxs, N_TEST)
        res = {"force_normal": 0, "abstain_novel": 0,
               "misattributed_known": 0}
        for i in test:
            w = txns[i]
            votes = [cls for s, cls in pats if s <= w]
            if not votes:
                res["force_normal"] += 1          # force mode: -> normal
                res["abstain_novel"] += 1         # abstain mode: -> novel
            else:
                pred = max(set(votes), key=votes.count)
                res["misattributed_known"] += 1   # forced into a known class
        per_held[held] = {k: round(v / N_TEST, 3) for k, v in res.items()}
    out = {"per_heldout": per_held,
           "note": "LLM reference (exp_cwru_lofo): novel=1.00, "
                   "misattributed=0.00 in all 15 cells"}
    with open(os.path.join(OUTDIR, "lofo_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for h, r in per_held.items():
        print(h, r)


if __name__ == "__main__":
    main()
