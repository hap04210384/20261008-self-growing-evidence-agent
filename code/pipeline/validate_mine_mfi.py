# -*- coding: utf-8 -*-
"""validate_mine_mfi.py —— 修复后重验 mine_mfi 与两个引擎 gold 的一致性。

1) chess@0.8 vs TensorFIM gold（位精确）；
2) chess AnyFIM expected（0.95 阈值档）vs mine_mfi(count 语义)。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import skab_spike as spike

TF_GOLD = os.path.join(ROOT, "code", "engine_tensorfim", "data", "gold_results",
                       "chess.txt-0.800000=Results.txt")
AF_EXP = os.path.join(ROOT, "code", "engine_anyfim", "results", "expected",
                      "chess.txt-0.950000=Results.txt")
CHESS = os.path.join(ROOT, "code", "engine_tensorfim", "data", "small",
                     "chess.txt")


def load_fimi_txns(path):
    txns = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = ln.strip()
        if ln:
            txns.append(tuple(int(x) for x in ln.split()))
    return txns


def parse_results_sets(path):
    """引擎 Results.txt -> set(frozenset(int items))"""
    import re
    pat = re.compile(r"^\d+th:\s+\d+\s+\{\s*([^}]*)\}")
    out = set()
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = pat.match(ln.strip())
        if m:
            out.add(frozenset(int(x) for x in m.group(1).split()))
    return out


def main():
    txns = load_fimi_txns(CHESS)
    n = len(txns)
    print(f"chess: {n} txns")

    # 1) TensorFIM gold @0.8
    gold = parse_results_sets(TF_GOLD)
    mfis, _ = spike.mine_mfi(txns, min_sup=0.8)
    mine = {frozenset(p) for p, _ in mfis}
    print(f"TF gold@0.8: {len(gold)}  mine: {len(mine)}  identical={gold == mine}")
    if gold != mine:
        print("  only gold:", len(gold - mine), " only mine:", len(mine - gold))

    # 2) AnyFIM expected @0.95
    af = parse_results_sets(AF_EXP)
    mc = max(1, int(round(0.95 * n)))
    mfis2, _ = spike.mine_mfi(txns, min_count=mc)
    mine2 = {frozenset(p) for p, _ in mfis2}
    print(f"AF exp@0.95(mc={mc}): {len(af)}  mine: {len(mine2)}  identical={af == mine2}")
    if af != mine2:
        only_af = af - mine2
        only_mn = mine2 - af
        print("  only anyfim:", len(only_af), [sorted(x) for x in list(only_af)[:4]])
        print("  only mine  :", len(only_mn), [sorted(x) for x in list(only_mn)[:4]])


if __name__ == "__main__":
    main()
