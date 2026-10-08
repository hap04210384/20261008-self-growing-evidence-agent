# -*- coding: utf-8 -*-
"""Replicate a transaction dataset N times (for scalability benchmarks).

Usage:
    python replicate.py <input.txt> <factor> <output.txt>

Example:
    python replicate.py accidents.txt 4 accidents_x4.txt
    python replicate.py pumsb.txt 8 pumsb_x8.txt

Replication preserves the mined MFI family (each transaction is duplicated
verbatim), so absolute support counts scale by exactly <factor> while the
output itemsets stay identical -- this is the replication invariance checked
in Section V-D of the paper.
"""
import sys

def main():
    src, factor, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    with open(src, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for _ in range(factor):
            for ln in lines:
                f.write(ln + "\n")
    print(f"{dst}: {len(lines) * factor} transactions ({len(lines)} x {factor})")

if __name__ == "__main__":
    main()
