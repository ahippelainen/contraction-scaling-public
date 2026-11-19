"""k_min(Q_{1,int}) <= 3 (Theorem 1.1(d); Appendix B, paragraph 'k_min(Q_{1,int}) <= 3'). The stored table data/map_3Q1int_fullcube.txt is an integer map for 3 Q_{1,int} on the full cube {0,1}^11 in the sense of Definition 2.8: pins (2.9) at the occurrence words of the 17 parties and at 0, ranges n_r(u) in {0, ..., 3 beta_r}, the contraction condition (2.10) on all 11,264 edges (sufficient by Lemma 2.11), and, directly, on all 2,096,128 pairs of points."""
from __future__ import annotations

import sys

import numpy as np

from carriers import Params, carrier
from certificates import FULLCUBE, load
from countmap import all_pairs_full_cube, check_full_cube


def main() -> bool:
    Q = carrier(1)[0].scaled(3 * Params(1).K)
    pts, table = load(FULLCUBE, Q)
    full = np.array_equal(np.sort(pts), np.arange(1 << len(Q.lhs)))
    T = np.zeros_like(table)
    T[pts] = table
    rep = check_full_cube(Q, T, 1)
    print(f"{FULLCUBE.name}: every point of the cube once: {full}; 3*Q_1int: {rep}; values in [{T.min()}, {T.max()}]")
    pairs, viol, worst = all_pairs_full_cube(Q, T, 1)
    bad = T.copy()
    bad[(1 << len(Q.lhs)) - 2, 1] += 1
    caught = not check_full_cube(Q, bad, 1).ok
    print(f"all pairs: {pairs} pairs, {viol} violations, worst excess {worst}; one-unit perturbation caught: {caught}")
    ok = full and rep.ok and rep.points == 2048 and rep.edges == 11264 and pairs == 2_096_128 and viol == 0 and caught
    print(("PASS" if ok else "FAIL") + ": integer map for 3*Q_1int on the full cube {0,1}^11, so k_min(Q_1int) <= 3")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
