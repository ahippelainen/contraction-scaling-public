"""An independent check that the paper's proofs do not use: the water-level map of Definition 4.8 is a real map for the carriers Q_1 on {0,1}^11 and Q_2 on {0,1}^17, checked exactly. For j = 2 this is a case of Theorem 5.7 with Proposition 6.1; Theorem 5.7 does not cover j = 1 (end of Section 5.3), where the paper rests on the two stored maps instead.

D_j is the lcm of the denominators of all values of the water-level map of the unscaled Q_j over all points of the full cube, compared with the recorded values 10,800 and 281,707,272,000. D_j * alpha and D_j * beta are integers, so the check runs on integers."""
from __future__ import annotations

import sys

import numpy as np

from carriers import carrier, map_M
from countmap import check_full_cube, scaled_ints
from hec import common_denominator

EXPECTED_D = {1: 10_800, 2: 281_707_272_000}


def main() -> bool:
    ok_all = True
    for j, expected in EXPECTED_D.items():
        Q, _ = carrier(j)
        L = len(Q.lhs)
        idx, vals = map_M(j, np.arange(1 << L, dtype=np.int64))
        D = common_denominator(vals[i] for i in np.unique(idx))
        table = scaled_ints(vals, D)[idx]
        rep = check_full_cube(Q, table, D)
        # a one-unit change of one value at a point that is not pinned must be caught
        bad = table.copy()
        bad[(1 << L) - 2, 1] += 1
        caught = not check_full_cube(Q, bad, D).ok
        all_tight = rep.tight_edges == rep.edges
        print(f"j={j}: L={L}, R={len(Q.rhs)}, D_{j} = {D} (recorded: {expected}); {rep}; every edge tight: {all_tight}; perturbation caught: {caught}", flush=True)
        ok_all &= rep.ok and all_tight and caught and D == expected
    print(("PASS" if ok_all else "FAIL") + ": the water-level map is a real map for Q_1 and Q_2 on the full cube, every edge tight (independent check, outside the paper's proofs)")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
