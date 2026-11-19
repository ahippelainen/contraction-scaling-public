"""Loader of the two stored integer maps in data/ (plain text).

File format: header lines '# lhs l coefficient parties' and '# rhs r coefficient parties' list the terms of 3*Q_{1,int} in the order of the columns and bits; each data line is a point u as a string of L bits, character l = the bit of LHS term l (the string lists the bits in LHS term order, starting with term 0), followed by the counts n_r(u) for r = 0..R-1.
For the files shipped here the order is that of the display after Theorem 1.1: bits l = 0..4 the box terms of b = 0..4, l = 5..9 the star terms of v = 0..4, l = 10 the term A q_B; columns r = 0 the term F, r = 1..20 the claw terms F - {q_w, p_T} with T = 01, 02, ..., 34 lexicographic and w in T increasing, r = 21 the term A.
The loader compares the header with the terms that carriers.carrier builds from the definition, and rejects a file in any other order. The conditions of a count map are checked by full_cube_map.py and lifted_map.py."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from hec import Inequality

DATA = Path(__file__).resolve().parent / "data"
FULLCUBE = DATA / "map_3Q1int_fullcube.txt"
REAL = DATA / "map_3Q1int_real.txt"


def load(path: Path, Q: Inequality) -> tuple[np.ndarray, np.ndarray]:
    """(points as ints with bit l = LHS term l, counts as an int64 array points x R) of a stored map for the inequality Q."""
    lhs, rhs, pts, rows = [], [], [], []
    with open(path) as f:
        for line in f:
            w = line.split()
            if line.startswith("# lhs ") or line.startswith("# rhs "):
                (lhs if w[1] == "lhs" else rhs).append((frozenset(w[4:]), Fraction(w[3])))
            elif not line.startswith("#"):
                pts.append(int(w[0][::-1], 2))
                rows.append([int(v) for v in w[1:]])
    named = lambda side: [(frozenset(Q.names[p] for p in t), a) for t, a in side]  # noqa: E731
    assert lhs == named(Q.lhs) and rhs == named(Q.rhs), f"{path.name}: term order differs from the inequality"
    table = np.array(rows, dtype=np.int64)
    assert table.shape == (len(pts), len(Q.rhs)) and len(set(pts)) == len(pts)
    return np.array(pts, dtype=np.int64), table
