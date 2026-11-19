"""Exact checks of Definition 2.8 (pins (2.9), ranges, contraction condition (2.10)) on the full cube, for an integer table (values already multiplied by a common denominator D)."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from hec import Inequality, words_to_int


@dataclass
class CubeReport:
    points: int
    pin_failures: int
    range_failures: int
    edges: int
    edge_violations: int
    tight_edges: int
    worst_excess: Fraction

    @property
    def ok(self) -> bool:
        return self.pin_failures == 0 and self.range_failures == 0 and self.edge_violations == 0

    def __str__(self) -> str:
        return (f"points {self.points}, pin failures {self.pin_failures}, range failures {self.range_failures}, "
                f"edges {self.edges}, edge violations {self.edge_violations}, tight edges {self.tight_edges}, worst excess {self.worst_excess}")


def scaled_ints(values: list[Fraction], D: int) -> np.ndarray:
    out = []
    for v in values:
        w = v * D
        assert w.denominator == 1
        out.append(int(w))
    return np.array(out, dtype=np.int64)


def check_full_cube(Q: Inequality, table: np.ndarray, D: int) -> CubeReport:
    """table[x, r] = D * n_r(x) for every x in {0,1}^L (row index = x with bit l = term l)."""
    L = len(Q.lhs)
    assert table.shape == (1 << L, len(Q.rhs)) and table.dtype == np.int64
    a = scaled_ints(Q.alpha, D)
    b = scaled_ints(Q.beta, D)
    assert int(np.abs(table).max()) * table.shape[1] < 2**62
    X, Y = Q.occ_x(), Q.occ_y()
    xi = words_to_int(X)
    pin_fail = int((table[xi] != Y.astype(np.int64) * b[None, :]).any(axis=1).sum()) + int((table[0] != 0).any())
    range_fail = int(((table < 0) | (table > b[None, :])).any(axis=1).sum())
    xs = np.arange(1 << L, dtype=np.int64)
    viol = tight = 0
    worst: int | None = None
    for l in range(L):
        lo = xs[(xs >> l) & 1 == 0]
        d = np.abs(table[lo] - table[lo | (1 << l)]).sum(axis=1) - a[l]
        viol += int((d > 0).sum())
        tight += int((d == 0).sum())
        mx = int(d.max())
        worst = mx if worst is None else max(worst, mx)
    return CubeReport(1 << L, pin_fail, range_fail, L << (L - 1), viol, tight, Fraction(worst, D))


def all_pairs_full_cube(Q: Inequality, table: np.ndarray, D: int) -> tuple[int, int, Fraction]:
    """(pairs, violations, worst excess) of the contraction condition over all pairs of the full cube; for small L only."""
    L = len(Q.lhs)
    a = scaled_ints(Q.alpha, D)
    xs = np.arange(1 << L, dtype=np.int64)
    dal = (((xs[:, None] >> np.arange(L)) & 1) * a[None, :]).sum(1)  # d_alpha(x, 0); d_alpha(x, y) = dal[x ^ y]
    pairs = viol = 0
    worst = None
    for x in range(1 << L):
        ys = xs[x + 1 :]
        d = np.abs(table[ys] - table[x]).sum(axis=1) - dal[ys ^ x]
        pairs += len(ys)
        viol += int((d > 0).sum())
        if len(ys):
            worst = int(d.max()) if worst is None else max(worst, int(d.max()))
    return pairs, viol, Fraction(worst, D)
