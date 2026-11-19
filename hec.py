"""Inequalities in purified form (2.1), Bell values (2.5), the superbalancing of Definition 7.1 and the realizable domain of Definition 2.12, written from the definitions of the paper."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import lcm

import numpy as np

Term = frozenset[int]


@dataclass(frozen=True)
class Inequality:
    """sum_l alpha_l S(I_l) >= sum_r beta_r S(J_r) on parties 0..n-1, purifier implicit (no term contains it)."""

    n: int
    names: tuple[str, ...]
    lhs: tuple[tuple[Term, Fraction], ...]
    rhs: tuple[tuple[Term, Fraction], ...]

    @property
    def alpha(self) -> list[Fraction]:
        return [a for _, a in self.lhs]

    @property
    def beta(self) -> list[Fraction]:
        return [b for _, b in self.rhs]

    def scaled(self, k: Fraction | int) -> Inequality:
        k = Fraction(k)
        return Inequality(self.n, self.names, tuple((t, k * a) for t, a in self.lhs), tuple((t, k * b) for t, b in self.rhs))

    def coefficients(self) -> dict[Term, Fraction]:
        q: dict[Term, Fraction] = {}
        for t, a in self.lhs:
            q[t] = q.get(t, Fraction(0)) + a
        for t, b in self.rhs:
            q[t] = q.get(t, Fraction(0)) - b
        return {t: v for t, v in q.items() if v != 0}

    def occ_x(self) -> np.ndarray:
        """Occurrence words x^(i): array (n, L) of 0/1."""
        return membership(self.n, [t for t, _ in self.lhs])

    def occ_y(self) -> np.ndarray:
        return membership(self.n, [t for t, _ in self.rhs])


def from_coefficients(n: int, names: tuple[str, ...], q: dict[Term, Fraction]) -> Inequality:
    lhs = tuple(sorted(((t, v) for t, v in q.items() if v > 0), key=lambda tv: (len(tv[0]), sorted(tv[0]))))
    rhs = tuple(sorted(((t, -v) for t, v in q.items() if v < 0), key=lambda tv: (len(tv[0]), sorted(tv[0]))))
    return Inequality(n, names, lhs, rhs)


def membership(n: int, terms: list[Term]) -> np.ndarray:
    m = np.zeros((n, len(terms)), dtype=np.int8)
    for l, t in enumerate(terms):
        m[list(t), l] = 1
    return m


def purify(t: Term, n: int, with_purifier: bool) -> Term:
    """Purified form of the set t (or t ∪ {O} if with_purifier): a set containing O is replaced by its complement."""
    return frozenset(range(n)) - t if with_purifier else t


def common_denominator(values) -> int:
    d = 1
    for v in values:
        d = lcm(d, Fraction(v).denominator)
    return d


def bell_matrix(Q: Inequality) -> np.ndarray:
    """Bell values Q.Bell_ij for all i, j in parties ∪ {O} (index n = O), as an object array of Fractions.

    Bell_ij(I) = 1 iff I contains exactly one of i, j; terms are purified, so O lies in none. Integer arithmetic on D * coefficients."""
    q = Q.coefficients()
    terms = list(q)
    D = common_denominator(q.values())
    w = np.array([int(q[t] * D) for t in terms], dtype=np.int64)
    assert int(np.abs(w).sum()) < 2**61
    P = membership(Q.n, terms).astype(np.int64)
    bO = P @ w
    G = (P * w) @ P.T
    full = np.zeros((Q.n + 1, Q.n + 1), dtype=np.int64)
    full[: Q.n, : Q.n] = bO[:, None] + bO[None, :] - 2 * G
    full[: Q.n, Q.n] = bO
    full[Q.n, : Q.n] = bO
    np.fill_diagonal(full, 0)
    out = np.empty(full.shape, dtype=object)
    for idx, v in np.ndenumerate(full):
        out[idx] = Fraction(int(v), D)
    return out


def superbalance(Q: Inequality) -> tuple[Inequality, np.ndarray]:
    """C(Q) = Q - 1/2 sum_{u<v} b_uv I(u:v) over u, v in parties ∪ {O}, eq. (7.2); returns C and the Bell matrix of Q."""
    n = Q.n
    b = bell_matrix(Q)
    q = dict(Q.coefficients())
    full = frozenset(range(n))

    def add(t: Term, v: Fraction) -> None:
        q[t] = q.get(t, Fraction(0)) + v

    for u in range(n + 1):
        for v in range(u + 1, n + 1):
            h = b[u, v] / 2
            if h == 0:
                continue
            if v == n:  # I(u:O) = S(u) + S(O) - S(uO), S(O) = S(all parties), S(uO) = S(all - u)
                add(frozenset([u]), -h)
                add(full, -h)
                add(purify(frozenset([u]), n, True), h)
            else:
                add(frozenset([u]), -h)
                add(frozenset([v]), -h)
                add(frozenset([u, v]), h)
    q = {t: v for t, v in q.items() if v != 0}
    return from_coefficients(n, Q.names, q), b


# ---------- bitmask helpers (cube points as integers, bit l = term l) ----------

def term_masks(terms: list[Term], n: int) -> np.ndarray:
    """Each term as an n-bit party mask (n <= 63)."""
    assert n <= 63
    return np.array([sum(1 << p for p in t) for t in terms], dtype=np.int64)


def realizable_full(terms: list[Term]) -> np.ndarray:
    """All x in {0,1}^L (as ints, bit l = term l) satisfying (R1) and (R2) of Definition 2.12; for L <= 26."""
    L = len(terms)
    xs = np.arange(1 << L, dtype=np.int64)
    ok = np.ones(xs.shape, dtype=bool)
    for l in range(L):
        on = (xs >> l) & 1 == 1
        disjoint = sum(1 << k for k in range(L) if k != l and not (terms[l] & terms[k]))
        sup = sum(1 << k for k in range(L) if k != l and terms[l] <= terms[k])
        ok &= ~on | (((xs & disjoint) == 0) & ((xs & sup) == sup))
    return xs[ok]


def words_to_int(words: np.ndarray) -> np.ndarray:
    L = words.shape[1]
    return (words.astype(np.int64) << np.arange(L, dtype=np.int64)).sum(axis=1)


def popcount64(a: np.ndarray) -> np.ndarray:
    return np.bitwise_count(a.astype(np.uint64)).astype(np.int64)
