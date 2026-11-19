"""The carrier Q_j = Q(T_j) of the family T_j of all (j+1)-subsets of B = {0, ..., 3j+1} and its water-level map, written from Definitions 4.2 and 4.8 of the paper."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import comb

import numpy as np

from hec import Inequality, common_denominator


@dataclass(frozen=True)
class Params:
    """m = 3j+2 box points, blocks of t = j+1 points, and the coefficients c, ws = omega, e = c_A of the carrier."""

    j: int

    @property
    def m(self) -> int:
        return 3 * self.j + 2

    @property
    def t(self) -> int:
        return self.j + 1

    @property
    def c(self) -> Fraction:
        m = self.m
        return Fraction(m - 1, m * comb(m - 1, 2 * self.j + 1))

    @property
    def ws(self) -> Fraction:
        return Fraction(self.m, self.t) - 1 - self.c

    @property
    def e(self) -> Fraction:
        return self.m * self.ws + Fraction(1, self.m)

    @property
    def n_parties(self) -> int:
        return 2 + self.m + comb(self.m, self.t)

    def coefficient_values(self) -> list[Fraction]:
        return [Fraction(1), self.ws, Fraction(1, self.m), Fraction(1), self.c, self.e]

    @property
    def K(self) -> int:
        """K_j of the paper: the least positive integer for which K_j * Q_j has integer coefficients, i.e. the lcm of the coefficients' denominators."""
        return common_denominator(self.coefficient_values())


def carrier(j: int) -> tuple[Inequality, list[tuple[int, frozenset[int]]]]:
    """Q_j (unscaled), parties A, H, q_0..q_{m-1}, p_T (T lexicographic), in the order of the display after Theorem 1.1 for j = 1.
    LHS term l: l < m the box term of b = l, m <= l < 2m the star term of v = l - m, l = 2m the term A q_B (the bit layout x = (S, U, eps) of Lemma 4.5(iii), used for every cube point below).
    RHS term r: r = 0 the term F (all parties), r = 1..|claws| the claw terms F - {q_w, p_T} for T lexicographic and w in T increasing, last the term A.
    Returns the inequality and the list of claw labels (w, T) in RHS order."""
    P = Params(j)
    m, t = P.m, P.t
    Ts = [frozenset(T) for T in combinations(range(m), t)]
    sep = "" if m <= 10 else "_"
    names = ["A", "H"] + [f"q{v}" for v in range(m)] + ["p" + sep.join(map(str, sorted(T))) for T in Ts]
    A, H = 0, 1
    q = [2 + v for v in range(m)]
    p = {T: 2 + m + i for i, T in enumerate(Ts)}
    allp = frozenset(range(len(names)))
    box = [frozenset({A, H} | {q[v] for v in range(m) if v != b} | {p[T] for T in Ts if b in T}) for b in range(m)]
    star = [frozenset({A} | {p[T] for T in Ts if v in T}) for v in range(m)]
    Qt = frozenset({A} | set(q))
    lhs = [(s, Fraction(1)) for s in box] + [(s, P.ws) for s in star] + [(Qt, Fraction(1, m))]
    claws = [(w, T) for T in Ts for w in sorted(T)]
    rhs = [(allp, Fraction(1))] + [(allp - {q[w], p[T]}, P.c) for w, T in claws] + [(frozenset({A}), P.e)]
    return Inequality(len(names), tuple(names), tuple(lhs), tuple(rhs)), claws


def map_M(j: int, xs: np.ndarray) -> tuple[np.ndarray, list[Fraction]]:
    """The water-level map of Definition 4.8 for T_j at the cube points xs (ints, bit l = LHS term l of carrier(j)).

    Returns (idx, vals): an int array (len(xs), R) of indices into the list vals of Fractions, so that M_r(x) = vals[idx[x, r]].
    lambda is computed exactly per distinct (s, q, u, class counts) key; points with equal keys have equal values on each edge class."""
    P = Params(j)
    m, c, ws = P.m, P.c, P.ws
    kappa = Fraction(1, m - 1)
    _, claws = carrier(j)
    nE = len(claws)
    xs = np.asarray(xs, dtype=np.int64)
    Sbits = (xs[:, None] >> np.arange(m)) & 1
    Ubits = (xs[:, None] >> (m + np.arange(m))) & 1
    qb = (xs >> (2 * m)) & 1
    s, u = Sbits.sum(1), Ubits.sum(1)
    Tmask = np.array([sum(1 << v for v in T) for _, T in claws], dtype=np.int64)
    wcol = np.array([w for w, _ in claws])
    Smask = (Sbits << np.arange(m)).sum(1)
    Umask = (Ubits << np.arange(m)).sum(1)
    w_in_S = Sbits[:, wcol].astype(bool)
    S_sub_T = (Smask[:, None] & ~Tmask[None, :]) == 0
    U_sub_T = (Umask[:, None] & ~Tmask[None, :]) == 0
    # claw class: 0 = w not in S, 1 = w in S and S subset T (not available), 2 = available at S; +3 if open at U (U not subset T). The ground and the height depend on the class only.
    cls = np.where(~w_in_S, 0, np.where(S_sub_T, 1, 2)) + 3 * (~U_sub_T)
    counts = np.stack([(cls == k).sum(1) for k in range(6)], axis=1)
    keys = np.concatenate([s[:, None], qb[:, None], u[:, None], counts], axis=1)
    ukeys, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()

    # |Av(S)| / |claws| by direct count, for one representative S of each size (T_j is invariant under all permutations of B)
    pbox = {}
    for ss in range(m + 1):
        S = set(range(ss))
        pbox[ss] = Fraction(sum(1 for w, T in claws if w in S and not S <= T), nE)

    def b_s(ss: int) -> Fraction:  # gamma(S) / c
        return Fraction(0) if ss <= 1 else Fraction(ss - 1) / ((m - 1) * pbox[ss])

    vals: list[Fraction] = []
    index: dict[Fraction, int] = {}

    def vid(v: Fraction) -> int:
        if v not in index:
            index[v] = len(vals)
            vals.append(v)
        return index[v]

    R = 2 + nE
    key_idx = np.zeros((len(ukeys), 8), dtype=np.int64)  # F, claw value for classes 0..5, E
    for kk, key in enumerate(ukeys):
        ss, qq, uu = int(key[0]), int(key[1]), int(key[2])
        cnt = [int(v) for v in key[3:]]
        alpha = ss + ws * uu + Fraction(qq, m)
        beta_p = (1 - Fraction(qq, m)) * b_s(ss) + kappa * qq
        Bv = [Fraction(0), kappa * qq, beta_p] * 2
        sigma = ws * uu if ss >= 1 else max(ws * uu + Fraction(qq, m) - 1, Fraction(0))
        lam = water_level(c, [(Bv[k], cnt[k]) for k in range(3, 6)], sigma)
        h = [Bv[k] if k < 3 else max(Bv[k], lam) for k in range(6)]
        E = c * sum(cnt[k] * max(h[k] - 1, Fraction(0)) for k in range(6))
        key_idx[kk, 0] = vid(min(Fraction(1), alpha))
        for k in range(6):
            key_idx[kk, 1 + k] = vid(c * min(Fraction(1), h[k]))
        key_idx[kk, 7] = vid(E)
    idx = np.empty((len(xs), R), dtype=np.int64)
    idx[:, 0] = key_idx[inv, 0]
    idx[:, 1 : 1 + nE] = np.take_along_axis(key_idx[inv, 1:7], cls, axis=1)
    idx[:, R - 1] = key_idx[inv, 7]
    return idx, vals


def water_level(c: Fraction, levels: list[tuple[Fraction, int]], sigma: Fraction) -> Fraction:
    """Least lam >= 0 with c * sum_k n_k (lam - B_k)^+ >= sigma."""
    if sigma <= 0:
        return Fraction(0)
    pts = sorted({B for B, n in levels if n > 0})
    if not pts:
        raise AssertionError("positive demand with no open claw")
    for i, B in enumerate(pts):
        nxt = pts[i + 1] if i + 1 < len(pts) else None
        slope = c * sum(n for b, n in levels if b <= B)
        base = c * sum(n * (B - b) for b, n in levels if b <= B)
        lam = B + (sigma - base) / slope
        if lam >= B and (nxt is None or lam <= nxt):
            return max(lam, Fraction(0))
    raise AssertionError("unreachable")
