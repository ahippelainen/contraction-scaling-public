"""Real(C(Q')) for a centered Q', built from Definition 2.12, and the lift of a map of Q' on Real(Q') to it by Definition 8.7 of the paper."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from hec import Inequality, realizable_full, superbalance, term_masks, words_to_int


@dataclass
class Superbalanced:
    """C(Q') laid out as: LHS = Q' LHS terms, then the pair terms S(ik), i < k, in lexicographic party order; RHS = Q' RHS terms, then the singletons S(i) in party order (parties with a_i = 0 omitted)."""

    Q: Inequality
    C: Inequality
    pairs: list[tuple[int, int]]
    a: np.ndarray  # pair coefficients a_ik (ints), in the order of pairs
    cap: np.ndarray  # singleton coefficients a_i (ints, index = party)


def layout(Q: Inequality) -> Superbalanced:
    Cgen, _ = superbalance(Q)
    q_old = Q.coefficients()
    extra = {t: v for t, v in Cgen.coefficients().items() if q_old.get(t) != v}
    assert all(Cgen.coefficients().get(t) == v for t, v in q_old.items()), "superbalancing changed a term of Q'"
    pairs = sorted(tuple(sorted(t)) for t, v in extra.items() if v > 0)
    assert all(len(p) == 2 for p in pairs) and all(len(t) == 1 for t, v in extra.items() if v < 0)
    a = np.array([int(extra[frozenset(p)]) for p in pairs], dtype=np.int64)
    cap = np.zeros(Q.n, dtype=np.int64)
    for t, v in extra.items():
        if v < 0:
            cap[next(iter(t))] = int(-v)
    lhs = Q.lhs + tuple((frozenset(p), Fraction(int(w))) for p, w in zip(pairs, a))
    rhs = Q.rhs + tuple((frozenset([i]), Fraction(int(cap[i]))) for i in range(Q.n) if cap[i] > 0)
    C = Inequality(Q.n, Q.names, lhs, rhs)
    assert C.coefficients() == Cgen.coefficients()
    return Superbalanced(Q, C, pairs, a, cap)


def lowbit(v: np.ndarray) -> np.ndarray:
    """Index of the lowest set bit of each (positive) int64."""
    return np.bitwise_count(((v & -v) - 1).astype(np.uint64)).astype(np.int64)


@dataclass
class RealC:
    x: np.ndarray  # Q'-part of the point (int, bit l = Q' term l)
    gam_lo: np.ndarray  # pair-term part, bits 0..63
    gam_hi: np.ndarray  # pair-term part, bits 64..
    kind: np.ndarray  # 0 empty, 1 single pair term, 2 fan (>= 2 leaves), 3 triangle
    center: np.ndarray  # fan / single pair term: center (single pair term: smaller party); triangle: -1
    leaves: np.ndarray  # party mask of the leaves (single pair term: the larger party); triangle: its three parties
    a_gamma: np.ndarray  # a(Gamma)


def realizable_C(S: Superbalanced) -> RealC:
    """Real(C) of Definition 2.12, taking the LHS terms in two groups: G1 = Q' terms, G2 = pair terms.
    A point is (x, Gamma) with x in Real(G1), Gamma in Real(G2), and R1/R2 for every (G1 term, G2 term) couple."""
    Q, pairs = S.Q, S.pairs
    t1 = [t for t, _ in Q.lhs]
    t2 = [frozenset(p) for p in pairs]
    n2 = len(t2)
    real1 = realizable_full(t1)
    pm = term_masks(t2, Q.n).tolist()
    # cross masks per G2 term e: G1 terms disjoint from e (R1), G1 terms containing e (R2: e on => a on)
    dis = [sum(1 << l for l, a in enumerate(t1) if not (a & e)) for e in t2]
    sup = [sum(1 << l for l, a in enumerate(t1) if e <= a) for e in t2]
    # R2 the other way (a subset of e, a on => e on), and R2 inside G2: none for a centered Q' (every G1 term contains the center, no pair does)
    assert not any(a <= e for a in t1 for e in t2) and not any(e < f for e in t2 for f in t2)
    adj = [sum(1 << f for f in range(n2) if f != e and pm[e] & pm[f]) for e in range(n2)]
    out: list[tuple[int, int, int, int, int, int, int]] = []  # (gamma, forbid, force, size, and-mask, or-mask, a(gamma))

    def grow(gamma: int, cand: int, forbid: int, force: int, size: int, andm: int, orm: int, asum: int) -> None:
        out.append((gamma, forbid, force, size, andm, orm, asum))
        while cand:
            e = (cand & -cand).bit_length() - 1
            cand &= cand - 1
            grow(gamma | 1 << e, cand & adj[e], forbid | dis[e], force | sup[e], size + 1, andm & pm[e], orm | pm[e], asum + int(S.a[e]))

    grow(0, (1 << n2) - 1, 0, 0, 0, (1 << Q.n) - 1, 0, 0)
    G = np.array([o[0] for o in out], dtype=object)
    forb = np.array([o[1] for o in out], dtype=np.int64)
    forc = np.array([o[2] for o in out], dtype=np.int64)
    size = np.array([o[3] for o in out], dtype=np.int64)
    andm = np.array([o[4] if o[3] else 0 for o in out], dtype=np.int64)
    orm = np.array([o[5] for o in out], dtype=np.int64)
    asum = np.array([o[6] for o in out], dtype=np.int64)
    kind = np.where(size == 0, 0, np.where(size == 1, 1, np.where(andm != 0, 2, 3)))
    assert np.all(np.bitwise_count(orm[kind == 3].astype(np.uint64)) == 3) and np.all(size[kind == 3] == 3)
    assert np.all(np.bitwise_count(andm[kind == 2].astype(np.uint64)) == 1)
    center = np.full(len(out), -1, dtype=np.int64)
    leaves = np.zeros(len(out), dtype=np.int64)
    center[kind == 2] = lowbit(andm[kind == 2])
    leaves[kind == 2] = orm[kind == 2] & ~(np.int64(1) << center[kind == 2])
    center[kind == 1] = lowbit(orm[kind == 1])
    leaves[kind == 1] = orm[kind == 1] & ~(np.int64(1) << center[kind == 1])
    leaves[kind == 3] = orm[kind == 3]
    keys, inv = np.unique(np.stack([forb, forc], 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(keys) + 1))
    gi_parts, x_parts = [], []
    for k, (fb, fc) in enumerate(keys):
        xs = real1[((real1 & fb) == 0) & ((real1 & fc) == fc)]
        cl = order[bounds[k] : bounds[k + 1]]
        gi_parts.append(np.repeat(cl, len(xs)))
        x_parts.append(np.tile(xs, len(cl)))
    gi = np.concatenate(gi_parts)
    x = np.concatenate(x_parts)
    srt = np.lexsort((x, gi))
    gi, x = gi[srt], x[srt]
    mask64 = (1 << 64) - 1
    lo = np.array([int(g) & mask64 for g in G], dtype=np.uint64)
    hi = np.array([int(g) >> 64 for g in G], dtype=np.uint64)
    return RealC(x, lo[gi], hi[gi], kind[gi], center[gi], leaves[gi], asum[gi])


def lift(S: Superbalanced, g: np.ndarray, P: RealC) -> tuple[np.ndarray, dict[str, bool]]:
    """The lift of g at the points P, cases (i)-(iv) of Definition 8.7, with C's RHS order; g[x, r] is the integer map of Q' (rows = all x in the cube, only rows in Real(Q') are read).
    Also returns, as diagnostics, the identity o^{u,{v}} = o^{v,{u}} stated after Definition 8.7 and the identity for theta_{u,{v}} of Proposition D.14(iii), evaluated at every single-pair-term point (the check of the lift does not depend on them)."""
    Q = S.Q
    R1 = len(Q.rhs)
    beta = np.array([int(b) for b in Q.beta], dtype=np.int64)
    alpha = np.array([int(a) for a in Q.alpha], dtype=np.int64)
    J = term_masks([t for t, _ in Q.rhs], Q.n)
    xw = words_to_int(Q.occ_x())
    Y = Q.occ_y().astype(np.int64) * beta[None, :]
    L = len(Q.lhs)
    cube = np.arange(1 << L, dtype=np.int64)
    dal = (((cube[:, None] >> np.arange(L)) & 1) * alpha[None, :]).sum(1)
    capv = S.cap
    n = len(P.x)
    out = np.zeros((n, R1 + Q.n), dtype=np.int64)  # singleton column i+R1 for party i (parties with cap 0 dropped at the end)

    def old_part(sel: np.ndarray, center: np.ndarray, leaves: np.ndarray) -> np.ndarray:
        x = P.x[sel]
        u = x & xw[center]
        inJ = ((J[None, :] >> center[:, None]) & 1) == 1
        meet = (J[None, :] & leaves[:, None]) != 0
        contains = (J[None, :] & leaves[:, None]) == leaves[:, None]
        mn = np.full((len(x), R1), np.iinfo(np.int64).max)
        for k in range(Q.n):
            has = ((leaves >> k) & 1) == 1
            if has.any():
                mn[has] = np.minimum(mn[has], g[x[has] & xw[k]])
        return np.where(inJ & ~meet, g[u], np.where(inJ & meet, beta[None, :], np.where(~inJ & contains, mn, 0)))

    def theta(sel: np.ndarray, center: np.ndarray, o: np.ndarray, a_gamma: np.ndarray) -> np.ndarray:
        x = P.x[sel]
        return a_gamma - dal[x ^ xw[center]] + np.abs(o - Y[center]).sum(1)

    s0 = np.flatnonzero(P.kind == 0)
    out[s0, :R1] = g[P.x[s0]]
    s2 = np.flatnonzero(P.kind == 2)
    o = old_part(s2, P.center[s2], P.leaves[s2])
    out[s2, :R1] = o
    out[s2, R1 + P.center[s2]] = theta(s2, P.center[s2], o, P.a_gamma[s2])
    s1 = np.flatnonzero(P.kind == 1)
    i, k = P.center[s1], lowbit(P.leaves[s1])
    o_i = old_part(s1, i, np.int64(1) << k)
    o_k = old_part(s1, k, np.int64(1) << i)
    both = (((J[None, :] >> i[:, None]) & 1) & ((J[None, :] >> k[:, None]) & 1)).astype(bool)
    x1 = P.x[s1]
    G_i = np.where(both, g[x1 & xw[i]], 0).sum(1)
    G_k = np.where(both, g[x1 & xw[k]], 0).sum(1)
    th_i = theta(s1, i, o_i, P.a_gamma[s1])
    diag = {"o^{u,{v}} = o^{v,{u}}": bool(np.array_equal(o_i, o_k)), "theta_{u,{v}} = G_uv(xi_u) - G_uv(xi_v)": bool(np.array_equal(th_i, G_i - G_k))}
    out[s1, :R1] = o_i
    out[s1, R1 + i] = np.maximum(G_i - G_k, 0)
    out[s1, R1 + k] = np.maximum(G_k - G_i, 0)
    s3 = np.flatnonzero(P.kind == 3)
    pins = np.zeros((Q.n, R1 + Q.n), dtype=np.int64)
    pins[:, :R1] = Y
    pins[np.arange(Q.n), R1 + np.arange(Q.n)] = capv
    tri = np.array([[p for p in range(Q.n) if v >> p & 1] for v in P.leaves[s3].tolist()], dtype=np.int64).reshape(-1, 3)
    pv = pins[tri]  # (points, 3, colours)
    out[s3] = pv.sum(1) - pv.max(1) - pv.min(1)
    keep = np.concatenate([np.arange(R1), R1 + np.flatnonzero(capv > 0)])
    return out[:, keep], diag
