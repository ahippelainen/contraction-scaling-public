"""The data of Q_{1,int} and C(Q_{1,int}) printed after Theorem 1.1, in Section 4.1, in Appendix F (eq. (F.1)) and in Table 1, computed from the definitions (Appendix B, paragraph 'The data of Q_{1,int} and C(Q_{1,int})').

Q_{1,int} = K_1 Q_1 is built by carriers.carrier from Definition 4.2 and compared term by term with the display after Theorem 1.1, which is transcribed below; C(Q_{1,int}) is built by hec.superbalance from Definition 7.1. The pair coefficients a_uv and the singleton coefficients a_u are read off the coefficients of C, not from the Bell values.
Usage: seventeen_party_data.py [--all]  (--all also prints a_uv for each of the 120 pairs)."""
from __future__ import annotations

import sys
from fractions import Fraction
from itertools import combinations

from carriers import Params, carrier
from hec import bell_matrix, superbalance

# Table 1: class of pair -> (number of pairs, Bell value of Q_{1,int}, a_uv)
TABLE = {
    "H, q_v": (5, 4, 2),
    "H, p_T": (10, 52, 26),
    "q_v, q_w": (10, 4, 2),
    "q_v, p_T, v in T": (20, 60, 30),
    "q_v, p_T, v not in T": (30, 36, 18),
    "p_T, p_T', |T cap T'| = 1": (30, 38, 19),
    "p_T, p_T', T cap T' empty": (15, 84, 42),
}
TOTALS = {"A": 117, "H": 50, "q": 42, "p": 46}
A_U = {"H": 270, "q": 238, "p": 380}


def display_Q1int() -> tuple[list[tuple[frozenset[str], int]], list[tuple[frozenset[str], int]]]:
    """The terms of the display after Theorem 1.1 in its order, as (set of party names, coefficient)."""
    B = range(5)
    Ts = list(combinations(B, 2))
    p = lambda T: "p" + "".join(map(str, T))  # noqa: E731
    P = {v: {p(T) for T in Ts if v in T} for v in B}
    F = frozenset({"A", "H"} | {f"q{v}" for v in B} | {p(T) for T in Ts})
    lhs = [(frozenset({"A", "H"} | {f"q{v}" for v in B if v != b} | P[b]), 10) for b in B]
    lhs += [(frozenset({"A"} | P[v]), 13) for v in B]
    lhs += [(frozenset({"A"} | {f"q{v}" for v in B}), 2)]
    rhs = [(F, 10)] + [(F - {f"q{w}", p(T)}, 2) for T in Ts for w in T] + [(frozenset({"A"}), 67)]
    return lhs, rhs


def pair_class(a: str, b: str) -> str:
    a, b = sorted((a, b), key=lambda s: ("Hqp".index(s[0]), s))
    if a == "H":
        return "H, q_v" if b[0] == "q" else "H, p_T"
    if b[0] == "q":
        return "q_v, q_w"
    if a[0] == "q":
        return "q_v, p_T, v in T" if a[1] in b[1:] else "q_v, p_T, v not in T"
    return "p_T, p_T', |T cap T'| = 1" if set(a[1:]) & set(b[1:]) else "p_T, p_T', T cap T' empty"


def main(show_all: bool) -> bool:
    P = Params(1)
    d = sum(1 for T in combinations(range(P.m), P.t) if 0 in T)
    ok = (d, P.c, P.ws, P.e) == (4, Fraction(1, 5), Fraction(13, 10), Fraction(67, 10))
    print(f"j = 1, all 10 two-element subsets of a 5-point box: d = {d}, c = {P.c}, omega = {P.ws}, c_A = {P.e}; Section 4.1: 4, 1/5, 13/10, 67/10")
    Q = carrier(1)[0].scaled(P.K)
    n, names = Q.n, Q.names
    ok &= P.K == 10 and n == 17 and (len(Q.lhs), len(Q.rhs)) == (11, 22)
    named = lambda side: [(frozenset(names[i] for i in t), a) for t, a in side]  # noqa: E731
    lhs_disp, rhs_disp = display_Q1int()
    same_disp = named(Q.lhs) == lhs_disp and named(Q.rhs) == rhs_disp
    ok &= same_disp
    print(f"Q_1int = {P.K} Q_1: N = {n}, L = {len(Q.lhs)}, R = {len(Q.rhs)}; equal to the display after Theorem 1.1 term by term, with its coefficients 10, 13, 2 and 10, 2, 67: {same_disp}")
    in_all = all(0 in t for t, _ in Q.lhs + Q.rhs)
    tot_l = [sum(int(a) for t, a in Q.lhs if i in t) for i in range(n)]
    tot_r = [sum(int(b) for t, b in Q.rhs if i in t) for i in range(n)]
    tot_ok = tot_l == tot_r and all(tot_l[i] == TOTALS[names[i][0]] for i in range(n))
    print(f"every term contains A: {in_all}; total coefficient of each party, LHS = RHS: {tot_l == tot_r}; A {tot_l[0]}, H {tot_l[1]}, q_v {sorted(set(tot_l[2:7]))}, p_T {sorted(set(tot_l[7:]))}")
    ok &= in_all and tot_ok

    b = bell_matrix(Q)  # index n is the purifier O
    pairs = [(i, k) for i in range(n + 1) for k in range(i + 1, n + 1)]
    zero_AO = all(b[i, k] == 0 for i, k in pairs if i == 0 or k == n)
    even = all(b[i, k].denominator == 1 and b[i, k].numerator % 2 == 0 for i, k in pairs)
    print(f"Bell values on all {len(pairs)} pairs of the {n} parties and O: zero on every pair with A or O: {zero_AO}; all even integers: {even}")
    ok &= len(pairs) == 153 and zero_AO and even

    C, _ = superbalance(Q)
    cq = C.coefficients()
    integral = all(v.denominator == 1 for v in cq.values())
    superbal = all(v == 0 for v in bell_matrix(C).ravel())
    print(f"C(Q_1int): L = {len(C.lhs)}, R = {len(C.rhs)}, integer coefficients: {integral}, all Bell values zero: {superbal}")
    ok &= (len(C.lhs), len(C.rhs)) == (131, 38) and integral and superbal

    found: dict[str, list[tuple[str, Fraction, Fraction]]] = {k: [] for k in TABLE}
    for i in range(1, n):
        for k in range(i + 1, n):
            auv = cq.get(frozenset([i, k]), Fraction(0))
            found[pair_class(names[i], names[k])].append((f"{names[i]} {names[k]}", b[i, k], auv))
            ok &= auv == b[i, k] / 2 > 0
    print(f"{'pair class':28s} {'pairs':>5s} {'Bell':>6s} {'a_uv':>6s}   Table 1: pairs, Bell, a_uv")
    for cls, (cnt, bell, auv) in TABLE.items():
        got = found[cls]
        bells, auvs = {x[1] for x in got}, {x[2] for x in got}
        same = len(got) == cnt and bells == {bell} and auvs == {auv}
        ok &= same
        print(f"{cls:28s} {len(got):5d} {'/'.join(map(str, sorted(bells))):>6s} {'/'.join(map(str, sorted(auvs))):>6s}   {cnt}, {bell}, {auv}: {'equal' if same else 'DIFFERENT'}")
    au = {names[i]: int(-cq.get(frozenset([i]), Fraction(0))) for i in range(1, n)}
    au_ok = all(au[nm] == A_U[nm[0]] for nm in au) and len(au) == 16
    print(f"singleton coefficients a_u (coefficient -a_u of S(u) in C): H {au['H']}, q_v {sorted(set(au[f'q{v}'] for v in range(5)))}, p_T {sorted(set(v for nm, v in au.items() if nm[0] == 'p'))}; Appendix F: 270, 238, 380: {au_ok}")
    ok &= au_ok
    if show_all:
        for cls, got in found.items():
            print(cls + ": " + ", ".join(f"{p} {a}" for p, _, a in got))
    print(("PASS" if ok else "FAIL") + ": the data of Q_1int and C(Q_1int) in Section 4.1, Appendix F and Table 1")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main("--all" in sys.argv[1:]) else 1)
