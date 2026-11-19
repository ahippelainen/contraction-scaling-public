"""k_min^A(C(Q_{1,int})) <= 3 (Theorem 1.1(d); Appendix B, paragraph 'k_min^A(C(Q_{1,int})) <= 3'). The stored integer map g for 3 Q_{1,int} on Real(Q_{1,int}) (data/map_3Q1int_real.txt, 486 points) is checked (pins, ranges, all 117,855 pairs); Real(C(Q_{1,int})) is rebuilt from Definition 2.12 (679,944 points); the lift of g by Definition 8.7 is computed and checked to be an integer map for 3 C(Q_{1,int}) there: pins at the occurrence words of the 17 parties and at 0, ranges, and the contraction condition on all 231,161,581,596 pairs (by the compiled program allpairs.c, which needs gcc with OpenMP)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from carriers import Params, carrier
from certificates import REAL, load
from hec import realizable_full, words_to_int
from lift import layout, lift, realizable_C

HERE = Path(__file__).resolve().parent
NC = 40


def check_on_points(alpha: np.ndarray, xs: np.ndarray, vals: np.ndarray) -> tuple[int, int, int]:
    """(pairs, violations, worst excess) over all pairs of a small point set of a cube with LHS weights alpha."""
    L = len(alpha)
    dal = (((np.arange(1 << L)[:, None] >> np.arange(L)) & 1) * alpha[None, :]).sum(1)
    pairs = viol = 0
    worst = -(1 << 40)
    for i in range(len(xs) - 1):
        ex = np.abs(vals[i + 1 :] - vals[i]).sum(1) - dal[xs[i + 1 :] ^ xs[i]]
        pairs += len(ex)
        viol += int((ex > 0).sum())
        worst = max(worst, int(ex.max()))
    return pairs, viol, worst


def dump(d: Path, P, vals: np.ndarray, alpha: np.ndarray, a: np.ndarray) -> None:
    d.mkdir(parents=True, exist_ok=True)
    assert vals.shape[1] <= NC and np.abs(vals).max() < 2**15 and np.abs(vals).sum(1).max() < 2**30
    pad = np.zeros((len(vals), NC), dtype=np.int16)
    pad[:, : vals.shape[1]] = vals
    L = len(alpha)
    dal = (((np.arange(1 << L)[:, None] >> np.arange(L)) & 1) * alpha[None, :]).sum(1).astype(np.int32)
    assert L == 11
    ws = sorted(set(a.tolist()))
    mlo = np.array([sum(1 << e for e in range(min(64, len(a))) if a[e] == w) for w in ws], dtype=np.uint64)
    mhi = np.array([sum(1 << (e - 64) for e in range(64, len(a)) if a[e] == w) for w in ws], dtype=np.uint64)
    (d / "meta.txt").write_text(f"{len(vals)} {len(ws)}\n")
    P.x.astype(np.uint16).tofile(d / "x.u16")
    P.gam_lo.astype(np.uint64).tofile(d / "glo.u64")
    P.gam_hi.astype(np.uint64).tofile(d / "ghi.u64")
    pad.tofile(d / "vals.i16")
    dal.tofile(d / "dal.i32")
    np.array(ws, dtype=np.int32).tofile(d / "wcls.i32")
    mlo.tofile(d / "mlo.u64")
    mhi.tofile(d / "mhi.u64")


def run_allpairs(exe: Path, d: Path) -> str:
    r = subprocess.run([str(exe), str(d)], capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8")})
    return r.stdout.strip() + r.stderr.strip()


def main() -> bool:
    k = 3
    Q = carrier(1)[0].scaled(Params(1).K * k)
    S = layout(Q)
    C = S.C
    alpha = np.array([int(v) for v in Q.alpha], dtype=np.int64)
    beta = np.array([int(v) for v in Q.beta], dtype=np.int64)
    print(f"3*C(Q_1int): L = {len(C.lhs)} ({len(S.pairs)} pair terms), R = {len(C.rhs)}; distinct pair weights {sorted(set(S.a.tolist()))}")

    realQ = realizable_full([t for t, _ in Q.lhs])
    pts, gR = load(REAL, Q)
    order = np.argsort(pts)
    pts, gR = pts[order], gR[order]
    same_domain = np.array_equal(pts, realQ)
    g = np.zeros((1 << len(Q.lhs), len(Q.rhs)), dtype=np.int64)
    g[pts] = gR
    X, Y = Q.occ_x(), Q.occ_y()
    xw = words_to_int(X)
    pins_ok = bool(np.array_equal(g[xw], Y * beta[None, :])) and not g[0].any() and set(xw.tolist()) <= set(realQ.tolist())
    ranges_ok = bool(((gR >= 0) & (gR <= beta[None, :])).all())
    pr, vi, wo = check_on_points(alpha, pts, gR)
    g_ok = len(realQ) == 486 and same_domain and pins_ok and ranges_ok and vi == 0 and pr == 117_855
    print(f"g ({REAL.name}): points = Real(3Q_1int) of Definition 2.12: {same_domain} ({len(realQ)} points); pins {pins_ok}, ranges {ranges_ok}, all {pr} pairs: {vi} violations, worst excess {wo}")

    P = realizable_C(S)
    ntri = int((P.kind == 3).sum())
    print(f"|Real(3C(Q_1int))| = {len(P.x)} (Gamma empty {int((P.kind == 0).sum())}, single pair term {int((P.kind == 1).sum())}, fan {int((P.kind == 2).sum())}, triangle {ntri}; triangles {len(set(P.leaves[P.kind == 3].tolist()))})", flush=True)
    vals, diag = lift(S, g, P)
    betaC = np.array([int(v) for v in C.beta], dtype=np.int64)
    rng_ok = bool(((vals >= 0) & (vals <= betaC[None, :])).all())
    # pins of C: the occurrence word of party p is (x_p, all pair terms containing p), with value (beta y_p, a_p on S(p)); the purifier pin is the value 0 at (0, empty)
    XC, YC = C.occ_x(), C.occ_y().astype(np.int64) * betaC[None, :]
    pin_fail = 0
    for p in range(C.n):
        gam = sum(1 << e for e, pr_ in enumerate(S.pairs) if p in pr_)
        at = np.flatnonzero((P.x == xw[p]) & (P.gam_lo == np.uint64(gam & (2**64 - 1))) & (P.gam_hi == np.uint64(gam >> 64)))
        assert len(at) == 1 and np.array_equal(XC[p, : len(Q.lhs)], X[p])
        pin_fail += int(not np.array_equal(vals[at[0]], YC[p]))
    at0 = np.flatnonzero((P.x == 0) & (P.gam_lo == 0) & (P.gam_hi == 0))
    pin_fail += int(len(at0) != 1 or vals[at0[0]].any())
    print(f"lift: ranges {rng_ok}, pin failures {pin_fail} (17 parties + purifier); at the single-pair-term points {diag}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "allpairs"
        subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-o", str(exe), str(HERE / "allpairs.c")], check=True)
        d = Path(tmp) / "lift"
        dump(d, P, vals, alpha, S.a)
        out = run_allpairs(exe, d)
        print("allpairs:", out, flush=True)
        # one unit added to one value at a point inside the first 40,000 points must produce violations
        bad = vals[:40000].copy()
        bad[12345, 3] += 1
        Pc = type(P)(*(getattr(P, f)[:40000] for f in ("x", "gam_lo", "gam_hi", "kind", "center", "leaves", "a_gamma")))
        dump(Path(tmp) / "control", Pc, bad, alpha, S.a)
        ctrl = run_allpairs(exe, Path(tmp) / "control")
    print("negative control (first 40,000 points, one unit added to one value):", ctrl)
    n = len(P.x)
    ok = g_ok and n == 679_944 and rng_ok and pin_fail == 0 and " violations 0 " in out and f"pairs {n * (n - 1) // 2} " in out and " violations 0 " not in ctrl
    print(("PASS" if ok else "FAIL") + ": the lift of g is an integer map for 3*C(Q_1int) on Real(C(Q_1int)), so k_min^A(C(Q_1int)) <= 3")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
