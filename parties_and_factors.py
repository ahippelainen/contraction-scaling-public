"""The numbers of parties N_1, N_2, N_3 = 17, 66, 343 of eq. (6.3), K_1 = 10, and the forced factors (2j+1)/gcd(2j+1, K_j) = 3, 5, 7 of Theorem 1.1(c) and Corollary 6.4 for j = 1, 2, 3, as stated in Section 1 (after Theorem 1.1) and Section 6.1."""
from __future__ import annotations

import sys
from math import gcd

from carriers import Params, carrier

CLAIMED = {1: (17, 3), 2: (66, 5), 3: (343, 7)}


def main() -> bool:
    ok = Params(1).K == 10
    for j, (N, f) in CLAIMED.items():
        P = Params(j)
        n_list = carrier(j)[0].n
        forced = (2 * j + 1) // gcd(2 * j + 1, P.K)
        print(f"j={j}: N_j = 3j+4+C(3j+2, j+1) = {P.n_parties}, parties built by carrier(j): {n_list}, K_j = {P.K}, forced factor (2j+1)/gcd(2j+1, K_j) = {forced}; paper: N_j = {N}, factor {f}")
        ok &= P.n_parties == n_list == N and forced == f
    print(("PASS" if ok else "FAIL") + ": N_j = 17, 66, 343 and forced factors 3, 5, 7 for j = 1, 2, 3, and K_1 = 10")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
