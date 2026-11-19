# Holographic entropy inequalities with arbitrarily large contraction scaling: code

Code and certificates for the computations in the paper *Balanced and superbalanced holographic entropy inequalities with arbitrarily large contraction scaling*. Each script reproduces the numbers of the statements listed next to it; numbers refer to the paper.

| Script | Reproduces |
|---|---|
| `seventeen_party_data.py` | The data of Appendix F and Table 2 (Appendix B, paragraph *The data of $Q_{1,\mathrm{int}}$ and $C(Q_{1,\mathrm{int}})$*): $Q_{1,\mathrm{int}}$ built from Definition 4.2 and equal to the display after Theorem 1.1 term by term, with $L=11$, $R=22$ and the party totals $117$, $50$, $42$, $46$; the Bell values (2.5) on all $153$ pairs of the $17$ parties and the purifier, zero on the pairs with $A$ or $O$ and even; the seven rows of Table 2; the superbalancing $C(Q_{1,\mathrm{int}})$ of (F.1) with integer coefficients, all Bell values zero, $L=131$, $R=38$ and $a_u=270$, $238$, $380$; and $d=4$, $c=\frac15$, $\omega=\frac{13}{10}$, $c_A=\frac{67}{10}$ of Section 4.1. With `--all` it also prints $a_{uv}$ for each of the $120$ pairs |
| `full_cube_map.py` | Theorem 1.1(a), (b) and (d) at $j=1$, the validity of $Q_{1,\mathrm{int}}$ and $k_{\min}(Q_{1,\mathrm{int}})\le3$: the stored integer map for $3Q_{1,\mathrm{int}}$ on $\{0,1\}^{11}$ ($2{,}048$ points, $22$ counts each) satisfies the pins (2.9), the ranges and the contraction condition (2.10) on the $11{,}264$ edges and on all $2{,}096{,}128$ pairs of points (Appendix B, paragraph *$k_{\min}(Q_{1,\mathrm{int}})\le3$*; Section 9.1) |
| `lifted_map.py` | Theorem 1.1(a), (b) and (d) at $j=1$, the validity of $C(Q_{1,\mathrm{int}})$ and $k_{\min}^{\mathcal A}(C(Q_{1,\mathrm{int}}))\le3$: the stored integer map $g$ for $3Q_{1,\mathrm{int}}$ on $\mathcal A(Q_{1,\mathrm{int}})$ ($486$ points) satisfies the pins, the ranges and the contraction condition on all $117{,}855$ pairs; $\mathcal A(C(Q_{1,\mathrm{int}}))$, rebuilt from Definition 2.12, has $679{,}944$ points; the lift of $g$ by Definition 8.7 is an integer map for $3C(Q_{1,\mathrm{int}})$ there, with the pins at the occurrence words of the $17$ parties and at $0$, the ranges, and the contraction condition on all $231{,}161{,}581{,}596$ pairs (Appendix B, paragraph *$k_{\min}^{\mathcal A}(C(Q_{1,\mathrm{int}}))\le3$*; Sections 8.4 and 9.1) |
| `water_level_map.py` | An independent check that the paper's proofs do not use: the water-level map of Definition 4.8 is a real map for the carriers $Q_1$ on $\{0,1\}^{11}$ and $Q_2$ on $\{0,1\}^{17}$, with common denominators $10{,}800$ and $281{,}707{,}272{,}000$, the pins of all parties and of the purifier, the ranges, and the contraction condition on all $11{,}264$ and $1{,}114{,}112$ edges, every edge with equality. For $j=2$ this is a case of Theorem 5.7 with Proposition 6.1; Theorem 5.7 does not cover $j=1$ (end of Section 5.3), where the paper rests on the two stored maps instead |
| `parties_and_factors.py` | $N_1,N_2,N_3=17,66,343$ by (6.3), $K_1=10$, and the forced factors $(2j+1)/\gcd(2j+1,K_j)=3,5,7$ of Theorem 1.1(c) and Corollary 6.4 for $j=1,2,3$ (Section 1, after Theorem 1.1; Section 6.1) |

`hec.py`, `carriers.py`, `countmap.py` and `lift.py` hold the definitions the scripts share: inequalities, Bell values, superbalancing and realizable domain; the carriers and the water-level map; the conditions of Definition 2.8 on the full cube; the realizable domain of the superbalancing and the lift. `certificates.py` loads the two stored maps in `data/`, and `allpairs.c` is the check on all pairs that `lifted_map.py` compiles and runs. In the printed output, `Real` stands for the realizable domain $\mathcal A$.

Each script prints the computed numbers next to the paper's (for `water_level_map.py`, next to the recorded denominators) and ends with a line `PASS: ...`, with exit status 0, or `FAIL: ...`. Every check is exact: rational numbers are multiplied by a common denominator and compared as integers, and no floating-point number enters a check. The two stored maps were found by integer programs; the scripts check each of them against Definition 2.8 directly, from the list of terms of the inequality, so how they were found plays no role. `seventeen_party_data.py`, `full_cube_map.py` and `lifted_map.py` perform the three computations of Appendix B, `parties_and_factors.py` recomputes numbers stated in the text, and `water_level_map.py` adds the check of the water-level map for $j=1,2$. The lower bounds (that $3$ divides every $k$ for which $kQ_{1,\mathrm{int}}$ or $kC(Q_{1,\mathrm{int}})$ has an integer map), the statements of Theorem 1.1 for general $j$, and Theorem 1.2 are proved in the text of the paper without computer assistance.

The paper and this code were produced by large language models; the paper's introduction states how, and what has and has not been checked.

## The stored maps

`data/map_3Q1int_fullcube.txt` is the integer map for $3Q_{1,\mathrm{int}}$ on the full cube ($2{,}048$ points), and `data/map_3Q1int_real.txt` the map $g$ on $\mathcal A(Q_{1,\mathrm{int}})$ ($486$ points). Both are plain text. The header lists the terms of $3Q_{1,\mathrm{int}}$ with their coefficients, one line `# lhs l coefficient parties` per LHS term and `# rhs r coefficient parties` per RHS term, in the order of the display after Theorem 1.1: bits $l=0,\dots,4$ are the box terms of $b=0,\dots,4$, bits $5,\dots,9$ the star terms of $v=0,\dots,4$, bit $10$ the term $Aq_B$; column $0$ is $S(F)$, columns $1,\dots,20$ the claw terms $S(F\setminus\{q_w,p_T\})$ with $T=01,02,\dots,34$ in lexicographic order and $w\in T$ increasing, and column $21$ is $S(A)$. Each data line is a point, a string of $11$ bits whose character $l$ is the bit of LHS term $l$, followed by its $22$ counts. `certificates.py` compares the header with the terms built from Definition 4.2 and rejects a file in any other order.

## Running

Python 3.12 (tested with 3.12.3) with the version in `requirements.txt`, and for `lifted_map.py` the C compiler `gcc` with OpenMP (tested with gcc 13.3), which the script calls itself and whose output it keeps in a temporary directory. From this folder:

```
pip install -r requirements.txt
python seventeen_party_data.py
python full_cube_map.py
python lifted_map.py
python water_level_map.py
python parties_and_factors.py
```

`lifted_map.py` runs the check on all pairs with 8 threads unless `OMP_NUM_THREADS` is set.

Measured on 2026-11-19 with Python 3.12.3, numpy 2.5.4 and gcc 13.3 on 7 cores of an AMD Ryzen 7 PRO 5750G: `seventeen_party_data.py`, `full_cube_map.py` and `parties_and_factors.py` under 1 s each, `water_level_map.py` 5 s (peak memory 0.8 GB), and `lifted_map.py` 4.5 min with `OMP_NUM_THREADS=7` (peak memory 0.85 GB), of which the check on all pairs took 266 s.

## License

MIT, see `LICENSE`.
