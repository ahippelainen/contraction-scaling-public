/* Contraction check of an integer count map on all pairs of a point set (condition (2.10) of Definition 2.8 of the paper):
   sum_r |n_r(P) - n_r(P')| <= d(P, P'), d = d_alpha on the x part + weighted Hamming distance on the pair-term part.
   Input directory (written by lifted_map.py): meta.txt "n nclass", x.u16, glo.u64, ghi.u64, vals.i16 (n x NC), dal.i32 (2^11), wcls.i32, mlo.u64, mhi.u64.
   Usage: allpairs DIR. Tiled over (i, j) blocks so that a j tile stays in cache while a block of i points sweeps it. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NC 40
#define BI 128
#define BJ 2048

static void *slurp(const char *dir, const char *name, size_t bytes) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    void *p = malloc(bytes);
    if (fread(p, 1, bytes, f) != bytes) { fprintf(stderr, "short read %s\n", path); exit(2); }
    fclose(f);
    return p;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s DIR\n", argv[0]); return 2; }
    const char *dir = argv[1];
    char mp[4096];
    snprintf(mp, sizeof mp, "%s/meta.txt", dir);
    FILE *mf = fopen(mp, "r");
    long n, nc;
    if (!mf || fscanf(mf, "%ld %ld", &n, &nc) != 2) { fprintf(stderr, "bad meta\n"); return 2; }
    fclose(mf);
    const uint16_t *x = slurp(dir, "x.u16", n * sizeof(uint16_t));
    const uint64_t *glo = slurp(dir, "glo.u64", n * sizeof(uint64_t));
    const uint64_t *ghi = slurp(dir, "ghi.u64", n * sizeof(uint64_t));
    const int16_t *v = slurp(dir, "vals.i16", (size_t)n * NC * sizeof(int16_t));
    const int32_t *dal = slurp(dir, "dal.i32", 2048 * sizeof(int32_t));
    const int32_t *w = slurp(dir, "wcls.i32", nc * sizeof(int32_t));
    const uint64_t *mlo = slurp(dir, "mlo.u64", nc * sizeof(uint64_t));
    const uint64_t *mhi = slurp(dir, "mhi.u64", nc * sizeof(uint64_t));

    uint64_t pairs = 0, viol = 0, tight = 0, edges = 0;
    long worst = -(1L << 40), wi = -1, wj = -1;
    double t0 = omp_get_wtime();
    long nblk = (n + BI - 1) / BI;
#pragma omp parallel
    {
        uint64_t lp = 0, lv = 0, lt = 0, le = 0;
        long lw = -(1L << 40), lwi = -1, lwj = -1;
#pragma omp for schedule(dynamic, 1)
        for (long b = 0; b < nblk; b++) {
            long i0 = b * BI, i1 = i0 + BI < n ? i0 + BI : n;
            for (long j0 = i0; j0 < n; j0 += BJ) {
                long j1 = j0 + BJ < n ? j0 + BJ : n;
                for (long i = i0; i < i1; i++) {
                    const int16_t *vi = v + (size_t)i * NC;
                    long js = j0 > i + 1 ? j0 : i + 1;
                    for (long j = js; j < j1; j++) {
                        const int16_t *vj = v + (size_t)j * NC;
                        int s = 0;
                        for (int r = 0; r < NC; r++) { int d = vi[r] - vj[r]; s += d < 0 ? -d : d; }
                        uint64_t dl = glo[i] ^ glo[j], dh = ghi[i] ^ ghi[j];
                        uint16_t dx = x[i] ^ x[j];
                        long d = dal[dx];
                        for (long c = 0; c < nc; c++) d += (long)w[c] * (__builtin_popcountll(dl & mlo[c]) + __builtin_popcountll(dh & mhi[c]));
                        long ex = (long)s - d;
                        lp++;
                        if (ex > 0) lv++;
                        if (ex == 0) lt++;
                        if (ex > lw) { lw = ex; lwi = i; lwj = j; }
                        if (__builtin_popcount(dx) + __builtin_popcountll(dl) + __builtin_popcountll(dh) == 1) le++;
                    }
                }
            }
        }
#pragma omp critical
        {
            pairs += lp; viol += lv; tight += lt; edges += le;
            if (lw > worst) { worst = lw; wi = lwi; wj = lwj; }
        }
    }
    printf("points %ld pairs %lu violations %lu tight %lu cube_edges %lu worst_excess %ld at %ld %ld threads %d seconds %.1f\n",
           n, (unsigned long)pairs, (unsigned long)viol, (unsigned long)tight, (unsigned long)edges, worst, wi, wj,
           omp_get_max_threads(), omp_get_wtime() - t0);
    return viol ? 1 : 0;
}
