"""
Benchmark: old vs new implementations of makeConnectionMatDegree and
makeInputMat(sparseFlag=True).

The "old" implementations below are reconstructed verbatim from the
pre-optimization versions of libRC.py (dense NxN mask+random array for
connection matrices; dense NxD random array sliced down for sparse
input matrices) so they can be timed head-to-head against the current
methods on the same Reservoir subclass.

Run with:
    python benchmarks/bench_connectivity.py
"""
import builtins
import sys
import os
import time

import numpy as np
from scipy.sparse import csr_matrix as sparseCsrMatrix
from scipy.sparse.linalg import eigs
from scipy.stats import uniform as statsUniform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libRC import mapRC


# ---------------------------------------------------------------------------
# Reconstructed pre-optimization implementations
# ---------------------------------------------------------------------------
def old_makeConnectionMatDegree(self, rho, degree=3, diag_vals=None, dist=statsUniform, loc=-1.0, scale=2.0):
    self.rho = rho
    self.degree = min(degree, self.N)
    mask = np.zeros((self.N, self.N))
    for i in range(self.N):
        idx = np.random.choice(np.arange(self.N), replace=False, size=self.degree)
        mask[i, idx] = 1
    self.A = sparseCsrMatrix(dist(loc, scale).rvs(size=(self.N, self.N)) * mask)
    if diag_vals is not None:
        self.A.setdiag(diag_vals)
        self.A.eliminate_zeros()
    maxEig = float(np.abs(eigs(self.A, k=1, which='LM', return_eigenvectors=False))[0])
    self.A = self.A.multiply(rho / maxEig).tocsr()


def old_makeInputMat(self, D, sigma, randMin=0.0, randMax=1.0, sparseFlag=True):
    self.D = D
    self.sigma = sigma
    self.B = sigma * np.random.uniform(low=randMin, high=randMax, size=(self.N, self.D))
    if sparseFlag:
        rows = np.arange(self.N)
        cols = rows % self.D
        self.B = sparseCsrMatrix((self.B[rows, cols], (rows, cols)), shape=(self.N, self.D))


def old_construct_only(N, degree, dist=statsUniform, loc=-1.0, scale=2.0):
    """Isolates just the matrix-construction cost (no spectral-radius solve)."""
    mask = np.zeros((N, N))
    for i in range(N):
        idx = np.random.choice(np.arange(N), replace=False, size=degree)
        mask[i, idx] = 1
    return sparseCsrMatrix(dist(loc, scale).rvs(size=(N, N)) * mask)


def new_construct_only(N, degree, dist=statsUniform, loc=-1.0, scale=2.0):
    rows = np.repeat(np.arange(N), degree)
    cols = np.empty(N * degree, dtype=int)
    for i in range(N):
        cols[i * degree:(i + 1) * degree] = np.random.choice(np.arange(N), replace=False, size=degree)
    vals = dist(loc, scale).rvs(size=N * degree)
    return sparseCsrMatrix((vals, (rows, cols)), shape=(N, N))


def _time_once(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def bench_construction_only():
    print("=" * 70)
    print("Construction step only (mask+dense-random vs direct COO), no eigs()")
    print("=" * 70)
    print(f"{'N':>8} {'degree':>7} {'old (s)':>12} {'new (s)':>12} {'speedup':>10}")
    for N, degree in [(200, 6), (500, 6), (1000, 6), (2000, 6), (4000, 6), (8000, 6)]:
        np.random.seed(0)
        t_old = min(_time_once(lambda: old_construct_only(N, degree)) for _ in range(5))
        np.random.seed(0)
        t_new = min(_time_once(lambda: new_construct_only(N, degree)) for _ in range(5))
        print(f"{N:>8} {degree:>7} {t_old:>12.5f} {t_new:>12.5f} {t_old / t_new:>9.1f}x")
    print()


def bench_connection_mat():
    print("=" * 70)
    print("makeConnectionMatDegree: old (dense NxN mask) vs new (direct COO)")
    print("=" * 70)
    print(f"{'N':>8} {'degree':>7} {'old (s)':>12} {'new (s)':>12} {'speedup':>10}")
    for N, degree in [(200, 6), (500, 6), (1000, 6), (2000, 6), (4000, 6)]:
        np.random.seed(0)
        RC_old = mapRC(N)
        t_old = min(_time_once(lambda: old_makeConnectionMatDegree(RC_old, rho=0.9, degree=degree)) for _ in range(5))

        np.random.seed(0)
        RC_new = mapRC(N)
        t_new = min(_time_once(lambda: RC_new.makeConnectionMatDegree(rho=0.9, degree=degree)) for _ in range(5))

        print(f"{N:>8} {degree:>7} {t_old:>12.5f} {t_new:>12.5f} {t_old / t_new:>9.1f}x")
    print()


def bench_input_mat():
    print("=" * 70)
    print("makeInputMat(sparseFlag=True): old (dense NxD then slice) vs new (direct)")
    print("=" * 70)
    print(f"{'N':>10} {'D':>4} {'old (s)':>12} {'new (s)':>12} {'speedup':>10}")
    for N, D in [(10_000, 3), (100_000, 3), (1_000_000, 3), (100_000, 20), (1_000_000, 50)]:
        np.random.seed(0)
        RC_old = mapRC(N)
        t_old = min(_time_once(lambda: old_makeInputMat(RC_old, D=D, sigma=0.5)) for _ in range(3))

        np.random.seed(0)
        RC_new = mapRC(N)
        t_new = min(_time_once(lambda: RC_new.makeInputMat(D=D, sigma=0.5)) for _ in range(3))

        print(f"{N:>10} {D:>4} {t_old:>12.5f} {t_new:>12.5f} {t_old / t_new:>9.1f}x")
    print()


if __name__ == "__main__":
    real_print = builtins.print
    # libRC methods print status lines on every call ("Connection matrix is
    # setup.", etc.); silence those so only benchmark output is shown.
    builtins.print = lambda *a, **k: None
    try:
        for fn in (bench_construction_only, bench_connection_mat, bench_input_mat):
            fn.__globals__["print"] = real_print
        bench_construction_only()
        bench_connection_mat()
        bench_input_mat()
    finally:
        builtins.print = real_print
