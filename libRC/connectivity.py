# -*- coding: utf-8 -*-
"""Connection (adjacency) and input matrix builders for Reservoir instances.

These are pure functions: they take N/D and distribution parameters and
return a matrix rather than mutating a Reservoir instance in place. The
Reservoir methods of the same name (see reservoir.py) call these and
assign the results onto self.

Every builder accepts an optional `rng` (a numpy.random.Generator or
RandomState). When omitted, the global `np.random` state is used, which
matches the historical behavior of this module. Passing an explicit
`rng` makes reservoir construction reproducible independent of global
RNG state -- useful for running multiple experiments (e.g. in parallel)
without them interfering with each other's random draws.
"""
import numpy as np
from scipy.sparse import random as sparseRandom
from scipy.sparse import csr_matrix as sparseCsrMatrix
from scipy.sparse import diags as sparseDiags
from scipy.sparse.linalg import eigs, ArpackNoConvergence
from scipy.stats import uniform as statsUniform


def _spectralRadius(A, tol=0, maxiter=None):
    """Largest-magnitude eigenvalue of sparse matrix A via ARPACK.

    Raises a clear RuntimeError instead of letting ArpackNoConvergence
    propagate -- this can happen for very sparse or degenerate matrices
    (e.g. degree=1). `tol`/`maxiter` are passed straight through to
    `eigs`; loosening `tol` trades exactness for speed at large N, at
    the cost of imprecision in the rescaled rho. Defaults reproduce the
    original (tight-tolerance) behavior exactly.
    """
    try:
        maxEig = eigs(A, k=1, which='LM', return_eigenvectors=False, tol=tol, maxiter=maxiter)
    except ArpackNoConvergence as exc:
        raise RuntimeError(
            "Spectral radius computation (ARPACK) failed to converge. This "
            "can happen for very sparse (low-degree) or degenerate "
            "connection matrices -- try a higher degree/density, or pass "
            "a looser `tol`/higher `maxiter`."
        ) from exc
    return float(np.abs(maxEig)[0])


def makeConnectionMatDegree(N, rho, degree=3, diag_vals=None, dist=statsUniform,
                             loc=-1.0, scale=2.0, rng=None, tol=0, maxiter=None):
    """Sparse NxN connection matrix with exactly `degree` connections per
    row, rescaled to spectral radius `rho`. Returns (A, degree_used) --
    degree_used is `degree` clamped to N.
    """
    degree = min(degree, N)
    r = rng if rng is not None else np.random
    rows = np.repeat(np.arange(N), degree)
    cols = np.empty(N * degree, dtype=int)
    for i in range(N):
        cols[i * degree:(i + 1) * degree] = r.choice(np.arange(N), replace=False, size=degree)
    vals = dist(loc, scale).rvs(size=N * degree, random_state=rng)
    A = sparseCsrMatrix((vals, (rows, cols)), shape=(N, N))
    if diag_vals is not None:
        A.setdiag(diag_vals)
        A.eliminate_zeros()
    maxEig = _spectralRadius(A, tol=tol, maxiter=maxiter)
    A = A.multiply(rho / maxEig).tocsr()
    return A, degree


def makeConnectionMatDensity(N, rho, density=0.02, diag_vals=None, dist=statsUniform,
                              loc=-1.0, scale=2.0, rng=None, tol=0, maxiter=None):
    """Sparse NxN connection matrix with the given fill density, rescaled
    to spectral radius `rho`."""
    A = sparseRandom(N, N, density=density,
                      data_rvs=lambda size: dist(loc, scale).rvs(size=size, random_state=rng),
                      random_state=rng)
    if diag_vals is not None:
        A.setdiag(diag_vals)
        A.eliminate_zeros()
    maxEig = _spectralRadius(A, tol=tol, maxiter=maxiter)
    A = A.multiply(rho / maxEig).tocsr()
    return A


def makeDiagConnectionMat(N, rho=1, randMin=-1.0, randMax=1.0, rng=None):
    """Diagonal-only NxN connection matrix rescaled to spectral radius `rho`."""
    r = rng if rng is not None else np.random
    diagElem = r.uniform(randMin, randMax, N)
    A = sparseDiags(diagElem).multiply(rho / max(diagElem))
    return A


def makeInputMat(N, D, sigma, randMin=0.0, randMax=1.0, sparseFlag=True, rng=None):
    """NxD input weight matrix. Sparse by default (one connection per row,
    cycling through the D input channels); dense if `sparseFlag=False`."""
    r = rng if rng is not None else np.random
    if sparseFlag:
        rows = np.arange(N)
        cols = rows % D
        vals = sigma * r.uniform(low=randMin, high=randMax, size=N)
        B = sparseCsrMatrix((vals, (rows, cols)), shape=(N, D))
    else:
        B = sigma * r.uniform(low=randMin, high=randMax, size=(N, D))
    return B
