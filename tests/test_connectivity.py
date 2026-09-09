import numpy as np
import pytest
from scipy.sparse import csr_matrix

from libRC import mapRC


def make_reservoir(N):
    return mapRC(N)


class TestMakeConnectionMatDegree:
    def test_shape_and_exact_degree_per_row(self):
        np.random.seed(0)
        N, degree = 40, 4
        RC = make_reservoir(N)
        RC.makeConnectionMatDegree(rho=0.9, degree=degree)
        assert RC.A.shape == (N, N)
        nnz_per_row = np.asarray((RC.A != 0).sum(axis=1)).flatten()
        assert np.all(nnz_per_row == degree)

    def test_spectral_radius_matches_rho(self):
        np.random.seed(1)
        N, rho = 60, 0.7
        RC = make_reservoir(N)
        RC.makeConnectionMatDegree(rho=rho, degree=6)
        sr = np.max(np.abs(np.linalg.eigvals(RC.A.toarray())))
        assert sr == pytest.approx(rho, abs=1e-6)

    def test_degree_larger_than_N_is_clamped(self):
        N = 5
        RC = make_reservoir(N)
        RC.makeConnectionMatDegree(rho=0.5, degree=100)
        assert RC.degree == N
        assert RC.A.shape == (N, N)

    def test_reproducible_with_seed(self):
        N = 20
        np.random.seed(42)
        RC1 = make_reservoir(N)
        RC1.makeConnectionMatDegree(rho=0.8, degree=3)
        np.random.seed(42)
        RC2 = make_reservoir(N)
        RC2.makeConnectionMatDegree(rho=0.8, degree=3)
        np.testing.assert_allclose(RC1.A.toarray(), RC2.A.toarray())

    def test_result_is_csr(self):
        RC = make_reservoir(30)
        RC.makeConnectionMatDegree(rho=0.9, degree=3)
        assert isinstance(RC.A, csr_matrix)

    def test_diag_vals_sets_diagonal(self):
        N = 10
        RC = make_reservoir(N)
        RC.makeConnectionMatDegree(rho=0.9, degree=3, diag_vals=0.0)
        assert np.allclose(RC.A.diagonal(), 0.0)


class TestMakeConnectionMatDensity:
    def test_shape_and_density_within_tolerance(self):
        np.random.seed(2)
        N, density = 100, 0.05
        RC = make_reservoir(N)
        RC.makeConnectionMatDensity(rho=0.9, density=density)
        assert RC.A.shape == (N, N)
        expected_nnz = density * N * N
        assert abs(RC.A.nnz - expected_nnz) < 0.3 * expected_nnz

    def test_spectral_radius_matches_rho(self):
        np.random.seed(3)
        N, rho = 80, 0.85
        RC = make_reservoir(N)
        RC.makeConnectionMatDensity(rho=rho, density=0.05)
        sr = np.max(np.abs(np.linalg.eigvals(RC.A.toarray())))
        assert sr == pytest.approx(rho, abs=1e-6)

    def test_reproducible_with_seed(self):
        N = 30
        np.random.seed(7)
        RC1 = make_reservoir(N)
        RC1.makeConnectionMatDensity(rho=0.8, density=0.1)
        np.random.seed(7)
        RC2 = make_reservoir(N)
        RC2.makeConnectionMatDensity(rho=0.8, density=0.1)
        np.testing.assert_allclose(RC1.A.toarray(), RC2.A.toarray())

    def test_result_is_csr(self):
        RC = make_reservoir(30)
        RC.makeConnectionMatDensity(rho=0.8, density=0.05)
        assert isinstance(RC.A, csr_matrix)


class TestSharedSignature:
    """Both builders now share the same (diag_vals, dist, loc, scale) dialect (issue #10)."""

    def test_both_accept_normal_distribution(self):
        from scipy.stats import norm

        np.random.seed(4)
        RC_deg = make_reservoir(20)
        RC_deg.makeConnectionMatDegree(rho=0.9, degree=3, dist=norm, loc=0.0, scale=1.0)

        RC_dens = make_reservoir(20)
        RC_dens.makeConnectionMatDensity(rho=0.9, density=0.1, dist=norm, loc=0.0, scale=1.0)

        assert RC_deg.A.shape == (20, 20)
        assert RC_dens.A.shape == (20, 20)
