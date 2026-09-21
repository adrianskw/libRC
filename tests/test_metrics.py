import numpy as np
import pytest

from src.metrics import infer_pc, period_steps


class TestInferPC:
    def test_diagonal_is_squared_correlation(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal((3, 500))
        p = y + 0.5 * rng.standard_normal((3, 500))
        pc = infer_pc(y, p)
        for k in range(3):
            assert pc[k, k] == pytest.approx(np.corrcoef(y[k], p[k])[0, 1] ** 2)

    def test_invariant_to_scale_and_offset_of_prediction(self):
        rng = np.random.default_rng(1)
        y = rng.standard_normal((2, 400))
        p = y + 0.3 * rng.standard_normal((2, 400))
        np.testing.assert_allclose(infer_pc(y, p), infer_pc(y, 5.0 * p + 7.0))

    def test_perfect_prediction_gives_one(self):
        y = np.random.default_rng(2).standard_normal((3, 300))
        np.testing.assert_allclose(np.diag(infer_pc(y, y)), 1.0)


class TestPeriodSteps:
    def test_recovers_the_period_of_a_sinusoid(self):
        t = np.arange(4000)
        period, n = period_steps(np.sin(2 * np.pi * t / 158.0))
        assert period == pytest.approx(158.0, abs=2.0) and n > 20

    def test_flat_signal_reports_no_oscillation(self):
        period, n = period_steps(np.full(2000, 0.3))
        assert np.isnan(period) and n < 2
