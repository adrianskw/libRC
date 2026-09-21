import numpy as np
import pytest

from src.observer import WindowObserver, window_features

LAGS = [0, 3, 6, 9]


def series(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)) * 0.1 + np.sin(np.arange(n) * 0.05)


class TestWindowFeatures:
    def test_columns_are_lagged_copies(self):
        x = np.arange(100.0)
        f = window_features(x, np.array([20, 21]), LAGS)
        np.testing.assert_array_equal(f, [[20, 17, 14, 11], [21, 18, 15, 12]])

    def test_rejects_windows_that_reach_before_the_record(self):
        with pytest.raises(ValueError):
            window_features(np.arange(50.0), np.array([5]), LAGS)


class TestWindowObserver:
    def test_ridge_recovers_a_linear_function_of_the_window(self):
        x = series()
        idx = np.arange(9, len(x))
        target = (2.0 * x[idx] - 1.0 * x[idx - 3] + 0.5 * x[idx - 9])[:, None]
        full = np.zeros((len(x), 1)); full[idx] = target
        obs = WindowObserver(LAGS, kind="ridge").fit(x, full, np.arange(9, 2000))
        pred = obs.predict(x, np.arange(2000, len(x)))
        np.testing.assert_allclose(pred, full[2000:], atol=2e-2)

    def test_mlp_beats_ridge_on_a_nonlinear_target(self):
        x = series(seed=1)
        idx = np.arange(9, len(x))
        y = np.zeros((len(x), 1)); y[idx, 0] = np.tanh(2 * x[idx]) * x[idx - 6] ** 2
        tr, va = np.arange(9, 2200), np.arange(2200, len(x))
        r = WindowObserver(LAGS, kind="ridge").fit(x, y, tr).predict(x, va)
        m = WindowObserver(LAGS, kind="mlp", epochs=150, n_seeds=1, device="cpu").fit(x, y, tr).predict(x, va)
        assert np.mean((m - y[va]) ** 2) < 0.5 * np.mean((r - y[va]) ** 2)

    def test_prediction_never_reads_future_measurements(self):
        x = series(seed=2)
        y = np.stack([x, np.roll(x, 4)], axis=1)
        obs = WindowObserver(LAGS, kind="ridge").fit(x, y, np.arange(9, 2000))
        t = 2500
        before = obs.predict(x, np.array([t]))
        x2 = x.copy(); x2[t + 1:] += 1000.0                  # corrupt everything after t
        np.testing.assert_array_equal(before, obs.predict(x2, np.array([t])))

    def test_output_shape_and_multitarget(self):
        x = series()
        y = np.stack([x, x ** 2, np.roll(x, 2)], axis=1)
        obs = WindowObserver(LAGS, kind="mlp", epochs=5, n_seeds=2, device="cpu").fit(x, y, np.arange(9, 2000))
        assert obs.predict(x, np.arange(2000, 2100)).shape == (100, 3)

    def test_bad_kind_rejected(self):
        with pytest.raises(ValueError):
            WindowObserver(LAGS, kind="forest")
