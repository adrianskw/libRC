import numpy as np
import pytest

from src import observables as obs


def write_log(tmp_path, n=50):
    header = 'VARIABLES = "Time (s)", "Discharge current (A)", "Thrust"\n'
    t = np.arange(1, n + 1) * 5e-8
    rows = np.column_stack([t, 4.0 + np.arange(n) * 0.01, np.full(n, 0.085)])
    with open(tmp_path / "log.dat", "w") as f:
        f.write(header)
        np.savetxt(f, rows)
    return rows


class TestLoaders:
    def test_load_log_columns_and_length(self, tmp_path):
        rows = write_log(tmp_path)
        log = obs.load_log(str(tmp_path))
        assert set(log) == {"Time (s)", "Discharge current (A)", "Thrust"}
        assert len(log["Thrust"]) == len(rows)

    def test_discharge_current_matches_column(self, tmp_path):
        rows = write_log(tmp_path)
        np.testing.assert_allclose(obs.load_discharge_current(str(tmp_path)), rows[:, 1])

    def test_alias_and_exact_header_agree(self, tmp_path):
        write_log(tmp_path)
        a = obs.load_observable(str(tmp_path), "id")
        b = obs.load_observable(str(tmp_path), "Discharge current (A)")
        np.testing.assert_array_equal(a, b)

    def test_unknown_column_raises(self, tmp_path):
        write_log(tmp_path)
        with pytest.raises(KeyError, match="no log.dat column"):
            obs.load_observable(str(tmp_path), "nonsense")


class TestZscore:
    def test_uses_train_statistics_only(self):
        x = np.concatenate([np.zeros(100), np.full(100, 1000.0)])
        xn, mean, std = obs.zscore(x, n_train=100)
        assert mean.item() == pytest.approx(0.0)
        assert xn[:100].std() == pytest.approx(0.0, abs=1e-6)
        assert xn[100:].mean() > 1e6        # held-out values are NOT re-centered

    def test_train_block_is_unit_scale(self):
        rng = np.random.default_rng(0)
        x = 5.0 + 3.0 * rng.standard_normal(1000)
        xn, _, _ = obs.zscore(x, n_train=800)
        assert xn[:800].mean() == pytest.approx(0.0, abs=1e-8)
        assert xn[:800].std() == pytest.approx(1.0, abs=1e-6)

    def test_multichannel_shapes_and_roundtrip(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((4, 500)) * np.array([[1], [2], [3], [4]]) + 7
        xn, mean, std = obs.zscore(x, n_train=400)
        assert mean.shape == (4, 1) and std.shape == (4, 1)
        np.testing.assert_allclose(xn * std + mean, x)


class TestSmooth:
    def test_window_one_is_identity(self):
        x = np.arange(10.0)
        np.testing.assert_array_equal(obs.smooth(x, 1), x)

    def test_causal_uses_only_the_past(self):
        x = np.zeros(20)
        x[10] = 5.0
        y = obs.smooth(x, 4, causal=True)
        assert np.all(y[:10] == 0.0)          # nothing leaks backwards
        assert y[10] == pytest.approx(5.0 / 4)

    def test_causal_start_uses_expanding_mean(self):
        y = obs.smooth(np.array([2.0, 4.0, 6.0, 8.0]), 3, causal=True)
        np.testing.assert_allclose(y, [2.0, 3.0, 4.0, 6.0])

    def test_constant_signal_unchanged_both_modes(self):
        x = np.full(30, 3.3)
        np.testing.assert_allclose(obs.smooth(x, 5, causal=True), x)
        np.testing.assert_allclose(obs.smooth(x, 5, causal=False), x)


class TestStackState:
    def test_order_is_respected(self):
        ch = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0]), "c": np.array([5.0, 6.0])}
        s = obs.stack_state(ch, ["c", "a"])
        np.testing.assert_array_equal(s, [[5.0, 6.0], [1.0, 2.0]])

    def test_latent_channels_split(self):
        z = np.arange(12.0).reshape(4, 3)
        ch = obs.latent_channels(z)
        assert list(ch) == ["z1", "z2", "z3"]
        np.testing.assert_array_equal(ch["z2"], z[:, 1])

    def test_unknown_channel_raises(self):
        with pytest.raises(KeyError):
            obs.stack_state({"a": np.zeros(3)}, ["a", "b"])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            obs.stack_state({"a": np.zeros(3), "b": np.zeros(4)}, ["a", "b"])
