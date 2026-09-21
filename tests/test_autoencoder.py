import numpy as np
import pytest
import torch

from src.autoencoder import ConvAE, decode_to_fields, relative_field_error

FIELDS = ["n_e", "phi", "T_e", "n_n", "n_i_dot", "v_i_z", "v_i_r"]
LOG = ["n_e", "n_n", "n_i_dot"]


def make_ckpt(mean=0.0, std=1.0):
    shape = (1, len(FIELDS), 1, 1)
    return {"field_names": FIELDS, "log_fields": LOG,
            "mean": np.full(shape, mean, dtype=np.float32), "std": np.full(shape, std, dtype=np.float32)}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return ConvAE(in_ch=len(FIELDS), latent_dim=3).eval()


class TestDecodeToFields:
    def test_output_shape(self, model):
        z = np.random.default_rng(0).standard_normal((7, 3)).astype(np.float32)
        assert decode_to_fields(model, make_ckpt(), z).shape == (7, len(FIELDS), 25, 50)

    def test_log_fields_are_positive_and_linear_fields_are_not_forced(self, model):
        z = np.random.default_rng(1).standard_normal((40, 3)).astype(np.float32) * 3
        out = decode_to_fields(model, make_ckpt(), z)
        for c, name in enumerate(FIELDS):
            if name in LOG:
                assert (out[:, c] > 0).all(), name
        assert (out[:, FIELDS.index("phi")] < 0).any()      # not log-transformed, so can be negative

    def test_chunking_does_not_change_the_result(self, model):
        z = np.random.default_rng(2).standard_normal((23, 3)).astype(np.float32)
        a = decode_to_fields(model, make_ckpt(), z, chunk=1000)
        b = decode_to_fields(model, make_ckpt(), z, chunk=5)
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)   # batch size can reorder float sums

    def test_undoes_the_zscore(self, model):
        z = np.random.default_rng(3).standard_normal((4, 3)).astype(np.float32)
        base = decode_to_fields(model, make_ckpt(0.0, 1.0), z)
        shifted = decode_to_fields(model, make_ckpt(2.0, 3.0), z)
        c = FIELDS.index("phi")                              # linear field: shifted = base * 3 + 2
        np.testing.assert_allclose(shifted[:, c], base[:, c] * 3.0 + 2.0, rtol=1e-4, atol=1e-4)


class TestRelativeFieldError:
    def test_zero_for_identical_arrays(self):
        x = np.random.default_rng(0).uniform(1, 2, (5, 7, 25, 50))
        np.testing.assert_array_equal(relative_field_error(x, x), np.zeros(7))

    def test_uniform_relative_perturbation(self):
        x = np.random.default_rng(1).uniform(1, 2, (5, 7, 25, 50))
        err = relative_field_error(1.1 * x, x)
        np.testing.assert_allclose(err, 0.1, rtol=1e-6)      # |0.1 x| / mean|x|, averaged = 0.1

    def test_returns_one_value_per_field(self):
        x = np.ones((3, 7, 25, 50))
        assert relative_field_error(x + 1, x).shape == (7,)
