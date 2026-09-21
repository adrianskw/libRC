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


from src.autoencoder import id_loss_terms, save_checkpoint, load_checkpoint  # noqa: E402


def frames(n=6, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).standard_normal((n, 7, 25, 50)).astype(np.float32))


class TestIdModes:
    def test_default_is_original_behavior(self):
        m = ConvAE(in_ch=7, latent_dim=3)
        assert m.id_mode == "none" and m.to_latent.out_features == 3
        recon, z = m(frames())
        assert recon.shape == (6, 7, 25, 50) and z.shape == (6, 3)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            ConvAE(id_mode="bogus")

    def test_supervised_keeps_three_encoder_outputs(self):
        m = ConvAE(in_ch=7, latent_dim=3, id_mode="supervised")
        assert m.to_latent.out_features == 3 and m.n_free == 2
        assert m(frames())[1].shape == (6, 3)

    def test_conditional_encodes_two_and_appends_id_exactly(self):
        m = ConvAE(in_ch=7, latent_dim=3, id_mode="conditional")
        assert m.to_latent.out_features == 2 and m.from_latent.in_features == 3
        x, id_n = frames(), torch.linspace(-1, 1, 6)
        assert m.encode(x).shape == (6, 2)
        recon, z = m(x, id_n)
        assert recon.shape == (6, 7, 25, 50) and z.shape == (6, 3)
        torch.testing.assert_close(z[:, 2], id_n)            # the third coordinate IS Id

    def test_conditional_requires_id(self):
        with pytest.raises(ValueError, match="needs id_n"):
            ConvAE(id_mode="conditional")(frames())

    def test_decoder_input_width_is_the_same_in_every_mode(self):
        for mode in ("none", "supervised", "conditional"):
            assert ConvAE(id_mode=mode).from_latent.in_features == 3     # decode() arity is unchanged


class TestIdLossTerms:
    def test_zero_in_none_mode(self):
        m = ConvAE(id_mode="none")
        a, b = id_loss_terms(m, torch.randn(8, 3), torch.randn(8))
        assert a.item() == 0.0 and b.item() == 0.0

    def test_supervised_term_is_mse_of_last_coordinate(self):
        m = ConvAE(id_mode="supervised")
        z, id_n = torch.randn(32, 3), torch.randn(32)
        term, _ = id_loss_terms(m, z, id_n)
        assert term.item() == pytest.approx(((z[:, -1] - id_n) ** 2).mean().item(), rel=1e-5)

    def test_conditional_has_no_supervised_term(self):
        m = ConvAE(id_mode="conditional")
        term, _ = id_loss_terms(m, torch.randn(16, 3), torch.randn(16))
        assert term.item() == 0.0

    def test_decorr_flags_free_coordinates_that_copy_id(self):
        m = ConvAE(id_mode="supervised")
        id_n = torch.randn(512)
        leaky = torch.stack([id_n, torch.randn(512), id_n], dim=1)     # z1 is a copy of Id
        clean = torch.stack([torch.randn(512), torch.randn(512), id_n], dim=1)
        assert id_loss_terms(m, leaky, id_n)[1].item() > 0.4
        assert id_loss_terms(m, clean, id_n)[1].item() < 0.02

    def test_gradients_flow_to_the_latent(self):
        m = ConvAE(id_mode="supervised")
        z = torch.randn(16, 3, requires_grad=True)
        term, decorr = id_loss_terms(m, z, torch.randn(16))
        (term + decorr).backward()
        assert z.grad is not None and torch.isfinite(z.grad).all()


class TestCheckpointRoundtrip:
    def test_mode_survives_save_and_load(self, tmp_path):
        m = ConvAE(in_ch=7, latent_dim=3, id_mode="conditional")
        p = str(tmp_path / "c.pt")
        save_checkpoint(p, m, np.zeros((1, 7, 1, 1)), np.ones((1, 7, 1, 1)), FIELDS, LOG, 3, id_mode="conditional")
        m2, ck = load_checkpoint(p)
        assert m2.id_mode == "conditional" and ck["id_mode"] == "conditional"
        x, id_n = frames(), torch.linspace(-1, 1, 6)
        torch.testing.assert_close(m.eval()(x, id_n)[0], m2(x, id_n)[0])

    def test_checkpoint_from_before_id_mode_existed_loads_as_none(self, tmp_path):
        m = ConvAE(in_ch=7, latent_dim=3)
        p = str(tmp_path / "old.pt")
        save_checkpoint(p, m, np.zeros((1, 7, 1, 1)), np.ones((1, 7, 1, 1)), FIELDS, LOG, 3)   # no id_mode key
        m2, ck = load_checkpoint(p)
        assert "id_mode" not in ck and m2.id_mode == "none"
