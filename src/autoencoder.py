# -*- coding: utf-8 -*-
"""
Convolutional autoencoder used to compress Hall-thruster simulation frames
(7 fields x 25x50 grid) into a low-dimensional latent trajectory for the
reservoir-computing pipeline. Canonical home for the ConvAE architecture and
its checkpoint format, shared by the train/evaluate/decode scripts under
HPHall-data-Summer24/scripts/.

The optional `id_mode` makes a measured scalar (the discharge current, z-scored as
`id_n`) part of the 3-vector the decoder sees; see HPHall-data-Summer24/scripts/
ID_TARGETING.md. "none" is the original behavior and the default.

  none         z = E(x) in R^D.                               Id is ignored.
  supervised   z = E(x) in R^D; last coordinate is trained    Encoder needs only x.
               toward id_n through an extra loss term.
  conditional  E(x) in R^(D-1); decoder input is [E(x), id_n]. Encoder needs x and id_n.

@author: Adrian Wong
"""
import numpy as np
import torch
import torch.nn as nn


ID_MODES = ("none", "supervised", "conditional")


class ConvAE(nn.Module):
    def __init__(self, in_ch=7, latent_dim=3, id_mode="none"):
        super().__init__()
        if id_mode not in ID_MODES:
            raise ValueError(f"id_mode must be one of {ID_MODES}, got {id_mode!r}")
        self.id_mode = id_mode
        self.latent_dim = latent_dim
        self.n_free = latent_dim - 1 if id_mode != "none" else latent_dim   # coordinates not tied to Id
        n_enc = latent_dim - 1 if id_mode == "conditional" else latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.enc_shape = (64, 7, 13)
        flat = 64 * 7 * 13
        self.to_latent = nn.Linear(flat, n_enc)
        self.from_latent = nn.Linear(latent_dim, flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=(0, 0)), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(16, in_ch, 3, stride=1, padding=1),
        )

    def encode(self, x):
        """Encoder output: D dims, or D-1 free dims in conditional mode."""
        return self.to_latent(self.enc(x).flatten(1))

    def encode_full(self, x, id_n=None):
        """The full D-vector the decoder consumes: [free..., id_n] in conditional mode."""
        z = self.encode(x)
        if self.id_mode != "conditional":
            return z
        if id_n is None:
            raise ValueError("conditional mode needs id_n to build the full latent vector")
        return torch.cat([z, id_n.reshape(-1, 1).to(z.dtype)], dim=1)

    def decode(self, z):
        h = self.from_latent(z).view(-1, *self.enc_shape)
        out = self.dec(h)
        out = out[:, :, :25, :50]
        if out.shape[-2:] != (25, 50):
            out = nn.functional.pad(out, (0, 50 - out.shape[-1], 0, 25 - out.shape[-2]))
        return out

    def forward(self, x, id_n=None):
        z = self.encode_full(x, id_n)
        return self.decode(z), z


def id_loss_terms(model, z, id_n):
    """Extra loss terms tying the latent to Id for one batch; both zero when `id_mode` is "none".

    Returns (id_term, decorr): the supervised MSE between the last latent coordinate and
    id_n (supervised mode only), and the mean squared batch correlation between each free
    coordinate and id_n, which penalizes the free coordinates for re-encoding Id.
    """
    zero = z.new_zeros(())
    if model.id_mode == "none" or id_n is None:
        return zero, zero
    id_n = id_n.reshape(-1).to(z.dtype)
    id_term = nn.functional.mse_loss(z[:, -1], id_n) if model.id_mode == "supervised" else zero
    free = z[:, :model.n_free]
    fc, ic = free - free.mean(0), id_n - id_n.mean()
    corr = (fc * ic[:, None]).mean(0) / (fc.std(0, unbiased=False) * ic.std(unbiased=False) + 1e-8)
    return id_term, (corr ** 2).mean()


def save_checkpoint(path, model, mean, std, field_names, log_fields, latent_dim, **extra):
    """Save a ConvAE checkpoint in the standard dict format used across the pipeline."""
    torch.save({
        "model_state": model.state_dict(),
        "mean": mean,
        "std": std,
        "field_names": field_names,
        "log_fields": list(log_fields),
        "latent_dim": latent_dim,
        **extra,
    }, path)


def load_checkpoint(path, device=None):
    """Load a ConvAE checkpoint, returning (model, ckpt) with model in eval mode on `device`."""
    ckpt = torch.load(path, weights_only=False, map_location=device)
    model = ConvAE(in_ch=len(ckpt["field_names"]), latent_dim=ckpt["latent_dim"],
                   id_mode=ckpt.get("id_mode", "none"))
    model.load_state_dict(ckpt["model_state"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, ckpt


def decode_to_fields(model, ckpt, z, device=None, chunk=1000):
    """Raw AE latents (n, latent_dim) -> physical fields (n, C, 25, 50).

    Runs the frozen decoder in chunks, then undoes the z-score and the log10 applied
    to ckpt["log_fields"] during training.
    """
    field_names = [str(s) for s in ckpt["field_names"]]
    log_fields = set(ckpt["log_fields"])
    out = np.empty((z.shape[0], len(field_names), 25, 50), dtype=np.float32)
    for i in range(0, z.shape[0], chunk):
        zt = torch.from_numpy(np.ascontiguousarray(z[i:i + chunk])).float()
        if device is not None:
            zt = zt.to(device)
        with torch.no_grad():
            norm = model.decode(zt).cpu().numpy()
        phys = norm * ckpt["std"] + ckpt["mean"]
        for c, name in enumerate(field_names):
            if name in log_fields:
                phys[:, c] = 10 ** phys[:, c]
        out[i:i + chunk] = phys
    return out


def relative_field_error(recon, true):
    """Per-field mean relative error, mean|recon - true| / mean|true|, over frames and pixels."""
    rel = np.abs(recon - true) / (np.abs(true).mean(axis=(0, 2, 3), keepdims=True) + 1e-30)
    return rel.mean(axis=(0, 2, 3))
