# -*- coding: utf-8 -*-
"""
Convolutional autoencoder used to compress Hall-thruster simulation frames
(7 fields x 25x50 grid) into a low-dimensional latent trajectory for the
reservoir-computing pipeline. Canonical home for the ConvAE architecture and
its checkpoint format, shared by the train/evaluate/decode scripts under
HPHall-data-Summer24/scripts/.

@author: Adrian Wong
"""
import torch
import torch.nn as nn


class ConvAE(nn.Module):
    def __init__(self, in_ch=7, latent_dim=3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.enc_shape = (64, 7, 13)
        flat = 64 * 7 * 13
        self.to_latent = nn.Linear(flat, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=(0, 0)), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(16, in_ch, 3, stride=1, padding=1),
        )

    def encode(self, x):
        return self.to_latent(self.enc(x).flatten(1))

    def decode(self, z):
        h = self.from_latent(z).view(-1, *self.enc_shape)
        out = self.dec(h)
        out = out[:, :, :25, :50]
        if out.shape[-2:] != (25, 50):
            out = nn.functional.pad(out, (0, 50 - out.shape[-1], 0, 25 - out.shape[-2]))
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


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
    model = ConvAE(in_ch=len(ckpt["field_names"]), latent_dim=ckpt["latent_dim"])
    model.load_state_dict(ckpt["model_state"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, ckpt
