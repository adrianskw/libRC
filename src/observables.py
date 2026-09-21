# -*- coding: utf-8 -*-
"""
Scalar observables measured alongside the field data (discharge current,
thrust, beam current, ...) and helpers for using them as extra reservoir
channels next to the autoencoder latents.

Every function here is pure numpy. Arrays follow the reservoir convention
(channels x time), so a state built by `stack_state` can be handed straight to
`Reservoir.listen()` / `train()`.

@author: Adrian Wong
"""
import numpy as np

# Short names for the log.dat columns used so far. Any column can also be
# requested by its full header text (e.g. "Thrust").
LOG_ALIASES = {
    "id": "Discharge current (A)",
    "vd": "Discharge voltage (V)",
    "thrust": "Thrust",
    "beam": "Total beam current (A)",
}


def load_log(base):
    """Read <base>/log.dat into {column header: 1-D array}, one value per simulation step."""
    path = f"{base}/log.dat"
    with open(path) as f:
        header = f.readline()
    names = [s.strip().strip('"').strip() for s in header.split("=", 1)[1].split(",")]
    data = np.loadtxt(path, skiprows=1)
    if data.ndim != 2 or data.shape[1] != len(names):
        raise ValueError(f"{path}: {len(names)} header columns but data shape {data.shape}")
    return {name: data[:, i] for i, name in enumerate(names)}


def load_observable(base, name):
    """One log.dat column by alias ('id', 'thrust', ...) or by its exact header text."""
    log = load_log(base)
    key = LOG_ALIASES.get(name, name)
    if key not in log:
        raise KeyError(f"no log.dat column {name!r}; available: {sorted(log)}")
    return log[key]


def load_discharge_current(base):
    """Discharge current Id(t) in amps."""
    return load_observable(base, "id")


def zscore(x, n_train, axis=-1):
    """Standardize with mean/std of the first `n_train` samples only.

    Returns (x_norm, mean, std) with mean/std kept broadcastable (keepdims), the
    same convention `train_rc.py` uses for the latents, so a normalized channel can
    be un-normalized with `x_norm * std + mean`.
    """
    x = np.asarray(x, dtype=float)
    train = np.take(x, np.arange(n_train), axis=axis)
    mean = train.mean(axis=axis, keepdims=True)
    std = train.std(axis=axis, keepdims=True) + 1e-8
    return (x - mean) / std, mean, std


def smooth(x, window, causal=True):
    """Moving average over `window` samples along the last axis.

    causal=True uses only past samples (what a hardware measurement could do), with
    an expanding window over the first window-1 samples; causal=False is centered.
    window <= 1 returns x unchanged.
    """
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    if not causal:
        pad = window // 2
        padded = np.concatenate(
            [np.repeat(x[..., :1], pad, axis=-1), x, np.repeat(x[..., -1:], window - 1 - pad, axis=-1)],
            axis=-1)
        csum = np.cumsum(padded, axis=-1)
        csum = np.concatenate([np.zeros_like(csum[..., :1]), csum], axis=-1)
        return (csum[..., window:] - csum[..., :-window]) / window
    csum = np.cumsum(x, axis=-1)
    out = csum.copy()
    out[..., window:] = csum[..., window:] - csum[..., :-window]
    counts = np.minimum(np.arange(1, x.shape[-1] + 1), window)
    return out / counts


def latent_channels(z):
    """Split a (M, D) latent array into {'z1': ..., 'z2': ..., ...}, each length M."""
    z = np.asarray(z)
    return {f"z{i + 1}": z[:, i] for i in range(z.shape[1])}


def stack_state(channels, order):
    """Stack named 1-D channels into a (len(order), M) state array, in `order`."""
    missing = [k for k in order if k not in channels]
    if missing:
        raise KeyError(f"unknown channels {missing}; available: {sorted(channels)}")
    lengths = {k: len(channels[k]) for k in order}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"channel lengths differ: {lengths}")
    return np.vstack([np.asarray(channels[k], dtype=float) for k in order])
