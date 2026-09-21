# -*- coding: utf-8 -*-
"""
Windowed observer: estimate the latent state from a window of past measurements of a
scalar observable (the discharge current).

A scalar observable of a >=3-D system determines the state only through its history
(Takens), so the input is Id(t), Id(t-s), ..., Id(t-W): a tapped delay line. The map from
that window to the latent vector is fit on the training block only, by ridge regression
or by a small MLP ensemble. It is causal (never reads Id after time t) and does not run
free, so it is the "infer" counterpart to the reservoir's infer mode, without the
exposure bias of feeding back its own estimates.

@author: Adrian Wong
"""
import numpy as np
import torch
import torch.nn as nn


def window_features(x, idx, lags):
    """Stack x[idx - lag] for each lag: (len(idx), len(lags)). Every idx must satisfy idx >= max(lags)."""
    x = np.asarray(x)
    idx = np.asarray(idx)
    if idx.min() < max(lags):
        raise ValueError(f"idx must be >= max(lags)={max(lags)} so the window stays inside the record")
    return np.stack([x[idx - l] for l in lags], axis=1)


class WindowObserver:
    """Id window -> latent vector.

    kind="ridge": closed-form ridge, alpha picked on the last 15% of the training block.
    kind="mlp"  : two hidden tanh layers, `n_seeds` independent fits averaged.
    train_noise : std (in units of the input's own std) of Gaussian noise added to the MLP
                  training inputs each step, to make the observer robust to measurement noise.
    """

    def __init__(self, lags, kind="mlp", hidden=128, epochs=300, n_seeds=3, lr=2e-3, weight_decay=1e-4,
                 train_noise=0.0, device=None, alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100)):
        if kind not in ("ridge", "mlp"):
            raise ValueError("kind must be 'ridge' or 'mlp'")
        self.lags = list(lags)
        self.kind, self.hidden, self.epochs, self.n_seeds = kind, hidden, epochs, n_seeds
        self.lr, self.weight_decay, self.train_noise, self.alphas = lr, weight_decay, train_noise, alphas
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, x, targets, train_idx):
        """x: (M,) measured series; targets: (M, K) latent series; train_idx: time indices to fit on."""
        train_idx = np.asarray(train_idx)
        X = window_features(x, train_idx, self.lags)
        Y = np.asarray(targets, dtype=float)[train_idx]
        self.x_mu, self.x_sd = X.mean(0), X.std(0) + 1e-8
        self.y_mu, self.y_sd = Y.mean(0), Y.std(0) + 1e-8
        Xs, Ys = (X - self.x_mu) / self.x_sd, (Y - self.y_mu) / self.y_sd
        if self.kind == "ridge":
            self._fit_ridge(Xs, Ys)
        else:
            self.nets = [self._fit_mlp(Xs, Ys, seed) for seed in range(self.n_seeds)]
        return self

    def predict(self, x, idx):
        Xs = (window_features(x, idx, self.lags) - self.x_mu) / self.x_sd
        if self.kind == "ridge":
            out = Xs @ self.w
        else:
            xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                out = np.mean([net(xt).cpu().numpy() for net in self.nets], axis=0)
        return out * self.y_sd + self.y_mu

    def _fit_ridge(self, Xs, Ys):
        cut = int(len(Xs) * 0.85)
        best_alpha, best_err = None, np.inf
        for a in self.alphas:
            w = np.linalg.solve(Xs[:cut].T @ Xs[:cut] + a * np.eye(Xs.shape[1]), Xs[:cut].T @ Ys[:cut])
            err = np.mean((Xs[cut:] @ w - Ys[cut:]) ** 2)
            if err < best_err:
                best_alpha, best_err = a, err
        self.alpha = best_alpha
        self.w = np.linalg.solve(Xs.T @ Xs + best_alpha * np.eye(Xs.shape[1]), Xs.T @ Ys)

    def _fit_mlp(self, Xs, Ys, seed):
        torch.manual_seed(seed)
        Xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        Yt = torch.tensor(Ys, dtype=torch.float32, device=self.device)
        net = nn.Sequential(nn.Linear(Xt.shape[1], self.hidden), nn.Tanh(),
                            nn.Linear(self.hidden, self.hidden), nn.Tanh(),
                            nn.Linear(self.hidden, Yt.shape[1])).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt), device=self.device)
            for i in range(0, len(Xt), 512):
                b = perm[i:i + 512]
                xb = Xt[b]
                if self.train_noise > 0:
                    xb = xb + self.train_noise * torch.randn_like(xb)
                opt.zero_grad()
                nn.functional.mse_loss(net(xb), Yt[b]).backward()
                opt.step()
        return net.eval()
