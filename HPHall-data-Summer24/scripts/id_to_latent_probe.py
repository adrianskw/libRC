"""Can the latent state z(t) be recovered from a window of PAST measured Id?  (no reservoir)

Takens-style check of whether the information exists, independent of the RC:
inputs are Id(t), Id(t-s), ... over a window, targets are z at t. Chronological
split identical to the pipeline (n_train = 17000); ridge and a small MLP.
Complements rc_with_id.py: if this works but the RC's Id-driven infer does not,
the information is present and the reservoir's infer path is the bottleneck.

Usage: python id_to_latent_probe.py [latent_dim]   (default 3)
Output: latent{D}/id_map/id_window_to_latent.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src import observables as obs  # noqa: E402

BASE = os.path.join(REPO_ROOT, "HPHall-data-Summer24")
LATENT_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3
os.makedirs(f"{BASE}/latent{LATENT_DIM}/id_map", exist_ok=True)
Id = obs.load_discharge_current(BASE)
lat = np.load(f"{BASE}/latent{LATENT_DIM}/latent_full.npz"); z = lat["z"].astype(float); n_train = int(lat["n_train"]); N = len(z)
Idn, _, _ = obs.zscore(Id, n_train)
zn, _, _ = obs.zscore(z.T, n_train); zn = zn.T


def feats(idx, lags):
    return np.stack([Idn[idx - l] for l in lags], axis=1)


def ridge(Xtr, Ytr, Xva, alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100)):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A, B = (Xtr - mu) / sd, (Xva - mu) / sd
    cut = int(len(A) * 0.85); best = (None, 1e18)
    for a in alphas:
        w = np.linalg.solve(A[:cut].T @ A[:cut] + a * np.eye(A.shape[1]), A[:cut].T @ Ytr[:cut])
        e = np.mean((A[cut:] @ w - Ytr[cut:]) ** 2)
        if e < best[1]: best = (a, e)
    w = np.linalg.solve(A.T @ A + best[0] * np.eye(A.shape[1]), A.T @ Ytr)
    return B @ w


def mlp(Xtr, Ytr, Xva, seed=0, epochs=300):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32); Yt = torch.tensor(Ytr, dtype=torch.float32)
    Xv = torch.tensor((Xva - mu) / sd, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(Xt.shape[1], 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, Yt.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            b = perm[i:i + 512]; opt.zero_grad(); nn.functional.mse_loss(net(Xt[b]), Yt[b]).backward(); opt.step()
    with torch.no_grad(): return net(Xv).numpy()


def r2(y, p): return 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


results = []
print("held-out R^2 of the latents recovered from a window of PAST Id only")
print(f"{'window (steps)':>16s} {'taps':>5s} | {'ridge z1':>8s} {'z2':>6s} {'z3':>6s} | {'MLP z1':>7s} {'z2':>6s} {'z3':>6s}")
for span, step in [(0, 1), (50, 5), (150, 5), (300, 10), (600, 20)]:
    lags = [0] if span == 0 else list(range(0, span + 1, step))
    start = max(lags)
    tr, va = np.arange(start, n_train), np.arange(n_train, N)
    Xtr, Xva = feats(tr, lags), feats(va, lags)
    pr = ridge(Xtr, zn[tr], Xva); pm = mlp(Xtr, zn[tr], Xva)
    a, b = r2(zn[va], pr), r2(zn[va], pm)
    print(f"{span:16d} {len(lags):5d} | {a[0]:8.3f} {a[1]:6.3f} {a[2]:6.3f} | {b[0]:7.3f} {b[1]:6.3f} {b[2]:6.3f}")
    results.append({"window_steps": span, "taps": len(lags), "ridge_r2": a.tolist(), "mlp_r2": b.tolist()})

with open(f"{BASE}/latent{LATENT_DIM}/id_map/id_window_to_latent.json", "w") as f:
    json.dump({"latent_dim": LATENT_DIM, "n_train": n_train, "results": results}, f, indent=2)
print(f"wrote {BASE}/latent{LATENT_DIM}/id_map/id_window_to_latent.json")
