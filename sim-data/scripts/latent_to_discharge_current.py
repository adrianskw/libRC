"""How well can the AE latent trajectory predict the discharge current Id(t)?

A probe of the autoencoder: if the latent (3-D or 2-D) keeps the physics that
sets Id, a mapping z -> Id should work well on held-out (later-in-time) frames.
Id comes from log.dat (col 2), one row per simulation step, aligned 1:1 with
the 20,000 field frames.

Mappings tried (all fit on the training block only, scored on the held-out
block, chronological split identical to the AE/RC's n_train):
  mean            predict the train mean (R^2 ~ 0 reference)
  linear z(t)     ridge on the instantaneous latent
  poly3 z(t)      ridge on cubic polynomial features of z(t)
  MLP z(t)        small MLP on the instantaneous latent
  linear window   ridge on z(t), z(t-5), ..., z(t-45)   (10 taps)
  MLP window      small MLP on the same 10-tap window
  fields->Id      reference: PCA-64 of the full normalized fields (same
                  log10/z-score as the AE) -> ridge / MLP. Shows what is
                  recoverable from the raw fields, i.e. what the latent could
                  at best retain.

Usage: python latent_to_discharge_current.py [latent_dim]   (default 3)
"""
import itertools
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LATENT_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT_DIR = f"{BASE}/latent{LATENT_DIM}"
RES_DIR = f"{OUT_DIR}/id_map"
os.makedirs(RES_DIR, exist_ok=True)

LAGS = [5 * i for i in range(10)]          # 0..45
MAX_LAG = max(LAGS)
N_SEEDS = 3
MLP_EPOCHS = 400
DEVICE = "cpu"

# ---- data ----
log = np.loadtxt(f"{BASE}/log.dat", skiprows=1)
Id = log[:, 1]
lat = np.load(f"{OUT_DIR}/latent_full.npz")
z = lat["z"].astype(np.float64)
n_train = int(lat["n_train"])
N = len(z)
assert len(Id) == N, (len(Id), N)
print(f"latent_dim={LATENT_DIM}  N={N}  n_train={n_train}  n_val={N - n_train}")

# time indices used by every model: start after the oldest tap so all models
# are scored on exactly the same held-out frames
tr_idx = np.arange(MAX_LAG, n_train)
va_idx = np.arange(n_train, N)
y_tr, y_va = Id[tr_idx], Id[va_idx]
y_mu, y_sd = y_tr.mean(), y_tr.std()


def window_feats(zz, idx, lags):
    return np.concatenate([zz[idx - l] for l in lags], axis=1)


def poly_feats(x, degree=3):
    cols = [np.ones(len(x))]
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(x.shape[1]), d):
            cols.append(np.prod(x[:, combo], axis=1))
    return np.stack(cols, axis=1)


def standardize(Xtr, Xva):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xva - mu) / sd


def metrics(y_true, y_pred):
    err = y_pred - y_true
    r2 = 1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    return dict(r2=float(r2), rmse_A=float(np.sqrt(np.mean(err ** 2))),
                mae_A=float(np.mean(np.abs(err))),
                corr=float(np.corrcoef(y_true, y_pred)[0, 1]))


def fit_ridge(Xtr, ytr, Xva):
    """Ridge with alpha chosen on the last 15% of the train block, refit on all of train."""
    Xtr_s, Xva_s = standardize(Xtr, Xva)
    ytr_c = (ytr - y_mu) / y_sd
    cut = int(len(Xtr_s) * 0.85)
    best = (None, np.inf)
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3]:
        A = Xtr_s[:cut]
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ ytr_c[:cut])
        e = np.mean((Xtr_s[cut:] @ w - ytr_c[cut:]) ** 2)
        if e < best[1]:
            best = (alpha, e)
    alpha = best[0]
    w = np.linalg.solve(Xtr_s.T @ Xtr_s + alpha * np.eye(Xtr_s.shape[1]), Xtr_s.T @ ytr_c)
    return (Xva_s @ w) * y_sd + y_mu, alpha


def fit_mlp(Xtr, ytr, Xva, hidden=64, seed=0):
    torch.manual_seed(seed)
    Xtr_s, Xva_s = standardize(Xtr, Xva)
    Xt = torch.tensor(Xtr_s, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor((ytr - y_mu) / y_sd, dtype=torch.float32, device=DEVICE)[:, None]
    Xv = torch.tensor(Xva_s, dtype=torch.float32, device=DEVICE)
    net = nn.Sequential(nn.Linear(Xt.shape[1], hidden), nn.Tanh(),
                        nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(MLP_EPOCHS):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            nn.functional.mse_loss(net(Xt[b]), yt[b]).backward()
            opt.step()
    with torch.no_grad():
        return net(Xv).squeeze(1).cpu().numpy() * y_sd + y_mu


def mlp_avg(Xtr, ytr, Xva):
    return np.mean([fit_mlp(Xtr, ytr, Xva, seed=s) for s in range(N_SEEDS)], axis=0)


results, preds = {}, {}


def record(name, pred, extra=None):
    results[name] = metrics(y_va, pred)
    if extra:
        results[name].update(extra)
    preds[name] = pred
    m = results[name]
    print(f"  {name:22s} R2={m['r2']:6.3f}  RMSE={m['rmse_A']:.3f} A  MAE={m['mae_A']:.3f} A  corr={m['corr']:.3f}")


print("\nheld-out block (last 15% of frames, later in time than all training):")
record("mean", np.full_like(y_va, y_mu))

Z_tr, Z_va = z[tr_idx], z[va_idx]
p, a = fit_ridge(Z_tr, y_tr, Z_va)
record("linear z(t)", p, {"alpha": a})
p, a = fit_ridge(poly_feats(standardize(Z_tr, Z_va)[0]), y_tr, poly_feats(standardize(Z_tr, Z_va)[1]))
record("poly3 z(t)", p, {"alpha": a})
record("MLP z(t)", mlp_avg(Z_tr, y_tr, Z_va))

W_tr, W_va = window_feats(z, tr_idx, LAGS), window_feats(z, va_idx, LAGS)
p, a = fit_ridge(W_tr, y_tr, W_va)
record("linear window", p, {"alpha": a})
record("MLP window", mlp_avg(W_tr, y_tr, W_va))

# ---- reference: full fields -> Id ----
print("\nreference from the full fields (PCA-64, same normalization as the AE):")
ckpt = torch.load(f"{OUT_DIR}/conv_ae_full.pt", weights_only=False, map_location="cpu")
mean, std = ckpt["mean"], ckpt["std"]
log_fields = set(ckpt["log_fields"])
field_names = [str(s) for s in ckpt["field_names"]]
ds = np.load(f"{BASE}/dataset_full.npz", allow_pickle=True, mmap_mode="r")
F = np.empty((N, len(field_names) * 25 * 50), dtype=np.float32)
for i in range(0, N, 2000):
    chunk = np.array(ds["data"][i:i + 2000], dtype=np.float32)
    for c, name in enumerate(field_names):
        if name in log_fields:
            chunk[:, c] = np.log10(np.clip(chunk[:, c], 1e-6, None))
    F[i:i + 2000] = ((chunk - mean) / std).reshape(len(chunk), -1)
Ft = torch.from_numpy(F)
mu = Ft[:n_train].mean(0)
_, _, V = torch.pca_lowrank(Ft[:n_train] - mu, q=64, center=False, niter=4)
P = ((Ft - mu) @ V).numpy().astype(np.float64)
del F, Ft
p, a = fit_ridge(P[tr_idx], y_tr, P[va_idx])
record("fields PCA64 linear", p, {"alpha": a})
record("fields PCA64 MLP", mlp_avg(P[tr_idx], y_tr, P[va_idx]))

with open(f"{RES_DIR}/metrics.json", "w") as f:
    json.dump({"latent_dim": LATENT_DIM, "n_train": n_train, "n_val": N - n_train,
               "id_train_mean_A": float(y_mu), "id_train_std_A": float(y_sd),
               "lags": LAGS, "results": results}, f, indent=2)
print(f"\nwrote {RES_DIR}/metrics.json")

# ---- plots ----
names = list(results)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(names[::-1], [results[k]["r2"] for k in names[::-1]], color="#4477aa")
ax.set_xlabel("held-out R$^2$ (discharge current)")
ax.axvline(0, color="k", lw=0.6)
ax.set_title(f"z -> Id mapping quality, latent_dim={LATENT_DIM}")
fig.tight_layout()
fig.savefig(f"{RES_DIR}/r2_bars.png", dpi=150)

show = ["MLP z(t)", "MLP window", "fields PCA64 MLP"]
Mp = min(1500, len(y_va))
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(y_va[:Mp], color="k", lw=1.4, label="true Id")
for k in show:
    ax.plot(preds[k][:Mp], lw=1, alpha=0.85, label=f"{k} (R2={results[k]['r2']:.2f})")
ax.set_xlabel("step into held-out block (50 ns/step)")
ax.set_ylabel("discharge current (A)")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(f"{RES_DIR}/id_timeseries.png", dpi=150)
print(f"wrote {RES_DIR}/r2_bars.png, id_timeseries.png")
