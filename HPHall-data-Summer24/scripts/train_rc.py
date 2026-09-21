"""Reservoir computer trained on the conv-AE's latent trajectory (N-D).

The AE is frozen and acts purely as a feature extractor (7 fields x 25x50
grid -> D latent numbers per frame, already computed in latentN/latent_full.npz).
This script drives a small mapRC (libRC) with that D-variable trajectory:

  listen  -- drive the reservoir with the true latent series (train portion)
  train   -- ridge-regression readout W so W@r reconstructs the driving signal
  echo    -- disconnect the driving signal and let the reservoir run on its
             own predictions (autonomous mode) for the held-out val window --
             this is the test of whether the RC learned the breathing-mode
             attractor itself, not just memorized the training trace
  infer   -- re-drive with only z1 observed and let the RC estimate the
             remaining latent channels from that single observed channel

Chronological train/val split matches the AE's own split (n_train=17000)
so "learned the dynamics" means "generalizes to unseen future cycles",
consistent with how the AE was validated.

Usage: python train_rc.py [latent_dim] [n_res]   (default 3, 200)
"""
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src import mapRC  # noqa: E402

BASE = os.path.join(REPO_ROOT, "HPHall-data-Summer24")
LATENT_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_RES = int(sys.argv[2]) if len(sys.argv) > 2 else 200       # reservoir size
OUT_DIR = f"{BASE}/latent{LATENT_DIM}"
RC_DIR = f"{OUT_DIR}/n{N_RES}"
os.makedirs(RC_DIR, exist_ok=True)

SEED = 0
RHO = 0.9           # spectral radius of the connection matrix
DEGREE = 10         # connections per reservoir node (5% density at N=200)
ALPHA = 1e-3        # ridge regularization for the readout fit
TRAIN_SKIP = 200    # discard this many listen steps (reservoir sync transient) from the fit
DRIVE_INDEX = [0]   # which latent channel stays observed during infer()

np.random.seed(SEED)

# ---- load the AE's latent trajectory (all 20,000 frames) ----
lat = np.load(f"{OUT_DIR}/latent_full.npz")
z = lat["z"]                      # (20000, 3)
frame_indices = lat["frame_indices"].astype(int)
n_train = int(lat["n_train"])     # 17000, same chronological split as the AE

z_t = z.T                         # libRC convention: (D, M)
D, M = z_t.shape
n_val = M - n_train

mean = z_t[:, :n_train].mean(axis=1, keepdims=True)
std = z_t[:, :n_train].std(axis=1, keepdims=True) + 1e-8
z_norm = (z_t - mean) / std

y_in = z_norm[:, :n_train]
y_val = z_norm[:, n_train:]

sigma = 2.0 / np.max(np.max(y_in, axis=1) - np.min(y_in, axis=1))
print(f"D={D} n_train={n_train} n_val={n_val} sigma={sigma:.4f}")

# ---- build + drive the reservoir ----
RC = mapRC(N_RES, bias=True)
RC.makeConnectionMatDegree(RHO, degree=DEGREE, loc=-1, scale=1)
RC.makeInputMat(D, sigma, randMin=-1, randMax=1)

RC.listen(y_in)
RC.train(y_in, start=TRAIN_SKIP, alpha=ALPHA)

# ---- echo: fully autonomous run over the held-out window ----
RC.echo(n_val)

# ---- infer: re-driven with only z1 observed ----
RC.infer(y_val[DRIVE_INDEX], DRIVE_INDEX)
PCmat = RC.inferPC(y_val)

# ---- metrics ----
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


echo_rmse = rmse(RC.y_echo, y_val)
infer_rmse = rmse(RC.y_infer, y_val)
labels = [f"z{i+1}" for i in range(D)]
print("\nnormalized-unit RMSE vs. held-out val window:")
for i, name in enumerate(labels):
    print(f"  {name}  echo={echo_rmse[i]:.3f}  infer={infer_rmse[i]:.3f}  "
          f"(std of val itself = {y_val[i].std():.3f})")
print(f"\ninferPC (unobserved-channel reconstruction, driving only z1):")
for i, name in enumerate(labels):
    if i in DRIVE_INDEX:
        continue
    print(f"  {name}: PC = {PCmat[i, i]:.3f}")

# ---- save trained RC + results for reuse ----
with open(f"{RC_DIR}/rc_model.pkl", "wb") as f:
    pickle.dump({
        "N": N_RES, "D": D, "rho": RHO, "degree": DEGREE, "alpha": ALPHA,
        "sigma": sigma, "seed": SEED, "A": RC.A, "B": RC.B, "W": RC.W,
        "mask": RC.mask, "mean": mean, "std": std, "n_train": n_train,
    }, f)
np.savez(
    f"{RC_DIR}/rc_results.npz",
    y_val=y_val, y_echo=RC.y_echo, y_infer=RC.y_infer,
    frame_indices_val=frame_indices[n_train:], mean=mean, std=std,
)
print(f"saved {RC_DIR}/rc_model.pkl, {RC_DIR}/rc_results.npz")

# ---- plots ----
Mplot = min(3000, n_val)
fig, axs = plt.subplots(D, 1, figsize=(10, 7), sharex=True)
fig.subplots_adjust(hspace=0.15)
for i in range(D):
    axs[i].plot(y_val[i, :Mplot], label="true", lw=1.2)
    axs[i].plot(RC.y_echo[i, :Mplot], label="echo (autonomous)", lw=1, alpha=0.85)
    axs[i].plot(RC.y_infer[i, :Mplot], label="infer (z1-driven)", lw=1, alpha=0.85)
    axs[i].set_ylabel(labels[i])
axs[0].legend(loc="upper right", fontsize=9)
axs[0].set_title(f"N={N_RES} reservoir: held-out latent trajectory vs. RC reconstruction")
axs[-1].set_xlabel("step (val window, 50 ns/step)")
fig.tight_layout()
fig.savefig(f"{RC_DIR}/rc_timeseries.png", dpi=150)
print(f"wrote {RC_DIR}/rc_timeseries.png")

fig = plt.figure(figsize=(15, 5))
titles = ["true (held-out)", "echo (autonomous)", "infer (z1-driven)"]
datasets = [y_val, RC.y_echo, RC.y_infer]
for k, (title, y) in enumerate(zip(titles, datasets)):
    if D >= 3:
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.plot(y[0], y[1], y[2], lw=0.5)
        ax.set_zlabel("z3")
    else:
        ax = fig.add_subplot(1, 3, k + 1)
        ax.plot(y[0], y[1], lw=0.5)
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
    ax.set_title(title)
fig.suptitle("Breathing-mode attractor: ground truth vs. reservoir reconstruction")
fig.tight_layout()
fig.savefig(f"{RC_DIR}/rc_attractor_compare.png", dpi=150)
print(f"wrote {RC_DIR}/rc_attractor_compare.png")
