"""Visualize AE reconstruction quality and the latent trajectory.

Usage: python evaluate_ae.py [latent_dim]   (default 3)
"""
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src.autoencoder import load_checkpoint  # noqa: E402

BASE = os.path.join(REPO_ROOT, "HPHall-data-Summer24")
LATENT_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT_DIR = f"{BASE}/latent{LATENT_DIM}"

npz = np.load(f"{BASE}/dataset.npz", allow_pickle=True)
data = npz["data"]
field_names = [str(s) for s in npz["field_names"]]
frame_indices = npz["frame_indices"]
x, y = npz["x"], npz["y"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, ckpt = load_checkpoint(f"{OUT_DIR}/conv_ae.pt", device=device)
mean, std = ckpt["mean"], ckpt["std"]
log_fields = set(ckpt["log_fields"])

# --- normalize full dataset the same way as training ---
data_t = data.copy()
for c, name in enumerate(field_names):
    if name in log_fields:
        data_t[:, c] = np.log10(np.clip(data_t[:, c], 1e-6, None))
data_norm = (data_t - mean) / std

with torch.no_grad():
    full_x = torch.from_numpy(data_norm).float().to(device)
    recon_norm, z_all = model(full_x)
    recon_norm = recon_norm.cpu().numpy()
    z_all = z_all.cpu().numpy()

# invert normalization back to physical units
recon_t = recon_norm * std + mean
recon = recon_t.copy()
for c, name in enumerate(field_names):
    if name in log_fields:
        recon[:, c] = 10 ** recon_t[:, c]

# ---- 1. reconstruction comparison for one mid-sequence frame ----
frame_i = len(frame_indices) // 2
fig, axes = plt.subplots(2, 7, figsize=(22, 6))
for c, name in enumerate(field_names):
    vmin, vmax = data[frame_i, c].min(), data[frame_i, c].max()
    axes[0, c].pcolormesh(x, y, data[frame_i, c], shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, c].set_title(name)
    axes[0, c].set_aspect("equal")
    axes[0, c].set_xticks([]); axes[0, c].set_yticks([])
    axes[1, c].pcolormesh(x, y, recon[frame_i, c], shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, c].set_aspect("equal")
    axes[1, c].set_xticks([]); axes[1, c].set_yticks([])
axes[0, 0].set_ylabel("original", fontsize=11)
axes[1, 0].set_ylabel("reconstructed", fontsize=11)
fig.suptitle(f"Frame {frame_indices[frame_i]}: original (top) vs AE reconstruction (bottom)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/ae_reconstruction.png", dpi=150)
print(f"wrote {OUT_DIR}/ae_reconstruction.png")

# ---- 2. per-channel reconstruction error (relative, over all frames) ----
rel_err = np.abs(recon - data) / (np.abs(data).mean(axis=(0, 2, 3), keepdims=True) + 1e-30)
mean_rel_err = rel_err.mean(axis=(0, 2, 3))
for name, err in zip(field_names, mean_rel_err):
    print(f"  {name:10s} mean relative error: {err:.3%}")

# ---- 3. latent trajectory colored by time (3D scatter, or 2D phase plot) ----
lat = np.load(f"{OUT_DIR}/latent.npz")
z = lat["z"]
n_train = int(lat["n_train"])
D_lat = z.shape[1]

fig = plt.figure(figsize=(8, 7))
if D_lat >= 3:
    ax = fig.add_subplot(111, projection="3d")
    sca = ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=frame_indices, cmap="viridis", s=6)
    ax.plot(z[:n_train, 0], z[:n_train, 1], z[:n_train, 2], color="gray", alpha=0.3, linewidth=0.5)
    ax.plot(z[n_train:, 0], z[n_train:, 1], z[n_train:, 2], color="red", alpha=0.5, linewidth=0.5, label="val (held-out, later in time)")
    ax.set_zlabel("z3")
else:
    ax = fig.add_subplot(111)
    sca = ax.scatter(z[:, 0], z[:, 1], c=frame_indices, cmap="viridis", s=6)
    ax.plot(z[:n_train, 0], z[:n_train, 1], color="gray", alpha=0.3, linewidth=0.5)
    ax.plot(z[n_train:, 0], z[n_train:, 1], color="red", alpha=0.5, linewidth=0.5, label="val (held-out, later in time)")
ax.set_xlabel("z1"); ax.set_ylabel("z2")
ax.set_title(f"{D_lat}D latent trajectory (color = frame index / time)")
ax.legend()
cbar = fig.colorbar(sca, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("frame index")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/latent_trajectory.png", dpi=150)
print(f"wrote {OUT_DIR}/latent_trajectory.png")

# ---- 4. each latent coordinate vs time ----
fig, axes = plt.subplots(D_lat, 1, figsize=(10, 2 * D_lat), sharex=True)
axes = np.atleast_1d(axes)
for i in range(D_lat):
    axes[i].plot(frame_indices, z[:, i], linewidth=0.8)
    axes[i].axvline(frame_indices[n_train], color="red", linestyle="--", alpha=0.5)
    axes[i].set_ylabel(f"z{i+1}")
axes[-1].set_xlabel("frame index")
axes[0].set_title("Latent coordinates vs. time (red line = train/val split)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/latent_vs_time.png", dpi=150)
print(f"wrote {OUT_DIR}/latent_vs_time.png")
