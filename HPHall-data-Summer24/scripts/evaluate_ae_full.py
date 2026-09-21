"""Evaluate the full-dataset fine-tuned AE: reconstruction quality + latent trajectory.

Usage: python evaluate_ae_full.py [latent_dim]   (default 3)
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

npz = np.load(f"{BASE}/dataset_full.npz", allow_pickle=True)
data = npz["data"]
field_names = [str(s) for s in npz["field_names"]]
frame_indices = npz["frame_indices"]
x, y = npz["x"], npz["y"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, ckpt = load_checkpoint(f"{OUT_DIR}/conv_ae_full.pt", device=device)
mean, std = ckpt["mean"], ckpt["std"]
log_fields = set(ckpt["log_fields"])

data_t = data.copy()
for c, name in enumerate(field_names):
    if name in log_fields:
        data_t[:, c] = np.log10(np.clip(data_t[:, c], 1e-6, None))
data_norm = (data_t - mean) / std

N = data.shape[0]
recon_norm = np.empty_like(data_norm, dtype=np.float32)
z_all = np.empty((N, ckpt["latent_dim"]), dtype=np.float32)
with torch.no_grad():
    for i in range(0, N, 2000):
        chunk = torch.from_numpy(data_norm[i:i + 2000]).float().to(device)
        r, z = model(chunk)
        recon_norm[i:i + 2000] = r.cpu().numpy()
        z_all[i:i + 2000] = z.cpu().numpy()

recon_t = recon_norm * std + mean
recon = recon_t.copy()
for c, name in enumerate(field_names):
    if name in log_fields:
        recon[:, c] = 10 ** recon_t[:, c]

frame_i = N // 2
fig, axes = plt.subplots(2, 7, figsize=(22, 6))
for c, name in enumerate(field_names):
    vmin, vmax = data[frame_i, c].min(), data[frame_i, c].max()
    axes[0, c].pcolormesh(x, y, data[frame_i, c], shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, c].set_title(name)
    axes[0, c].set_aspect("equal"); axes[0, c].set_xticks([]); axes[0, c].set_yticks([])
    axes[1, c].pcolormesh(x, y, recon[frame_i, c], shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, c].set_aspect("equal"); axes[1, c].set_xticks([]); axes[1, c].set_yticks([])
axes[0, 0].set_ylabel("original", fontsize=11)
axes[1, 0].set_ylabel("reconstructed", fontsize=11)
fig.suptitle(f"Frame {frame_indices[frame_i]} (full 20k-frame model): original vs reconstruction")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/ae_reconstruction_full.png", dpi=150)
print(f"wrote {OUT_DIR}/ae_reconstruction_full.png")

rel_err = np.abs(recon - data) / (np.abs(data).mean(axis=(0, 2, 3), keepdims=True) + 1e-30)
mean_rel_err = rel_err.mean(axis=(0, 2, 3))
err_summary = []
for name, err in zip(field_names, mean_rel_err):
    print(f"  {name:10s} mean relative error: {err:.3%}")
    err_summary.append((name, float(err) * 100))

lat = np.load(f"{OUT_DIR}/latent_full.npz")
z = lat["z"]
n_train = int(lat["n_train"])
D_lat = z.shape[1]

fig = plt.figure(figsize=(8, 7))
if D_lat >= 3:
    ax = fig.add_subplot(111, projection="3d")
    sca = ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=frame_indices, cmap="viridis", s=2, alpha=0.5)
    ax.set_zlabel("z3")
else:
    ax = fig.add_subplot(111)
    sca = ax.scatter(z[:, 0], z[:, 1], c=frame_indices, cmap="viridis", s=2, alpha=0.5)
ax.set_xlabel("z1"); ax.set_ylabel("z2")
ax.set_title(f"{D_lat}D latent trajectory, all 20,000 frames")
cbar = fig.colorbar(sca, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("frame index")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/latent_trajectory_full.png", dpi=150)
print(f"wrote {OUT_DIR}/latent_trajectory_full.png")

import json
with open(f"{OUT_DIR}/err_summary_full.json", "w") as f:
    json.dump(err_summary, f)
print("wrote err_summary_full.json:", err_summary)
