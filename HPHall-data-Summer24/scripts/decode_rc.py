"""Decode the RC's echo/infer latent predictions back to physical fields.

Closes the full loop: true fields -> [frozen AE encoder] -> N-D latent ->
[RC, driven or autonomous] -> predicted N-D latent -> [frozen AE decoder]
-> predicted 7-field x 25x50 grid. Compares against the true fields at
the same simulation frames to see what the RC's latent-space errors
(and, for echo, its phase drift) actually look like once mapped back
onto the physics.

Usage: python decode_rc.py [latent_dim] [n_res]   (default 3, 200)
"""
import json
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src.autoencoder import load_checkpoint, decode_to_fields, relative_field_error  # noqa: E402

BASE = os.path.join(REPO_ROOT, "HPHall-data-Summer24")
LATENT_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_RES = int(sys.argv[2]) if len(sys.argv) > 2 else 200
OUT_DIR = f"{BASE}/latent{LATENT_DIM}"
RC_DIR = f"{OUT_DIR}/n{N_RES}"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- frozen AE decoder ----
model, ckpt = load_checkpoint(f"{OUT_DIR}/conv_ae_full.pt", device=device)
field_names = list(ckpt["field_names"])


# ---- RC latent predictions (still in RC's own z-scored space) ----
rc = np.load(f"{RC_DIR}/rc_results.npz")
lat_mean, lat_std = rc["mean"], rc["std"]          # (D,1), latent normalization used by train_rc.py
frame_indices_val = rc["frame_indices_val"].astype(int)
n_val = len(frame_indices_val)

def to_ae_latent(y_norm):
    return (y_norm * lat_std + lat_mean).T          # (D,n) -> (n,D) raw AE latent

z_true = to_ae_latent(rc["y_val"])
z_echo = to_ae_latent(rc["y_echo"])
z_infer = to_ae_latent(rc["y_infer"])

# ---- true fields for the same val-window frames ----
ds = np.load(f"{BASE}/dataset_full.npz", allow_pickle=True, mmap_mode="r")
n_train = int(np.load(f"{OUT_DIR}/latent_full.npz")["n_train"])
true_fields = np.asarray(ds["data"][n_train:n_train + n_val])   # (n_val,7,25,50)
x, y = np.asarray(ds["x"]), np.asarray(ds["y"])

# ---- decode all three in chunks ----
recon_true_latent = decode_to_fields(model, ckpt, z_true, device)   # AE's own round-trip error only (no RC involved) -- baseline
recon_infer = decode_to_fields(model, ckpt, z_infer, device)
recon_echo = decode_to_fields(model, ckpt, z_echo, device)

# ---- per-field relative error, full val window ----
def field_err(recon, true=true_fields):
    return relative_field_error(recon, true)


err_baseline = field_err(recon_true_latent)
err_infer = field_err(recon_infer)
err_echo = field_err(recon_echo)

print(f"{'field':10s}  {'AE-only':>9s}  {'RC-infer':>9s}  {'RC-echo':>9s}")
for i, name in enumerate(field_names):
    print(f"{name:10s}  {err_baseline[i]:8.2%}  {err_infer[i]:8.2%}  {err_echo[i]:8.2%}")

# ---- echo error vs. time-into-freerun (demonstrates phase-drift growth) ----
def windowed_err(recon, true, w=150):
    n = recon.shape[0]
    errs = []
    for i in range(0, n - w, w):
        rel = np.abs(recon[i:i + w] - true[i:i + w]) / (np.abs(true[i:i + w]).mean() + 1e-30)
        errs.append(rel.mean())
    return np.array(errs)


echo_err_curve = windowed_err(recon_echo, true_fields)
infer_err_curve = windowed_err(recon_infer, true_fields)

fig, ax = plt.subplots(figsize=(8, 4.5))
steps = np.arange(len(echo_err_curve)) * 150
ax.plot(steps, echo_err_curve * 100, "o-", label="RC-echo (autonomous)")
ax.plot(steps, infer_err_curve * 100, "o-", label="RC-infer (z1-driven)")
ax.set_xlabel("steps into held-out window")
ax.set_ylabel("mean relative field error (%), all 7 fields")
ax.set_title("Decoded-field error vs. free-run length")
ax.legend()
fig.tight_layout()
fig.savefig(f"{RC_DIR}/rc_decode_error_vs_time.png", dpi=150)
print(f"wrote {RC_DIR}/rc_decode_error_vs_time.png")

# ---- visual panel: original vs AE-only vs infer-decoded vs echo-decoded, one frame ----
DENSITY_FIELDS = {"n_e", "n_n", "n_i_dot"}   # number densities: show on a log10 color scale
frame_i = n_val // 6   # early in the val window, before echo has fully dephased
fig, axes = plt.subplots(4, len(field_names), figsize=(22, 11))
row_data = [true_fields[frame_i], recon_true_latent[frame_i], recon_infer[frame_i], recon_echo[frame_i]]
row_labels = ["original", "AE only\n(true latent)", "RC-infer\n(z1-driven)", "RC-echo\n(autonomous)"]
for c, name in enumerate(field_names):
    is_density = name in DENSITY_FIELDS
    col_data = [np.log10(np.clip(d[c], 1e-30, None)) if is_density else d[c] for d in row_data]
    vmin, vmax = col_data[0].min(), col_data[0].max()
    for r, data in enumerate(col_data):
        axes[r, c].pcolormesh(x, y, data, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[r, c].set_aspect("equal"); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, c].set_title(f"{name} (log$_{{10}}$)" if is_density else name)
for r, label in enumerate(row_labels):
    axes[r, 0].set_ylabel(label, fontsize=10)
fig.suptitle(f"Frame {frame_indices_val[frame_i]} (val step {frame_i}): decoded through frozen AE decoder")
fig.tight_layout()
fig.savefig(f"{RC_DIR}/rc_decode_panel.png", dpi=150)
print(f"wrote {RC_DIR}/rc_decode_panel.png")

with open(f"{RC_DIR}/rc_decode_err_summary.json", "w") as f:
    json.dump({
        "field_names": field_names,
        "err_baseline_pct": (err_baseline * 100).tolist(),
        "err_infer_pct": (err_infer * 100).tolist(),
        "err_echo_pct": (err_echo * 100).tolist(),
    }, f, indent=2)
print(f"wrote {RC_DIR}/rc_decode_err_summary.json")
