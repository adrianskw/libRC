"""Compare the latent trajectories learned independently by the three
conv-AE capacity variants (light/base/heavy from ae_size_sweep.py).

Each AE was trained from scratch, so its 3-D latent basis is arbitrary:
there's no reason channel 1 of "light" should correspond to channel 1 of
"heavy" -- the encoder can output any rotation/reflection/permutation of
an equivalent embedding. But all three see the exact same frames in the
exact same order, so we can align them post hoc: z-score each variant's
latent channels (own train-set mean/std), then find the permutation +
sign flip of one variant's channels that best correlates with a
reference ("base"), via a linear-sum-assignment on the |correlation|
matrix. That lets the raw time series be overlaid meaningfully instead
of just comparing three arbitrarily-oriented axis systems.

Usage: python compare_ae_latents.py [latent_dim]
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LATENT_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SWEEP_DIR = f"{BASE}/latent{LATENT_DIM}/ae_size_sweep"

VARIANTS = ["light", "base", "heavy"]
REF = "base"


def load(tag):
    d = np.load(f"{SWEEP_DIR}/{tag}/latent_full.npz")
    z, frame_indices, n_train = d["z"], d["frame_indices"], int(d["n_train"])
    zt = z.T  # (D, M), matches the convention used everywhere else
    mean = zt[:, :n_train].mean(axis=1, keepdims=True)
    std = zt[:, :n_train].std(axis=1, keepdims=True) + 1e-8
    return (zt - mean) / std, frame_indices, n_train


data = {tag: load(tag) for tag in VARIANTS}
frame_indices = data[REF][1]
n_train = data[REF][2]
D = data[REF][0].shape[0]

ref_z = data[REF][0]


def align_to_ref(z, ref_z):
    """Return (permuted+sign-flipped z, matched correlation per channel)."""
    corr = np.corrcoef(np.vstack([ref_z, z]))[:D, D:]  # corr[i,j] = ref chan i vs z chan j
    row_ind, col_ind = linear_sum_assignment(-np.abs(corr))
    matched_corr = corr[row_ind, col_ind]
    z_aligned = np.zeros_like(z)
    for i, j, c in zip(row_ind, col_ind, matched_corr):
        z_aligned[i] = z[j] * np.sign(c)
    return z_aligned, matched_corr[np.argsort(row_ind)]


aligned = {REF: ref_z}
match_corr = {REF: np.ones(D)}
for tag in VARIANTS:
    if tag == REF:
        continue
    aligned[tag], match_corr[tag] = align_to_ref(data[tag][0], ref_z)
    print(f"{tag:6s} -> {REF}: matched |correlation| per aligned channel = "
          f"{[f'{c:.3f}' for c in match_corr[tag]]}")

# ---- time series overlay, held-out val window ----
Mplot = min(3000, ref_z.shape[1] - n_train)
start = n_train
fig, axs = plt.subplots(D, 1, figsize=(11, 2.4 * D), sharex=True)
colors = {"light": "tab:orange", "base": "tab:blue", "heavy": "tab:green"}
for i in range(D):
    for tag in VARIANTS:
        axs[i].plot(aligned[tag][i, start:start + Mplot], color=colors[tag], lw=1,
                    alpha=0.85, label=f"{tag} (r={match_corr[tag][i]:.2f})" if tag != REF else tag)
    axs[i].set_ylabel(f"z{i+1} (aligned, z-scored)")
axs[0].legend(loc="upper right", fontsize=9)
axs[0].set_title("Latent time series: light vs. base vs. heavy AE (held-out window, aligned to base)")
axs[-1].set_xlabel("step (val window)")
fig.tight_layout()
fig.savefig(f"{SWEEP_DIR}/latent_timeseries_compare.png", dpi=150)
print(f"wrote {SWEEP_DIR}/latent_timeseries_compare.png")

# ---- 3D attractor overlay ----
if D >= 3:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for tag in VARIANTS:
        z = aligned[tag][:, start:start + Mplot]
        ax.plot(z[0], z[1], z[2], lw=0.6, alpha=0.7, color=colors[tag], label=tag)
    ax.set_xlabel("z1"); ax.set_ylabel("z2"); ax.set_zlabel("z3")
    ax.legend()
    ax.set_title("Breathing-mode attractor: light vs. base vs. heavy AE (aligned)")
    fig.tight_layout()
    fig.savefig(f"{SWEEP_DIR}/latent_attractor_compare.png", dpi=150)
    print(f"wrote {SWEEP_DIR}/latent_attractor_compare.png")

print("\nsummary: mean |matched correlation| to base, per variant:")
for tag in VARIANTS:
    print(f"  {tag:6s}: {np.mean(np.abs(match_corr[tag])):.4f}")
