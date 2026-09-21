"""Sweep diffRC's integration step ds together with connectivity degree
(0..10), at fixed N=200, on the frozen AE's latent trajectory. Stays
entirely in latent space -- no decoding through the AE here.

This supersedes the earlier single-ds (0.5) diffRC degree sweep: ds
turned out to matter a lot (the ds=0.5 sweep was noisy/non-monotonic and
had a catastrophic infer failure at degree=0), so here we scan ds across
[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1] and repeat the full degree sweep
at each value.

degree=0 special case: see sweep_degree.py -- makeConnectionMatDegree
can't rescale an all-zero matrix to a target spectral radius, so we
build A as an explicit all-zero sparse matrix directly for that case.

Usage: python sweep_ds.py [latent_dim] [n_res]
"""
import json
import pickle
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src import diffRC  # noqa: E402

BASE = os.path.join(REPO_ROOT, "HPHall-data-Summer24")

LATENT_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_RES = int(sys.argv[2]) if len(sys.argv) > 2 else 200

OUT_DIR = f"{BASE}/latent{LATENT_DIM}"
SWEEP_DIR = f"{OUT_DIR}/n{N_RES}/degree_sweep_diff"
os.makedirs(SWEEP_DIR, exist_ok=True)

SEED = 0
RHO = 0.9
ALPHA = 1e-3
TRAIN_SKIP = 200
DRIVE_INDEX = [0]
DEGREES = list(range(0, 11))
DS_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

# ---- load the AE's latent trajectory (shared across every run) ----
lat = np.load(f"{OUT_DIR}/latent_full.npz")
z = lat["z"]
frame_indices = lat["frame_indices"].astype(int)
n_train = int(lat["n_train"])

z_t = z.T
D, M = z_t.shape
n_val = M - n_train

mean = z_t[:, :n_train].mean(axis=1, keepdims=True)
std = z_t[:, :n_train].std(axis=1, keepdims=True) + 1e-8
z_norm = (z_t - mean) / std

y_in = z_norm[:, :n_train]
y_val = z_norm[:, n_train:]

sigma = 2.0 / np.max(np.max(y_in, axis=1) - np.min(y_in, axis=1))
print(f"diffRC ds-sweep: D={D} N={N_RES} n_train={n_train} n_val={n_val} sigma={sigma:.4f}")


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def infer_pc(y_true, y_infer):
    yy = y_true - y_true.mean(axis=1, keepdims=True)
    yi = y_infer - y_infer.mean(axis=1, keepdims=True)
    return (yi @ yy.T) ** 2 / ((yi @ yi.T) * (yy @ yy.T))


def period_steps(z1, distance=100, win=41):
    from scipy.signal import find_peaks
    z1d = z1 - np.convolve(z1, np.ones(win) / win, mode="same")
    peaks, _ = find_peaks(z1d, distance=distance, prominence=0.4 * np.std(z1d))
    if len(peaks) < 2:
        return float("nan")
    return float(np.median(np.diff(peaks)))


p_true = period_steps(y_val[0])

all_summary = {}
for ds in DS_VALUES:
    ds_dir = f"{SWEEP_DIR}/ds{ds}"
    os.makedirs(ds_dir, exist_ok=True)
    summary = []
    for degree in DEGREES:
        np.random.seed(SEED)
        print(f"\n=== ds={ds} degree={degree} ===")

        RC = diffRC(N_RES, ds=ds, bias=False)

        if degree == 0:
            RC.rho = RHO
            RC.degree = 0
            RC.A = csr_matrix((N_RES, N_RES))
        else:
            RC.makeConnectionMatDegree(RHO, degree=degree, loc=-1, scale=1)

        RC.makeInputMat(D, sigma, randMin=-1, randMax=1)

        RC.listen(y_in)
        RC.train(y_in, start=TRAIN_SKIP, alpha=ALPHA)
        RC.echo(n_val)
        RC.infer(y_val[DRIVE_INDEX], DRIVE_INDEX)

        pc = infer_pc(y_val, RC.y_infer)
        echo_rmse = rmse(RC.y_echo, y_val)
        infer_rmse = rmse(RC.y_infer, y_val)
        p_echo = period_steps(RC.y_echo[0])
        period_mismatch = 100 * abs(p_echo - p_true) / p_true if np.isfinite(p_echo) else float("nan")

        deg_dir = f"{ds_dir}/degree{degree}"
        os.makedirs(deg_dir, exist_ok=True)
        with open(f"{deg_dir}/rc_model.pkl", "wb") as f:
            pickle.dump({
                "N": N_RES, "D": D, "rho": RHO, "degree": degree, "alpha": ALPHA,
                "sigma": sigma, "seed": SEED, "rc_type": "diff", "ds": ds,
                "A": RC.A, "B": RC.B, "W": RC.W, "mask": RC.mask,
                "mean": mean, "std": std, "n_train": n_train,
            }, f)
        np.savez(
            f"{deg_dir}/rc_results.npz",
            y_val=y_val, y_echo=RC.y_echo, y_infer=RC.y_infer,
            frame_indices_val=frame_indices[n_train:], mean=mean, std=std,
        )

        row = dict(
            ds=ds, degree=degree, fit_error=float(RC.fitError),
            echo_rmse=echo_rmse.tolist(), infer_rmse=infer_rmse.tolist(),
            echo_rmse_mean=float(echo_rmse.mean()), infer_rmse_mean=float(infer_rmse.mean()),
            infer_pc=[float(pc[i, i]) for i in range(D) if i not in DRIVE_INDEX],
            period_true=p_true, period_echo=p_echo, period_mismatch_pct=period_mismatch,
        )
        summary.append(row)
        with open(f"{deg_dir}/metrics.json", "w") as f:
            json.dump(row, f, indent=2)

        print(f"  fit_error={row['fit_error']:.4f}  echo_rmse_mean={row['echo_rmse_mean']:.3f}  "
              f"infer_rmse_mean={row['infer_rmse_mean']:.3f}  infer_pc={row['infer_pc']}  "
              f"period_echo={p_echo}  mismatch%={period_mismatch}")

    with open(f"{ds_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    all_summary[ds] = summary
    print(f"wrote {ds_dir}/summary.json")

with open(f"{SWEEP_DIR}/summary_by_ds.json", "w") as f:
    json.dump({str(k): v for k, v in all_summary.items()}, f, indent=2)
print(f"\nwrote {SWEEP_DIR}/summary_by_ds.json")

# ---- aggregate plot: one line per ds, vs degree ----
import matplotlib.pyplot as plt
import matplotlib.cm as cm

fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
colors = cm.viridis(np.linspace(0, 0.9, len(DS_VALUES)))
for c, ds in zip(colors, DS_VALUES):
    summary = all_summary[ds]
    degrees = [r["degree"] for r in summary]
    echo_rmse_mean = [r["echo_rmse_mean"] for r in summary]
    infer_pc_mean = [np.mean(r["infer_pc"]) if r["infer_pc"] else np.nan for r in summary]
    period_mismatch = [r["period_mismatch_pct"] for r in summary]

    axs[0].plot(degrees, echo_rmse_mean, "o-", color=c, label=f"ds={ds}")
    axs[1].plot(degrees, infer_pc_mean, "o-", color=c, label=f"ds={ds}")
    axs[2].plot(degrees, period_mismatch, "o-", color=c, label=f"ds={ds}")

axs[0].set_xlabel("degree"); axs[0].set_ylabel("mean echo RMSE (normalized latent)")
axs[0].set_title("Echo RMSE vs. degree, by ds"); axs[0].set_yscale("log")
axs[1].set_xlabel("degree"); axs[1].set_ylabel("mean infer PC (unobserved ch.)")
axs[1].set_title("Infer reconstruction vs. degree, by ds"); axs[1].set_ylim(0, 1.02)
axs[2].set_xlabel("degree"); axs[2].set_ylabel("echo period mismatch (%)")
axs[2].set_title("Autonomous period error vs. degree, by ds"); axs[2].set_yscale("log")
axs[2].legend(fontsize=8, loc="upper right")

fig.suptitle(f"diffRC: ds x degree sweep, N={N_RES}, D={D}")
fig.tight_layout()
fig.savefig(f"{SWEEP_DIR}/ds_degree_sweep_summary.png", dpi=150)
print(f"wrote {SWEEP_DIR}/ds_degree_sweep_summary.png")

# ---- heatmap: infer PC (mean over unobserved channels) over ds x degree ----
pc_grid = np.full((len(DS_VALUES), len(DEGREES)), np.nan)
rmse_grid = np.full((len(DS_VALUES), len(DEGREES)), np.nan)
for i, ds in enumerate(DS_VALUES):
    for row in all_summary[ds]:
        j = DEGREES.index(row["degree"])
        pc_grid[i, j] = np.mean(row["infer_pc"]) if row["infer_pc"] else np.nan
        rmse_grid[i, j] = row["echo_rmse_mean"]

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
im0 = axs[0].imshow(pc_grid, aspect="auto", cmap="viridis", vmin=0, vmax=1)
axs[0].set_xticks(range(len(DEGREES))); axs[0].set_xticklabels(DEGREES)
axs[0].set_yticks(range(len(DS_VALUES))); axs[0].set_yticklabels(DS_VALUES)
axs[0].set_xlabel("degree"); axs[0].set_ylabel("ds")
axs[0].set_title("Infer PC (mean, unobserved ch.)")
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(np.log10(rmse_grid), aspect="auto", cmap="magma_r")
axs[1].set_xticks(range(len(DEGREES))); axs[1].set_xticklabels(DEGREES)
axs[1].set_yticks(range(len(DS_VALUES))); axs[1].set_yticklabels(DS_VALUES)
axs[1].set_xlabel("degree"); axs[1].set_ylabel("ds")
axs[1].set_title("log10(echo RMSE)")
fig.colorbar(im1, ax=axs[1])

fig.suptitle(f"diffRC: ds x degree grid, N={N_RES}, D={D}")
fig.tight_layout()
fig.savefig(f"{SWEEP_DIR}/ds_degree_heatmap.png", dpi=150)
print(f"wrote {SWEEP_DIR}/ds_degree_heatmap.png")
