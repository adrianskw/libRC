"""Sweep the reservoir connectivity degree (0..10) at fixed N=200, on the
frozen AE's latent trajectory, for either a mapRC (discrete forward map) or
a diffRC (ODE-integrated) reservoir. Stays entirely in latent space --
no decoding through the AE here, this is purely about how much internal
reservoir connectivity the readout needs to reproduce the dynamics.

degree=0 is a special case: makeConnectionMatDegree can't rescale an
all-zero matrix to a target spectral radius (nothing to rescale), so for
degree=0 we build A as an explicit all-zero matrix directly. This is
still a meaningful configuration -- the reservoir has no internal
node-to-node coupling, so any recurrence in echo/infer mode comes only
through the output-feedback loop (A@r term is exactly zero every step).

Usage: python sweep_degree.py [map|diff] [latent_dim] [n_res] [ds]
  ds only matters for diffRC (default 0.5).
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.sparse import csr_matrix

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src import mapRC, diffRC  # noqa: E402

BASE = os.path.join(REPO_ROOT, "HPHall-data-Summer24")

RC_TYPE = sys.argv[1] if len(sys.argv) > 1 else "map"
LATENT_DIM = int(sys.argv[2]) if len(sys.argv) > 2 else 3
N_RES = int(sys.argv[3]) if len(sys.argv) > 3 else 200
DS = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

assert RC_TYPE in ("map", "diff")

OUT_DIR = f"{BASE}/latent{LATENT_DIM}"
SWEEP_DIR = f"{OUT_DIR}/n{N_RES}/degree_sweep_{RC_TYPE}"
import os
os.makedirs(SWEEP_DIR, exist_ok=True)

SEED = 0
RHO = 0.9
ALPHA = 1e-3
TRAIN_SKIP = 200
DRIVE_INDEX = [0]
DEGREES = list(range(0, 11))

# ---- load the AE's latent trajectory (shared across every degree) ----
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
print(f"RC_TYPE={RC_TYPE} D={D} N={N_RES} n_train={n_train} n_val={n_val} sigma={sigma:.4f}")


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

summary = []
for degree in DEGREES:
    np.random.seed(SEED)
    print(f"\n=== degree={degree} ===")

    if RC_TYPE == "map":
        RC = mapRC(N_RES, bias=True)
    else:
        RC = diffRC(N_RES, ds=DS, bias=False)

    if degree == 0:
        # no internal connectivity -- rescaling to rho is meaningless for an
        # all-zero matrix, so build it directly instead of going through
        # makeConnectionMatDegree (which requires a nonzero spectral radius).
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

    deg_dir = f"{SWEEP_DIR}/degree{degree}"
    os.makedirs(deg_dir, exist_ok=True)
    with open(f"{deg_dir}/rc_model.pkl", "wb") as f:
        pickle.dump({
            "N": N_RES, "D": D, "rho": RHO, "degree": degree, "alpha": ALPHA,
            "sigma": sigma, "seed": SEED, "rc_type": RC_TYPE, "ds": DS if RC_TYPE == "diff" else None,
            "A": RC.A, "B": RC.B, "W": RC.W, "mask": RC.mask,
            "mean": mean, "std": std, "n_train": n_train,
        }, f)
    np.savez(
        f"{deg_dir}/rc_results.npz",
        y_val=y_val, y_echo=RC.y_echo, y_infer=RC.y_infer,
        frame_indices_val=frame_indices[n_train:], mean=mean, std=std,
    )

    row = dict(
        degree=degree, fit_error=float(RC.fitError),
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

with open(f"{SWEEP_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nwrote {SWEEP_DIR}/summary.json and per-degree results under {SWEEP_DIR}/degree*/")

# ---- summary plot: echo/infer RMSE and infer PC vs degree ----
import matplotlib.pyplot as plt

degrees = [r["degree"] for r in summary]
echo_rmse_mean = [r["echo_rmse_mean"] for r in summary]
infer_rmse_mean = [r["infer_rmse_mean"] for r in summary]
infer_pc_mean = [np.mean(r["infer_pc"]) if r["infer_pc"] else np.nan for r in summary]
period_mismatch = [r["period_mismatch_pct"] for r in summary]

fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
axs[0].plot(degrees, echo_rmse_mean, "o-", label="echo")
axs[0].plot(degrees, infer_rmse_mean, "o-", label="infer")
axs[0].set_xlabel("degree"); axs[0].set_ylabel("mean RMSE (normalized latent)")
axs[0].set_title("Echo/infer RMSE vs. degree"); axs[0].legend()

axs[1].plot(degrees, infer_pc_mean, "o-", color="tab:green")
axs[1].set_xlabel("degree"); axs[1].set_ylabel("mean infer PC (unobserved ch.)")
axs[1].set_title("Infer reconstruction quality vs. degree")
axs[1].set_ylim(0, 1.02)

axs[2].plot(degrees, period_mismatch, "o-", color="tab:red")
axs[2].set_xlabel("degree"); axs[2].set_ylabel("echo period mismatch (%)")
axs[2].set_title("Autonomous period error vs. degree")

fig.suptitle(f"{'mapRC' if RC_TYPE=='map' else f'diffRC (ds={DS})'}: degree sweep, N={N_RES}, D={D}")
fig.tight_layout()
fig.savefig(f"{SWEEP_DIR}/degree_sweep_summary.png", dpi=150)
print(f"wrote {SWEEP_DIR}/degree_sweep_summary.png")
