"""Does AE reconstruction quality actually matter to the downstream RC?

ae_size_sweep.py showed that a lighter conv-AE (33K params) reconstructs
noticeably worse than the base (89K) or heavy (270K) AE, but that its
latent trajectory is still recognizably the same breathing-mode
attractor (compare_ae_latents.py: matched channel correlations all
> 0.8, mostly > 0.95). This script asks the more relevant question for
the RC pipeline: does that gap in AE fidelity translate into a gap in
how well a reservoir can learn and reproduce the *dynamics* of each
AE's latent trajectory?

For each of the three AE sizes (light/base/heavy from ae_size_sweep.py),
this fits the best mapRC (degree=1, N=200) and best diffRC (ds=0.3,
degree=2, N=200) configs found by the earlier degree/ds sweeps, staying
in latent space (no decode), and compares echo RMSE / infer PC across
all 3 AE sizes x 2 RC types.

Usage: python ae_size_rc_compare.py [latent_dim] [n_res]
"""
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src import mapRC, diffRC  # noqa: E402

BASE = os.path.join(REPO_ROOT, "HPHall-data-Summer24")

LATENT_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_RES = int(sys.argv[2]) if len(sys.argv) > 2 else 200

SWEEP_DIR = f"{BASE}/latent{LATENT_DIM}/ae_size_sweep"
AE_TAGS = ["light", "base", "heavy"]
RC_CONFIGS = [
    dict(rc_type="map", degree=1, ds=None, label="mapRC (deg=1)"),
    dict(rc_type="diff", degree=2, ds=0.3, label="diffRC (ds=0.3, deg=2)"),
]

SEED = 0
RHO = 0.9
ALPHA = 1e-3
TRAIN_SKIP = 200
DRIVE_INDEX = [0]


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


def load_ae_latent(tag):
    d = np.load(f"{SWEEP_DIR}/{tag}/latent_full.npz")
    z, n_train = d["z"], int(d["n_train"])
    z_t = z.T
    D, M = z_t.shape
    mean = z_t[:, :n_train].mean(axis=1, keepdims=True)
    std = z_t[:, :n_train].std(axis=1, keepdims=True) + 1e-8
    z_norm = (z_t - mean) / std
    return z_norm[:, :n_train], z_norm[:, n_train:], D


def run_rc(cfg, y_in, y_val, D):
    np.random.seed(SEED)
    sigma = 2.0 / np.max(np.max(y_in, axis=1) - np.min(y_in, axis=1))
    n_val = y_val.shape[1]

    if cfg["rc_type"] == "map":
        RC = mapRC(N_RES, bias=True)
    else:
        RC = diffRC(N_RES, ds=cfg["ds"], bias=False)
    RC.makeConnectionMatDegree(RHO, degree=cfg["degree"], loc=-1, scale=1)
    RC.makeInputMat(D, sigma, randMin=-1, randMax=1)

    RC.listen(y_in)
    RC.train(y_in, start=TRAIN_SKIP, alpha=ALPHA)
    RC.echo(n_val)
    RC.infer(y_val[DRIVE_INDEX], DRIVE_INDEX)

    p_true = period_steps(y_val[0])
    p_echo = period_steps(RC.y_echo[0])
    pc = infer_pc(y_val, RC.y_infer)
    echo_rmse = rmse(RC.y_echo, y_val)
    infer_rmse = rmse(RC.y_infer, y_val)

    return dict(
        fit_error=float(RC.fitError),
        echo_rmse_mean=float(echo_rmse.mean()), infer_rmse_mean=float(infer_rmse.mean()),
        infer_pc=[float(pc[i, i]) for i in range(D) if i not in DRIVE_INDEX],
        period_true=p_true, period_echo=p_echo,
        period_mismatch_pct=100 * abs(p_echo - p_true) / p_true if np.isfinite(p_echo) else float("nan"),
    )


ae_meta = json.load(open(f"{SWEEP_DIR}/summary.json"))

results = {}
for tag in AE_TAGS:
    y_in, y_val, D = load_ae_latent(tag)
    for cfg in RC_CONFIGS:
        print(f"\n=== AE={tag} ({ae_meta[tag]['n_params']:,} params)  RC={cfg['label']} ===")
        m = run_rc(cfg, y_in, y_val, D)
        print(f"  echo_rmse={m['echo_rmse_mean']:.3f}  infer_pc={m['infer_pc']}  "
              f"period_mismatch%={m['period_mismatch_pct']:.2f}")
        results[(tag, cfg["label"])] = m

with open(f"{SWEEP_DIR}/rc_by_ae_size.json", "w") as f:
    json.dump({f"{tag}|{label}": v for (tag, label), v in results.items()}, f, indent=2)

# ---- comparison plot ----
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
x = np.arange(len(AE_TAGS))
width = 0.35
colors = {"mapRC (deg=1)": "tab:blue", "diffRC (ds=0.3, deg=2)": "tab:orange"}

for j, cfg in enumerate(RC_CONFIGS):
    label = cfg["label"]
    echo = [results[(tag, label)]["echo_rmse_mean"] for tag in AE_TAGS]
    pc_mean = [np.mean(results[(tag, label)]["infer_pc"]) for tag in AE_TAGS]
    mismatch = [results[(tag, label)]["period_mismatch_pct"] for tag in AE_TAGS]
    off = (j - 0.5) * width
    axs[0].bar(x + off, echo, width, label=label, color=colors[label])
    axs[1].bar(x + off, pc_mean, width, label=label, color=colors[label])
    axs[2].bar(x + off, mismatch, width, label=label, color=colors[label])

for ax, title, ylab in zip(
    axs, ["Echo RMSE by AE size", "Mean infer PC by AE size", "Echo period mismatch by AE size"],
    ["mean echo RMSE (normalized latent)", "mean infer PC (unobserved ch.)", "period mismatch (%)"],
):
    ax.set_xticks(x); ax.set_xticklabels([f"{t}\n({ae_meta[t]['n_params']:,})" for t in AE_TAGS])
    ax.set_title(title); ax.set_ylabel(ylab); ax.legend(fontsize=8)
axs[1].set_ylim(0, 1.02)

fig.suptitle("Does AE size affect downstream RC quality? (best mapRC vs. best diffRC, per AE size)")
fig.tight_layout()
fig.savefig(f"{SWEEP_DIR}/rc_by_ae_size.png", dpi=150)
print(f"\nwrote {SWEEP_DIR}/rc_by_ae_size.png and {SWEEP_DIR}/rc_by_ae_size.json")

print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
print(f"{'AE':8s}  {'params':>9s}  {'RC':24s}  {'echo_rmse':>10s}  {'infer_pc':>18s}  {'mismatch%':>10s}")
for tag in AE_TAGS:
    for cfg in RC_CONFIGS:
        m = results[(tag, cfg["label"])]
        pc_str = "/".join(f"{v:.3f}" for v in m["infer_pc"])
        print(f"{tag:8s}  {ae_meta[tag]['n_params']:9,d}  {cfg['label']:24s}  "
              f"{m['echo_rmse_mean']:10.3f}  {pc_str:>18s}  {m['period_mismatch_pct']:10.2f}")
