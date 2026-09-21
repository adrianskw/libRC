"""Compare a linear (ridge) readout against a small MLP readout, on the
two best RC configurations identified by the earlier degree/ds sweeps:

  mapRC:  N=200, degree=1        (near-best echo RMSE AND near-best infer PC)
  diffRC: N=200, ds=0.3, degree=2 (best point in the whole ds x degree grid)

Stays entirely in latent space (no decode through the AE) -- this is
purely about whether replacing the fitted linear readout y = W@r with a
small trainable MLP(r) improves the reservoir's autonomous (echo) and
driven (infer) reconstruction of the breathing-mode attractor, holding
the reservoir itself (A, B, and hence the r trajectory it produces when
driven) fixed.

The MLP readout: r (N,) -> Linear(N, N) -> activation (tanh/relu) ->
Linear(N, D). Trained with Adam + weight decay on exactly the same
(r[t], y_in[t]) pairs the ridge regression baseline fits -- r[t] was
produced by driving the reservoir with y_in[t-1], so the target
convention (state -> current input) is identical to train_rc.py /
sweep_degree.py / sweep_ds.py.

Both readouts are then dropped into the same free-run (echo) and
partially-driven (infer) loops used by libRC.Reservoir.echo()/infer(),
reimplemented here by hand so a nonlinear readout can be substituted in
place of the library's hardwired W@r.

Usage: python mlp_readout.py [latent_dim] [n_res] [activ] [epochs]
  activ: tanh (default) or relu
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src import mapRC, diffRC  # noqa: E402

BASE = os.path.join(REPO_ROOT, "HPHall-data-Summer24")

LATENT_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_RES = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ACTIV_NAME = sys.argv[3] if len(sys.argv) > 3 else "tanh"
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 2000

assert ACTIV_NAME in ("tanh", "relu")

OUT_DIR = f"{BASE}/latent{LATENT_DIM}/n{N_RES}"
SEED = 0
RHO = 0.9
ALPHA = 1e-3          # ridge alpha, matches sweeps
WEIGHT_DECAY = 1e-4    # MLP's analogue of ridge's L2 regularization
LR = 1e-3
TRAIN_SKIP = 200
DRIVE_INDEX = [0]

# the two best configs found by sweep_degree.py / sweep_ds.py
CONFIGS = [
    dict(rc_type="map", degree=1, ds=None),
    dict(rc_type="diff", degree=2, ds=0.3),
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- load the AE's latent trajectory ----
lat = np.load(f"{BASE}/latent{LATENT_DIM}/latent_full.npz")
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
print(f"D={D} N={N_RES} n_train={n_train} n_val={n_val} sigma={sigma:.4f} device={device}")


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


def build_rc(cfg):
    np.random.seed(SEED)
    if cfg["rc_type"] == "map":
        RC = mapRC(N_RES, bias=True)
    else:
        RC = diffRC(N_RES, ds=cfg["ds"], bias=False)
    RC.makeConnectionMatDegree(RHO, degree=cfg["degree"], loc=-1, scale=1)
    RC.makeInputMat(D, sigma, randMin=-1, randMax=1)
    return RC


def eval_metrics(y_echo, y_infer):
    pc = infer_pc(y_val, y_infer)
    echo_rmse = rmse(y_echo, y_val)
    infer_rmse = rmse(y_infer, y_val)
    p_echo = period_steps(y_echo[0])
    period_mismatch = 100 * abs(p_echo - p_true) / p_true if np.isfinite(p_echo) else float("nan")
    return dict(
        echo_rmse=echo_rmse.tolist(), infer_rmse=infer_rmse.tolist(),
        echo_rmse_mean=float(echo_rmse.mean()), infer_rmse_mean=float(infer_rmse.mean()),
        infer_pc=[float(pc[i, i]) for i in range(D) if i not in DRIVE_INDEX],
        period_true=p_true, period_echo=p_echo, period_mismatch_pct=period_mismatch,
    )


# ---- linear (ridge) readout: reuse the library exactly as the sweeps did ----
def run_linear(RC, r_listen):
    RC.r = r_listen.copy()
    RC.train(y_in, start=TRAIN_SKIP, alpha=ALPHA)
    RC.echo(n_val)
    RC.infer(y_val[DRIVE_INDEX], DRIVE_INDEX)
    metrics = eval_metrics(RC.y_echo, RC.y_infer)
    metrics["fit_error"] = float(RC.fitError)
    return metrics


# ---- MLP readout: r -> Linear(N,N) -> activ -> Linear(N,D) ----
class MLPReadout(nn.Module):
    def __init__(self, N, D, activ_name):
        super().__init__()
        activ = nn.Tanh() if activ_name == "tanh" else nn.ReLU()
        self.net = nn.Sequential(nn.Linear(N, N), activ, nn.Linear(N, D))

    def forward(self, x):
        return self.net(x)


def train_mlp(r_listen, start=TRAIN_SKIP, end=None):
    end = end or r_listen.shape[1]
    torch.manual_seed(SEED)
    model = MLPReadout(N_RES, D, ACTIV_NAME).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    X = torch.from_numpy(r_listen[:, start:end].T).float().to(device)
    Y = torch.from_numpy(y_in[:, start:end].T).float().to(device)
    loss_fn = nn.MSELoss()
    for epoch in range(EPOCHS):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()
        if epoch % (EPOCHS // 10 or 1) == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:5d}  train MSE = {loss.item():.6f}")
    model.eval()
    return model, float(loss.item())


def mlp_forward(model, r_col):
    with torch.no_grad():
        x = torch.from_numpy(r_col).float().to(device).unsqueeze(0)
        return model(x).squeeze(0).cpu().numpy()


def mlp_forward_batch(model, r_mat):
    with torch.no_grad():
        x = torch.from_numpy(r_mat.T).float().to(device)
        return model(x).cpu().numpy().T


def run_mlp(RC, r_listen):
    model, fit_error = train_mlp(r_listen)

    # ---- echo: fully autonomous, same recurrence structure as Reservoir.echo() ----
    r_echo = np.zeros((N_RES, n_val))
    r_echo[:, 0] = RC.step(r_listen[:N_RES, -1], mlp_forward(model, r_listen[:N_RES, -1]))
    for i in range(1, n_val):
        y_prev = mlp_forward(model, r_echo[:, i - 1])
        r_echo[:, i] = RC.step(r_echo[:, i - 1], y_prev)
    y_echo = mlp_forward_batch(model, r_echo)

    # ---- infer: driven only on z1, same structure as Reservoir.infer() ----
    r_infer = np.zeros((N_RES, n_val))
    r_infer[:, 0] = RC.step(r_listen[:N_RES, -1], mlp_forward(model, r_listen[:N_RES, -1]))
    for i in range(1, n_val):
        yTemp = mlp_forward(model, r_infer[:, i - 1])
        yTemp[DRIVE_INDEX] = y_val[DRIVE_INDEX, i - 1]
        r_infer[:, i] = RC.step(r_infer[:, i - 1], yTemp)
    y_infer = mlp_forward_batch(model, r_infer)

    metrics = eval_metrics(y_echo, y_infer)
    metrics["fit_error"] = fit_error
    return metrics, y_echo, y_infer


# ---- run both configs, both readouts ----
import matplotlib.pyplot as plt

all_results = {}
for cfg in CONFIGS:
    tag = cfg["rc_type"] if cfg["rc_type"] == "map" else f"diff_ds{cfg['ds']}"
    print(f"\n{'='*70}\n{tag}  degree={cfg['degree']}\n{'='*70}")

    RC = build_rc(cfg)
    RC.listen(y_in)
    r_listen = RC.r.copy()

    print("\n-- linear (ridge) readout --")
    lin = run_linear(RC, r_listen)
    print(f"  fit_error={lin['fit_error']:.4f}  echo_rmse_mean={lin['echo_rmse_mean']:.3f}  "
          f"infer_pc={lin['infer_pc']}")

    print(f"\n-- MLP readout ({ACTIV_NAME}, hidden={N_RES}) --")
    mlp, y_echo_mlp, y_infer_mlp = run_mlp(RC, r_listen)
    print(f"  fit_error={mlp['fit_error']:.4f}  echo_rmse_mean={mlp['echo_rmse_mean']:.3f}  "
          f"infer_pc={mlp['infer_pc']}")

    out_dir = f"{OUT_DIR}/mlp_readout_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(dict(config=cfg, linear=lin, mlp=mlp, activ=ACTIV_NAME, epochs=EPOCHS), f, indent=2)

    # timeseries comparison plot: true vs linear-echo vs mlp-echo (+ same for infer)
    Mplot = min(3000, n_val)
    fig, axs = plt.subplots(D, 2, figsize=(14, 7), sharex=True)
    for i in range(D):
        axs[i, 0].plot(y_val[i, :Mplot], "k-", lw=1.2, label="true")
        axs[i, 0].plot(RC.y_echo[i, :Mplot], lw=1, alpha=0.85, label="linear echo")
        axs[i, 0].plot(y_echo_mlp[i, :Mplot], lw=1, alpha=0.85, label="MLP echo")
        axs[i, 0].set_ylabel(f"z{i+1}")
        axs[i, 1].plot(y_val[i, :Mplot], "k-", lw=1.2, label="true")
        axs[i, 1].plot(RC.y_infer[i, :Mplot], lw=1, alpha=0.85, label="linear infer")
        axs[i, 1].plot(y_infer_mlp[i, :Mplot], lw=1, alpha=0.85, label="MLP infer")
    axs[0, 0].set_title("Echo (autonomous): linear vs. MLP readout")
    axs[0, 1].set_title("Infer (z1-driven): linear vs. MLP readout")
    axs[0, 0].legend(fontsize=8); axs[0, 1].legend(fontsize=8)
    fig.suptitle(f"{tag}, degree={cfg['degree']}: linear vs. MLP readout")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/readout_compare.png", dpi=150)
    print(f"wrote {out_dir}/readout_compare.png and {out_dir}/metrics.json")

    all_results[tag] = dict(config=cfg, linear=lin, mlp=mlp)

with open(f"{BASE}/latent{LATENT_DIM}/n{N_RES}/mlp_readout_summary.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
print(f"{'config':16s}  {'readout':8s}  {'echo_rmse':>10s}  {'infer_pc (z2/z3)':>20s}  {'period_mismatch%':>18s}")
for tag, res in all_results.items():
    for name, m in (("linear", res["linear"]), ("mlp", res["mlp"])):
        pc_str = "/".join(f"{x:.3f}" for x in m["infer_pc"])
        print(f"{tag:16s}  {name:8s}  {m['echo_rmse_mean']:10.3f}  {pc_str:>20s}  {m['period_mismatch_pct']:18.2f}")
