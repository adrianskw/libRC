"""Instead of a readout Wout that sees only the instantaneous reservoir
state r(t), give it a short window of the reservoir's own recent history:
r(t), r(t-5), r(t-10), ..., r(t-45) -- 10 taps spaced 5 steps apart,
reaching back 45 steps (roughly 1/3 of the ~158-step breathing-mode
period). This is still a *linear* readout (ridge regression), just over
a wider, explicitly time-delayed feature vector -- a tapped-delay-line /
FIR-style readout, as opposed to the nonlinear MLP readout tried in
mlp_readout.py.

Compared on the same two configs used throughout: best mapRC (N=200,
degree=1) and best diffRC (N=200, ds=0.3, degree=2).

Training careful point: the delay taps must not reach back into the
reservoir's initial (zero-state) transient. The usual TRAIN_SKIP=200
warm-up discard is extended so that even the *oldest* tap used
(t - 45) still lands at or after TRAIN_SKIP, i.e. ridge-regression
training starts at t = TRAIN_SKIP + max(lag).

Usage: python delay_embedding_readout.py [latent_dim] [n_res]
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

OUT_DIR = f"{BASE}/latent{LATENT_DIM}/n{N_RES}"
SEED = 0
RHO = 0.9
ALPHA = 1e-3                              # baseline (instantaneous-state) readout, unchanged
DELAY_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]  # delay-line readout: alpha sweep
TRAIN_SKIP = 200
DRIVE_INDEX = [0]

TAP_SPACING = 5
N_TAPS = 10
LAGS = [i * TAP_SPACING for i in range(N_TAPS)]   # [0, 5, 10, ..., 45]
MAX_LAG = LAGS[-1]
BUF_LEN = MAX_LAG + 1                              # 46 raw states held in the rolling buffer

CONFIGS = [
    dict(rc_type="map", degree=1, ds=None, bias=True),
    dict(rc_type="diff", degree=2, ds=0.3, bias=False),
]

# ---- load the AE's latent trajectory ----
lat = np.load(f"{BASE}/latent{LATENT_DIM}/latent_full.npz")
z = lat["z"]
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
print(f"D={D} N={N_RES} n_train={n_train} n_val={n_val} sigma={sigma:.4f}  "
      f"lags={LAGS} (max_lag={MAX_LAG})")


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
        RC = mapRC(N_RES, bias=cfg["bias"])
    else:
        RC = diffRC(N_RES, ds=cfg["ds"], bias=cfg["bias"])
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


# ---- baseline: library's standard instantaneous-state ridge readout ----
def run_linear(RC, r_listen):
    RC.r = r_listen.copy()
    RC.train(y_in, start=TRAIN_SKIP, alpha=ALPHA)
    RC.echo(n_val)
    RC.infer(y_val[DRIVE_INDEX], DRIVE_INDEX)
    metrics = eval_metrics(RC.y_echo, RC.y_infer)
    metrics["fit_error"] = float(RC.fitError)
    return metrics


# ---- tapped-delay-line readout: Wout looks at r(t), r(t-5), ..., r(t-45) ----
def make_feature(buf, bias):
    """buf: (N, BUF_LEN) array, columns ordered oldest -> newest (buf[:,-1] = r(t))."""
    feat = np.concatenate([buf[:, BUF_LEN - 1 - lag] for lag in LAGS])
    if bias:
        feat = np.concatenate([feat, [1.0]])
    return feat


def make_feature_matrix(r_listen, start, end, bias):
    """Vectorized construction of features for t in [start, end)."""
    T = end - start
    feat_blocks = [r_listen[:, start - lag:end - lag] for lag in LAGS]
    Phi = np.concatenate(feat_blocks, axis=0)   # (N*N_TAPS, T)
    if bias:
        Phi = np.vstack([Phi, np.ones((1, T))])
    return Phi


def train_delay_readout(r_listen, bias, alpha):
    start = TRAIN_SKIP + MAX_LAG
    Phi = make_feature_matrix(r_listen, start, n_train, bias)
    Y = y_in[:, start:n_train]
    F = Phi.shape[0]
    RRT = Phi @ Phi.T + alpha * np.eye(F)
    URT = Y @ Phi.T
    Wout = np.linalg.solve(RRT, URT.T).T   # (D, F)
    y_est = Wout @ Phi
    fit_error = float(np.sqrt(np.mean((y_est - Y) ** 2)))
    return Wout, fit_error


def run_delay(RC, r_listen, bias, alpha):
    Wout, fit_error = train_delay_readout(r_listen, bias, alpha)

    def predict(buf):
        return Wout @ make_feature(buf, bias)

    init_buf = r_listen[:, n_train - BUF_LEN:n_train].copy()

    # ---- echo: fully autonomous ----
    y_boundary = predict(init_buf)
    buf = init_buf.copy()
    r_next = RC.step(buf[:, -1], y_boundary)
    buf = np.concatenate([buf[:, 1:], r_next[:, None]], axis=1)

    y_echo = np.zeros((D, n_val))
    y_echo[:, 0] = predict(buf)
    for i in range(1, n_val):
        r_next = RC.step(buf[:, -1], y_echo[:, i - 1])
        buf = np.concatenate([buf[:, 1:], r_next[:, None]], axis=1)
        y_echo[:, i] = predict(buf)

    # ---- infer: driven only on z1 ----
    buf = init_buf.copy()
    r_next = RC.step(buf[:, -1], y_boundary)
    buf = np.concatenate([buf[:, 1:], r_next[:, None]], axis=1)

    y_infer = np.zeros((D, n_val))
    y_infer[:, 0] = predict(buf)
    y_infer[DRIVE_INDEX, 0] = y_val[DRIVE_INDEX, 0]
    for i in range(1, n_val):
        yTemp = y_infer[:, i - 1].copy()
        yTemp[DRIVE_INDEX] = y_val[DRIVE_INDEX, i - 1]
        r_next = RC.step(buf[:, -1], yTemp)
        buf = np.concatenate([buf[:, 1:], r_next[:, None]], axis=1)
        y_infer[:, i] = predict(buf)
        y_infer[DRIVE_INDEX, i] = y_val[DRIVE_INDEX, i]

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

    print("\n-- baseline: instantaneous-state linear readout --")
    lin = run_linear(RC, r_listen)
    print(f"  fit_error={lin['fit_error']:.4f}  echo_rmse_mean={lin['echo_rmse_mean']:.3f}  "
          f"infer_pc={lin['infer_pc']}")

    print(f"\n-- tapped-delay-line readout ({N_TAPS} taps, spacing={TAP_SPACING}, "
          f"max_lag={MAX_LAG}), alpha sweep --")
    delay_by_alpha = {}
    y_echo_by_alpha, y_infer_by_alpha = {}, {}
    for alpha in DELAY_ALPHAS:
        delay, y_echo_delay, y_infer_delay = run_delay(RC, r_listen, cfg["bias"], alpha)
        print(f"  alpha={alpha:<8g} fit_error={delay['fit_error']:.4f}  "
              f"echo_rmse_mean={delay['echo_rmse_mean']:.3f}  infer_pc={delay['infer_pc']}")
        delay_by_alpha[alpha] = delay
        y_echo_by_alpha[alpha] = y_echo_delay
        y_infer_by_alpha[alpha] = y_infer_delay

    best_alpha = min(DELAY_ALPHAS, key=lambda a: delay_by_alpha[a]["echo_rmse_mean"])
    best_delay = delay_by_alpha[best_alpha]
    print(f"  -> best alpha by echo RMSE: {best_alpha:g}")

    out_dir = f"{OUT_DIR}/delay_readout_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(
            dict(config=cfg, lags=LAGS, linear=lin,
                 delay_by_alpha={str(a): m for a, m in delay_by_alpha.items()},
                 best_alpha=best_alpha),
            f, indent=2,
        )

    # alpha-sweep summary plot: echo RMSE and infer PC vs. alpha (log x-axis)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    echo_vals = [delay_by_alpha[a]["echo_rmse_mean"] for a in DELAY_ALPHAS]
    pc_vals = [np.mean(delay_by_alpha[a]["infer_pc"]) for a in DELAY_ALPHAS]
    axs[0].axhline(lin["echo_rmse_mean"], color="k", ls="--", lw=1, label="instantaneous baseline")
    axs[0].plot(DELAY_ALPHAS, echo_vals, "o-", color="tab:orange", label="delay-line")
    axs[0].set_xscale("log"); axs[0].set_xlabel("ridge alpha"); axs[0].set_ylabel("echo RMSE (mean)")
    axs[0].set_title("Echo RMSE vs. alpha"); axs[0].legend(fontsize=8)
    axs[1].axhline(np.mean(lin["infer_pc"]), color="k", ls="--", lw=1, label="instantaneous baseline")
    axs[1].plot(DELAY_ALPHAS, pc_vals, "o-", color="tab:orange", label="delay-line")
    axs[1].set_xscale("log"); axs[1].set_xlabel("ridge alpha"); axs[1].set_ylabel("mean infer PC")
    axs[1].set_title("Infer PC vs. alpha"); axs[1].legend(fontsize=8)
    fig.suptitle(f"{tag}, degree={cfg['degree']}: delay-line readout, alpha sweep")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/alpha_sweep.png", dpi=150)

    # timeseries comparison at the best alpha found
    Mplot = min(3000, n_val)
    fig, axs = plt.subplots(D, 2, figsize=(14, 7), sharex=True)
    for i in range(D):
        axs[i, 0].plot(y_val[i, :Mplot], "k-", lw=1.2, label="true")
        axs[i, 0].plot(RC.y_echo[i, :Mplot], lw=1, alpha=0.85, label="instantaneous echo")
        axs[i, 0].plot(y_echo_by_alpha[best_alpha][i, :Mplot], lw=1, alpha=0.85,
                       label=f"delay-line echo (alpha={best_alpha:g})")
        axs[i, 0].set_ylabel(f"z{i+1}")
        axs[i, 1].plot(y_val[i, :Mplot], "k-", lw=1.2, label="true")
        axs[i, 1].plot(RC.y_infer[i, :Mplot], lw=1, alpha=0.85, label="instantaneous infer")
        axs[i, 1].plot(y_infer_by_alpha[best_alpha][i, :Mplot], lw=1, alpha=0.85,
                       label=f"delay-line infer (alpha={best_alpha:g})")
    axs[0, 0].set_title("Echo (autonomous): instantaneous vs. delay-line readout")
    axs[0, 1].set_title("Infer (z1-driven): instantaneous vs. delay-line readout")
    axs[0, 0].legend(fontsize=8); axs[0, 1].legend(fontsize=8)
    fig.suptitle(f"{tag}, degree={cfg['degree']}: instantaneous vs. {N_TAPS}-tap delay-line readout "
                 f"(alpha={best_alpha:g})")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/readout_compare.png", dpi=150)
    print(f"wrote {out_dir}/readout_compare.png, {out_dir}/alpha_sweep.png, {out_dir}/metrics.json")

    all_results[tag] = dict(config=cfg, linear=lin, best_alpha=best_alpha, best_delay=best_delay,
                             delay_by_alpha=delay_by_alpha)

with open(f"{OUT_DIR}/delay_readout_summary.json", "w") as f:
    json.dump(
        {tag: dict(config=r["config"], linear=r["linear"], best_alpha=r["best_alpha"],
                    best_delay=r["best_delay"],
                    delay_by_alpha={str(a): m for a, m in r["delay_by_alpha"].items()})
         for tag, r in all_results.items()},
        f, indent=2,
    )

print(f"\n{'='*70}\nSUMMARY (best alpha per config, by echo RMSE)\n{'='*70}")
print(f"{'config':16s}  {'readout':22s}  {'echo_rmse':>10s}  {'infer_pc (z2/z3)':>20s}  {'period_mismatch%':>18s}")
for tag, res in all_results.items():
    pc_str = "/".join(f"{x:.3f}" for x in res["linear"]["infer_pc"])
    print(f"{tag:16s}  {'instantaneous':22s}  {res['linear']['echo_rmse_mean']:10.3f}  "
          f"{pc_str:>20s}  {res['linear']['period_mismatch_pct']:18.2f}")
    m = res["best_delay"]
    pc_str = "/".join(f"{x:.3f}" for x in m["infer_pc"])
    label = f"delay-line (a={res['best_alpha']:g})"
    print(f"{tag:16s}  {label:22s}  {m['echo_rmse_mean']:10.3f}  "
          f"{pc_str:>20s}  {m['period_mismatch_pct']:18.2f}")
