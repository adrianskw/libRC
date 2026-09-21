"""Figures for the Id report (build_report_id.py). Each function returns a matplotlib Figure.

Reads only artifacts already on disk: id_study/summary.json (evaluate_id_variants.py), the
observer / RC run directories, log.dat, and the AE checkpoints.
"""
import base64
import io
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = os.path.join(REPO_ROOT, "HPHall-data-Summer24")
sys.path.insert(0, REPO_ROOT)
from src import observables  # noqa: E402

TEAL, RUST, INK, GRAY, AMBER, BLUE = "#0F766E", "#B5502D", "#211D14", "#8C8570", "#C9962B", "#3B6EA5"
FIELD_NAMES = ["n_e", "phi", "T_e", "n_n", "n_i_dot", "v_i_z", "v_i_r"]
FIELD_TEX = {"n_e": "$n_e$", "phi": r"$\phi$", "T_e": "$T_e$", "n_n": "$n_n$", "n_i_dot": r"$\dot n_i$",
             "v_i_z": "$v_{i,z}$", "v_i_r": "$v_{i,r}$"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#CBC3AA", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#E1DBCA", "grid.linewidth": 0.6, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})


def b64(fig, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_summary():
    return json.load(open(f"{BASE}/id_study/summary.json"))


def _series():
    lat = np.load(f"{BASE}/latent3/latent_full.npz")
    z, n_train = lat["z"], int(lat["n_train"])
    id_n, _, _ = observables.zscore(observables.load_discharge_current(BASE), n_train)
    zn, _, _ = observables.zscore(z.T, n_train)
    return id_n, zn.T, n_train


def fig_id_vs_latent(n=900):
    id_n, zn, n_train = _series()
    sl = slice(n_train, n_train + n)
    t = np.arange(n)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.5]})
    axs[0].plot(t, id_n[sl], color=RUST, lw=1.3)
    axs[0].set_ylabel("Id (z-scored)")
    for k, c in enumerate((TEAL, BLUE, AMBER)):
        axs[1].plot(t, zn[sl, k], color=c, lw=1.2, label=f"z{k + 1}")
    axs[1].set_ylabel("latent (z-scored)")
    axs[1].set_xlabel("step into held-out block (50 ns/step)")
    axs[1].legend(ncol=3, loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


def fig_option1_seed_spread(summary):
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    rng = np.random.default_rng(0)
    groups = []
    ctrl = [v for v in summary if v == "baseline" or v.startswith("ctrl")]
    z1 = [pc for v in ctrl for row in summary[v]["rc"]["z1drive"]["pc_per_seed"] for pc in row]
    idd = [pc for v in ctrl for row in summary[v]["rc"]["id4drive"]["pc_per_seed"] for pc in row]
    for i, (label, vals, col) in enumerate([("z1-driven infer\n(z2, z3 scored)", z1, TEAL), ("Id-driven infer\n(z1, z2, z3 scored)", idd, RUST)]):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=26, color=col, alpha=0.7, edgecolor="white", linewidth=0.4)
        ax.hlines(np.mean(vals), i - 0.3, i + 0.3, color=INK, lw=2)
        ax.text(i + 0.33, np.mean(vals), f"mean {np.mean(vals):.2f}", va="center", fontsize=9, color=INK)
        groups.append(label)
    ax.axhline(0.9, color=GRAY, ls="--", lw=1)
    ax.text(1.42, 0.91, "criterion 0.9", ha="right", fontsize=9, color=GRAY)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(groups)
    ax.set_xlim(-0.5, 1.7)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("held-out inferPC")
    n_ae, n_seed = len(ctrl), len(summary[ctrl[0]]["rc"]["z1drive"]["pc_per_seed"])
    ax.set_title(f"{n_ae} independently trained AEs x {n_seed} reservoir seeds", fontsize=10, color=GRAY)
    fig.tight_layout()
    return fig


def fig_window_length():
    res = json.load(open(f"{BASE}/latent3/id_map/id_window_to_latent.json"))["results"]
    span = [r["window_steps"] for r in res]
    mlp = [np.mean(r["mlp_r2"]) for r in res]
    rdg = [np.mean(r["ridge_r2"]) for r in res]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(span, mlp, "o-", color=TEAL, lw=2, label="MLP")
    ax.plot(span, rdg, "s--", color=BLUE, lw=1.6, label="ridge")
    ax.axvline(158, color=GRAY, ls=":", lw=1)
    ax.text(160, 0.05, "one breathing\nperiod (~158 steps)", fontsize=8.5, color=GRAY)
    ax.set_xlabel("window of past Id (steps)")
    ax.set_ylabel("held-out R$^2$ of the latent\n(mean of z1, z2, z3)")
    ax.set_ylim(-0.05, 1.02)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def fig_observer_timeseries(n=700):
    r = np.load(f"{BASE}/latent3/observer_mlp150/results.npz")
    rc = np.load(f"{BASE}/latent3/n200/rc_study_id4drive_decoded/results.npz")
    z_true, z_obs = r["z_true"], r["z_pred"]
    zn_mean, zn_std = z_true.mean(0), z_true.std(0)
    t = np.arange(n)
    fig, axs = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True)
    for k in range(3):
        axs[k].plot(t, (z_true[:n, k] - zn_mean[k]) / zn_std[k], color=INK, lw=1.5, label="true")
        axs[k].plot(t, (z_obs[:n, k] - zn_mean[k]) / zn_std[k], color=TEAL, lw=1.2, label="window observer")
        axs[k].plot(t, rc["y_infer"][k, :n], color=RUST, lw=1.0, alpha=0.8, label="reservoir, Id-driven (seed 0)")
        axs[k].set_ylabel(f"z{k + 1}")
    axs[0].legend(ncol=3, loc="upper right", fontsize=8.5, frameon=False)
    axs[-1].set_xlabel("step into held-out block (50 ns/step)")
    fig.tight_layout()
    return fig


def fig_field_errors(summary):
    b = summary["baseline"]
    obs = b["observer"]["decoded_field_error_pct"]
    rc1 = b["rc"]["z1drive"]["decoded"]["decoded_field_error_pct"]
    rcid = b["rc"]["id4drive"]["decoded"]["decoded_field_error_pct"]
    series = [("AE floor", obs["ae_only"], GRAY), ("window observer (Id only)", obs["observer"], TEAL),
              ("reservoir, z1-driven (needs z1)", rc1["infer"], BLUE), ("reservoir, Id-driven (seed 0)", rcid["infer"], RUST)]
    x = np.arange(len(FIELD_NAMES))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, (label, d, col) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, [d[f] for f in FIELD_NAMES], w, color=col, label=label)
    ax.scatter(x, [obs["persistence"][f] for f in FIELD_NAMES], marker="_", s=900, color=AMBER, lw=2.6, zorder=5, label="persistence")
    ax.set_xticks(x)
    ax.set_xticklabels([FIELD_TEX[f] for f in FIELD_NAMES])
    ax.set_ylabel("mean relative error (%)")
    ax.legend(ncol=3, fontsize=8.5, frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def fig_noise(summary):
    b = summary["baseline"]
    runs = [("trained on clean Id", b["observer"]["noise_sweep"], TEAL)]
    if "observer_mlp150_tn0.05" in b.get("observer_extras", {}):
        runs.append(("trained with 5% noise", b["observer_extras"]["observer_mlp150_tn0.05"]["noise_sweep"], BLUE))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
    for label, rows, col in runs:
        xs = [100 * r["noise_frac_of_id_std"] for r in rows]
        axs[0].plot(xs, [r["mean_pc"] for r in rows], "o-", color=col, lw=2, label=label)
        axs[1].plot(xs, [r["mean_decoded_error_pct"] for r in rows], "o-", color=col, lw=2, label=label)
    axs[0].set_ylabel("mean latent PC")
    axs[1].set_ylabel("mean decoded field error (%)")
    for a in axs:
        a.set_xlabel("noise added to the Id input (% of Id std)")
    axs[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- option 2 ----
def _order(summary):
    ctrl = [v for v in ("baseline", "ctrl_s0", "ctrl_s1", "ctrl_s2") if v in summary]
    sup = sorted([v for v in summary if v.startswith("sup")], key=lambda v: (float(v.split("_w")[1].split("_d")[0]), float(v.split("_d")[-1])))
    cond = sorted(v for v in summary if v.startswith("cond"))
    return ctrl + sup + cond


def _color(v):
    return GRAY if (v == "baseline" or v.startswith("ctrl")) else (TEAL if v.startswith("sup") else RUST)


def pretty(v):
    if v == "baseline":
        return "production\nAE"
    if v.startswith("ctrl"):
        return f"control\nseed {v[-1]}"
    if v.startswith("sup"):
        lam, mu = v.split("_w")[1].split("_d")
        return f"sup\n$\\lambda$={lam}\n$\\mu$={mu}"
    return f"cond\n$\\mu$={v.split('_d')[1]}"


def fig_option2_overview(summary):
    order = _order(summary)
    x = np.arange(len(order))
    col = [_color(v) for v in order]
    fig, axs = plt.subplots(3, 1, figsize=(10, 8.6), sharex=True)
    err = [summary[v]["static"]["mean_field_error_pct_all_frames"] for v in order]
    ctrl_err = [summary[v]["static"]["mean_field_error_pct_all_frames"] for v in order if _color(v) == GRAY]
    axs[0].bar(x, err, color=col)
    axs[0].axhspan(min(ctrl_err), max(ctrl_err), color=GRAY, alpha=0.18, zorder=0)
    axs[0].set_ylim(min(err) * 0.95, max(err) * 1.03)
    axs[0].set_ylabel("mean decoded field\nerror, all frames (%)")
    axs[0].set_title("Reconstruction cost (shaded: spread of the four controls)", fontsize=10, color=GRAY, loc="left")

    lin = [summary[v]["coupling"]["free_coordinates_linear_r2"] for v in order]
    mlp = [summary[v]["coupling"]["free_coordinates_mlp_r2"] for v in order]
    axs[1].bar(x, mlp, color=col)
    axs[1].scatter(x, lin, marker="_", s=420, color=INK, lw=2, zorder=5, label="linear map")
    axs[1].set_ylabel("held-out R$^2$ of Id\nfrom z1, z2 alone")
    axs[1].set_ylim(0, 1.02)
    axs[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    axs[1].set_title("Leakage: how much Id the two free coordinates still carry (bar: MLP)", fontsize=10, color=GRAY, loc="left")

    z3 = [summary[v]["coupling"]["single_coordinate_linear_r2"][2] for v in order]
    axs[2].bar(x, z3, color=col)
    axs[2].set_ylabel("held-out R$^2$ of Id\nfrom z3 alone (linear)")
    axs[2].set_ylim(0, 1.02)
    axs[2].set_title("How exactly the third coordinate tracks Id", fontsize=10, color=GRAY, loc="left")
    axs[2].set_xticks(x)
    axs[2].set_xticklabels([pretty(v) for v in order], fontsize=8.3)
    fig.tight_layout()
    return fig


def fig_option2_rc(summary):
    order = [v for v in _order(summary) if "rc" in summary[v]]
    x = np.arange(len(order))
    col = [_color(v) for v in order]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    m1 = [np.mean(summary[v]["rc"]["z1drive"]["pc_mean"]) for v in order]
    s1 = [np.mean(summary[v]["rc"]["z1drive"]["pc_std"]) for v in order]
    axs[0].bar(x, m1, yerr=s1, color=col, capsize=3, ecolor=INK)
    axs[0].set_title("driven by z1 (not measurable)", fontsize=10, color=GRAY, loc="left")
    idv = [v for v in order if "z3drive" in summary[v]["rc"]]
    xi = np.arange(len(idv))
    m3 = [np.mean(summary[v]["rc"]["z3drive"]["pc_mean"]) for v in idv]
    s3 = [np.mean(summary[v]["rc"]["z3drive"]["pc_std"]) for v in idv]
    axs[1].bar(xi, m3, yerr=s3, color=[_color(v) for v in idv], capsize=3, ecolor=INK)
    ctrl_id = np.mean([np.mean(summary[v]["rc"]["id4drive"]["pc_mean"]) for v in order if "id4drive" in summary[v]["rc"]])
    axs[1].axhline(ctrl_id, color=GRAY, ls="--", lw=1.2)
    axs[1].text(len(idv) - 0.6, ctrl_id + 0.02, "option 1: Id as a 4th channel", ha="right", fontsize=8.5, color=GRAY)
    axs[1].set_title("driven by Id (z3 is Id)", fontsize=10, color=GRAY, loc="left")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([pretty(v) for v in order], fontsize=7.6)
    axs[1].set_xticks(xi)
    axs[1].set_xticklabels([pretty(v) for v in idv], fontsize=7.6)
    axs[0].set_ylabel("held-out inferPC, mean over scored channels\n(error bar: std over reservoir seeds)")
    axs[0].set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


def fig_option2_observer(summary):
    order = [v for v in _order(summary) if "observer" in summary[v]]
    x = np.arange(len(order))
    col = [_color(v) for v in order]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axs[0].bar(x, [np.mean([c["pc"] for c in summary[v]["observer"]["per_channel"].values()]) for v in order], color=col)
    axs[0].set_ylim(0.85, 1.0)
    axs[0].set_ylabel("mean latent PC")
    axs[0].set_title("window observer: latent recovered from Id", fontsize=10, color=GRAY, loc="left")
    ratio = [np.mean(list(summary[v]["observer"]["ratio_to_persistence_full"].values())) for v in order]
    axs[1].bar(x, ratio, color=col)
    axs[1].set_ylabel("mean decoded error / persistence")
    axs[1].set_title("decoded fields (lower is better)", fontsize=10, color=GRAY, loc="left")
    for a in axs:
        a.set_xticks(x)
        a.set_xticklabels([pretty(v) for v in order], fontsize=7.6)
    fig.tight_layout()
    return fig


def fig_decode_panel(step=500):
    import torch
    from src.autoencoder import load_checkpoint, decode_to_fields
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(f"{BASE}/latent3/conv_ae_full.pt", device=dev)
    lat = np.load(f"{BASE}/latent3/latent_full.npz")
    n_train = int(lat["n_train"])
    ds = np.load(f"{BASE}/dataset_full.npz", allow_pickle=True, mmap_mode="r")
    x, y = np.asarray(ds["x"]), np.asarray(ds["y"])
    obs = np.load(f"{BASE}/latent3/observer_mlp150/results.npz")
    rc = np.load(f"{BASE}/latent3/n200/rc_study_id4drive_decoded/results.npz")
    zi = (rc["y_infer"][:3, step:step + 1] * rc["std"][:3] + rc["mean"][:3]).T
    frames = [np.asarray(ds["data"][n_train + step]),
              decode_to_fields(model, ckpt, lat["z"][n_train + step:n_train + step + 1], dev)[0],
              decode_to_fields(model, ckpt, obs["z_pred"][step:step + 1], dev)[0],
              decode_to_fields(model, ckpt, zi, dev)[0]]
    rows = ["original", "AE only\n(true latent)", "window observer\n(Id only)", "reservoir\nId-driven (seed 0)"]
    logf = {"n_e", "n_n", "n_i_dot"}
    fig, axs = plt.subplots(4, 7, figsize=(15, 6.2))
    for c, name in enumerate(FIELD_NAMES):
        cols = [np.log10(np.clip(f[c], 1e-30, None)) if name in logf else f[c] for f in frames]
        lo, hi = cols[0].min(), cols[0].max()
        for r in range(4):
            axs[r, c].pcolormesh(x, y, cols[r], shading="auto", cmap="viridis", vmin=lo, vmax=hi)
            axs[r, c].set_aspect("equal")
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
            axs[r, c].grid(False)
        axs[0, c].set_title(FIELD_TEX[name] + (r" ($\log_{10}$)" if name in logf else ""), fontsize=10)
    for r, label in enumerate(rows):
        axs[r, 0].set_ylabel(label, fontsize=9)
    fig.tight_layout()
    return fig
