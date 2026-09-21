"""Reservoir computer on the AE latents plus measured scalar observables (option 1).

The AE is untouched. The reservoir state is any chosen set of channels: the AE
latents (z1, z2, ...) and/or observables from log.dat (id = discharge current,
thrust, ...). Every channel is z-scored on the train block only. The headline
experiment is infer mode driven by a *measurable* channel, e.g. Id alone: the
reservoir is shown only Id and must reconstruct the latent state, which decodes
back to the full 7-field plasma state.

Presets (same script, only flags differ):
  baseline   --channels z1,z2,z3      --drive z1   (reproduces train_rc.py)
  id-drive   --channels z1,z2,z3,id   --drive id
  z1-drive   --channels z1,z2,z3,id   --drive z1   (how well is Id itself inferred)

Usage: python rc_with_id.py [--latent-dim 3] [--n-res 200] [--channels ...]
                            [--drive ...] [--id-smooth W] [--seed S] [--variant V] [--tag NAME] [--no-decode]
Output: latent{D}/n{N}/rc_{tag}/{config.json, metrics.json, results.npz, *.png}
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src import mapRC, observables  # noqa: E402
from src.metrics import period_steps  # noqa: E402

BASE = os.path.join(REPO_ROOT, "HPHall-data-Summer24")

# Reservoir constants, identical to train_rc.py so results sit beside its baseline.
RHO = 0.9
DEGREE = 10
ALPHA = 1e-3
TRAIN_SKIP = 200

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--latent-dim", type=int, default=3)
ap.add_argument("--n-res", type=int, default=200)
ap.add_argument("--channels", default="z1,z2,z3,id", help="comma list: z1..zD and/or log.dat observables")
ap.add_argument("--drive", default="id", help="comma list of channels observed during infer")
ap.add_argument("--id-smooth", type=int, default=0, help="causal moving-average window (steps) on 'id'; 0 = off")
ap.add_argument("--seed", type=int, default=0, help="RNG seed for the reservoir draw (train_rc.py uses 0)")
ap.add_argument("--variant", default="", help="use the AE variant in latent{D}_{variant}/ (see train_ae_id.py)")
ap.add_argument("--tag", default=None)
ap.add_argument("--no-decode", action="store_true", help="skip decoding to physical fields")
args = ap.parse_args()

channels = [c.strip() for c in args.channels.split(",")]
drive = [c.strip() for c in args.drive.split(",")]
if not set(drive) <= set(channels):
    sys.exit(f"--drive {drive} must be a subset of --channels {channels}")
SEED = args.seed
tag = args.tag or (f"{'-'.join(channels)}_drive-{'-'.join(drive)}" + (f"_smooth{args.id_smooth}" if args.id_smooth else "")
                   + (f"_seed{SEED}" if SEED else ""))
OUT_DIR = f"{BASE}/latent{args.latent_dim}" + (f"_{args.variant}" if args.variant else "")
RUN_DIR = f"{OUT_DIR}/n{args.n_res}/rc_{tag}"
os.makedirs(RUN_DIR, exist_ok=True)

np.random.seed(SEED)

# ---- assemble the state: latents + observables, all aligned 1:1 with the frames ----
lat = np.load(f"{OUT_DIR}/latent_full.npz")
z = lat["z"]
frame_indices = lat["frame_indices"].astype(int)
n_train = int(lat["n_train"])
available = observables.latent_channels(z)
for name in channels:
    if name not in available:
        obs = observables.load_observable(BASE, name)
        if len(obs) != len(z):
            sys.exit(f"observable {name!r} has {len(obs)} rows but there are {len(z)} frames")
        if name == "id" and args.id_smooth > 1:
            obs = observables.smooth(obs, args.id_smooth, causal=True)
        available[name] = obs

state = observables.stack_state(available, channels)          # (D, M), reservoir convention
D, M = state.shape
n_val = M - n_train
state_norm, mean, std = observables.zscore(state, n_train)    # train-block statistics only
y_in, y_val = state_norm[:, :n_train], state_norm[:, n_train:]
drive_idx = [channels.index(c) for c in drive]

sigma = 2.0 / np.max(np.max(y_in, axis=1) - np.min(y_in, axis=1))
print(f"tag={tag}  channels={channels}  drive={drive}  D={D} n_train={n_train} n_val={n_val} sigma={sigma:.4f}")

# ---- build, drive, fit ----
RC = mapRC(args.n_res, bias=True)
RC.makeConnectionMatDegree(RHO, degree=DEGREE, loc=-1, scale=1)
RC.makeInputMat(D, sigma, randMin=-1, randMax=1)
RC.listen(y_in)
RC.train(y_in, start=TRAIN_SKIP, alpha=ALPHA)
RC.echo(n_val)
RC.infer(y_val[drive_idx], drive_idx)
PCmat = RC.inferPC(y_val)


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


echo_rmse, infer_rmse = rmse(RC.y_echo, y_val), rmse(RC.y_infer, y_val)
metrics = {"channels": channels, "drive": drive, "n_train": n_train, "n_val": n_val, "per_channel": {}}
print("\nchannel   echo_RMSE  infer_RMSE  inferPC   (normalized units)")
for i, name in enumerate(channels):
    pc = None if i in drive_idx else float(PCmat[i, i])
    metrics["per_channel"][name] = {"echo_rmse": float(echo_rmse[i]), "infer_rmse": float(infer_rmse[i]), "infer_pc": pc}
    print(f"  {name:6s}  {echo_rmse[i]:9.3f}  {infer_rmse[i]:10.3f}  " + ("  (driven)" if pc is None else f"{pc:8.3f}"))

# ---- breathing-mode period of the free run vs. the truth (the report's echo metric) ----
if "z1" in channels:
    p_true, _ = period_steps(y_val[channels.index("z1")])
    p_echo, n_echo_peaks = period_steps(RC.y_echo[channels.index("z1")])
    metrics["echo_period"] = {"true_steps": p_true, "echo_steps": p_echo, "echo_peaks": n_echo_peaks,
                              "mismatch_pct": None if np.isnan(p_echo) else float(100 * abs(p_echo - p_true) / p_true)}
    tail = slice(n_val // 2, None)   # a free run that has collapsed to a fixed point has ~0 spread here
    metrics["echo_tail_std_ratio"] = float(RC.y_echo[:, tail].std(axis=1).mean() / y_val[:, tail].std(axis=1).mean())
    print(f"echo period: true {p_true:.1f} steps, echo {p_echo:.1f} steps; echo/true late-run spread {metrics['echo_tail_std_ratio']:.2f}")

# ---- decode to physical fields (only meaningful when all latents are in the state) ----
latent_names = [f"z{i + 1}" for i in range(args.latent_dim)]
if not args.no_decode and all(n in channels for n in latent_names):
    import torch
    from src.autoencoder import load_checkpoint, decode_to_fields, relative_field_error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(f"{OUT_DIR}/conv_ae_full.pt", device=device)
    field_names = [str(s) for s in ckpt["field_names"]]
    zi = [channels.index(n) for n in latent_names]

    def to_z(y_norm):
        return (y_norm[zi] * std[zi] + mean[zi]).T

    ds = np.load(f"{BASE}/dataset_full.npz", allow_pickle=True, mmap_mode="r")
    true_fields = np.asarray(ds["data"][n_train:n_train + n_val])
    persist = np.broadcast_to(np.asarray(ds["data"][n_train - 1]), true_fields.shape)  # hold last training frame

    recon = {"ae_only": decode_to_fields(model, ckpt, to_z(y_val), device),
             "infer": decode_to_fields(model, ckpt, to_z(RC.y_infer), device),
             "echo": decode_to_fields(model, ckpt, to_z(RC.y_echo), device)}
    H = 100   # matches the FNO study's rollout horizon
    err = {k: relative_field_error(v, true_fields) for k, v in recon.items()}
    err["persistence"] = relative_field_error(persist, true_fields)
    err100 = {k: relative_field_error(v[:H], true_fields[:H]) for k, v in recon.items()}
    err100["persistence"] = relative_field_error(persist[:H], true_fields[:H])

    metrics["decoded_field_error_pct"] = {k: dict(zip(field_names, (100 * v).tolist())) for k, v in err.items()}
    metrics["decoded_ratio_to_persistence_full"] = {
        k: dict(zip(field_names, (err[k] / err["persistence"]).tolist())) for k in ("infer", "echo")}
    metrics["decoded_ratio_to_persistence_first100"] = {
        k: dict(zip(field_names, (err100[k] / err100["persistence"]).tolist())) for k in ("infer", "echo")}
    print(f"\ndecoded relative field error, full held-out window (%), and infer/persistence ratio:")
    print(f"  {'field':9s} {'AE-only':>8s} {'infer':>8s} {'echo':>8s} {'persist':>8s} {'infer/persist':>14s}")
    for i, name in enumerate(field_names):
        print(f"  {name:9s} {100*err['ae_only'][i]:8.2f} {100*err['infer'][i]:8.2f} {100*err['echo'][i]:8.2f} "
              f"{100*err['persistence'][i]:8.2f} {err['infer'][i]/err['persistence'][i]:14.3f}")

# ---- save: config (everything needed to reproduce), metrics, arrays, plots ----
try:
    git_rev = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception:
    git_rev = None
config = {"tag": tag, "latent_dim": args.latent_dim, "n_res": args.n_res, "channels": channels, "drive": drive,
          "id_smooth": args.id_smooth, "variant": args.variant, "seed": SEED, "rho": RHO, "degree": DEGREE, "alpha": ALPHA,
          "train_skip": TRAIN_SKIP, "sigma": float(sigma), "n_train": n_train, "git_rev": git_rev,
          "command": " ".join(sys.argv)}
with open(f"{RUN_DIR}/config.json", "w") as f:
    json.dump(config, f, indent=2)
with open(f"{RUN_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
np.savez(f"{RUN_DIR}/results.npz", y_val=y_val, y_echo=RC.y_echo, y_infer=RC.y_infer, mean=mean, std=std,
         channels=np.array(channels), drive_idx=np.array(drive_idx), frame_indices_val=frame_indices[n_train:])

Mplot = min(3000, n_val)
fig, axs = plt.subplots(D, 1, figsize=(10, 1.9 * D + 1), sharex=True)
axs = np.atleast_1d(axs)
for i, name in enumerate(channels):
    axs[i].plot(y_val[i, :Mplot], label="true", lw=1.2)
    axs[i].plot(RC.y_echo[i, :Mplot], label="echo (autonomous)", lw=1, alpha=0.85)
    axs[i].plot(RC.y_infer[i, :Mplot], label="infer", lw=1, alpha=0.85)
    axs[i].set_ylabel(name + (" *" if i in drive_idx else ""))
axs[0].legend(loc="upper right", fontsize=8)
axs[0].set_title(f"{tag}: held-out block (* = observed during infer)")
axs[-1].set_xlabel("step (val window, 50 ns/step)")
fig.tight_layout()
fig.savefig(f"{RUN_DIR}/rc_timeseries.png", dpi=150)
print(f"\nsaved {RUN_DIR}/ (config.json, metrics.json, results.npz, rc_timeseries.png)")
