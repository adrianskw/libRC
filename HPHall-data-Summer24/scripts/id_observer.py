"""Windowed Id observer (option 1, revised): latent state from a window of past Id.

Replaces the reservoir's infer path with a direct causal regression, Id(t), Id(t-5), ...,
Id(t-span) -> latent(t), fit on the train block only (src/observer.py). Scored with the
first report's metrics on the same held-out block (frames 17,001-20,000):

  - latent PC (squared correlation, the report's inferPC) and RMSE in normalized units;
  - per-field decoded relative error through the frozen AE decoder, next to the AE-only
    floor and a persistence baseline (hold the last training frame), and their ratio;
  - a measurement-noise sweep: Gaussian noise (fraction of Id's std) added to the Id input
    at test time, since a real measurement is not clean.

Works on any AE variant: for a conditional variant the third latent coordinate *is* the
measured Id, so it is used as given rather than estimated.

Usage: python id_observer.py [--variant V] [--kind mlp|ridge] [--span 150] [--step 5]
                             [--train-noise 0] [--tag NAME]
Output: latent{D}[_V]/observer_{tag}/{config.json, metrics.json, results.npz}
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src import observables  # noqa: E402
from src.autoencoder import load_checkpoint, decode_to_fields, relative_field_error  # noqa: E402
from src.observer import WindowObserver  # noqa: E402

BASE = os.path.join(REPO_ROOT, "HPHall-data-Summer24")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--latent-dim", type=int, default=3)
ap.add_argument("--variant", default="")
ap.add_argument("--kind", choices=("mlp", "ridge"), default="mlp")
ap.add_argument("--span", type=int, default=150, help="window length in steps (150 ~ one breathing period)")
ap.add_argument("--step", type=int, default=5, help="spacing between taps")
ap.add_argument("--train-noise", type=float, default=0.0, help="Gaussian noise (units of input std) added while training")
ap.add_argument("--noise-levels", type=float, nargs="*", default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default=None)
args = ap.parse_args()

OUT_DIR = f"{BASE}/latent{args.latent_dim}" + (f"_{args.variant}" if args.variant else "")
tag = args.tag or f"{args.kind}{args.span}" + (f"_tn{args.train_noise:g}" if args.train_noise else "")
RUN_DIR = f"{OUT_DIR}/observer_{tag}"
os.makedirs(RUN_DIR, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

lat = np.load(f"{OUT_DIR}/latent_full.npz")
z, n_train = lat["z"].astype(float), int(lat["n_train"])
M, D = z.shape
n_val = M - n_train
Id = observables.load_discharge_current(BASE)
id_n, _, _ = observables.zscore(Id, n_train)
z_norm, z_mean, z_std = observables.zscore(z.T, n_train)
z_norm = z_norm.T                                     # (M, D), normalized on the train block
lags = list(range(0, args.span + 1, args.step))
tr_idx, va_idx = np.arange(max(lags), n_train), np.arange(n_train, M)

model, ckpt = load_checkpoint(f"{OUT_DIR}/conv_ae_full.pt", device="cuda" if torch.cuda.is_available() else "cpu")
device = next(model.parameters()).device
field_names = [str(s) for s in ckpt["field_names"]]
id_mode = ckpt.get("id_mode", "none")
z3_is_id = id_mode == "conditional"                    # third coordinate is the measured Id, exactly

ds = np.load(f"{BASE}/dataset_full.npz", allow_pickle=True, mmap_mode="r")
true_fields = np.asarray(ds["data"][n_train:n_train + n_val])
persist = np.broadcast_to(np.asarray(ds["data"][n_train - 1]), true_fields.shape)

obs = WindowObserver(lags, kind=args.kind, train_noise=args.train_noise).fit(id_n, z_norm, tr_idx)


def score(id_input):
    """Predict the held-out latents from `id_input` (the Id series the observer is allowed to see)."""
    pred_norm = obs.predict(id_input, va_idx)                              # (n_val, D)
    if z3_is_id:
        pred_norm[:, 2] = z_norm[va_idx, 2]
    y, p = z_norm[va_idx], pred_norm
    yc, pc_ = y - y.mean(0), p - p.mean(0)
    per = {}
    for k in range(D):
        r2 = 1 - np.sum((y[:, k] - p[:, k]) ** 2) / np.sum(yc[:, k] ** 2)
        pc = (pc_[:, k] @ yc[:, k]) ** 2 / ((pc_[:, k] @ pc_[:, k]) * (yc[:, k] @ yc[:, k]))
        per[f"z{k + 1}"] = {"pc": float(pc), "r2": float(r2), "rmse": float(np.sqrt(np.mean((y[:, k] - p[:, k]) ** 2)))}
    z_pred_raw = pred_norm * z_std[:, 0] + z_mean[:, 0]
    return per, z_pred_raw


per_channel, z_pred = score(id_n)
recon_obs = decode_to_fields(model, ckpt, z_pred, device)
recon_ae = decode_to_fields(model, ckpt, z[va_idx], device)
H = 100
err = {"ae_only": relative_field_error(recon_ae, true_fields), "observer": relative_field_error(recon_obs, true_fields),
       "persistence": relative_field_error(persist, true_fields)}
err100 = {"observer": relative_field_error(recon_obs[:H], true_fields[:H]),
          "persistence": relative_field_error(persist[:H], true_fields[:H])}
metrics = {
    "variant": args.variant or "baseline", "id_mode": id_mode, "kind": args.kind, "span": args.span, "taps": len(lags),
    "train_noise": args.train_noise, "n_train": n_train, "n_val": n_val, "per_channel": per_channel,
    "decoded_field_error_pct": {k: dict(zip(field_names, (100 * v).tolist())) for k, v in err.items()},
    "ratio_to_persistence_full": dict(zip(field_names, (err["observer"] / err["persistence"]).tolist())),
    "ratio_to_persistence_first100": dict(zip(field_names, (err100["observer"] / err100["persistence"]).tolist())),
}

print(f"variant={metrics['variant']} id_mode={id_mode} {args.kind} window={args.span} steps ({len(lags)} taps)")
print("  latent channel   PC     R2    RMSE")
for k, m in per_channel.items():
    print(f"  {k:14s} {m['pc']:.3f}  {m['r2']:.3f}  {m['rmse']:.3f}")
print(f"  {'field':9s} {'AE-only':>8s} {'observer':>9s} {'persist':>8s} {'obs/persist':>12s}")
for i, name in enumerate(field_names):
    print(f"  {name:9s} {100*err['ae_only'][i]:8.2f} {100*err['observer'][i]:9.2f} {100*err['persistence'][i]:8.2f} "
          f"{err['observer'][i]/err['persistence'][i]:12.3f}")

# ---- measurement-noise sweep (test-time noise on the Id the observer sees) ----
rng = np.random.default_rng(args.seed)
noise_rows = []
for level in args.noise_levels:
    noisy = id_n + level * rng.standard_normal(id_n.shape)
    per, zp = score(noisy)
    e = relative_field_error(decode_to_fields(model, ckpt, zp, device), true_fields)
    noise_rows.append({"noise_frac_of_id_std": level, "mean_pc": float(np.mean([v["pc"] for v in per.values()])),
                       "pc": {k: v["pc"] for k, v in per.items()},
                       "mean_decoded_error_pct": float(100 * e.mean()),
                       "mean_ratio_to_persistence": float((e / err["persistence"]).mean())})
metrics["noise_sweep"] = noise_rows
print("  noise (frac of Id std)  mean PC   mean decoded err %   mean ratio to persistence")
for r in noise_rows:
    print(f"  {r['noise_frac_of_id_std']:>22g}  {r['mean_pc']:7.3f}  {r['mean_decoded_error_pct']:18.2f}  {r['mean_ratio_to_persistence']:10.3f}")

try:
    git_rev = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception:
    git_rev = None
with open(f"{RUN_DIR}/config.json", "w") as f:
    json.dump({**vars(args), "tag": tag, "lags": lags, "git_rev": git_rev, "command": " ".join(sys.argv)}, f, indent=2)
with open(f"{RUN_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
np.savez(f"{RUN_DIR}/results.npz", z_pred=z_pred, z_true=z[va_idx], id_val=Id[va_idx], frame_indices_val=lat["frame_indices"][va_idx])
print(f"saved {RUN_DIR}/")
