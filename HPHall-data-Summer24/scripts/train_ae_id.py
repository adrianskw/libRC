"""Train the conv autoencoder with the discharge current tied into the latent (option 2).

Follows the production recipe of train_ae.py + train_ae_full.py exactly (stage 1: 200
epochs on the 2,000-frame subsample, batch 32, lr 1e-3; stage 2: warm start, 60 epochs on
all 20,000 frames, batch 64, lr 5e-4; chronological 85/15 split; log10 on n_e/n_n/n_i_dot;
same normalization), so that `--id-mode none` is a same-machine control and any difference
to the other modes is attributable to how Id enters the model.

  --id-mode none         plain AE (control)
  --id-mode supervised   3 free encoder outputs; the last is trained toward z-scored Id
                         (loss + id_weight * MSE(z3, Id_n))
  --id-mode conditional  2 free encoder outputs; the decoder is handed [z1, z2, Id_n]
  --decorr-weight mu     penalize the free coordinates for correlating with Id (batch corr^2)

Id is z-scored with train-block statistics. Everything is written to latent{D}_{variant}/
in the same layout as latent{D}/ (conv_ae.pt, conv_ae_full.pt, latent_full.npz) plus
train_history.json and config.json, so downstream scripts only need a different directory.

Usage: python train_ae_id.py --id-mode supervised --id-weight 1 --variant sup_w1
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src.autoencoder import ConvAE, ID_MODES, id_loss_terms, save_checkpoint  # noqa: E402
from src import observables  # noqa: E402

BASE = os.path.join(REPO_ROOT, "HPHall-data-Summer24")
LOG_FIELDS = {"n_e", "n_n", "n_i_dot"}
VAL_FRACTION = 0.15

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--id-mode", choices=ID_MODES, default="none")
ap.add_argument("--id-weight", type=float, default=1.0, help="lambda on MSE(z3, Id_n) (supervised mode)")
ap.add_argument("--decorr-weight", type=float, default=0.0, help="mu on the free-coordinate/Id decorrelation penalty")
ap.add_argument("--latent-dim", type=int, default=3)
ap.add_argument("--variant", required=True, help="output goes to latent{D}_{variant}/")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--stage1-epochs", type=int, default=200)
ap.add_argument("--stage2-epochs", type=int, default=60)
args = ap.parse_args()

OUT_DIR = f"{BASE}/latent{args.latent_dim}_{args.variant}"
os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
np.random.seed(args.seed)

Id_all = observables.load_discharge_current(BASE)      # row k-1 <-> frame k


def prepare(path):
    """Load a dataset, log-transform, z-score (train recipe), and align Id to its frames."""
    npz = np.load(path, allow_pickle=True)
    data = npz["data"]
    field_names = [str(s) for s in npz["field_names"]]
    frame_indices = npz["frame_indices"].astype(int)
    data_t = data.copy()
    for c, name in enumerate(field_names):
        if name in LOG_FIELDS:
            data_t[:, c] = np.log10(np.clip(data_t[:, c], 1e-6, None))
    mean = data_t.mean(axis=(0, 2, 3), keepdims=True)
    std = data_t.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    x = ((data_t - mean) / std).astype(np.float32)
    n_train = len(x) - int(len(x) * VAL_FRACTION)
    id_raw = Id_all[frame_indices - 1]
    id_n, id_mean, id_std = observables.zscore(id_raw, n_train)
    return dict(x=x, id_n=id_n.astype(np.float32), id_mean=id_mean.item(), id_std=id_std.item(), mean=mean, std=std,
                field_names=field_names, frame_indices=frame_indices, n_train=n_train)


def run_stage(model, d, epochs, lr, batch_size, label):
    n_train = d["n_train"]
    tr = TensorDataset(torch.from_numpy(d["x"][:n_train]), torch.from_numpy(d["id_n"][:n_train]))
    va = TensorDataset(torch.from_numpy(d["x"][n_train:]), torch.from_numpy(d["id_n"][n_train:]))
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    # Shuffled (own generator, so the training RNG stream is untouched): decorrelation is a batch statistic, and
    # contiguous 64-step slices are dominated by slow trends, which inflated the logged val_decorr in early runs.
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(0))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    hist = []

    def batch_losses(xb, ib):
        xb, ib = xb.to(device), ib.to(device)
        recon, z = model(xb, ib)
        rec = mse(recon, xb)
        id_term, decorr = id_loss_terms(model, z, ib)
        return rec, id_term, decorr

    for epoch in range(1, epochs + 1):
        model.train()
        sums = np.zeros(3)
        for xb, ib in tr_loader:
            rec, id_term, decorr = batch_losses(xb, ib)
            loss = rec + args.id_weight * id_term + args.decorr_weight * decorr
            opt.zero_grad()
            loss.backward()
            opt.step()
            sums += np.array([rec.item(), id_term.item(), decorr.item()]) * len(xb)
        tr_m = sums / n_train
        model.eval()
        sums = np.zeros(3)
        with torch.no_grad():
            for xb, ib in va_loader:
                rec, id_term, decorr = batch_losses(xb, ib)
                sums += np.array([rec.item(), id_term.item(), decorr.item()]) * len(xb)
        va_m = sums / len(va)
        hist.append({"epoch": epoch, "train_recon": tr_m[0], "train_id": tr_m[1], "train_decorr": tr_m[2],
                     "val_recon": va_m[0], "val_id": va_m[1], "val_decorr": va_m[2]})
        if epoch == 1 or epoch % max(1, epochs // 10) == 0:
            print(f"[{label}] epoch {epoch:4d}  recon train {tr_m[0]:.5f} val {va_m[0]:.5f} | "
                  f"id val {va_m[1]:.4f} | decorr val {va_m[2]:.4f}", flush=True)
    return hist


t0 = time.time()
print(f"device={device} id_mode={args.id_mode} lambda={args.id_weight} mu={args.decorr_weight} seed={args.seed} -> {OUT_DIR}")
C = 7

# ---- stage 1: 2,000-frame subsample from scratch ----
d1 = prepare(f"{BASE}/dataset.npz")
model = ConvAE(in_ch=C, latent_dim=args.latent_dim, id_mode=args.id_mode).to(device)
hist1 = run_stage(model, d1, args.stage1_epochs, 1e-3, 32, "stage1")
save_checkpoint(f"{OUT_DIR}/conv_ae.pt", model, d1["mean"], d1["std"], d1["field_names"], LOG_FIELDS, args.latent_dim,
                id_mode=args.id_mode, id_mean=d1["id_mean"], id_std=d1["id_std"])

# ---- stage 2: warm start on all 20,000 frames ----
d2 = prepare(f"{BASE}/dataset_full.npz")
hist2 = run_stage(model, d2, args.stage2_epochs, 5e-4, 64, "stage2")

# ---- outputs ----
extra = dict(id_mode=args.id_mode, id_mean=d2["id_mean"], id_std=d2["id_std"], id_weight=args.id_weight,
             decorr_weight=args.decorr_weight, seed=args.seed, warm_started_from="stage 1 (in memory)")
save_checkpoint(f"{OUT_DIR}/conv_ae_full.pt", model, d2["mean"], d2["std"], d2["field_names"], LOG_FIELDS,
                args.latent_dim, **extra)

model.eval()
x_all, id_all = torch.from_numpy(d2["x"]), torch.from_numpy(d2["id_n"])
with torch.no_grad():
    z_all = torch.cat([model.encode_full(x_all[i:i + 2000].to(device), id_all[i:i + 2000].to(device)).cpu()
                       for i in range(0, len(x_all), 2000)]).numpy()
np.savez(f"{OUT_DIR}/latent_full.npz", z=z_all, frame_indices=d2["frame_indices"], n_train=d2["n_train"], id_n=d2["id_n"])

try:
    git_rev = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception:
    git_rev = None
with open(f"{OUT_DIR}/train_history.json", "w") as f:
    json.dump({"stage1": hist1, "stage2": hist2}, f)
with open(f"{OUT_DIR}/config.json", "w") as f:
    json.dump({**vars(args), "git_rev": git_rev, "command": " ".join(sys.argv), "device": device,
               "final_val_recon": hist2[-1]["val_recon"], "final_val_id": hist2[-1]["val_id"],
               "final_val_decorr": hist2[-1]["val_decorr"], "train_seconds": time.time() - t0}, f, indent=2)
print(f"wrote {OUT_DIR}/ (conv_ae.pt, conv_ae_full.pt, latent_full.npz, train_history.json, config.json) "
      f"in {time.time() - t0:.0f}s; final val recon {hist2[-1]['val_recon']:.5f}")
