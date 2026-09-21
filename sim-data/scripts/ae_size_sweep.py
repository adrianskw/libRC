"""Autoencoder capacity sweep: does a lighter conv-AE hurt reconstruction
much, and does a heavier one help much, relative to the 16/32/64-channel
architecture used throughout the rest of this project's pipeline ("base")?

The motivation: the RC itself is a deliberately small, lightweight model
(N=200 nodes, a linear readout). It's worth knowing whether the AE
feeding it latent trajectories could also be made much smaller without
giving up much reconstruction fidelity, or whether there's real value
left on the table by going bigger.

Trains three ConvAEs that are architecturally identical (3 conv + 3
deconv layers, same kernel/stride/padding, same 3-D latent bottleneck)
and differ only in channel width:

  light:  8  / 16 / 32   channels  (~1/3 the parameters of base)
  base:   16 / 32 / 64   channels  (matches conv_ae_full.pt's architecture)
  heavy:  32 / 64 / 128  channels  (~3x the parameters of base)

All three are trained from scratch (no warm-start pretrain, unlike the
main pipeline's conv_ae_full.pt) for the same number of epochs on the
full 20,000-frame dataset, with an identical optimizer/schedule/split,
so parameter count is the only thing that differs.

Usage: python ae_size_sweep.py [latent_dim] [epochs]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LATENT_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 60

OUT_DIR = f"{BASE}/latent{LATENT_DIM}/ae_size_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = f"{BASE}/dataset_full.npz"
BATCH_SIZE = 64
LR = 5e-4
VAL_FRACTION = 0.15
SEED = 0
LOG_FIELDS = {"n_e", "n_n", "n_i_dot"}

VARIANTS = [
    ("light", (8, 16, 32)),
    ("base", (16, 32, 64)),
    ("heavy", (32, 64, 128)),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

npz = np.load(DATA_PATH, allow_pickle=True)
data = npz["data"]
field_names = list(npz["field_names"])
frame_indices = npz["frame_indices"]
x_grid, y_grid = npz["x"], npz["y"]
N, C, J, I = data.shape
print(f"data: {data.shape}, fields: {field_names}")

data_t = data.copy()
for c, name in enumerate(field_names):
    if name in LOG_FIELDS:
        data_t[:, c] = np.log10(np.clip(data_t[:, c], 1e-6, None))

mean = data_t.mean(axis=(0, 2, 3), keepdims=True)
std = data_t.std(axis=(0, 2, 3), keepdims=True) + 1e-8
data_norm = (data_t - mean) / std

n_val = int(N * VAL_FRACTION)
n_train = N - n_val
train_x = torch.from_numpy(data_norm[:n_train]).float()
val_x = torch.from_numpy(data_norm[n_train:]).float()

train_loader = DataLoader(TensorDataset(train_x), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_x), batch_size=BATCH_SIZE, shuffle=False)


class ConvAE(nn.Module):
    """Same depth/stride structure as the main pipeline's ConvAE; channel
    widths (c1, c2, c3) are the only free parameter."""
    def __init__(self, in_ch, latent_dim, channels):
        super().__init__()
        c1, c2, c3 = channels
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(c2, c3, 3, 2, 1), nn.ReLU(),
        )
        self.enc_shape = (c3, 7, 13)
        flat = c3 * 7 * 13
        self.to_latent = nn.Linear(flat, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 3, stride=2, padding=1, output_padding=(0, 0)), nn.ReLU(),
            nn.ConvTranspose2d(c2, c1, 3, stride=2, padding=1, output_padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(c1, in_ch, 3, stride=1, padding=1),
        )

    def encode(self, x):
        return self.to_latent(self.enc(x).flatten(1))

    def decode(self, z):
        h = self.from_latent(z).view(-1, *self.enc_shape)
        return self.dec(h)[:, :, :25, :50]

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train_variant(tag, channels):
    print(f"\n{'='*70}\n{tag}  channels={channels}\n{'='*70}")
    torch.manual_seed(SEED)
    model = ConvAE(in_ch=C, latent_dim=LATENT_DIM, channels=channels).to(device)
    params = n_params(model)
    print(f"parameters: {params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            running += loss.item() * batch.size(0)
        train_loss = running / n_train

        model.eval()
        with torch.no_grad():
            running = 0.0
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                running += loss_fn(recon, batch).item() * batch.size(0)
            val_loss = running / n_val

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"epoch {epoch:4d}  train {train_loss:.5f}  val {val_loss:.5f}")

    variant_dir = f"{OUT_DIR}/{tag}"
    os.makedirs(variant_dir, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(), "mean": mean, "std": std,
            "field_names": field_names, "log_fields": list(LOG_FIELDS),
            "latent_dim": LATENT_DIM, "channels": channels, "n_params": params,
        },
        f"{variant_dir}/conv_ae.pt",
    )

    # ---- per-field relative reconstruction error, full dataset ----
    model.eval()
    recon_norm = np.empty_like(data_norm, dtype=np.float32)
    z_all = np.empty((N, LATENT_DIM), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, 2000):
            chunk = torch.from_numpy(data_norm[i:i + 2000]).float().to(device)
            r, z = model(chunk)
            recon_norm[i:i + 2000] = r.cpu().numpy()
            z_all[i:i + 2000] = z.cpu().numpy()

    recon_t = recon_norm * std + mean
    recon = recon_t.copy()
    for c, name in enumerate(field_names):
        if name in LOG_FIELDS:
            recon[:, c] = 10 ** recon_t[:, c]

    rel_err = np.abs(recon - data) / (np.abs(data).mean(axis=(0, 2, 3), keepdims=True) + 1e-30)
    mean_rel_err = rel_err.mean(axis=(0, 2, 3))
    err_by_field = {name: float(err) * 100 for name, err in zip(field_names, mean_rel_err)}
    for name, err in err_by_field.items():
        print(f"  {name:10s} mean relative error: {err:.2f}%")

    np.savez(f"{variant_dir}/latent_full.npz", z=z_all, frame_indices=frame_indices, n_train=n_train)

    result = dict(
        tag=tag, channels=list(channels), n_params=params,
        final_train_loss=train_losses[-1], final_val_loss=val_losses[-1],
        best_val_loss=min(val_losses), err_by_field_pct=err_by_field,
        train_losses=train_losses, val_losses=val_losses,
    )
    with open(f"{variant_dir}/result.json", "w") as f:
        json.dump(result, f, indent=2)

    # one example frame, original vs reconstruction, for a visual sanity check
    frame_i = N // 2
    fig, axes = plt.subplots(2, len(field_names), figsize=(22, 6))
    for c, name in enumerate(field_names):
        vmin, vmax = data[frame_i, c].min(), data[frame_i, c].max()
        axes[0, c].pcolormesh(x_grid, y_grid, data[frame_i, c], shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, c].set_title(name)
        axes[0, c].set_aspect("equal"); axes[0, c].set_xticks([]); axes[0, c].set_yticks([])
        axes[1, c].pcolormesh(x_grid, y_grid, recon[frame_i, c], shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, c].set_aspect("equal"); axes[1, c].set_xticks([]); axes[1, c].set_yticks([])
    axes[0, 0].set_ylabel("original", fontsize=11)
    axes[1, 0].set_ylabel("reconstructed", fontsize=11)
    fig.suptitle(f"{tag} ({params:,} params): frame {frame_indices[frame_i]}")
    fig.tight_layout()
    fig.savefig(f"{variant_dir}/reconstruction_example.png", dpi=150)
    print(f"wrote {variant_dir}/reconstruction_example.png, result.json, conv_ae.pt, latent_full.npz")

    return result


results = {tag: train_variant(tag, channels) for tag, channels in VARIANTS}

with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(results, f, indent=2)

# ---- comparison plots ----
tags = [t for t, _ in VARIANTS]
params_list = [results[t]["n_params"] for t in tags]
val_list = [results[t]["final_val_loss"] for t in tags]

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].plot(params_list, val_list, "o-", color="tab:blue")
for t, p, v in zip(tags, params_list, val_list):
    axs[0].annotate(f"{t}\n({p:,})", (p, v), textcoords="offset points", xytext=(8, 4), fontsize=9)
axs[0].set_xscale("log"); axs[0].set_yscale("log")
axs[0].set_xlabel("parameter count"); axs[0].set_ylabel("final val MSE (normalized units)")
axs[0].set_title("AE capacity vs. reconstruction loss")

field_names_order = field_names
bar_w = 0.25
xpos = np.arange(len(field_names_order))
for i, t in enumerate(tags):
    errs = [results[t]["err_by_field_pct"][n] for n in field_names_order]
    axs[1].bar(xpos + (i - 1) * bar_w, errs, width=bar_w, label=f"{t} ({params_list[i]:,})")
axs[1].set_xticks(xpos); axs[1].set_xticklabels(field_names_order, rotation=30, ha="right")
axs[1].set_ylabel("mean relative error (%)")
axs[1].set_title("Per-field reconstruction error by AE size")
axs[1].legend(fontsize=8)

fig.suptitle(f"Conv-AE capacity sweep, latent_dim={LATENT_DIM}, {EPOCHS} epochs each")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/capacity_sweep_summary.png", dpi=150)
print(f"\nwrote {OUT_DIR}/capacity_sweep_summary.png and {OUT_DIR}/summary.json")

print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
print(f"{'variant':8s}  {'params':>10s}  {'val_loss':>10s}  {'mean field err%':>16s}")
for t in tags:
    r = results[t]
    mean_err = np.mean(list(r["err_by_field_pct"].values()))
    print(f"{t:8s}  {r['n_params']:10,d}  {r['final_val_loss']:10.5f}  {mean_err:16.2f}")
