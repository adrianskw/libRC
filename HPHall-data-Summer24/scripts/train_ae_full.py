"""Warm-start fine-tune of the conv autoencoder on the full 20,000-frame dataset.

Loads weights from conv_ae.pt (trained on the 2,000-frame subsample) and
continues training on all 20,000 frames at a lower learning rate.

Usage: python train_ae_full.py [latent_dim]   (default 3)
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src.autoencoder import ConvAE, save_checkpoint  # noqa: E402

BASE = os.path.join(REPO_ROOT, "HPHall-data-Summer24")
LATENT_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT_DIR = f"{BASE}/latent{LATENT_DIM}"
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = f"{BASE}/dataset_full.npz"
WARM_START_PATH = f"{OUT_DIR}/conv_ae.pt"
MODEL_PATH = f"{OUT_DIR}/conv_ae_full.pt"
BATCH_SIZE = 64
EPOCHS = 60
LR = 5e-4
VAL_FRACTION = 0.15
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

npz = np.load(DATA_PATH, allow_pickle=True)
data = npz["data"]
field_names = list(npz["field_names"])
frame_indices = npz["frame_indices"]
N, C, J, I = data.shape
print(f"data: {data.shape}, fields: {field_names}")

LOG_FIELDS = {"n_e", "n_n", "n_i_dot"}
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
print(f"train: {train_x.shape}, val: {val_x.shape}")

train_loader = DataLoader(TensorDataset(train_x), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_x), batch_size=BATCH_SIZE, shuffle=False)


model = ConvAE(in_ch=C, latent_dim=LATENT_DIM).to(device)

warm = torch.load(WARM_START_PATH, weights_only=False)
model.load_state_dict(warm["model_state"])
print(f"warm-started from {WARM_START_PATH} (trained on {warm.get('field_names')})")

opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# baseline loss before any fine-tuning, to quantify the warm start's head start
model.eval()
with torch.no_grad():
    warm_val = 0.0
    for (batch,) in val_loader:
        batch = batch.to(device)
        recon, _ = model(batch)
        warm_val += loss_fn(recon, batch).item() * batch.size(0)
    warm_val /= n_val
print(f"pre-fine-tune val loss on full data: {warm_val:.5f}")

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
    if epoch % 5 == 0 or epoch == 1:
        print(f"epoch {epoch:4d}  train {train_loss:.5f}  val {val_loss:.5f}")

save_checkpoint(
    MODEL_PATH, model, mean, std, field_names, LOG_FIELDS, LATENT_DIM,
    warm_started_from=WARM_START_PATH, pre_finetune_val_loss=warm_val,
)
print("Saved model:", MODEL_PATH)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(train_losses, label="train")
ax.plot(val_losses, label="val")
ax.axhline(warm_val, color="gray", linestyle="--", linewidth=1, label="pre-fine-tune val")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE (normalized units)")
ax.set_yscale("log")
ax.legend()
ax.set_title("Conv-AE fine-tuning on full 20,000-frame dataset")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/ae_loss_curve_full.png", dpi=150)
print(f"wrote {OUT_DIR}/ae_loss_curve_full.png")

model.eval()
with torch.no_grad():
    full_x = torch.from_numpy(data_norm).float().to(device)
    z_chunks = []
    for i in range(0, N, 2000):
        z_chunks.append(model.encode(full_x[i:i + 2000]).cpu())
    z_all = torch.cat(z_chunks, 0).numpy()

np.savez(
    f"{OUT_DIR}/latent_full.npz",
    z=z_all, frame_indices=frame_indices, n_train=n_train,
)
print(f"wrote {OUT_DIR}/latent_full.npz, z_all shape:", z_all.shape)
