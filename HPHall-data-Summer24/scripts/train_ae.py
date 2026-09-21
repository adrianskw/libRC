"""Train a convolutional autoencoder with an N-D latent space on Hall thruster frames.

Usage: python train_ae.py [latent_dim]   (default 3)
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

DATA_PATH = f"{BASE}/dataset.npz"
MODEL_PATH = f"{OUT_DIR}/conv_ae.pt"
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3
VAL_FRACTION = 0.15
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

npz = np.load(DATA_PATH, allow_pickle=True)
data = npz["data"]  # (N, 7, 25, 50)
field_names = list(npz["field_names"])
frame_indices = npz["frame_indices"]
N, C, J, I = data.shape
print(f"data: {data.shape}, fields: {field_names}")

# log-transform strictly-positive, wide-dynamic-range densities before z-score
LOG_FIELDS = {"n_e", "n_n", "n_i_dot"}
data_t = data.copy()
for c, name in enumerate(field_names):
    if name in LOG_FIELDS:
        data_t[:, c] = np.log10(np.clip(data_t[:, c], 1e-6, None))

mean = data_t.mean(axis=(0, 2, 3), keepdims=True)
std = data_t.std(axis=(0, 2, 3), keepdims=True) + 1e-8
data_norm = (data_t - mean) / std

# chronological split: last VAL_FRACTION of the (already time-ordered) frames held out
n_val = int(N * VAL_FRACTION)
n_train = N - n_val
train_x = torch.from_numpy(data_norm[:n_train]).float()
val_x = torch.from_numpy(data_norm[n_train:]).float()
print(f"train: {train_x.shape}, val: {val_x.shape}")

train_loader = DataLoader(TensorDataset(train_x), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_x), batch_size=BATCH_SIZE, shuffle=False)


model = ConvAE(in_ch=C, latent_dim=LATENT_DIM).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# sanity-check shapes once
with torch.no_grad():
    test_out, test_z = model(train_x[:2].to(device))
    print("recon shape:", test_out.shape, "latent shape:", test_z.shape)
assert test_out.shape[-2:] == (25, 50)

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
    if epoch % 10 == 0 or epoch == 1:
        print(f"epoch {epoch:4d}  train {train_loss:.5f}  val {val_loss:.5f}")

save_checkpoint(MODEL_PATH, model, mean, std, field_names, LOG_FIELDS, LATENT_DIM)
print("Saved model:", MODEL_PATH)

# --- plots ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(train_losses, label="train")
ax.plot(val_losses, label="val")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE (normalized units)")
ax.set_yscale("log")
ax.legend()
ax.set_title("Conv-AE training curve")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/ae_loss_curve.png", dpi=150)
print(f"wrote {OUT_DIR}/ae_loss_curve.png")

# full-dataset latent trajectory (chronological)
model.eval()
with torch.no_grad():
    full_x = torch.from_numpy(data_norm).float().to(device)
    z_all = model.encode(full_x).cpu().numpy()

np.savez(
    f"{OUT_DIR}/latent.npz",
    z=z_all, frame_indices=frame_indices, n_train=n_train,
)
print(f"wrote {OUT_DIR}/latent.npz, z_all shape:", z_all.shape)
