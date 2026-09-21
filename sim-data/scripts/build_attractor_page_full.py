"""Generate the standalone interactive attractor HTML page from the full 20,000-frame run.

Same layout/design as build_attractor_page.py, but sources data from the
full-dataset warm-started model instead of the 2,000-frame subsample.
"""
import json
import os
import re

import numpy as np
import torch
from scipy.signal import find_peaks

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---- load latent trajectory (all 20,000 frames) ----
lat = np.load(f"{BASE}/latent_full.npz")
z = lat["z"]
frame_indices = lat["frame_indices"].astype(int)
n_train = int(lat["n_train"])
n = len(frame_indices)

# ---- checkpoint metadata ----
ckpt = torch.load(f"{BASE}/conv_ae_full.pt", weights_only=False, map_location="cpu")
pre_finetune_val = ckpt["pre_finetune_val_loss"]

# ---- final train/val loss: parse from the training run's own log ----
log_path = r"C:\Users\Adrian\AppData\Local\Temp\claude\C--Users-Adrian-Documents-GitHub-AE-RC\cbaba7cc-69e4-433b-8be7-3de9cd84232f\tasks\bzgn9k9pe.output"
with open(log_path) as f:
    log_text = f.read()
epoch_lines = re.findall(r"epoch\s+(\d+)\s+train\s+([\d.]+)\s+val\s+([\d.]+)", log_text)
final_epoch, final_train, final_val = epoch_lines[-1]
final_train, final_val = float(final_train), float(final_val)
print(f"parsed final epoch {final_epoch}: train={final_train:.5f} val={final_val:.5f} "
      f"(pre-fine-tune val was {pre_finetune_val:.5f})")

# ---- parse real sim params from parm.in ----
with open(f"{BASE}/parm.in") as f:
    parm_text = f.read()


def parm(name):
    m = re.search(rf"^{name}\s+([0-9eE.+-]+)", parm_text, re.MULTILINE)
    return float(m.group(1))


dt0 = parm("DT_0")
n_its = int(parm("N_ITS"))

# ---- measure breathing-mode period from z1 (full resolution now: dt = 1 iter) ----
z1 = z[:, 0]
z1_detrend = z1 - np.convolve(z1, np.ones(41) / 41, mode="same")
peaks, _ = find_peaks(z1_detrend, distance=100, prominence=0.4 * np.std(z1_detrend))
peak_frames = frame_indices[peaks]
period_iters = float(np.median(np.diff(peak_frames)))
period_s = period_iters * dt0
freq_hz = 1.0 / period_s

print(f"dt0={dt0:.2e}s  peaks found={len(peaks)}  period={period_iters:.1f} iters "
      f"= {period_s*1e6:.2f} us  freq={freq_hz/1e3:.2f} kHz")

# ---- per-field reconstruction error, computed on the full dataset ----
with open(f"{BASE}/err_summary_full.json") as f:
    err_pairs = json.load(f)
err_data = [{"name": name, "pct": pct} for name, pct in sorted(err_pairs, key=lambda p: p[1])]

# ---- subsample the 3D scatter for the browser (20,000 pts still fine for WebGL,
#      but the line trace between them is redrawn as a lighter every-Nth polyline
#      so it doesn't look like a solid smear) ----
data_json = json.dumps({
    "z1": np.round(z[:, 0], 4).tolist(),
    "z2": np.round(z[:, 1], 4).tolist(),
    "z3": np.round(z[:, 2], 4).tolist(),
    "frame": frame_indices.tolist(),
    "n_train": n_train,
}, separators=(",", ":"))
err_json = json.dumps(err_data, separators=(",", ":"))

with open(f"{BASE}/scripts/build_attractor_page.py", encoding="utf-8") as f:
    src = f.read()

template_start = src.index('TEMPLATE = r"""') + len('TEMPLATE = r"""')
template_end = src.index('"""', template_start)
TEMPLATE = src[template_start:template_end]

html = TEMPLATE
html = html.replace("__DATA_JSON__", data_json)
html = html.replace("__ERR_JSON__", err_json)
html = html.replace("__N_ITS__", f"{n_its:,}")
html = html.replace("__N__", f"{n:,}")
html = html.replace("__FREQ_KHZ__", f"{freq_hz/1e3:.1f}")
html = html.replace("__PERIOD_US__", f"{period_s*1e6:.2f}")
html = html.replace("__PERIOD_ITS__", f"{period_iters:.0f}")
html = html.replace("__SIM_MS__", f"{n_its*dt0*1e3:.2f}")

# page-specific text tweaks for the full run: mention warm start, update MSE stat,
# drop the "of 20,000" qualifier since it's now the complete run, and swap the
# grid-points/stride framing since every frame is now included.
html = html.replace(
    "__N__ <small>of __N_ITS__</small>".replace("__N__", f"{n:,}").replace("__N_ITS__", f"{n_its:,}"),
    f"{n:,} <small>all frames</small>",
)
html = html.replace(
    '<p class="stat-label">Val. MSE</p><p class="stat-value">0.0220 <small>normalized</small></p>',
    f'<p class="stat-label">Val. MSE</p><p class="stat-value">{final_val:.4f} <small>normalized</small></p>',
)
html = html.replace(
    "every 10th frame encoded",
    "every frame encoded (warm-started from the 2,000-frame subsample model)",
)
html = html.replace(
    "Plotted against each other across __N__ snapshots, those 3 numbers trace a closed loop: the discharge's breathing-mode limit cycle, recovered without ever telling the network a cycle exists.".replace("__N__", f"{n:,}"),
    f"Plotted against each other across all {n:,} snapshots &mdash; 10&times; the resolution of the first pass &mdash; those 3 numbers trace a closed loop: the discharge's breathing-mode limit cycle, recovered without ever telling the network a cycle exists.",
)

out_path = f"{BASE}/latent_attractor.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", out_path, len(html), "bytes")
