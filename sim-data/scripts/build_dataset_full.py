"""Extract ALL 20,000 Tecplot frames from parsed_files.tgz into a single array.

Same format as build_dataset.py but STRIDE=1 (every frame). Output: dataset_full.npz.
"""
import os
import re
import tarfile
import time

import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARCHIVE = os.path.join(BASE, "parsed_files.tgz")
OUT = os.path.join(BASE, "dataset_full.npz")
STRIDE = 1
NI, NJ = 50, 25

pat = re.compile(r"movie_tecplot_(\d{5})\.dat$")

t0 = time.time()
with tarfile.open(ARCHIVE, "r:gz") as tf:
    members = tf.getmembers()
    wanted = []
    for m in members:
        match = pat.search(m.name)
        if match:
            idx = int(match.group(1))
            if (idx - 1) % STRIDE == 0:
                wanted.append((idx, m))
    wanted.sort(key=lambda t: t[0])
    print(f"Found {len(members)} members, selecting {len(wanted)} frames (stride={STRIDE})", flush=True)

    frames = []
    frame_indices = []
    var_names = None
    x = y = None

    for i, (idx, member) in enumerate(wanted):
        f = tf.extractfile(member)
        raw = f.read().decode("ascii", errors="strict")
        lines = raw.split("\n")
        header = lines[0]
        data = np.loadtxt(lines[2:2 + NI * NJ])

        if var_names is None:
            var_names = [v.strip().strip('"') for v in header.split("=")[1].split(",")]

        arr = data.reshape(NJ, NI, len(var_names))
        if x is None:
            x = arr[:, :, 0].astype(np.float32)
            y = arr[:, :, 1].astype(np.float32)

        fields = np.moveaxis(arr[:, :, 2:], -1, 0).astype(np.float32)
        frames.append(fields)
        frame_indices.append(idx)

        if (i + 1) % 2000 == 0:
            print(f"  parsed {i + 1}/{len(wanted)}  ({time.time() - t0:.1f}s elapsed)", flush=True)

data = np.stack(frames, axis=0)
field_names = var_names[2:]
frame_indices = np.array(frame_indices, dtype=np.int64)

print("data shape:", data.shape)
print("field names:", field_names)
print(f"total time: {time.time() - t0:.1f}s")

np.savez_compressed(
    OUT,
    data=data,
    field_names=np.array(field_names),
    frame_indices=frame_indices,
    x=x,
    y=y,
)
print("Saved:", OUT)
