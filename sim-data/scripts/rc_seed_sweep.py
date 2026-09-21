"""Seed-variance check for rc_with_id.py configurations.

Runs rc_with_id.py once per reservoir seed (skipping seeds whose metrics.json
already exists, unless --force), then tabulates inferPC across seeds. Exists
because a single reservoir draw can badly misrepresent a configuration:
z1-driven infer is stable to ~0.002 across seeds, Id-driven infer is not.

Usage: python rc_seed_sweep.py [--seeds 8] [--latent-dim 3] [--n-res 200] [--force]
Output: latent{D}/n{N}/rc_seed_sweep_summary.json
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))

CONFIGS = {   # name -> (channels, drive)
    "id-drive": ("z1,z2,z3,id", "id"),
    "baseline": ("z1,z2,z3", "z1"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--latent-dim", type=int, default=3)
ap.add_argument("--n-res", type=int, default=200)
ap.add_argument("--force", action="store_true")
args = ap.parse_args()
RC_ROOT = f"{BASE}/latent{args.latent_dim}/n{args.n_res}"

summary = {}
for name, (channels, drive) in CONFIGS.items():
    rows = []
    for s in range(args.seeds):
        tag = f"sweep_{name}_s{s}"
        metrics_path = f"{RC_ROOT}/rc_{tag}/metrics.json"
        if args.force or not os.path.exists(metrics_path):
            subprocess.run([sys.executable, f"{HERE}/rc_with_id.py", "--latent-dim", str(args.latent_dim),
                            "--n-res", str(args.n_res), "--channels", channels, "--drive", drive,
                            "--seed", str(s), "--no-decode", "--tag", tag],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rows.append(json.load(open(metrics_path))["per_channel"])
    undriven = [c for c in channels.split(",") if c not in drive.split(",") and c.startswith("z")]
    if name == "id-drive":
        undriven = ["z1", "z2", "z3"]
    pc = np.array([[r[c]["infer_pc"] for c in undriven] for r in rows])
    summary[name] = {"channels": undriven, "pc_per_seed": pc.tolist(), "mean": pc.mean(0).tolist(),
                     "std": pc.std(0).tolist(), "min": float(pc.min()), "max": float(pc.max())}
    print(f"{name:9s} inferPC {undriven}: mean {np.round(pc.mean(0), 3)}  std {np.round(pc.std(0), 3)}  "
          f"range [{pc.min():.3f}, {pc.max():.3f}]  over {args.seeds} seeds")

with open(f"{RC_ROOT}/rc_seed_sweep_summary.json", "w") as f:
    json.dump({"seeds": args.seeds, "summary": summary}, f, indent=2)
print(f"wrote {RC_ROOT}/rc_seed_sweep_summary.json")
