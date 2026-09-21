"""Train the full matrix of Id-aware autoencoder variants (skips any already trained).

One place that defines which variants exist, so the experiment is reproducible with a
single command. Each runs train_ae_id.py, which writes latent3_{name}/.

  ctrl_s{0,1,2}   id_mode=none, three seeds: the training-variance floor. The report notes
                  the RC is sensitive to which local minimum an AE lands in, so a single
                  control would not say how much of any difference is just seed noise.
  sup_w{L}_d{M}   supervised, lambda=L on MSE(z3, Id_n), decorrelation weight mu=M
  cond_d{M}       conditional (decoder handed the measured Id), decorrelation weight mu=M

Usage: python run_id_variants.py [--only name ...] [--force]
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))

VARIANTS = {
    "ctrl_s0": ["--id-mode", "none", "--seed", "0"],
    "ctrl_s1": ["--id-mode", "none", "--seed", "1"],
    "ctrl_s2": ["--id-mode", "none", "--seed", "2"],
}
for lam in (0.1, 1, 10):
    for mu in (0, 0.1):
        VARIANTS[f"sup_w{lam:g}_d{mu:g}"] = ["--id-mode", "supervised", "--id-weight", str(lam), "--decorr-weight", str(mu)]
for mu in (0, 0.1):
    VARIANTS[f"cond_d{mu:g}"] = ["--id-mode", "conditional", "--decorr-weight", str(mu)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for name, flags in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        if not args.force and os.path.exists(f"{BASE}/latent3_{name}/config.json"):
            print(f"skip {name} (already trained)")
            continue
        print(f"=== training {name}: {' '.join(flags)}", flush=True)
        subprocess.run([sys.executable, f"{HERE}/train_ae_id.py", "--variant", name, *flags], check=True)
