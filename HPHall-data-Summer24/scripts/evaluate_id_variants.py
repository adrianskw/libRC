"""Score every Id-aware AE variant with the first report's metrics; write id_study/summary.json.

For each variant (the original production AE as "baseline", plus everything defined in
run_id_variants.py):

  static     per-field decoded reconstruction error over all frames (the report's "AE floor"),
             final validation MSE
  coupling   how Id relates to the latent, held-out: R^2 of each coordinate alone, of the full
             latent (linear / MLP) and, as a leakage measure, of the two free coordinates alone
  observer   the windowed Id observer (id_observer.py): latent PC, decoded error, noise sweep
  rc         reservoir infer over several seeds (rc_with_id.py): inferPC, echo period mismatch,
             collapse-to-fixed-point count, plus one decoded run for per-field error

Everything heavy is delegated to the single-purpose scripts and cached on disk, so re-running
only fills in what is missing. Held-out block throughout: frames 17,001-20,000.

Usage: python evaluate_id_variants.py [--only V ...] [--seeds 5] [--skip-rc] [--skip-observer]
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = os.path.join(REPO_ROOT, "HPHall-data-Summer24")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)
from src import observables  # noqa: E402
from src.autoencoder import load_checkpoint, decode_to_fields  # noqa: E402
from run_id_variants import VARIANTS  # noqa: E402

PY = sys.executable
BASELINE_VAL_MSE = 0.02114        # production AE, printed by train_ae_full.py and quoted in the first report
OBSERVER_EXTRAS = [("ridge", 150, 0.0), ("mlp", 50, 0.0), ("mlp", 300, 0.0), ("mlp", 150, 0.05)]

ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="*")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--skip-rc", action="store_true")
ap.add_argument("--skip-observer", action="store_true")
args = ap.parse_args()

names = ["baseline"] + list(VARIANTS)
if args.only:
    names = [n for n in names if n in args.only]


def vdir(name):
    return f"{BASE}/latent3" + ("" if name == "baseline" else f"_{name}")


def vflag(name):
    return [] if name == "baseline" else ["--variant", name]


def is_control(name):
    return name == "baseline" or name.startswith("ctrl")


# ---------------- static + coupling (in process) ----------------
def ridge_r2(Xtr, ytr, Xva, yva, alpha=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A, B = (Xtr - mu) / sd, (Xva - mu) / sd
    A1, B1 = np.c_[A, np.ones(len(A))], np.c_[B, np.ones(len(B))]
    w = np.linalg.solve(A1.T @ A1 + alpha * np.eye(A1.shape[1]), A1.T @ ytr)
    return float(1 - np.sum((yva - B1 @ w) ** 2) / np.sum((yva - yva.mean()) ** 2))


def mlp_r2(Xtr, ytr, Xva, yva, seeds=3, epochs=200):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)[:, None]
    Xv = torch.tensor((Xva - mu) / sd, dtype=torch.float32, device=dev)
    preds = []
    for s in range(seeds):
        torch.manual_seed(s)
        net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 64),
                                  torch.nn.Tanh(), torch.nn.Linear(64, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(epochs):
            perm = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(Xt), 512):
                b = perm[i:i + 512]
                opt.zero_grad()
                torch.nn.functional.mse_loss(net(Xt[b]), yt[b]).backward()
                opt.step()
        with torch.no_grad():
            preds.append(net(Xv).squeeze(1).cpu().numpy())
    p = np.mean(preds, axis=0)
    return float(1 - np.sum((yva - p) ** 2) / np.sum((yva - yva.mean()) ** 2))


def static_and_coupling(name):
    d = vdir(name)
    lat = np.load(f"{d}/latent_full.npz")
    z, n_train = lat["z"].astype(float), int(lat["n_train"])
    M = len(z)
    model, ckpt = load_checkpoint(f"{d}/conv_ae_full.pt", device="cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    field_names = [str(s) for s in ckpt["field_names"]]
    ds = np.load(f"{BASE}/dataset_full.npz", allow_pickle=True, mmap_mode="r")
    err_sum, true_sum, val_err_sum, val_true_sum = 0.0, 0.0, 0.0, 0.0
    for i in range(0, M, 2000):
        true = np.asarray(ds["data"][i:i + 2000])
        rec = decode_to_fields(model, ckpt, z[i:i + 2000], device)
        e, t = np.abs(rec - true).sum(axis=(0, 2, 3)), np.abs(true).sum(axis=(0, 2, 3))
        err_sum, true_sum = err_sum + e, true_sum + t
        v0 = max(n_train - i, 0)
        if v0 < len(true):
            val_err_sum = val_err_sum + np.abs(rec[v0:] - true[v0:]).sum(axis=(0, 2, 3))
            val_true_sum = val_true_sum + np.abs(true[v0:]).sum(axis=(0, 2, 3))
    static = {"field_error_pct_all_frames": dict(zip(field_names, (100 * err_sum / true_sum).tolist())),
              "field_error_pct_heldout": dict(zip(field_names, (100 * val_err_sum / val_true_sum).tolist())),
              "val_mse": BASELINE_VAL_MSE if name == "baseline" else json.load(open(f"{d}/config.json"))["final_val_recon"]}
    static["mean_field_error_pct_all_frames"] = float(np.mean(list(static["field_error_pct_all_frames"].values())))

    id_n, _, _ = observables.zscore(observables.load_discharge_current(BASE), n_train)
    tr, va = slice(0, n_train), slice(n_train, M)
    coupling = {
        "single_coordinate_linear_r2": [ridge_r2(z[tr, k:k + 1], id_n[tr], z[va, k:k + 1], id_n[va]) for k in range(3)],
        "all_latents_linear_r2": ridge_r2(z[tr], id_n[tr], z[va], id_n[va]),
        "all_latents_mlp_r2": mlp_r2(z[tr], id_n[tr], z[va], id_n[va]),
        "free_coordinates_linear_r2": ridge_r2(z[tr, :2], id_n[tr], z[va, :2], id_n[va]),
        "free_coordinates_mlp_r2": mlp_r2(z[tr, :2], id_n[tr], z[va, :2], id_n[va]),
    }
    return static, coupling


# ---------------- subprocess jobs (cached) ----------------
def run(cmd, out_json, env=None):
    if os.path.exists(out_json):
        return
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)


def rc_jobs(name):
    """(cmd, metrics path) for every RC run this variant needs."""
    root = f"{vdir(name)}/n200"
    if is_control(name):
        cfgs = {"z1drive": ("z1,z2,z3", "z1"), "id4drive": ("z1,z2,z3,id", "id")}
    else:
        cfgs = {"z1drive": ("z1,z2,z3", "z1"), "z3drive": ("z1,z2,z3", "z3")}
    jobs = []
    for cname, (ch, dr) in cfgs.items():
        for s in range(args.seeds):
            tag = f"study_{cname}_s{s}"
            jobs.append(([PY, f"{HERE}/rc_with_id.py", "--channels", ch, "--drive", dr, "--seed", str(s), "--no-decode",
                          "--tag", tag, *vflag(name)], f"{root}/rc_{tag}/metrics.json"))
        tag = f"study_{cname}_decoded"
        jobs.append(([PY, f"{HERE}/rc_with_id.py", "--channels", ch, "--drive", dr, "--seed", "0", "--tag", tag,
                      *vflag(name)], f"{root}/rc_{tag}/metrics.json"))
    return cfgs, jobs


def summarize_rc(name, cfgs):
    root = f"{vdir(name)}/n200"
    out = {}
    for cname, (ch, dr) in cfgs.items():
        per_seed = [json.load(open(f"{root}/rc_study_{cname}_s{s}/metrics.json")) for s in range(args.seeds)]
        chans = [c for c in ch.split(",") if c not in dr.split(",") and c != "id"]
        pcs = np.array([[m["per_channel"][c]["infer_pc"] for c in chans] for m in per_seed])
        mism = [m["echo_period"]["mismatch_pct"] for m in per_seed]
        ok = [x for x in mism if x is not None]
        out[cname] = {
            "channels": ch, "drive": dr, "scored_channels": chans, "seeds": args.seeds,
            "pc_per_seed": pcs.tolist(), "pc_mean": pcs.mean(0).tolist(), "pc_std": pcs.std(0).tolist(),
            "pc_min": float(pcs.min()), "pc_max": float(pcs.max()),
            "echo_period_mismatch_pct_per_seed": mism,
            "echo_period_mismatch_pct_mean_of_oscillating": float(np.mean(ok)) if ok else None,
            "echo_collapsed_seeds": int(sum(m is None for m in mism)),
            "decoded": json.load(open(f"{root}/rc_study_{cname}_decoded/metrics.json")),
        }
    return out


def observer_cmd(name, kind, span, tn):
    cmd = [PY, f"{HERE}/id_observer.py", "--kind", kind, "--span", str(span), "--train-noise", str(tn), *vflag(name)]
    tag = f"{kind}{span}" + (f"_tn{tn:g}" if tn else "")
    return cmd, f"{vdir(name)}/observer_{tag}/metrics.json"


# ---------------- orchestrate ----------------
env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
pool = ThreadPoolExecutor(max_workers=args.workers)
futures, rc_cfgs = [], {}
if not args.skip_rc:
    for name in names:
        rc_cfgs[name], jobs = rc_jobs(name)
        futures += [pool.submit(run, cmd, path, env) for cmd, path in jobs]
    print(f"queued {len(futures)} RC runs on {args.workers} workers", flush=True)

summary_path = f"{BASE}/id_study/summary.json"
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

for name in names:
    print(f"--- {name}: static + coupling", flush=True)
    st, co = static_and_coupling(name)
    entry = summary.get(name, {})
    entry.update({"id_mode": ("none" if is_control(name) else ("conditional" if name.startswith("cond") else "supervised")),
                  "static": st, "coupling": co})
    if not args.skip_observer:
        cmd, path = observer_cmd(name, "mlp", 150, 0.0)
        print(f"--- {name}: observer", flush=True)
        run(cmd, path)
        entry["observer"] = json.load(open(path))
        if name == "baseline":
            entry["observer_extras"] = {}
            for kind, span, tn in OBSERVER_EXTRAS:
                cmd, path = observer_cmd(name, kind, span, tn)
                run(cmd, path)
                entry["observer_extras"][os.path.basename(os.path.dirname(path))] = json.load(open(path))
    summary[name] = entry
    json.dump(summary, open(summary_path, "w"), indent=2)

for f in futures:
    f.result()
if not args.skip_rc:
    for name in names:
        summary[name]["rc"] = summarize_rc(name, rc_cfgs[name])
    json.dump(summary, open(summary_path, "w"), indent=2)
print(f"wrote {summary_path}")
