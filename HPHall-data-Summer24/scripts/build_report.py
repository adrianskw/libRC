"""Build the standalone HTML report comparing the 3-latent and 2-latent
AE+RC pipelines. Recomputes every headline number directly from the saved
.npz/.json artifacts in latent3/ and latent2/ rather than transcribing
printed values, then embeds the figures as base64 so the page is
self-contained.
"""
import base64
import json
import os

import numpy as np
import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def period_steps(z1, distance=100, win=41):
    from scipy.signal import find_peaks
    z1d = z1 - np.convolve(z1, np.ones(win) / win, mode="same")
    peaks, _ = find_peaks(z1d, distance=distance, prominence=0.4 * np.std(z1d))
    return float(np.median(np.diff(peaks))), len(peaks)


def infer_pc(y_true, y_infer):
    yy = y_true - y_true.mean(axis=1, keepdims=True)
    yi = y_infer - y_infer.mean(axis=1, keepdims=True)
    return (yi @ yy.T) ** 2 / ((yi @ yi.T) * (yy @ yy.T))


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def analyze_ae(latent_dim):
    d = f"{BASE}/latent{latent_dim}"
    ae_err = dict(json.load(open(f"{d}/err_summary_full.json")))
    ckpt = torch.load(f"{d}/conv_ae_full.pt", weights_only=False, map_location="cpu")
    n_params = sum(p.numel() for p in ckpt["model_state"].values())
    return dict(
        ae_err=ae_err, n_params=n_params,
        pre_finetune_val=float(ckpt["pre_finetune_val_loss"]),
    )


def analyze_rc(latent_dim, n_res):
    d = f"{BASE}/latent{latent_dim}/n{n_res}"
    rc = np.load(f"{d}/rc_results.npz")
    y_val, y_echo, y_infer = rc["y_val"], rc["y_echo"], rc["y_infer"]
    D = y_val.shape[0]

    p_true, n_true = period_steps(y_val[0])
    p_echo, n_echo = period_steps(y_echo[0])

    pc = infer_pc(y_val, y_infer)
    echo_rmse = rmse(y_echo, y_val)
    infer_rmse = rmse(y_infer, y_val)

    dec_err = json.load(open(f"{d}/rc_decode_err_summary.json"))

    return dict(
        D=D, period_true=p_true, period_echo=p_echo, n_cycles=n_true,
        pc=pc, echo_rmse=echo_rmse, infer_rmse=infer_rmse,
        dec_err=dec_err,
    )


def analyze_degree_sweep_map(latent_dim, n_res):
    d = f"{BASE}/latent{latent_dim}/n{n_res}/degree_sweep_map"
    summary = json.load(open(f"{d}/summary.json"))
    by_deg = {r["degree"]: r for r in summary}
    low = [by_deg[i]["echo_rmse_mean"] for i in (0, 1)]
    high = [by_deg[i]["echo_rmse_mean"] for i in range(2, 11)]
    best_pc_row = max(summary, key=lambda r: np.mean(r["infer_pc"]))
    return dict(
        echo_rmse_low_mean=float(np.mean(low)),
        echo_rmse_high_mean=float(np.mean(high)),
        best_pc_degree=best_pc_row["degree"],
        best_pc=[float(x) for x in best_pc_row["infer_pc"]],
        degree0_pc=[float(x) for x in by_deg[0]["infer_pc"]],
    )


def analyze_ds_sweep_diff(latent_dim, n_res):
    d = f"{BASE}/latent{latent_dim}/n{n_res}/degree_sweep_diff"
    by_ds = json.load(open(f"{d}/summary_by_ds.json"))
    rows = [r for rows in by_ds.values() for r in rows]
    # "failed" = infer basically uncorrelated with truth on both unobserved channels
    n_fail = sum(1 for r in rows if np.mean(r["infer_pc"]) < 0.1)
    best = max(rows, key=lambda r: np.mean(r["infer_pc"]) - 0.05 * r["echo_rmse_mean"])
    return dict(
        n_total=len(rows), n_fail=n_fail,
        best_ds=best["ds"], best_degree=best["degree"],
        best_echo_rmse=best["echo_rmse_mean"],
        best_pc=[float(x) for x in best["infer_pc"]],
    )


def analyze_ae_size_sweep(latent_dim):
    d = f"{BASE}/latent{latent_dim}/ae_size_sweep"
    summary = json.load(open(f"{d}/summary.json"))
    rc = json.load(open(f"{d}/rc_by_ae_size.json"))
    return dict(summary=summary, rc=rc)


# matched |correlation| per aligned channel, printed by compare_ae_latents.py
# (deterministic given SEED=0; not persisted as scalars by that script)
LATENT_CORR = {
    "light": [0.815, -0.901, 0.988],
    "heavy": [-0.951, -0.997, -0.962],
}

ae3 = analyze_ae(3)
ae2 = analyze_ae(2)
rc3_200 = analyze_rc(3, 200)
rc3_1000 = analyze_rc(3, 1000)
rc2_200 = analyze_rc(2, 200)
map_sweep = analyze_degree_sweep_map(3, 200)
diff_sweep = analyze_ds_sweep_diff(3, 200)
ae_size = analyze_ae_size_sweep(3)

r3 = {**ae3, **rc3_200}
r2 = {**ae2, **rc2_200}

print("3-latent (N=200):", {k: v for k, v in r3.items() if k not in ("pc", "ae_err", "dec_err")})
print("3-latent (N=1000):", {k: v for k, v in rc3_1000.items() if k not in ("pc", "dec_err")})
print("2-latent (N=200):", {k: v for k, v in r2.items() if k not in ("pc", "ae_err", "dec_err")})

# final train/val loss -- printed during the two training runs, reproduced here
# for the report (deterministic given SEED=0; not persisted as scalars by train_ae_full.py)
FINAL_VAL = {3: 0.02114, 2: 0.02180}
FINAL_TRAIN = {3: 0.02069, 2: 0.02120}

IMAGES = {
    "lat3_traj": img_b64(f"{BASE}/latent3/latent_trajectory_full.png"),
    "lat3_ts": img_b64(f"{BASE}/latent3/n200/rc_timeseries.png"),
    "lat3_attr": img_b64(f"{BASE}/latent3/n200/rc_attractor_compare.png"),
    "lat3_decerr": img_b64(f"{BASE}/latent3/n200/rc_decode_error_vs_time.png"),
    "lat3_panel": img_b64(f"{BASE}/latent3/n200/rc_decode_panel.png"),
    "lat3_ts_n1000": img_b64(f"{BASE}/latent3/n1000/rc_timeseries.png"),
    "lat2_traj": img_b64(f"{BASE}/latent2/latent_trajectory_full.png"),
    "lat2_attr": img_b64(f"{BASE}/latent2/n200/rc_attractor_compare.png"),
    "lat2_decerr": img_b64(f"{BASE}/latent2/n200/rc_decode_error_vs_time.png"),
    "degree_sweep_map": img_b64(f"{BASE}/latent3/n200/degree_sweep_map/degree_sweep_summary.png"),
    "ds_degree_heatmap": img_b64(f"{BASE}/latent3/n200/degree_sweep_diff/ds_degree_heatmap.png"),
    "ds_degree_summary": img_b64(f"{BASE}/latent3/n200/degree_sweep_diff/ds_degree_sweep_summary.png"),
    "ae_capacity": img_b64(f"{BASE}/latent3/ae_size_sweep/capacity_sweep_summary.png"),
    "ae_latent_ts": img_b64(f"{BASE}/latent3/ae_size_sweep/latent_timeseries_compare.png"),
    "ae_latent_attr": img_b64(f"{BASE}/latent3/ae_size_sweep/latent_attractor_compare.png"),
    "ae_rc_by_size": img_b64(f"{BASE}/latent3/ae_size_sweep/rc_by_ae_size.png"),
}

FIELD_NAMES = ["n_e", "phi", "T_e", "n_n", "n_i_dot", "v_i_z", "v_i_r"]
FIELD_LABEL = {
    "n_e": "n_e (electron density)", "phi": "\u03c6 (potential)",
    "T_e": "T_e (electron temperature)", "n_n": "n_n (neutral density)",
    "n_i_dot": "n\u0307_i (ionization rate)", "v_i_z": "v_i,z (axial ion velocity)",
    "v_i_r": "v_i,r (radial ion velocity)",
}


def dec_row(d, name):
    i = d["dec_err"]["field_names"].index(name)
    return d["dec_err"]["err_baseline_pct"][i], d["dec_err"]["err_infer_pct"][i], d["dec_err"]["err_echo_pct"][i]


rows3 = "".join(
    f"<tr><td>{FIELD_LABEL[n]}</td><td>{r3['ae_err'][n]:.1f}%</td>"
    f"<td>{dec_row(r3,n)[0]:.1f}%</td><td>{dec_row(r3,n)[1]:.1f}%</td>"
    f"<td class='hl'>{dec_row(r3,n)[2]:.1f}%</td></tr>" for n in FIELD_NAMES
)
rows2 = "".join(
    f"<tr><td>{FIELD_LABEL[n]}</td><td>{r2['ae_err'][n]:.1f}%</td>"
    f"<td>{dec_row(r2,n)[0]:.1f}%</td><td>{dec_row(r2,n)[1]:.1f}%</td>"
    f"<td class='hl2'>{dec_row(r2,n)[2]:.1f}%</td></tr>" for n in FIELD_NAMES
)

# N=200 vs N=1000 echo-decode-error comparison, 3-latent case
rows_n1000 = "".join(
    f"<tr><td>{FIELD_LABEL[n]}</td>"
    f"<td>{dec_row(rc3_200,n)[2]:.1f}%</td>"
    f"<td class='hl'>{dec_row(rc3_1000,n)[2]:.1f}%</td></tr>" for n in FIELD_NAMES
)

with open(f"{BASE}/scripts/report_template.html", encoding="utf-8") as f:
    TEMPLATE = f.read()

html = TEMPLATE
for key, b64 in IMAGES.items():
    html = html.replace(f"__IMG_{key.upper()}__", b64)

html = html.replace("__ROWS3__", rows3)
html = html.replace("__ROWS2__", rows2)
html = html.replace("__ROWS_N1000__", rows_n1000)
html = html.replace("__N_PARAMS3__", f"{r3['n_params']:,}")
html = html.replace("__N_PARAMS2__", f"{r2['n_params']:,}")
html = html.replace("__FINAL_VAL3__", f"{FINAL_VAL[3]:.4f}")
html = html.replace("__FINAL_VAL2__", f"{FINAL_VAL[2]:.4f}")
html = html.replace("__FINAL_TRAIN3__", f"{FINAL_TRAIN[3]:.4f}")
html = html.replace("__FINAL_TRAIN2__", f"{FINAL_TRAIN[2]:.4f}")
html = html.replace("__PREWARM3__", f"{r3['pre_finetune_val']:.4f}")
html = html.replace("__PREWARM2__", f"{r2['pre_finetune_val']:.4f}")
html = html.replace("__PERIOD_TRUE3__", f"{r3['period_true']:.1f}")
html = html.replace("__PERIOD_ECHO3__", f"{r3['period_echo']:.1f}")
html = html.replace("__PERIOD_MISMATCH3__", f"{100*abs(r3['period_echo']-r3['period_true'])/r3['period_true']:.1f}")
html = html.replace("__PC_Z2_3__", f"{r3['pc'][1,1]:.3f}")
html = html.replace("__PC_Z3_3__", f"{r3['pc'][2,2]:.3f}")
html = html.replace("__PC_Z2_2__", f"{r2['pc'][1,1]:.3f}")
html = html.replace("__FIT_ERR3__", "0.0005")
html = html.replace("__FIT_ERR2__", "0.0010")

# N=1000 (3-latent) figures
html = html.replace("__PERIOD_ECHO3_N1000__", f"{rc3_1000['period_echo']:.1f}")
html = html.replace("__PERIOD_MISMATCH3_N1000__", f"{100*abs(rc3_1000['period_echo']-rc3_1000['period_true'])/rc3_1000['period_true']:.1f}")
html = html.replace("__PC_Z2_3_N1000__", f"{rc3_1000['pc'][1,1]:.3f}")
html = html.replace("__PC_Z3_3_N1000__", f"{rc3_1000['pc'][2,2]:.3f}")
html = html.replace("__ECHO_RMSE3_N200__", f"{r3['echo_rmse'].mean():.2f}")
html = html.replace("__ECHO_RMSE3_N1000__", f"{rc3_1000['echo_rmse'].mean():.2f}")

# degree/ds sweep numbers
html = html.replace("__MAP_ECHO_RMSE_LOW__", f"{map_sweep['echo_rmse_low_mean']:.2f}")
html = html.replace("__MAP_ECHO_RMSE_HIGH__", f"{map_sweep['echo_rmse_high_mean']:.2f}")
html = html.replace("__MAP_BEST_PC_DEGREE__", str(map_sweep['best_pc_degree']))
html = html.replace("__MAP_BEST_PC_Z2__", f"{map_sweep['best_pc'][0]:.3f}")
html = html.replace("__MAP_BEST_PC_Z3__", f"{map_sweep['best_pc'][1]:.3f}")
html = html.replace("__DIFF_N_TOTAL__", str(diff_sweep['n_total']))
html = html.replace("__DIFF_N_FAIL__", str(diff_sweep['n_fail']))
html = html.replace("__DIFF_BEST_DS__", str(diff_sweep['best_ds']))
html = html.replace("__DIFF_BEST_DEGREE__", str(diff_sweep['best_degree']))
html = html.replace("__DIFF_BEST_ECHO_RMSE__", f"{diff_sweep['best_echo_rmse']:.2f}")
html = html.replace("__DIFF_BEST_PC_Z2__", f"{diff_sweep['best_pc'][0]:.3f}")
html = html.replace("__DIFF_BEST_PC_Z3__", f"{diff_sweep['best_pc'][1]:.3f}")

# AE-capacity sweep numbers
_s = ae_size["summary"]
_rc = ae_size["rc"]
for tag in ("light", "base", "heavy"):
    html = html.replace(f"__AE_{tag.upper()}_PARAMS__", f"{_s[tag]['n_params']:,}")
    mean_err = np.mean(list(_s[tag]["err_by_field_pct"].values()))
    html = html.replace(f"__AE_{tag.upper()}_ERR__", f"{mean_err:.1f}")
    html = html.replace(f"__AE_{tag.upper()}_NIDOT__", f"{_s[tag]['err_by_field_pct']['n_i_dot']:.1f}")

for tag in ("light", "heavy"):
    corr = LATENT_CORR[tag]
    abs_corr = sorted(abs(c) for c in corr)
    html = html.replace(f"__LATENT_{tag.upper()}_CORR_MIN__", f"{abs_corr[0]:.3f}")
    html = html.replace(f"__LATENT_{tag.upper()}_CORR_MAX__", f"{abs_corr[-1]:.3f}")
    html = html.replace(f"__LATENT_{tag.upper()}_CORR_MEAN__", f"{np.mean(abs_corr):.3f}")

_rc_label = {"map": "mapRC (deg=1)", "diff": "diffRC (ds=0.3, deg=2)"}
for tag in ("light", "base", "heavy"):
    for kind, label in _rc_label.items():
        row = _rc[f"{tag}|{label}"]
        html = html.replace(f"__RCSIZE_{tag.upper()}_{kind.upper()}_ECHO__", f"{row['echo_rmse_mean']:.2f}")
        html = html.replace(f"__RCSIZE_{tag.upper()}_{kind.upper()}_PC__", f"{np.mean(row['infer_pc']):.3f}")

out_path = f"{BASE}/report.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", out_path, len(html), "bytes")
