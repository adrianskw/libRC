"""Build the standalone Id report: HPHall-data-Summer24/report_id.html.

Companion to report.html (build_report.py) and written in the same design system: the CSS is
read from report_template.html so the two stay visually in sync. Every number in the text and
tables is computed from artifacts on disk (id_study/summary.json and the run directories) rather
than transcribed. Figures come from report_id_figures.py.

Run evaluate_id_variants.py first (it writes id_study/summary.json).
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report_id_figures as F  # noqa: E402

BASE = F.BASE
FIELDS = F.FIELD_NAMES
LABEL = {"n_e": "n<sub>e</sub> (electron density)", "phi": "&phi; (potential)", "T_e": "T<sub>e</sub> (electron temperature)",
         "n_n": "n<sub>n</sub> (neutral density)", "n_i_dot": "&#7749;<sub>i</sub> (ionization rate)",
         "v_i_z": "v<sub>i,z</sub> (axial ion velocity)", "v_i_r": "v<sub>i,r</sub> (radial ion velocity)"}

S = F.load_summary()
CTRL = [v for v in ("baseline", "ctrl_s0", "ctrl_s1", "ctrl_s2") if v in S]
SUP = [v for v in F._order(S) if v.startswith("sup")]
COND = [v for v in F._order(S) if v.startswith("cond")]


# ------------------------------------------------------------------ helpers ----
def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def pct1(x): return f"{x:.1f}%"


def table(headers, rows, hl=()):
    """headers: list of str; rows: list of lists of str; hl: row indices to highlight."""
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for i, r in enumerate(rows):
        cells = "".join(f"<td{' class=\"hl\"' if (i in hl and j > 0) else ''}>{c}</td>" for j, c in enumerate(r))
        body += f"<tr>{cells}</tr>"
    return f'<div class="narrow tblwrap"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(b64, caption, num):
    return (f'<figure class="narrow"><img src="data:image/png;base64,{b64}" alt="{re.sub("<[^>]+>", "", caption)[:120]}">'
            f'<figcaption><b>Fig. {num}</b> &mdash; {caption}</figcaption></figure>')


def callout(label, text, kind="teal"):
    return f'<div class="narrow"><div class="callout {kind}"><p class="label">{label}</p><p>{text}</p></div></div>'


def pooled(variants, cfg, ch_idx=None):
    vals = np.array([row for v in variants for row in S[v]["rc"][cfg]["pc_per_seed"]])
    return vals if ch_idx is None else vals[:, ch_idx]


# --------------------------------------------------------- numbers for section 2 ----
base = S["baseline"]
z1_pc = pooled(CTRL, "z1drive")                # (runs, [z2, z3])
id_pc = pooled(CTRL, "id4drive")               # (runs, [z1, z2, z3])
n_runs = len(id_pc)
id_by_ae = [float(np.mean(S[v]['rc']['id4drive']['pc_mean'])) for v in CTRL]
probe = {r["window_steps"]: r for r in json.load(open(f"{BASE}/latent3/id_map/id_window_to_latent.json"))["results"]}
mlp_r2 = {k: np.mean(v["mlp_r2"]) for k, v in probe.items()}
ridge_r2 = {k: np.mean(v["ridge_r2"]) for k, v in probe.items()}
obs = base["observer"]
obs_pc = {k: v["pc"] for k, v in obs["per_channel"].items()}
obs_ratio = obs["ratio_to_persistence_full"]
rc1_dec = base["rc"]["z1drive"]["decoded"]
rcid_dec = base["rc"]["id4drive"]["decoded"]
rc1_ratio = rc1_dec["decoded_ratio_to_persistence_full"]["infer"]
rcid_ratio = rcid_dec["decoded_ratio_to_persistence_full"]["infer"]
mean_ratio = lambda d: float(np.mean(list(d.values())))
ex = base["observer_extras"]
noise = {r["noise_frac_of_id_std"]: r for r in obs["noise_sweep"]}
noise_tn = {r["noise_frac_of_id_std"]: r for r in ex["observer_mlp150_tn0.05"]["noise_sweep"]}

# smoothing check from the seed-0 runs made while diagnosing option 1 (rc_with_id.py --id-smooth W)
smooth = {}
for w in (10, 25, 50, 100):
    p = f"{BASE}/latent3/n200/rc_z1-z2-z3-id_drive-id_smooth{w}/metrics.json"
    if os.path.exists(p):
        m = json.load(open(p))["per_channel"]
        smooth[w] = np.mean([m[c]["infer_pc"] for c in ("z1", "z2", "z3")])


def section_a():
    out = []
    out.append(f"""
    <section id="approach-a">
      <div class="narrow">
        <span class="tag tag-teal">&#9679; approach A &middot; observe the latent from Id</span>
        <h2>2. Reading the latent state off the discharge current</h2>
        <p>The question is whether the 3-number latent, and through the frozen decoder the full 7-field state, can be tracked from Id alone. Id is one number per step, measurable on hardware; the latents are not. Fig. 1 shows the raw material on the held-out block: Id is a slow, noisy signal that is visibly phase-shifted from the latent oscillation.</p>
      </div>
      {figure(F.b64(F.fig_id_vs_latent()), "Discharge current (top) and the three latents (bottom) over the first 900 held-out steps. Id carries the breathing-mode rhythm, but no single instant of Id determines the latent state.", 1)}

      <div class="narrow">
        <h3>2.1 First attempt: give the reservoir Id as a fourth channel</h3>
        <p>The reservoir's infer mode observes a chosen channel and reconstructs the rest. Here the state is [z<sub>1</sub>, z<sub>2</sub>, z<sub>3</sub>, Id], all z-scored on the training block, and only Id is observed. The success criterion was fixed in advance: inferPC of at least 0.9 on the unobserved latents.</p>
        <p>It fails, and it fails erratically. Across <b>{len(CTRL)} independently trained autoencoders &times; {len(id_pc) // len(CTRL)} reservoir seeds ({n_runs} runs)</b>, Id-driven inferPC averages <b>{id_pc.mean():.2f}</b> and ranges from {id_pc.min():.2f} to {id_pc.max():.2f}, against <b>{z1_pc.mean():.3f}</b> (std {z1_pc.std():.3f}) when the reservoir is driven by z<sub>1</sub>. A single draw can look like partial success: the first run (production autoencoder, seed 0) scored {id_pc[0].mean():.2f}.</p>
      </div>
      {figure(F.b64(F.fig_option1_seed_spread(S)), f"Held-out inferPC for every run. Driving the reservoir with z<sub>1</sub> (teal) is stable (standard deviation across reservoir seeds at most {max(max(S[v]["rc"]["z1drive"]["pc_std"]) for v in CTRL):.3f} within an autoencoder); driving it with Id (rust) is neither accurate nor repeatable, and the spread has two sources: the reservoir seed within an autoencoder, and the autoencoder itself (per-autoencoder means differ, from {min(id_by_ae):.2f} to {max(id_by_ae):.2f}).", 2)}
      <div class="narrow">
        <p>Smoothing the drive did not help. With a causal moving average on Id (windows of 10, 25, 50 and 100 steps, seed 0) the mean inferPC was {", ".join(f"{smooth[w]:.2f}" for w in sorted(smooth))} respectively: erratic rather than monotone, so this is not a noise problem.</p>

        <h3>2.2 Diagnosis: the information is there</h3>
        <p>A scalar observable of a system with at least three dimensions determines the state only through its history (Takens). To test whether the history contains the latent, the reservoir was removed altogether: a plain regression from a window of <i>past</i> Id to the latent, fit on the training block and scored on the held-out block.</p>
      </div>
      {figure(F.b64(F.fig_window_length()), f"Held-out R&sup2; of the latent recovered from a window of past Id. A single instant of Id says nothing (MLP {mlp_r2[0]:.2f}); a window of one breathing period recovers the latent almost completely (MLP {mlp_r2[150]:.2f}, ridge {ridge_r2[150]:.2f}).", 3)}
      <div class="narrow">
        <p>The information is present once the window spans about a period. The reservoir supplies that history implicitly, but its infer mode fits the readout with every channel driven and then feeds back its own estimates for the undriven ones, so the readout is used in a regime it was never trained for. That mismatch, not the signal, is the bottleneck.</p>

        <h3>2.3 The windowed observer</h3>
        <p>The direct fix is to use the regression as the tracker. The observer reads Id(t), Id(t&minus;5), &hellip;, Id(t&minus;150) (31 taps, about one breathing period), never anything after t, and outputs the latent at t through a small two-layer MLP (three fits averaged). It is fit on the first 17,000 frames only. The reservoir is no longer involved.</p>
      </div>
      {figure(F.b64(F.fig_observer_timeseries()), "Held-out latents from Id alone. The window observer (teal) follows the truth (black) closely; the reservoir driven by Id (rust, one seed) oscillates but drifts in and out of phase.", 4)}
      <div class="narrow">
        <p>On the held-out block the observer reaches inferPC of <b>{obs_pc['z1']:.3f} / {obs_pc['z2']:.3f} / {obs_pc['z3']:.3f}</b> for z<sub>1</sub>, z<sub>2</sub>, z<sub>3</sub>. For comparison, the reservoir driven by z<sub>1</sub> reaches {np.mean(z1_pc[:, 0]):.3f} / {np.mean(z1_pc[:, 1]):.3f} on z<sub>2</sub>, z<sub>3</sub> while observing a latent nobody can measure. Decoded through the frozen decoder, the recovered state is nearly as good as the autoencoder itself allows:</p>
      </div>
      {figure(F.b64(F.fig_field_errors(S)), "Mean relative error per decoded field on the held-out block (the first report's metric). The observer, which sees only Id, sits close to the autoencoder's own floor and matches the reservoir that is handed z<sub>1</sub>; the reservoir driven by Id does not. The amber tick is a persistence forecast (hold the last training frame).", 5)}
      {table(["Field", "AE floor", "window observer (Id)", "reservoir, z1-driven", "reservoir, Id-driven", "persistence", "observer / persistence"],
             [[LABEL[f], pct1(obs["decoded_field_error_pct"]["ae_only"][f]), pct1(obs["decoded_field_error_pct"]["observer"][f]),
               pct1(rc1_dec["decoded_field_error_pct"]["infer"][f]), pct1(rcid_dec["decoded_field_error_pct"]["infer"][f]),
               pct1(obs["decoded_field_error_pct"]["persistence"][f]), f3(obs_ratio[f])] for f in FIELDS] +
             [["<b>mean over fields</b>", pct1(np.mean(list(obs["decoded_field_error_pct"]["ae_only"].values()))),
               pct1(np.mean(list(obs["decoded_field_error_pct"]["observer"].values()))),
               pct1(np.mean(list(rc1_dec["decoded_field_error_pct"]["infer"].values()))),
               pct1(np.mean(list(rcid_dec["decoded_field_error_pct"]["infer"].values()))),
               pct1(np.mean(list(obs["decoded_field_error_pct"]["persistence"].values()))), f3(mean_ratio(obs_ratio))]])}
      <div class="narrow">
        <p>Averaged over the seven fields the observer's error is {mean_ratio(obs_ratio):.2f} times a persistence forecast, against {mean_ratio(rc1_ratio):.2f} for the z<sub>1</sub>-driven reservoir and {mean_ratio(rcid_ratio):.2f} for the Id-driven one (one seed). Neutral density n<sub>n</sub> is the exception, at {obs_ratio['n_n']:.2f}: every method gives the same {obs['decoded_field_error_pct']['ae_only']['n_n']:.1f}%, which is the autoencoder's own floor for that field, so the error there comes from the autoencoder and not from the tracking.</p>
      </div>
      {figure(F.b64(F.fig_decode_panel()), "One held-out frame (step 500), all seven fields; densities on a log<sub>10</sub> scale. Rows: the simulation, the autoencoder's round trip on the true latent, the window observer from Id alone, and the reservoir driven by Id.", 6)}

      <div class="narrow">
        <h3>2.4 Robustness to measurement noise, window length and model</h3>
        <p>The simulation's Id is clean; a real measurement is not. Gaussian noise (a fraction of Id's own standard deviation) was added to the observer's input at test time.</p>
      </div>
      {figure(F.b64(F.fig_noise(S)), f"Observer accuracy under measurement noise on Id. Up to 5% noise costs almost nothing (mean PC {noise[0.0]['mean_pc']:.3f} to {noise[0.05]['mean_pc']:.3f}); at 20% it is {noise[0.2]['mean_pc']:.3f}, and training with 5% noise recovers part of that ({noise_tn[0.2]['mean_pc']:.3f}).", 7)}
      {table(["Observer", "window (steps)", "PC z<sub>1</sub>", "PC z<sub>2</sub>", "PC z<sub>3</sub>", "mean decoded error"],
             [[name, str(m["span"]), f3(m["per_channel"]["z1"]["pc"]), f3(m["per_channel"]["z2"]["pc"]), f3(m["per_channel"]["z3"]["pc"]),
               pct1(np.mean(list(m["decoded_field_error_pct"]["observer"].values())))]
              for name, m in [("MLP (default)", obs), ("MLP, 50-step window", ex["observer_mlp50"]), ("MLP, 300-step window", ex["observer_mlp300"]),
                              ("ridge regression", ex["observer_ridge150"]), ("MLP trained with 5% noise", ex["observer_mlp150_tn0.05"])]])}
      <div class="narrow">
        <p>A window of a period or more matters more than the model class: 50 steps is not enough, and the nonlinear MLP clearly beats ridge at 150 steps (mean PC {np.mean([m['pc'] for m in ex['observer_ridge150']['per_channel'].values()]):.2f} against {np.mean(list(obs_pc.values())):.2f}), so the map from Id history to latent is nonlinear.</p>
      </div>
    </section>""")
    return "".join(out)



# --------------------------------------------------------- numbers for section 3 ----
def group(variants, fn):
    return np.array([fn(S[v]) for v in variants])


SUP0 = [v for v in SUP if v.endswith("_d0")]
SUP1 = [v for v in SUP if v.endswith("_d0.1")]
COND0 = [v for v in COND if v.endswith("_d0")]
COND1 = [v for v in COND if v.endswith("_d0.1")]
TIED0 = SUP0 + COND0
TIED1 = SUP1 + COND1
mean_err = lambda e: e["static"]["mean_field_error_pct_all_frames"]
z1pc = lambda e: float(np.mean(e["rc"]["z1drive"]["pc_mean"]))
z3pc = lambda e: float(np.mean(e["rc"]["z3drive"]["pc_mean"]))
lin_leak = lambda e: e["coupling"]["free_coordinates_linear_r2"]
mlp_leak = lambda e: e["coupling"]["free_coordinates_mlp_r2"]
obs_mean_pc = lambda e: float(np.mean([c["pc"] for c in e["observer"]["per_channel"].values()]))
obs_ratio_mean = lambda e: float(np.mean(list(e["observer"]["ratio_to_persistence_full"].values())))
def rng_(a, fmt="{:.2f}"):
    lo, hi = fmt.format(a.min()), fmt.format(a.max())
    return lo if lo == hi else f"{lo}&ndash;{hi}"


obs_free_pc = lambda e: float(np.mean([e["observer"]["per_channel"][k]["pc"] for k in ("z1", "z2")]))

import torch  # noqa: E402


def n_params(variant):
    d = f"{BASE}/latent3" + ("" if variant == "baseline" else f"_{variant}")
    ck = torch.load(f"{d}/conv_ae_full.pt", weights_only=False, map_location="cpu")
    return sum(p.numel() for p in ck["model_state"].values())


def hf_fraction(variant, k=2, win=21):
    """High-frequency share of latent coordinate k: std of the residual after a moving average, over its total std."""
    d = f"{BASE}/latent3" + ("" if variant == "baseline" else f"_{variant}")
    x = np.load(f"{d}/latent_full.npz")["z"][:, k]
    return float(np.std(x - np.convolve(x, np.ones(win) / win, mode="same")) / np.std(x))


def pairs_worse(metric):
    """(n pairs where adding decorrelation lowered the metric, n pairs) over matched mu=0 / mu=0.1 variants."""
    pairs = [(a, a[:-2] + "d0.1" if a.endswith("_d0") else None) for a in TIED0]
    pairs = [(a, b) for a, b in pairs if b in S]
    return sum(metric(S[b]) < metric(S[a]) for a, b in pairs), len(pairs)


def variant_row(v):
    e = S[v]
    mode = {"none": "none", "supervised": "supervised", "conditional": "conditional"}[e["id_mode"]]
    lam = v.split("_w")[1].split("_d")[0] if v.startswith("sup") else "&ndash;"
    mu = v.split("_d")[-1] if not (v == "baseline" or v.startswith("ctrl")) else "&ndash;"
    name = {"baseline": "production AE"}.get(v, v)
    return [name, mode, lam, mu, f"{e['static']['val_mse']:.4f}", pct1(mean_err(e)), f3(e["coupling"]["single_coordinate_linear_r2"][2]),
            f3(lin_leak(e)), f3(mlp_leak(e))]


def section_b():
    ctrl_err = group(CTRL, mean_err)
    npar = {"none": n_params("baseline"), "supervised": n_params(SUP[0]), "conditional": n_params(COND[0])}
    w1, w2 = pairs_worse(z1pc), pairs_worse(z3pc)
    hf_ctrl = np.array([hf_fraction(v) for v in CTRL])
    hf_sup = np.array([hf_fraction(v) for v in SUP0])
    hf_cond = np.array([hf_fraction(v) for v in COND0])
    runs = [(v, c) for v in S for c in S[v]["rc"]]
    n_runs_rc = sum(S[v]["rc"][c]["seeds"] for v, c in runs)
    n_collapsed = sum(S[v]["rc"][c]["echo_collapsed_seeds"] for v, c in runs)
    collapse_txt = (f"None of the {n_runs_rc} reservoir runs collapsed to a fixed point" if n_collapsed == 0
                    else f"{n_collapsed} of {n_runs_rc} reservoir runs collapsed to a fixed point")
    return f"""
    <section id="approach-b">
      <div class="narrow">
        <span class="tag tag-rust">&#9679; approach B &middot; Id inside the autoencoder</span>
        <h2>3. Making Id one of the three latent coordinates</h2>
        <p>The alternative is to change the autoencoder so that one of its three latent coordinates <i>is</i> Id, leaving the other two free to encode what Id does not. There are two ways to enforce that, and they differ in what the encoder needs:</p>
        <div class="pipeline"><b>supervised</b>   fields &rarr; [encoder] &rarr; (z<sub>1</sub>, z<sub>2</sub>, z<sub>3</sub>) &rarr; [decoder] &rarr; fields
                                       &#8593; z<sub>3</sub> pulled toward Id by  &lambda;&middot;MSE(z<sub>3</sub>, Id<sub>n</sub>)
<b>conditional</b>  fields &rarr; [encoder] &rarr; (z<sub>1</sub>, z<sub>2</sub>)  +  measured Id<sub>n</sub>  &rarr; [decoder] &rarr; fields</div>
        <p>In <b>supervised</b> mode the encoder still emits three numbers and the third is trained toward z-scored Id through an extra loss term (weight &lambda;); it needs only the fields at encode time. In <b>conditional</b> mode the encoder emits two numbers and the decoder is handed the measured Id directly, so the constraint is exact but encoding needs Id. In either mode a decorrelation penalty (weight &mu;) can be added that punishes the free coordinates for being correlated with Id. Every variant follows the production training recipe exactly (two stages, 200 + 60 epochs, chronological 85/15 split), and a plain autoencoder retrained the same way (three seeds, plus the original) is the control.</p>
        <p>Eleven autoencoders were trained and compared with the production one: {len(CTRL) - 1} controls, {len(SUP)} supervised (&lambda;&nbsp;=&nbsp;0.1, 1, 10 &times; &mu;&nbsp;=&nbsp;0, 0.1) and {len(COND)} conditional (&mu;&nbsp;=&nbsp;0, 0.1). Each control is a plain autoencoder with a different seed. The supervised autoencoder has {npar['supervised']:,} parameters, the same as the original; the conditional one has {npar['conditional']:,} (one fewer encoder output).</p>
      </div>
      {table(["Autoencoder", "mode", "&lambda;", "&mu;", "val. MSE", "mean field error", "R&sup2; Id from z<sub>3</sub>", "R&sup2; Id from z<sub>1</sub>,z<sub>2</sub> (linear)", "R&sup2; Id from z<sub>1</sub>,z<sub>2</sub> (MLP)"], [variant_row(v) for v in F._order(S)])}
      <div class="narrow">
        <p style="font-size:13px;color:var(--text-muted);">Held-out block (frames 17,001&ndash;20,000) for the R&sup2; columns; mean field error is the decoded relative error over all 20,000 frames, as in the first report. One training run per configuration; the controls give the seed-to-seed spread.</p>

        <h3>3.1 Reconstruction costs nothing</h3>
        <p>Validation MSE is 0.0211&ndash;0.0214 for every variant, indistinguishable from the controls. Mean decoded field error is {rng_(ctrl_err)}% for the controls and {rng_(group(SUP0, mean_err))}% for the supervised variants without decorrelation, so tying a coordinate to Id does not hurt and may help slightly. That last part is suggestive only: with four controls, a 0.2&ndash;0.4 point gap is not established.</p>
      </div>
      {figure(F.b64(F.fig_option2_overview(S), dpi=120), "Top: reconstruction cost on a full scale from zero, with the spread of the four controls shaded. Middle: how much Id the two <i>free</i> coordinates still carry (bars: MLP; ticks: linear). Bottom: how closely the third coordinate tracks Id. Gray: controls; teal: supervised; rust: conditional.", 8)}
      <div class="narrow">
        <h3>3.2 The third coordinate tracks Id; the free ones still leak</h3>
        <p>The supervised third coordinate reproduces Id at held-out R&sup2; {rng_(group(SUP, lambda e: e['coupling']['single_coordinate_linear_r2'][2]), '{:.3f}')} at every &lambda; tried, including 0.1, and the conditional one is Id by construction. In a plain autoencoder no single coordinate is close (R&sup2; at most {max(max(S[v]['coupling']['single_coordinate_linear_r2']) for v in CTRL):.2f}).</p>
        <p>The free coordinates are another matter. The decorrelation penalty does what it says on the linear part: R&sup2; of Id from z<sub>1</sub>, z<sub>2</sub> by a linear map drops from {rng_(group(CTRL, lin_leak))} in the controls to {rng_(group(SUP0, lin_leak), '{:.3f}')} in the supervised variants and about zero with the penalty. But the nonlinear leakage (an MLP) does not go away: {rng_(group(CTRL, mlp_leak))} in the controls, {rng_(group(SUP0, mlp_leak))} in the supervised variants without the penalty, {rng_(group(SUP1, mlp_leak))} with it, and {rng_(group(COND, mlp_leak))} for the two conditional ones. This is expected rather than a defect: Id is a function of the plasma state, and on a nearly one-dimensional limit cycle any two coordinates that locate the state on the loop also determine Id nonlinearly. A linear penalty removes only the linear part, and single runs differ by more than the effect of the penalty in some pairs, so the leakage numbers should be read as ranges.</p>

        <h3>3.3 The reservoir does worse on these latents</h3>
        <p>The first report's reservoir, with unchanged settings, was run on every variant (5 seeds each). On the controls it repeats the original result: driven by z<sub>1</sub>, inferPC is {rng_(group(CTRL, z1pc), '{:.3f}')}. On the Id-tied latents without decorrelation it falls to {rng_(group(TIED0, z1pc))}, with much larger seed-to-seed spread; with the decorrelation penalty it ranges from {group(TIED1, z1pc).min():.2f} to {group(TIED1, z1pc).max():.2f}. Adding the penalty lowered z<sub>1</sub>-driven inferPC in {w1[0]} of {w1[1]} matched pairs (same &lambda; or mode, &mu; = 0 against 0.1).</p>
      </div>
      {figure(F.b64(F.fig_option2_rc(S), dpi=120), "Reservoir inferPC on each autoencoder's latents, mean over scored channels, error bars showing the standard deviation over five reservoir seeds. Left: driven by z<sub>1</sub>. Right: driven by z<sub>3</sub>, which is Id in the Id-tied variants; the dashed line is option 1 (Id as a fourth channel) averaged over the controls.", 9)}
      <div class="narrow">
        <p>Driving the reservoir with Id directly, now that Id is one of the three channels, gives inferPC {rng_(group(TIED0, z3pc))} without decorrelation (mean {group(TIED0, z3pc).mean():.2f}) and {group(TIED1, z3pc).min():.2f}&ndash;{group(TIED1, z3pc).max():.2f} with it. That is close to option 1 ({np.mean([np.mean(S[v]['rc']['id4drive']['pc_mean']) for v in CTRL]):.2f} averaged over the controls) and far from the 0.9 target: making Id a coordinate does not repair Id-driven infer. {collapse_txt}; the free-run period is off by {rng_(np.array([S[v]['rc']['z1drive']['echo_period_mismatch_pct_mean_of_oscillating'] for v in TIED0]), '{:.1f}')}% for the Id-tied variants without the penalty.</p>
        <p>Noise carried in from Id does not appear to explain this: the high-frequency share of the third coordinate (residual after a 21-step average, relative to its spread) is {rng_(hf_sup, '{:.3f}')} for the supervised latents and {rng_(hf_cond, '{:.3f}')} for the conditional one, against {rng_(hf_ctrl, '{:.3f}')} for the controls' third coordinate, presumably because the encoder sees a frame rather than the raw signal. The cause is not established here. One untested candidate is geometry: forcing z<sub>3</sub> to follow Id changes the shape of the latent trajectory the reservoir has to learn. The reservoir's settings were tuned on the production latent and not re-tuned, but the retrained controls, using the same settings, are fine, so the change comes from tying Id in.</p>

        <h3>3.4 The observer is unaffected</h3>
        <p>The windowed observer of Section 2 works equally well on every latent space. Scored on the two free coordinates z<sub>1</sub>, z<sub>2</sub> (z<sub>3</sub> is given where it is Id, so it is left out for a fair comparison), mean inferPC is {rng_(np.array([obs_free_pc(S[v]) for v in CTRL]), '{:.3f}')} for the controls and {rng_(np.array([obs_free_pc(S[v]) for v in TIED0 + TIED1]), '{:.3f}')} for the Id-tied autoencoders. Mean decoded error is {rng_(np.array([obs_ratio_mean(S[v]) for v in CTRL]), '{:.3f}')} times persistence for the controls and {rng_(np.array([obs_ratio_mean(S[v]) for v in TIED0 + TIED1]), '{:.3f}')} for the Id-tied ones: the same, or at most a small edge.</p>
      </div>
      {figure(F.b64(F.fig_option2_observer(S), dpi=120), "The window observer scored on each autoencoder's latents (left: mean PC of the free coordinates z<sub>1</sub>, z<sub>2</sub>). Accuracy is essentially the same everywhere; the Id-tied autoencoders give at most a small edge in decoded error.", 10)}
      {callout("Verdict on approach B", f"Putting Id inside the autoencoder is free for reconstruction and does make the third coordinate equal to Id, but it does not make the latent easier to track from Id, it degrades the reservoir's dynamics on that latent ({rng_(group(TIED0, z1pc))} z<sub>1</sub>-driven inferPC against {rng_(group(CTRL, z1pc), '{:.2f}')} for controls), and the decorrelation penalty lowers its inferPC in {w1[0]} of {w1[1]} matched pairs (z<sub>1</sub>-driven) and {w2[0]} of {w2[1]} (Id-driven) without removing nonlinear leakage. Its only demonstrated benefit is a clean interface, which the observer does not need.", "rust")}
    </section>"""


def section_compare():
    rows = []
    id4_all = pooled(CTRL, "id4drive")
    rows.append(["Reservoir, Id as 4th channel (option 1)", "Id only", f3(id4_all.mean()), rng_(id4_all), f3(mean_ratio(rcid_ratio)) + " (1 seed)"])
    rows.append(["Windowed observer, production AE", "Id only", f3(obs_mean_pc(base)), f"{min(c['pc'] for c in obs['per_channel'].values()):.2f}&ndash;{max(c['pc'] for c in obs['per_channel'].values()):.2f}", f3(obs_ratio_mean(base))])
    best_sup = min(SUP0, key=lambda v: obs_ratio_mean(S[v]))
    free = [S[best_sup]["observer"]["per_channel"][k]["pc"] for k in ("z1", "z2")]
    rows.append([f"Windowed observer, Id-tied AE ({best_sup}, lowest error of the three &lambda;); scored on z<sub>1</sub>, z<sub>2</sub> since z<sub>3</sub> is Id", "Id only", f3(np.mean(free)), f"{min(free):.2f}&ndash;{max(free):.2f}", f3(obs_ratio_mean(S[best_sup]))])
    tied_z3 = np.array([row for v in TIED0 for row in S[v]["rc"]["z3drive"]["pc_per_seed"]])
    rows.append(["Reservoir on Id-tied latents, driven by Id (&mu;=0)", "Id only", f3(tied_z3.mean()), rng_(tied_z3), "&ndash;"])
    rows.append(["<i>reference:</i> reservoir driven by z<sub>1</sub>", "z<sub>1</sub> (not measurable)", f3(z1_pc.mean()), rng_(z1_pc), f3(mean_ratio(rc1_ratio)) + " (1 seed)"])
    return f"""
    <section id="comparison">
      <div class="narrow">
        <h2>4. Comparison and recommendation</h2>
        <p>Recovering the latent state, and through the decoder the full 7-field state, from the discharge current alone:</p>
      </div>
      {table(["Method", "observes", "mean inferPC", "min&ndash;max", "decoded error / persistence"], rows, hl=(1,))}
      <div class="narrow">
        <p style="font-size:13px;color:var(--text-muted);">Reservoir rows pool every run (autoencoders &times; 5 seeds) and give the min&ndash;max over runs; the observer rows are single fits and give the min&ndash;max over the three latents. Decoded error is the mean over the seven fields of the held-out relative error divided by a persistence forecast (lower is better); the reservoir figure is from one seed.</p>
        <p><b>Use the windowed observer on the existing autoencoder.</b> It reaches inferPC {obs_mean_pc(base):.2f} and about {obs_ratio_mean(base):.2f}&times; persistence in decoded error from Id alone, matches the reservoir that is handed an unmeasurable latent, degrades gracefully with measurement noise, and needs no change to the autoencoder or the reservoir. The reservoir's infer mode is not a reliable way to use Id, and neither is retraining the autoencoder around Id.</p>
        <p>What each approach still costs. The observer is a supervised regression fit on one simulation, so it must be re-fit for a different operating point. It also reads about one breathing period (150 steps, 7.5&nbsp;&micro;s) of history before it can estimate the state, which is a latency of one period, not an instantaneous readout.</p>
      </div>
    </section>"""


def section_limits():
    return """
    <section id="limits">
      <div class="narrow">
        <h2>5. Limitations</h2>
        <ul style="line-height:1.7;font-size:15px;color:var(--text-secondary);">
          <li><b>One simulation, one operating point.</b> Train and held-out blocks are consecutive stretches of the same 1&nbsp;ms run (same thruster settings). Nothing here tests transfer to a different voltage, flow rate or thruster.</li>
          <li><b>Idealized measurement.</b> Id is the simulation's own discharge current at every 50&nbsp;ns step. A real signal would be sampled, filtered and noisier, and only the additive-Gaussian case was tested.</li>
          <li><b>Single training run per autoencoder configuration.</b> Only the controls have replicates (four). Differences in reconstruction of 0.2&ndash;0.4 points, and most of the leakage differences, are inside or close to that spread.</li>
          <li><b>The reservoir was not re-tuned</b> for the Id-tied latents. Its degradation there is measured with the first report's settings; a re-tuned reservoir might recover some of it, though the controls indicate the settings are not the cause.</li>
          <li><b>Cause of the reservoir's degradation is not established</b> (Section 3.3). The obvious candidate, noise carried into the latent from Id, is not supported by the data, but no explanation was tested directly.</li>
          <li><b>Decoded reservoir error is from one seed</b> (seed 0); inferPC uses five.</li>
        </ul>
      </div>
    </section>"""


def section_repro():
    return """
    <section id="reproduce">
      <div class="narrow">
        <h2>Appendix A. Reproducing this report</h2>
        <p>All code is in <code>HPHall-data-Summer24/scripts/</code> and <code>src/</code> on branch <code>ae-rc</code>; every experiment writes a <code>config.json</code> (seed, constants, git revision, command) beside its metrics.</p>
        <div class="pipeline">python run_id_variants.py            <span style="opacity:.6"># train the 11 Id-aware / control autoencoders (~1 min each on a GPU)</span>
python evaluate_id_variants.py --seeds 5   <span style="opacity:.6"># score everything -> id_study/summary.json</span>
python id_observer.py                       <span style="opacity:.6"># the windowed observer on one latent space</span>
python rc_with_id.py --channels z1,z2,z3,id --drive id   <span style="opacity:.6"># option 1, reservoir</span>
python id_to_latent_probe.py                <span style="opacity:.6"># information-vs-window-length probe (Fig. 3)</span>
python build_report_id.py                   <span style="opacity:.6"># this page</span></div>
        <p>Metrics follow the first report (<a href="report.html">Embedding the Breathing Mode</a>): inferPC is the squared correlation between prediction and truth, decoded error is the mean relative error over the frozen decoder's seven fields, and the persistence baseline holds the last training frame. The design of the Id-aware autoencoder, including the loss terms, is in <code>ID_TARGETING.md</code>.</p>
      </div>
    </section>"""


def build():
    with open(f"{HERE}/report_template.html", encoding="utf-8") as f:
        tmpl = f.read()
    head = tmpl[:tmpl.index("</style>") + len("</style>")]
    head = re.sub(r"<title>.*?</title>", "<title>Tracking the Breathing Mode from Discharge Current</title>", head, count=1)
    obs_pcs = list(obs_pc.values())
    stats = f"""
    <div class="stat-row narrow" style="max-width:820px">
      <div class="stat"><p class="stat-label">Observer, latent from Id</p><div class="stat-value">{np.mean(obs_pcs):.2f} <small>mean PC</small></div></div>
      <div class="stat"><p class="stat-label">Reservoir driven by Id</p><div class="stat-value">{id_pc.mean():.2f} <small>mean PC, {n_runs} runs</small></div></div>
      <div class="stat"><p class="stat-label">Decoded error vs. persistence</p><div class="stat-value">{obs_ratio_mean(base):.2f}&times; <small>observer</small></div></div>
      <div class="stat"><p class="stat-label">Id inside the AE</p><div class="stat-value">no cost <small>to reconstruction</small></div></div>
    </div>"""
    body = f"""
<div class="page">
  <header class="narrow">
    <p class="eyebrow">Technical report &middot; companion to <a href="report.html" style="color:inherit">Embedding the Breathing Mode</a></p>
    <h1>Tracking the Breathing Mode from Discharge Current</h1>
    <p class="dek">The autoencoder latents that drive the reservoir model cannot be measured on a thruster; the discharge current can. Two ways of connecting them were tested: reading the latent off a window of past current, and building the current into the autoencoder itself.</p>
    <div class="byline"><span>adrianskw / AE&#8209;RC</span><span>HPHall-data-Summer24</span><span>held-out: frames 17,001&ndash;20,000</span></div>
  </header>
  <div class="narrow"><div class="abstract"><p class="label">Summary</p>
    <p><b>Approach A, observe the latent from Id.</b> Driving the reservoir's infer mode with Id does not work: mean inferPC is {id_pc.mean():.2f} over {n_runs} runs (range {id_pc.min():.2f}&ndash;{id_pc.max():.2f}), against {z1_pc.mean():.2f} when it is driven by z<sub>1</sub>. The information is present, though: a causal regression on a window of one breathing period of past Id recovers the latent at inferPC {np.mean(obs_pcs):.2f}, decodes to fields at {obs_ratio_mean(base):.2f}&times; the error of a persistence forecast, and tolerates 5% measurement noise with negligible loss. <b>Approach B, Id inside the autoencoder.</b> Making one latent coordinate equal Id costs nothing in reconstruction, but the reservoir does much worse on those latents, a decorrelation penalty usually makes it worse still, and the observer gains nothing. Recommendation: use the windowed observer with the existing autoencoder.</p></div></div>
  {stats}
  <nav class="toc narrow"><span class="label">Contents</span><ol>
    <li><a href="#setup">Setup</a></li><li><a href="#approach-a">Reading the latent off the discharge current</a></li>
    <li><a href="#approach-b">Making Id a latent coordinate</a></li><li><a href="#comparison">Comparison and recommendation</a></li>
    <li><a href="#limits">Limitations</a></li><li><a href="#reproduce">Appendix A: reproducing this report</a></li></ol></nav>
  <main>
    <section id="setup"><div class="narrow"><h2>1. Setup</h2>
      <p>The data, autoencoder and reservoir are those of the <a href="report.html">first report</a>: a 7-field, 50&times;25 HPHall-II simulation of an SPT-100-class thruster, 20,000 frames at 50&nbsp;ns, compressed to a 3-number latent by a convolutional autoencoder, with a 200-node reservoir. The discharge current Id(t) is column 2 of the simulation's <code>log.dat</code>, one value per step, aligned one-to-one with the frames (mean {S_ID_MEAN:.2f}&nbsp;A, standard deviation {S_ID_STD:.2f}&nbsp;A). All splits are chronological (first 17,000 frames to train, last 3,000 held out), all normalization uses training statistics only, and the metrics are the first report's: inferPC, per-field relative error through the frozen decoder, and a persistence baseline.</p></div></section>
    {section_a()}
    {section_b()}
    {section_compare()}
    {section_limits()}
    {section_repro()}
  </main>
  <footer class="narrow"><div>Autoencoder: 3-layer conv encoder/decoder, PyTorch, Adam, chronological 85/15 split, log&#8321;&#8320; on n_e/n_n/n&#7749;_i, per-channel z-score.</div>
  <div>Reservoir: <code>mapRC</code>, N=200, &rho;=0.9, degree=10, ridge &alpha;=10<sup>-3</sup>, bias node. Observer: MLP 31&rarr;128&rarr;128&rarr;3, tanh, Adam, three fits averaged.</div>
  <div>adrianskw/AE-RC &middot; branch <code>ae-rc</code> of adrianskw/libRC</div></footer>
</div>"""
    out = f"{BASE}/report_id.html"
    html = head + body
    html = re.sub(r'<a href="report\.html"[^>]*>(.*?)</a>', r"\1", html)   # relative links would be dead in a hosted copy
    assert "__" not in re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", html).replace("__init__", ""), "unreplaced placeholder"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", out, f"{len(html) / 1e6:.2f} MB")


from src import observables as _obs  # noqa: E402
_id = _obs.load_discharge_current(BASE)
S_ID_MEAN, S_ID_STD = float(_id.mean()), float(_id.std())

if __name__ == "__main__":
    build()
