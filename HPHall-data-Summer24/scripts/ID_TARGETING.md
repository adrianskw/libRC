# Discharge current (Id) as a state variable

`HPHall-data-Summer24/log.dat` column 2 is the discharge current Id(t), one row per
simulation step, aligned 1:1 with the 20,000 field frames (`MOVIE_ITS 1`,
`DT_0 5e-8`). Id is measurable on a real thruster; the AE latents are not. The
question is whether the latent state can be tracked from Id alone.

Splits everywhere: chronological, `n_train = 17000`, held-out = frames
17,001-20,000. Interpreter: `/home/adrian/Projects/.venv/bin/python`.

## Status

| step | script | result |
|---|---|---|
| does the latent hold Id? | `latent_to_discharge_current.py` | yes for 3-D (R² 0.94, windowed MLP); no for 2-D (0.55) |
| **option 1**: 4-channel RC, Id-driven infer | `rc_with_id.py`, `rc_seed_sweep.py` | **fails the pre-set criterion** (below) |
| is the information there at all? | `id_to_latent_probe.py` | **yes**: 0.96-0.98 from a ~1-period window of Id |
| option 2: Id as a latent coordinate | not implemented | design below |

## Option 1 - 4-channel reservoir, AE untouched

State `[z1, z2, z3, Id]`, every channel z-scored on the train block only. Infer mode
observes only the drive channel and reconstructs the rest. Success criterion, fixed
before running: Id-driven inferPC >= 0.9 on z2 and z3.

```
python rc_with_id.py --channels z1,z2,z3    --drive z1 --tag baseline   # reproduces train_rc.py
python rc_with_id.py --channels z1,z2,z3,id --drive id --tag id-drive
python rc_with_id.py --channels z1,z2,z3,id --drive z1 --tag z1-drive
python rc_seed_sweep.py                                                 # 8-seed variance
```

Each run writes `latent{D}/n{N}/rc_{tag}/{config.json, metrics.json, results.npz, rc_timeseries.png}`.
`config.json` holds every constant, the seed, the git revision and the command line.

Results, inferPC over 8 reservoir seeds (N=200, rho=0.9, degree=10):

| configuration | unobserved channels | mean | std | range |
|---|---|---:|---:|---|
| z1-driven, 3 channels (baseline) | z2, z3 | 0.979, 0.944 | 0.001, 0.003 | 0.940-0.980 |
| **Id-driven, 4 channels** | z1, z2, z3 | 0.36, 0.33, 0.37 | ~0.19 | 0.19-0.72 |

- **The criterion fails robustly.** Seed 0 (0.71) was the lucky end of a wide,
  apparently bimodal spread (seeds 0 and 6 near 0.7, the rest 0.2-0.3). A single
  draw would have overstated it. Always judge Id-driven configurations over seeds.
- Even so, Id-driven infer beats persistence on all 7 decoded fields (seed 0:
  infer/persistence 0.56-0.73), just far from z1-driven quality (0.19-0.73).
- Causal smoothing of the Id drive (`--id-smooth 10/25/50/100`) did **not** help and
  behaved erratically (PC 0.01, 0.01, 0.58, 0.09): it changes the seed lottery, not
  the physics. Not a noise problem.
- Reverse direction: z1-driven infer recovers Id at only PC 0.15.

### Why: the information is there, the reservoir's infer path is the bottleneck

`id_to_latent_probe.py` regresses z(t) on a window of past Id with no reservoir:

| Id window (steps) | taps | MLP R² z1 / z2 / z3 | ridge R² z1 / z2 / z3 |
|---:|---:|---|---|
| 0 (instantaneous) | 1 | 0.08 / 0.00 / 0.00 | 0.08 / 0.00 / -0.02 |
| 50 | 11 | 0.75 / 0.77 / 0.80 | 0.41 / 0.55 / 0.58 |
| **150 (~1 period)** | 31 | **0.975 / 0.964 / 0.957** | 0.80 / 0.81 / 0.78 |
| 300 | 31 | 0.964 / 0.948 / 0.928 | 0.83 / 0.83 / 0.81 |

Instantaneous Id says nothing about the latent state; about one breathing period of
history recovers it well. This is the Takens picture: a scalar observable of a
>=3-D system determines the state only through its history. The reservoir supplies
that history implicitly, but `Reservoir.infer()` trains the readout with every
channel driven and then feeds back its own estimates for the undriven ones, which
is the same exposure-bias mismatch seen in the MLP and delay-embedding readout
experiments. Id is also a high-frequency, noisy drive.

### Next steps, cheapest first
1. **Windowed observer**: a direct Id-window -> z regression (already R² ~0.96) as
   the tracker, with no reservoir in the loop.
2. **Delay-embedding readout on the RC** (precedent: `delay_embedding_readout.py`):
   give the readout a window of reservoir states while Id-driven.
3. **Observer-mode reservoir**: drive with Id only and fit the readout for exactly
   that regime. Needs input/output dimensions to differ, which `Reservoir` does not
   currently allow (input `D` = output `D`).

## Option 2 - Id as one of the three latent coordinates (design, not built)

The decoder always receives a 3-vector `[a, b, c]`. Option 2 makes `c` correspond to
Id so the encoder's two free coordinates are trained to encode what Id does not.
Notation: `x` frame, `E` encoder, `D` decoder, `Id_n` z-scored Id (train statistics),
`lambda` and `mu` loss weights.

### Mode `supervised` - Id is a target the encoder must hit (soft)

    z = E(x) in R^3                    # z1, z2 free; z3 is the Id slot
    L = MSE(x, D(z))
        + lambda * MSE(z3, Id_n)       # z3 trained to BE normalized Id
        + mu * decorr(z1, z2 ; Id_n)   # optional, see leakage

- Same architecture as today (`to_latent` already emits 3 numbers); one extra loss
  term, so existing 3-D checkpoints warm-start directly.
- `lambda -> infinity` approaches a hard constraint. Finite `lambda` lets the
  encoder trade Id accuracy for reconstruction; the residual measures the conflict.
- Feasible: Id is nearly a function of a single frame (fields -> Id, R² = 0.97),
  residual floor ~0.08 A RMSE.
- The encoder needs only the fields at encode time.

### Mode `conditional` - Id is supplied to the decoder (hard)

    z_f = E(x) in R^2
    L = MSE(x, D([z_f, Id_n]))

- Exact by construction: Id is an input, not a prediction. A conditional autoencoder.
- `to_latent` shrinks to 2 outputs while `from_latent` still consumes 3, so
  `decode()` keeps its arity. `forward(x, id_n)` concatenates.
- Encoding a frame into the full 3-vector requires measured Id, so data must carry
  Id aligned under shuffling (`TensorDataset(train_x, train_id)`).

| | supervised | conditional |
|---|---|---|
| "z3 == Id" enforced by | loss term (soft) | architecture (exact) |
| encoder outputs | 3 | 2 |
| Id needed at encode time | no | yes |
| checkpoint additions | `id_mode`, `id_mean/std` | same, plus 2-dim `to_latent` |

### Leakage: the free coordinates need not encode the complement

Nothing stops z1, z2 from also encoding Id, since Id is a function of the fields.
That wastes capacity and defeats "optimized for what Id does not cover". Mitigations:

1. batch decorrelation `mu * mean_k corr(z_k, Id_n)^2`, k in {1,2}: cheap, linear only;
2. adversarial or HSIC penalty for nonlinear leakage, only if (1) leaks.

Measure leakage post hoc: fit `[z1, z2] -> Id` with `latent_to_discharge_current.py`;
the goal is low held-out R² from the free coordinates alone. Caveat: instantaneous
decorrelation is not dynamical independence. Id and the latents are coupled through
time (Id from a latent window, R² 0.94), and the RC learns exactly that coupling.

### How it plugs in without breaking anything

- `src/autoencoder.py`: `ConvAE(..., id_mode="none" | "supervised" | "conditional")`.
  `"none"` is today's behavior and the default, so existing checkpoints load unchanged
  (`load_checkpoint` treats missing keys as `"none"`). One class, one loss: a flag,
  not a fork.
- `train_ae*.py`: `--id-mode`, `--id-weight`, `--decorr-weight`, `--tag`; output to
  `latent3{tag}/` so the baseline `latent3/conv_ae_full.pt` is never overwritten.
- `latent_full.npz` keeps three columns, so `train_rc.py`, `decode_rc.py` and the
  sweep scripts run unchanged. The Id column must round-trip through the same
  normalization convention `decode_rc.py` uses for the RC's mean/std.
- Sweep once built: `lambda in {0.1, 1, 10}`, `mu in {0, 0.1}`; score reconstruction
  vs `latent3/err_summary_full.json`, z3-vs-Id R², leakage R², and RC Id-driven
  inferPC over seeds, head to head with option 1.

### Open question option 1 raised

Making Id a coordinate does not by itself remove the observability problem: the
other two coordinates must still be inferred from Id's *history*, and instantaneous
Id carries no information about them (R² ~0.08). Option 2 gives a cleaner interface
(the third coordinate is Id, readable with no regression) and plausibly a small
reconstruction cost, since Id adds only +0.003 of field variance on top of a free
3-D code. Whether it makes the RC's Id-driven infer any *easier* is unproven, and
option 1's result suggests the windowed observer is the higher-value next step.

## Related: the 2-D latent

Not pursued further. Poincare-Bendixson rules out chaos in a 2-D autonomous flow, and
the 2-D pipeline collapsed to a fixed point on free-run. Summarized as Appendix A of
`report.html` (`build_report.py` / `report_template.html`).
