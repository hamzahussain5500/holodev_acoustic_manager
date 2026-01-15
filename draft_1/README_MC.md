# HoloOcean EKF Monte Carlo Harness

This repo contains a small set of scripts to run and evaluate a 6D EKF (IMU + DVL + depth + acoustic) in HoloOcean, plus trajectory generators and consistency metrics.

## Scripts

- `current_acoustic_EKF_patched.py`
  - Runs a single EKF simulation in HoloOcean using IMU/DVL/depth/acoustic fusion.
  - Accepts a trajectory choice and target names for acoustic ranging.
  - Produces on-screen metrics (RMSE, NEES/NIS) and plots when run directly.

- `monte_carlo_runner.py`
  - Monte Carlo harness that launches many independent EKF trials with different seeds.
  - Aggregates RMSE/final-error stats, NEES/NIS summaries, and run-level consistency rate.
  - Writes `summary.json`, `per_run.csv`, `per_run.jsonl`, and plots under the chosen output directory.

- `trajectory.py`
  - Provides waypoint generators: `lawnmower`, `spiral`, `concentric`, `figure8` (lemniscate).
  - The runner/ EKF selects a trajectory via `--trajectory` (MC) or `--trajectory` (single run).

- `validation_metrics.py`
  - Consistency utilities: NEES/NIS calculation, chi-square bounds, and optional downsampled mean-NEES test to mitigate time-correlation.

- `kalman_utils.py`
  - EKF core (state, predict, linear update, range update) and RMSE helper.

## Common options (applied through `config_overrides` into the EKF)

Process / measurement tuning:
- `--q-vel-std` : accel process noise std (sigma_a) used in EKF Q.
- `--meas-scale`: scales DVL/depth/acoustic measurement stds uniformly.
- `--dvl-extra-std`, `--depth-extra-std`, `--range-extra-std`: injected extra measurement noise.
- `--imu-accel-extra-std`, `--imu-bias-rw-std`: injected IMU noise / bias random walk.

Consistency controls (MC runner):
- `--consistency-percent` (default 90): required % of NEES samples inside per-sample chi² gate.
- `--consistency-avg-band LO HI`: optional band on avg NEES as multiples of dof (e.g., `0.7 1.3`).
- `--consistency-require-ds`: require downsampled mean-NEES chi² test to pass.
- `--nees-ds-stride`, `--nees-ds-alpha`: stride/alpha for the downsampled mean-NEES test.

Trajectory selection:
- `--trajectory {lawnmower,spiral,concentric,figure8}`

Currents and targets:
- `--currents {on,off}`
- `--target-name usv1 --target-name usv2` (repeatable; acoustic targets)

## How to run

### Single EKF run (with plots)
```bash
python current_acoustic_EKF_patched.py --trajectory spiral --target-name usv1 --target-name usv2
```
Runs one simulated mission with a spiral path; shows RMSE/NEES/NIS and plots.

### Monte Carlo batch
```bash
# Basic 20-run batch, no currents, default lawnmower
python monte_carlo_runner.py --runs 20 --seed 0 --currents off

# Tuned noise + spiral trajectory + consistency gates
python monte_carlo_runner.py \
  --runs 30 --seed 0 --currents off --trajectory spiral \
  --q-vel-std 0.22 --meas-scale 1.5 \
  --consistency-percent 90 --consistency-avg-band 0.7 1.3 \
  --consistency-require-ds --nees-ds-stride 10

# Save decimated timeseries for each run
python monte_carlo_runner.py --runs 10 --seed 100 --save-timeseries decimated --out results_mc_ts
```

Outputs go to `--out` (default `results_mc/`):
- `summary.json`: mean/std/CI for RMSE, final error (plus median/p90/p95), NEES/NIS means, run-consistency rate.
- `per_run.csv` / `per_run.jsonl`: per-trial metrics including `run_consistent` flag.
- Plots: histograms/CDF/boxplots in the output directory.
- Optional timeseries: `timeseries/run_XXX.npz` and `timeseries_mean.npz` when enabled.

### Trajectory-only inspection
Call the factory directly (Python REPL) to visualize or reuse waypoints:
```python
from trajectory import build_trajectory
wps = build_trajectory("figure8", {"figure8_scale": 25.0, "figure8_turns": 2})
```

## Notes on consistency
- Run-level consistency uses: (a) percent of NEES samples inside per-sample chi² gate, AND
  optional (b) avg-NEES band, AND optional (c) downsampled mean-NEES chi² test. This is less brittle than a strict full-length mean-NEES test on correlated data.
- NIS/NEES are reported per sensor and for the full/position state; per-run `run_consistent` is stored in CSV/JSONL.

## Tips
- Start with `--currents off` for repeatability; enable later if desired.
- Tune `--q-vel-std` and `--meas-scale` together to steer NEES/NIS toward expected dof.
- Use `--save-timeseries decimated` during sweeps to keep files small.
