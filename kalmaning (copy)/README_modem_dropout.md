# Modem Dropout Test

This script runs a single Monte Carlo trial to assess estimator robustness to modem dropouts and overlaps.

## What it does
- Builds a trajectory (lawnmower, spiral, concentric, figure8) and simulates an EKF with acoustic/DVL/depth inputs.
- Applies a user-specified dropout schedule per modem, or a built-in overlapping schedule (`--use-default-overlaps`) that includes a full-outage window for all modems.
- Runs the filter (`run_single_trial`) and collects time series (truth, estimate, covariance, NEES/NIS, gating decisions).
- Emits plots and CSV/JSON summaries under `results_modem_dropout/<timestamp>_seed<seed>/`.
- Prints key metrics to the terminal for quick, paper-ready reporting.

## Key arguments
- `--duration-sec`: simulation length in seconds (default 180).
- `--trajectory`: `lawnmower|spiral|concentric|figure8`.
- `--targets`: list of modems to include (default all: usv1-4).
- `--dropout`: manual schedule, e.g. `"usv2:30-60;usv3:80-110"` (quote to avoid shell splitting).
- `--use-default-overlaps`: use built-in staggered + triple-overlap + full-outage schedule (mutually exclusive with `--dropout`).
- `--seed`: RNG seed.

## Outputs
- `summary.json`: seed, RMSEs, final error, runtime, dropout intervals, disabled seconds, config, NEES, NIS snapshots.
- `timeseries.csv`: per-timestep truth/estimate, error norm, covariance diagonals, NEES, NIS, and enabled flags per modem.
- Plots (in `plots/`):
  - `position_error_main.png`: raw + rolling-median error with dropout shading and phase medians.
  - `position_error_raw.png`: raw error with dropouts.
  - `position_error_cdf.png`: empirical CDF by phases.
  - `uncertainty_vs_time.png`: sqrt(trace(Ppos)) over time with dropouts.
  - `xy_traj.png`: ground-truth vs estimate XY path.

## Terminal metrics
The script now prints consolidated metrics after each run:
- Position error stats: RMSE, median, p90, p95, max.
- Pos uncertainty proxy: median/p90/p95 of sqrt(trace(Ppos)).
- NEES (pos and full): mean/median/p90/p95; mean-to-expected ratio for pos (dof≈3).
- Acoustic gating: counts and fraction used vs skipped; NIS quantiles for used updates.
- Uptime/downtime per requested modem relative to the scenario duration.

## Example commands
- Built-in overlaps (covers all modems and includes a full outage):
  - `python modem_dropout_test.py --duration-sec 900 --use-default-overlaps`
- Manual overlaps for a subset (quote the string):
  - `python modem_dropout_test.py --duration-sec 900 --targets usv2 usv4 --dropout "usv2:200-400;usv4:350-550"`

## Notes
- Quoting the `--dropout` string is required so semicolons are not split by the shell.
- Plots use a non-interactive backend when no display is present; they are saved to disk.
