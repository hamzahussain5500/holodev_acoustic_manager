# Adaptive SBL / EKF Toolkit

This folder contains simulation tools to study adaptive single-beacon localization (SBL) for a USV-AUV team. The goal is to dynamically reconfigure four hull-mounted acoustic modems on a USV to balance position accuracy and energy.

## How the pieces support the research plan
- Uncertainty quantification: `current_acoustic_EKF_patched.py` runs an EKF that fuses IMU, DVL, depth, and acoustic ranges, producing NEES/NIS and covariance traces.
- Geometry/observability: `sbl_geometry.py` + `fim_gdop_runner.py` compute FIM/GDOP/CRLB over trajectories and modem subsets to predict accuracy from geometry.
- Adaptive logic scaffolding: modem dropout/switching is emulated in `modem_dropout_test.py`, and sensor subset sweeps (`run_sensor_combos.py`) + Monte Carlo batches (`monte_carlo_runner.py`) provide data to drive future multi-objective optimization (accuracy vs. energy).

## Core simulation and fusion
- `current_acoustic_EKF_patched.py` – Single-run EKF simulator in HoloOcean. Takes trajectory and modem targets; logs timeseries, NEES/NIS, RMSE. Supports modem dropout schedules (ignore acoustic updates in windows).
- `kalman_utils.py` – Minimal 6D EKF implementation (predict, linear updates, range update) and RMSE helper.
- `trajectory.py` – Waypoint factories: lawnmower, spiral, concentric circles, figure-eight.
- `uncertainty_utils.py` – Covariance ellipse/ellipsoid utilities and 1D bounds for visualizing uncertainty.
- `validation_metrics.py` – NEES/NIS calculators and consistency tests (per-sample and downsampled mean NEES).

## Experiment runners
- `modem_switching_validation_fixed.py` – Policy/manual modem switching harness. Geometry policy (GDOP + size_penalty + dwell/margin hysteresis) and weighted multi-objective policy (observability, energy via SOC, mission uncertainty target) with dropout support and 0–4 active modems. Outputs config.json, selector_meta.json, timeseries.csv/plots to `results_modem_switching/`.
  Example (GDOP policy, dropouts, allow zero-beacon): `python modem_switching_validation_fixed.py --mode policy --traj spiral --duration 240 --seed 1 --dropout usv3:30-60 usv4:80-120 --allow-zero-beacons --size-penalty 1.0`
  Example (weighted policy): `python modem_switching_validation_fixed.py --mode policy --policy-type weighted --traj spiral --duration 240 --seed 1 --dropout usv3:30-60 --allow-zero-beacons --soc-init 1.0 --base-drain-w 0.1 --beacon-drain-w 0.5 --target-unc-xy 5 --target-unc-3d 8 --score-margin 0.05`
- `modem_dropout_test.py` – Runs a single EKF mission with scheduled modem dropouts; saves timeseries.csv, plots (pos error, uncertainty, NIS, XY), summary.json under `results_modem_dropout/<timestamp>_seedX/`.
  Example: `python modem_dropout_test.py --seed 0 --duration-sec 180 --trajectory spiral --targets usv1 usv2 usv3 usv4 --dropout "usv2:30-60;usv4:90-140"`
- `run_sensor_combos.py` – Quickly compares sensor subsets (IMU-only up to all modems) on a spiral; prints metrics and can plot covariance ellipses.
- `monte_carlo_runner.py` – Batch runner: launches many EKF trials with varied seeds; aggregates RMSE, final error, NEES/NIS consistency, optional timeseries. Outputs summary.json, per_run.csv/jsonl, plots, timeseries under `results_mc*/`.
- `fim_gdop_runner.py` – Geometry-only analysis using FIM/GDOP/CRLB on trajectories and modem subsets; saves CSV/NPZ/plots to `results_fim_gdop/`. Can load trajectories/beacons from NPZ/CSV or use synthetic defaults.
- `sbl_geometry.py` – Library for FIM/GDOP/CRLB over all modem subsets along a trajectory (3D and XY projections).

## Utilities and data
- `sensor_combo_summary.csv` – Collected metrics from sensor subset sweeps.
- `results_*` folders – Artifacts from past runs (Monte Carlo batches, dropout experiments, FIM/GDOP studies). Safe to delete/regenerate.
- `README_MC.md` – Original quickstart for the Monte Carlo harness (kept for reference).

## Typical workflows
- Single EKF trial (no dropouts): `python current_acoustic_EKF_patched.py --trajectory spiral --target-name usv1 --target-name usv2`
- Adaptive switching study (policy): `python modem_switching_validation_fixed.py --mode policy --traj spiral --duration 240 --seed 1 --dropout usv3:30-60 usv4:80-120 --allow-zero-beacons --size-penalty 1.0`
- Manual dropout emulation: `python modem_switching_validation_fixed.py --mode manual --traj spiral --duration 180 --dropout usv2:30-60 usv4:90-140`
- Sensor subset sweep: `python run_sensor_combos.py --seed 123` to compare IMU/DVL/depth/acoustic combos.
- Monte Carlo robustness: `python monte_carlo_runner.py --runs 30 --seed 0 --trajectory spiral --out results_mc_tuned`
- Geometry ranking: `python fim_gdop_runner.py --out results_fim_gdop --sigma-r 0.1 --decimate 10 --mode logdet`

## How it fits together
1) EKF runs (single or MC) quantify pose uncertainty and consistency for given modem sets and noise models.
2) Geometry analysis scores modem subsets before running the EKF, predicting observability and likely accuracy.
3) Adaptive switching experiments emulate modem scheduling decisions with hysteresis, GDOP/rank feasibility, and energy costs (size_penalty, zero-beacon allowance). Outputs (active_set traces, GDOP, pos error, selector_meta) feed a future multi-objective optimizer that trades accuracy against modem usage/energy under dropouts.

## Notes for first-time users
- All plotting uses headless Matplotlib (`Agg`); outputs are written to the respective results folders.
- Seeds are explicit flags (`--seed N`) to make trials reproducible; output folders are timestamped for traceability.
- Acoustic targets are named `usv1`..`usv4`; pass them via `--targets` (dropout test) or `--target-name` (EKF runner).
- Noise/consistency tuning: adjust `q_vel_std`, `meas-scale`, or per-sensor extra noise via CLI flags in the runners to steer NEES/NIS toward expected degrees of freedom.
