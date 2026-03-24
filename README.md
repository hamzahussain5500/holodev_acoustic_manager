# Adaptive SBL / EKF Toolkit — Integrated Guide

## Abstract (project context)
Short-baseline (SBL) acoustic positioning systems provide essential navigation capabilities for numerous underwater tasks, but face practical limitations including noise, multipath interference, geometric dilution of precision (GDOP), and hardware failures. The operation of these acoustic arrays is further limited by energy constraints on uncrewed surface vehicles (USVs). Existing research has shown that positional accuracy is related to modem array geometry through Fisher Information Matrix (FIM) analysis, and that the state estimation can be improved by fusing data from sensors such as the Doppler Velocity Logger (DVL) and the Inertial Measurement Unit (IMU). However, there is limited research on managing the acoustic array configuration based on mission requirements and energy constraints.

This research aims to contribute to the development of an adaptive SBL management framework that dynamically reconfigures four hull-mounted acoustic modems on a USV to balance positional accuracy requirements with energy consumption during collaborative USV–AUV operations.

The proposed methodology involves:
1) Quantifying AUV uncertainty using Extended Kalman Filtering with acoustic ranges, IMU data, and DVL velocity estimates.
2) Assessing observability across modem configurations using FIM/GDOP metrics to predict positional accuracy.
3) Proposing a multi-objective optimizer that selects modem configurations to maintain mission-specific uncertainty bounds while minimizing energy consumption.

This work advances autonomous navigation by providing an integrated framework for adaptive acoustic array management, with potential extensions to mission duration and operational reliability for marine robotics. Initial validation is conducted via simulation studies, with possible controlled field experiments.

## Repository overview (how everything maps to the abstract)
This folder contains simulation tools to study adaptive SBL for a USV–AUV team. The toolchain is designed to:
- Quantify estimation uncertainty with an EKF that fuses IMU, DVL, depth, and acoustic ranges.
- Evaluate geometry and observability across modem subsets using FIM/GDOP/CRLB.
- Emulate modem outages and adaptive scheduling to explore energy–accuracy trade-offs.
- Run Monte Carlo experiments for statistically robust comparisons.

## 1) EKF uncertainty quantification
Script: current_acoustic_EKF_patched.py

Purpose:
- Run a single EKF simulation in HoloOcean that fuses IMU, DVL, depth, and acoustic ranges.
- Produce time series, RMSE, NEES/NIS consistency metrics, and plots.
- Support adaptive modem selection via a selector function.

Key flow:
- Initialize HoloOcean, build waypoints from trajectory.py, and run one EKF heartbeat per env.step.
- Dynamic `dt` is computed each heartbeat and used in prediction (`ekf.dt = dt_step`).
- DVL/depth updates are asynchronous multi-rate updates via per-sensor due scheduling.
- Asynchronous acoustic scheduling: one request in flight, responses processed in the main loop.
- Acoustic latency policy is configurable (`skip`, `apply_current`, `rewind`) with max-age and fixed-lag buffer controls.
- Optional selector_fn chooses which beacons are active at each tick.

Outputs:
- Metrics: RMSE, final error, NEES/NIS coverage.
- Time series: positions, covariance, NIS, active set, GDOP (if selector provides it).
- Plots via plot_ekf_outputs(...).

Run examples:
- Single run: python current_acoustic_EKF_patched.py --trajectory spiral --target-name usv1 --target-name usv2
- With viewport: python current_acoustic_EKF_patched.py --trajectory spiral --show-viewport

## 2) Geometry and observability analysis (FIM/GDOP)
Script: fim_gdop_runner.py

Purpose:
- Evaluate geometry along a trajectory without running the EKF.
- Compute FIM log-det, GDOP, rank, and CRLB for all subsets.
- Track best subsets by GDOP or log-det (3D and XY modes).

Key flow:
- Load trajectory (generated or NPZ/CSV input) and beacon positions.
- Enumerate subsets, compute Jacobians, and derive FIM metrics.
- Save time series, summaries, and plots.

Outputs:
- metrics_timeseries.npz, summary.csv, summary_2d.csv
- best_subset_timeseries.csv and 2D counterpart
- Plots for log-det, GDOP, rank, CRLB over time

Run example:
- python fim_gdop_runner.py --out results_fim_gdop --sigma-r 0.1 --decimate 10 --mode logdet

## 3) Modem dropout analysis
Script: modem_dropout_test.py

Purpose:
- Stress the EKF with planned modem outages.
- Quantify how partial or full acoustic loss affects error and uncertainty.

Key flow:
- Parse dropout schedule (manual or built-in overlap schedule).
- Run a single EKF trial via run_single_trial(...).
- Summarize error, NEES/NIS, uncertainty proxy, and modem uptime.

Outputs:
- summary.json, timeseries.csv
- Plots: position error with dropout shading, uncertainty vs time, CDFs, XY trajectory

Run example:
- python modem_dropout_test.py --duration-sec 900 --use-default-overlaps

## 4) Adaptive modem switching
Script: modem_switching_validation_fixed.py

Purpose:
- Validate switching behavior under manual or policy-driven selection.
- Compare geometry-based, weighted multi-objective, and v2 selectors.

Modes:
- manual: dropouts mask beacons; active set is the remaining targets.
- policy: gdop, weighted, or v2 selection with dwell/margin hysteresis.

Policy highlights:
- GDOP policy: objective = GDOP + size_penalty; feasibility by rank and GDOP thresholds.
- Weighted policy: combines observability, energy, and mission-target uncertainty into a normalized score.
- V2 policy: uncertainty + size + rank deficit + energy penalty with SOC-aware scaling.

Outputs:
- config.json, selector_meta.json, timeseries.csv, summary.json
- Plots: active_count, active_set, GDOP, score components, SOC, position error

Run examples:
- Weighted policy: python modem_switching_validation_fixed.py --mode policy --policy-type weighted --traj spiral --duration 120 --seed 0 --make-plots
- GDOP policy: python modem_switching_validation_fixed.py --mode policy --policy-type gdop --traj lawnmower --duration 90 --seed 1

### AdaptiveModemManagerV2 (detailed working)
AdaptiveModemManagerV2 is a geometry-and-uncertainty driven selector used by the v2 policy. It does not run the EKF; it consumes the EKF position and covariance and returns a chosen beacon subset plus detailed metrics.

Inputs (per decision):
- a_pos: AUV position (x,y,z).
- P: covariance (either full EKF covariance with position in the first 3 states or a 3x3 position covariance).
- available_ids: list of beacon IDs currently available (dropouts are handled here).
- depth_available: selects XY or full 3D geometry scoring.

Core steps:
1) Uncertainty gating:
	- Compute a scalar uncertainty value from P (trace(P) or sqrt(trace(Ppos))).
	- If below an off-threshold, acoustics can be gated off (zero-beacon).
	- If above an on-threshold, acoustics are enabled.
	- Gate hysteresis is controlled by off_unc_mult/on_unc_mult and gate_min_dwell_steps.

2) Subset enumeration:
	- Enumerates all subsets between min_subset_size and max_subset_size (including 0 if allowed).
	- Computes geometry metrics per subset: rank, GDOP, FIM log-det, CRLB std (XY and 3D).

3) Objective scoring (lower is better):
	- Predict posterior trace using the linearized update:
	  P+ = P - P H^T (H P H^T + R)^{-1} H P.
	- Score = trace(P+) + size_penalty * (#beacons) + rank_deficit_penalty * deficit + energy_penalty.
	- Rank deficit penalizes subsets below rank 2 (XY) or rank 3 (3D).
	- Energy penalty uses a normalized power model (base + per-beacon draw).
	- When SOC <= low_power_soc, energy and size penalties increase via low-power multipliers.

4) Switching hysteresis:
	- If min_dwell_steps has not elapsed, the current subset is held.
	- Otherwise, a switch occurs only if the best candidate is better than current by switch_margin.

Outputs:
- Selected IDs plus SelectionMetrics (rank_xy/3d, gdop_xy/3d, fim_logdet_xy/3d, crlb_std, score, reason, soc).
- Reasons include: battery_depleted, uncertainty_gated_off, dwell_hold, switched, held_margin.

Energy model:
- Optional; enabled when battery_wh > 0 and energy_weight > 0.
- SOC is updated every step; if SOC <= soc_min, acoustics are forced off.

Practical configuration notes:
- target_unc_xy/target_unc_3d with off_unc_mult enable uncertainty-based gating.
- min_subset_size=0 allows acoustics-off under low uncertainty or low SOC.
- prefer_smaller breaks ties in favor of smaller subsets.
- If you want 1-beacon operation under low SOC, reduce rank_deficit_penalty or its low-power multiplier.

## 5) Monte Carlo evaluation
Script: monte_carlo_runner.py

Purpose:
- Compare algorithms and selectors across many seeds.
- Aggregate RMSE, NEES/NIS coverage, energy usage, switching statistics, and CI bands.

Configuration:
- Uses mc_config.yaml for nearly all parameters.
- CLI overrides: --outdir, --duration, --runs, --max-workers (process pool size; omit/0 for auto).

Parallelism:
- Seeds run via a process pool; tune concurrency with --max-workers.

Algorithms:
- Baselines: imu_dvl_depth, imu_dvl_depth_all4
- Adaptive: gdop, weighted, v2 (or explicit adaptive_gdop/adaptive_weighted/adaptive_v2)

Outputs:
- tables/summary_metrics.csv, tables/policy_summary.csv
- figures/*.png (CDFs, boxplots, CI bands, energy/accuracy tradeoffs, GDOP/FIM)
- trials/seed_####/... per-run CSV and summary

Run example:
- python monte_carlo_runner.py --outdir results_mc/spiral_T10_N5 --duration 10 --runs 5 --max-workers 4

## Core libraries and utilities
- kalman_utils.py: EKF predict/update and range update helpers.
- trajectory.py: spiral, lawnmower, concentric, figure8 waypoint generators.
- uncertainty_utils.py: covariance ellipses/ellipsoids.
- validation_metrics.py: NEES/NIS calculators and consistency checks.
- sbl_geometry.py: core FIM/GDOP/CRLB calculations and subset evaluation.

## Typical workflows (end-to-end)
1) Run the EKF once to validate sensors and noise: current_acoustic_EKF_patched.py
2) Evaluate geometry sensitivity: fim_gdop_runner.py
3) Stress outages: modem_dropout_test.py
4) Validate switching policies: modem_switching_validation_fixed.py
5) Scale up comparisons: monte_carlo_runner.py with mc_config.yaml

## Notes
- All plotting uses headless Matplotlib (Agg); outputs are written to results folders.
- Acoustic targets are named usv1..usv4.
- Seeds are explicit to keep comparisons reproducible.
