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
Scripts: `modem_switching_validation_fixed.py` (harness), `adaptive_modem_manager_v2.py` (V2 policy)
Comparative runner: `run_comparative_analysis.sh`
Full reference: `README_modem_switching_validation_fixed.md`

### Purpose
Validate acoustic beacon subset switching by running the EKF with a dynamic active set at each tick. Supports four strategies: manual staged dropout (baseline), GDOP geometry policy, weighted multi-objective policy, and V2 posterior covariance approximation policy. The active beacon subset is re-evaluated at 100 Hz by the selector and only the ranges from currently active beacons are fused into the EKF.

### Modes
- **manual** — dropout windows per beacon mask beacons out; active set = remaining targets
- **policy** — one of: `gdop`, `weighted`, `v2`

### Policy methodology

#### GDOP policy (GeometryPolicySelector)
Selects the beacon subset minimizing:
```
obj(S) = GDOP(S) + eff_penalty × |S|

eff_penalty = size_penalty                       (normal)
            = size_penalty × energy_size_mult    (when SOC ≤ low_power_soc)
```
GDOP is derived from the 2D or 3D Fisher Information Matrix: `J = H^T H / σ_r²`, `GDOP = sqrt(trace(J^{-1}))`.
Feasibility requires `rank(J) ≥ rank_req` and `GDOP ≤ gdop_thresh`. Falls back to best nonzero if none feasible.
Switching hysteresis: holds for `min_dwell_sec`; switches if improved by `switch_margin` or if fewer beacons with equal objective (power save).
The energy model (optional) tracks SOC and triples the size penalty when `SOC ≤ low_power_soc`, driving an energy-triggered subset reduction.

#### Weighted policy (WeightedPolicySelector)
Scores each subset with a normalized three-term objective:
```
score(S) = w_obs × f_obs(S)  +  w_energy × f_energy(S)  +  w_mission × f_mission(S)
```
- `f_obs` — normalized logdet(FIM); higher = more observable geometry
- `f_energy` — predicted SOC after using S × relative power saving vs max subset
- `f_mission` — how well posterior uncertainty matches `target_unc_xy`

Phase weights (w_obs, w_energy, w_mission):

| Phase | w_obs | w_energy | w_mission |
|-------|-------|----------|-----------|
| survey | 0.85 | 0.08 | 0.07 |
| cruise | 0.60 | 0.25 | 0.15 |
| transit | 0.25 | 0.60 | 0.15 |
| low_power | 0.15 | 0.75 | 0.10 |

Phase is set by `--phase-schedule` or forced to `low_power` when `SOC ≤ low_power_soc`.
Hysteresis: dwell timer resets only on actual subset changes (not on every tick). Switches if `score_best ≥ score_current + score_margin`, or fewer beacons within `power_save_tol`.
Zero-beacon is allowed only when `sqrt(trace(P_xy)) ≤ off_unc_mult × target_unc_xy`.

#### V2 policy (AdaptiveModemManagerV2)
Minimizes a posterior-covariance-based objective:
```
score(S) = trace(P+_pos(S)) + eff_size_pen × |S|
         + eff_rank_pen × rank_deficit(S) + energy_penalty(S)
```
Posterior covariance approximation (linearized Kalman update):
```
H  = range Jacobian for subset S
P+ = P - P H^T (H P H^T + σ_r² I)^{-1} H P
```
This predicts the position covariance reduction that would result from fusing subset S.

Low-power multipliers activate when `SOC ≤ v2_low_power_soc`:
```
eff_size_pen  = size_penalty × v2_size_penalty_mult
eff_rank_pen  = rank_deficit_penalty × v2_rank_deficit_mult
energy_weight = energy_weight × v2_energy_mult
```

Uncertainty gating (hysteresis):
```
off_threshold = off_unc_mult × target_unc_xy
on_threshold  = on_unc_mult  × target_unc_xy  (default: 1.25 × off_threshold)

acoustics ON  → gate OFF if sqrt(trace(P_xy)) < off_threshold
acoustics OFF → gate ON  if sqrt(trace(P_xy)) > on_threshold
```

Switching hysteresis uses step count (`min_dwell_steps = min_dwell_sec × 100 Hz`); switches only if improvement exceeds `switch_margin`.

### Bug fixes applied (2026-03-31 → 2026-04-01)

**Bug 1 — V2 permanent dwell-hold lock** (`adaptive_modem_manager_v2.py`):
The `held_margin` path reset `_dwell = 1`, below `min_dwell_steps`, so the next call immediately entered dwell_hold again — an infinite lock. Fix: set `_dwell = min_dwell_steps`.

**Bug 2 — V2 initial 4→2 jump from empty seed** (`adaptive_modem_manager_v2.py`):
`_current_ids` initialized to `[]`. First call compared any subset vs empty (score = trace(P), very large), so 2-beacon always won. Fix: pre-seed `_current_ids = list(beacon_positions.keys())`.

**Bug 3 — Weighted policy dwell timer never expired** (`modem_switching_validation_fixed.py`):
`last_switch_t = t` set unconditionally at 100 Hz, so `t - last_switch_t ≈ 0.01 s` < `min_dwell_sec` always. Fix: only reset when active subset actually changes.

**Enhancement — GDOP policy energy model:**
Added `EnergyModel` to `GeometryPolicySelector`: when `SOC ≤ low_power_soc` the size penalty multiplies by `energy_size_mult` (default 3×). In near-degenerate geometry (18 m SBL at ~450 m range) this is needed to trigger any switching.

**Bug 4 — V2 rank deficit penalty incorrectly applied to 0-beacon subset** (`adaptive_modem_manager_v2.py`):
`deficit = max(0, target_rank - rank)` with `rank=0` for 0-beacon gives `deficit=2`, adding `rank_deficit_penalty×2=10` to the 0-beacon score. This blocked all acoustics-off states. Fix: `deficit = 0 if not subset else max(0, target_rank - int(rank))` in all 3 score computation locations.

**Bug 5 — V2 0-beacon won immediately in scoring after rank deficit fix** (`adaptive_modem_manager_v2.py`):
After fixing rank deficit, 0-beacon scored ~1.72 < 2-beacon ~1.84 at any SOC, causing chattering every dwell period. Fix: Added `score_zero_below_soc` parameter — 0-beacon enters scoring only when `SOC ≤ score_zero_below_soc`. Above this threshold, only the uncertainty gate can produce acoustics-off.

**Bug 6 — V2 uncertainty gate fired immediately after subset switch** (`adaptive_modem_manager_v2.py`):
After 4→2 switch at t~15s, `_gate_dwell` had already accumulated 1500 steps from t=0 (EKF converged early). Gate fired within 1 step (2→0 at t=16s). Fix: Reset `_gate_dwell = 0` in the switch branch.

**Bug 7 — V2 gate dwell not reset on subset switch** (`adaptive_modem_manager_v2.py`):
`_gate_dwell` reset only in the `dwell_hold` branch, not in the `switched` branch. After any subset change, the gate could fire within one step because `_gate_dwell` had pre-accumulated from before the switch. Fix: added `self._gate_dwell = 0` in the switch branch.

**Enhancement — Weighted policy DEFAULT_PHASE_WEIGHTS for n=4 survey selection** (`modem_switching_validation_fixed.py`):
Old survey weights `(0.6, 0.2, 0.2)` gave `w_obs/w_energy=3.0` < 4.2 threshold for n=4 to win over n=2 in near-degenerate geometry. Updated to `(0.85, 0.08, 0.07)`, giving `w_obs/w_energy=10.6`. Full updated weights: cruise `(0.60, 0.25, 0.15)`, transit `(0.25, 0.60, 0.15)`, low_power `(0.15, 0.75, 0.10)`.

### Validated comparative results (spiral, 180 s, seed 0)

Energy calibration: 10 Wh battery, P_base=4 W, P_beacon=3 W, drain_scale=20.

| Policy | RMSE (m) | Switches | Active-set counts | Switch events |
|--------|----------|----------|-------------------|---------------|
| Manual | 0.863 | 3 | {1,2,3,4} | t=45s: 4→3, t=90s: 3→2, t=135s: 2→1 |
| GDOP | 0.962 | 1 | {2,3} | t=55s: 3→2 (energy, SOC=0.60) |
| Weighted | 1.020 | 2 | {0,2,4} | t=60s: 4→2 (phase), t=135s: 2→0 (battery) |
| **V2** | 1.287 | **4** | {0,2,4} | t=15s: 4→2 (energy), t=99s: 2→0 (gate), t=138s: 0→2 (drift), t=153s: 2→0 (energy) |

V2 demonstrates the richest switching behavior with 4 events driven by 3 independent mechanisms: energy scoring, uncertainty gate, and energy-based off at low SOC. It is the only policy that autonomously re-enables acoustics after gate-off. Higher RMSE for V2/Weighted reflects periods with 0 active beacons — the trade-off for energy savings.

### Reproducing the comparative results

The easiest way is to run the comparative script, which handles all four policies and computes drain scale automatically:

```bash
cd kalmaning/
bash run_comparative_analysis.sh
```

Or individually (see `README_modem_switching_validation_fixed.md` for full parameter tables).
Note: the commands below use `--drain-scale 20.0` which is pre-calibrated for 180 s duration with the given battery parameters. If you change `--duration`, recompute drain scale as: `0.95 × battery_wh × 3600 / ((base_drain_w + 2 × beacon_drain_w) × duration)`.

**Manual baseline:**
```bash
python3 modem_switching_validation_fixed.py \
  --outdir results_comparative/manual --mode manual \
  --traj spiral --duration 180 --seed 0 \
  --dropout usv4:45-180 usv3:90-180 usv2:135-180 \
  --sigma-r 0.5 --make-plots
```

**GDOP policy:**
```bash
python3 modem_switching_validation_fixed.py \
  --outdir results_comparative/gdop --mode policy --policy-type gdop \
  --traj spiral --duration 180 --seed 0 \
  --sigma-r 0.5 --gdop-xy 30.0 --gdop-3d 40.0 \
  --min-beacons-xy 2 --min-beacons-3d 3 \
  --min-dwell-sec 15.0 --switch-margin 2.0 --size-penalty 0.5 \
  --battery-wh 10.0 --base-drain-w 4.0 --beacon-drain-w 3.0 \
  --drain-scale 20.0 --soc-init 1.0 --soc-min 0.05 --low-power-soc 0.6 \
  --make-plots
```

**Weighted policy:**
```bash
python3 modem_switching_validation_fixed.py \
  --outdir results_comparative/weighted --mode policy --policy-type weighted \
  --traj spiral --duration 180 --seed 0 \
  --sigma-r 0.5 --gdop-xy 8.0 --gdop-3d 10.0 \
  --min-beacons-xy 2 --min-beacons-3d 3 \
  --min-dwell-sec 10.0 --score-margin 0.03 --power-save-tol 0.0 \
  --phase-schedule "survey:0-60,transit:60-120,low_power:120-180" \
  --target-unc-xy 0.5 --target-unc-3d 0.8 --off-unc-mult 1.5 \
  --allow-zero-beacons \
  --battery-wh 10.0 --base-drain-w 4.0 --beacon-drain-w 3.0 \
  --drain-scale 20.0 --soc-init 1.0 --soc-min 0.05 --low-power-soc 0.3 \
  --energy-weight 0.3 --make-plots
```

**V2 policy:**
```bash
python3 modem_switching_validation_fixed.py \
  --outdir results_comparative/v2 --mode policy --policy-type v2 \
  --traj spiral --duration 180 --seed 0 \
  --sigma-r 0.5 --min-beacons-xy 0 --min-beacons-3d 0 --allow-zero-beacons \
  --gdop-xy 8.0 --gdop-3d 10.0 \
  --min-dwell-sec 15.0 --switch-margin 0.005 --size-penalty 0.0 \
  --target-unc-xy 1.5 --target-unc-3d 2.0 --off-unc-mult 0.55 \
  --battery-wh 10.0 --base-drain-w 4.0 --beacon-drain-w 3.0 \
  --drain-scale 20.0 --soc-init 1.0 --soc-min 0.05 \
  --energy-weight 0.195 \
  --v2-low-power-soc 0.45 --v2-energy-mult 5.0 \
  --v2-size-penalty-mult 1.0 --v2-rank-deficit-penalty 5.0 --v2-rank-deficit-mult 0.5 \
  --v2-score-zero-below-soc 0.4 \
  --make-plots
```

### Outputs per run
- `config.json` — full CLI args and scenario parameters
- `selector_meta.json` — per-tick log: `t`, `selected`, `reason`, `rank`, `gdop`, `score`, `soc`, `phase`
- `timeseries.csv` — merged EKF + selector: `t`, `pos_err`, `active_count`, `active_set`, `gdop_xy`, `soc`, ...
- `summary.json` — `rmse_pos`, `final_pos_err`
- Figures: `fig_active_count.png`, `fig_active_set.png`, `fig_pos_err.png`, `fig_score_vs_time.png`, `fig_soc_vs_time.png`

## 5) Monte Carlo evaluation
Script: monte_carlo_runner.py

Purpose:
- Compare algorithms and selectors across many seeds.
- Aggregate RMSE, NEES/NIS coverage, energy usage, switching statistics, and CI bands.

Configuration:
- Uses mc_config.yaml for nearly all parameters.
- CLI overrides: --outdir, --duration, --runs, --max-workers (process pool size; omit to use value from mc_config.yaml, or set explicitly).

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
