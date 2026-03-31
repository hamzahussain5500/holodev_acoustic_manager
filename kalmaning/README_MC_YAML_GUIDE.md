# Monte Carlo YAML Parameter Guide

This guide explains every key in mc_config.yaml, what it does, and what values are recommended for stable experiments.

## 1) How config is resolved

Order of precedence for the new runner (`run_mc_new()`):

1. Built-in defaults from `DEFAULT_MC_CONFIG`
2. Values from mc_config.yaml
3. CLI overrides (only `--outdir`, `--duration`, `--runs`, `--max-workers`)

Notes:
- If `seeds: null`, seeds become `0..runs-1`.
- If `seeds` list is provided, `runs` is reset to `len(seeds)`.

## 2) Parameter reference

## Experiment setup

- `algorithms`: list of algorithms to run.
  - Typical: `[imu_dvl_depth, imu_dvl_depth_all4, adaptive]`
  - Also valid: `adaptive_gdop`, `adaptive_weighted`, `adaptive_v2`
- `adaptive_policy`: used only when `algorithms` contains `adaptive`.
  - Values: `gdop`, `weighted`, `v2`
- `targets`: active beacon names. Usually all four.
  - Typical: `[usv1, usv2, usv3, usv4]`
- `seeds`: list of integer seeds or `null`.
  - Recommended for papers: explicit list (reproducible).

## Randomization and injected noise

- `x0_pos_std`, `x0_vel_std`: initial state perturbation standard deviations.
- `imu_accel_extra_std`, `imu_bias_rw_std`, `dvl_extra_std`, `depth_extra_std`, `range_extra_std`:
  additive sensor noise scales.

Recommended baseline:
- Keep extra noise at `0.0` initially.
- Sweep one noise source at a time for ablations.

## Geometry / switching controls

- `sigma_r`: range noise used by geometry scoring.
- `gdop_xy`, `gdop_3d`: geometry acceptability thresholds.
- `switch_margin`: minimum objective improvement required to switch.
- `min_dwell_sec`: minimum time between switches.
- `size_penalty`: penalizes larger beacon sets.
- `rank_req_xy`, `rank_req_3d`: observability rank requirements.
- `min_beacons_xy`, `min_beacons_3d`: hard minimum active beacons.
- `allow_zero_beacons`: if true, selector may disable all beacons.
- `score_margin`, `power_save_tol`: mainly for weighted policy.
- `phase_schedule`, `mission_phase`: weighted policy phase logic.
- `churn_interval_sec`, `churn_score_eps`: anti-churn behavior.

Recommended stable values:
- `min_dwell_sec: 3.0` to `5.0`
- `switch_margin: 0.1` (avoid flip-flop)
- `size_penalty: 0.05` to `0.2`
- `min_beacons_xy: 2`
- `min_beacons_3d: 3`
- `allow_zero_beacons: false`

## Energy model and uncertainty gating

- `battery_wh`, `base_drain_w`, `beacon_drain_w`, `drain_scale`
- `soc_init`, `soc_min`, `low_power_soc`
- `target_unc_xy`, `target_unc_3d`, `off_unc_mult`

Recommended baseline values:
- `battery_wh: 100.0`
- `base_drain_w: 5.0`
- `beacon_drain_w: 6.0`
- `drain_scale: 1.0`
- `soc_init: 1.0`
- `soc_min: 0.0`
- `low_power_soc: 0.2`
- `target_unc_xy: 5.0`
- `target_unc_3d: 8.0`
- `off_unc_mult: 1.5`

Avoid very aggressive values like `target_unc_xy=0.3~0.5` with large drain terms unless intentionally stress testing.

## V2-specific tuning

- `energy_weight`: global energy term weight.
- `v2_low_power_soc`: SOC threshold for low-power mode.
- `v2_energy_mult`: energy term multiplier in low-power mode.
- `v2_size_penalty_mult`: set-size penalty multiplier in low-power mode.
- `v2_rank_deficit_mult`, `v2_rank_deficit_penalty`: rank-deficit penalty terms.
- `ticks_per_sec`: conversion for dwell-step logic.

Recommended starting point (`adaptive_policy: v2`):
- `energy_weight: 0.1`
- `v2_low_power_soc: 0.3`
- `v2_energy_mult: 3.0`
- `v2_size_penalty_mult: 2.0`
- `v2_rank_deficit_mult: 1.0`
- `v2_rank_deficit_penalty: 10.0`
- `ticks_per_sec: 100.0`

## 3) Suggested YAML profiles

## A) Balanced (recommended first)

```yaml
algorithms: [imu_dvl_depth, imu_dvl_depth_all4, adaptive]
adaptive_policy: v2
targets: [usv1, usv2, usv3, usv4]
seeds: [0,1,2,3,4,5,6,7,8,9]

x0_pos_std: 1.0
x0_vel_std: 0.1
imu_accel_extra_std: 0.0
imu_bias_rw_std: 0.0
dvl_extra_std: 0.0
depth_extra_std: 0.0
range_extra_std: 0.0

sigma_r: 0.5
gdop_xy: 12.0
gdop_3d: 15.0
switch_margin: 0.1
min_dwell_sec: 3.0
size_penalty: 0.1
rank_req_xy: 2
rank_req_3d: 3
min_beacons_xy: 2
min_beacons_3d: 3
allow_zero_beacons: false
score_margin: 0.05
power_save_tol: 0.02
phase_schedule: null
mission_phase: cruise
churn_interval_sec: 0.0
churn_score_eps: 0.02

battery_wh: 100.0
base_drain_w: 5.0
beacon_drain_w: 6.0
drain_scale: 1.0
soc_init: 1.0
soc_min: 0.0
low_power_soc: 0.2
target_unc_xy: 5.0
target_unc_3d: 8.0
off_unc_mult: 1.5

energy_weight: 0.1
v2_low_power_soc: 0.3
v2_energy_mult: 3.0
v2_size_penalty_mult: 2.0
v2_rank_deficit_mult: 1.0
v2_rank_deficit_penalty: 10.0
ticks_per_sec: 100.0
max_workers: 2
```

## B) Accuracy-first

Key changes from balanced:
- `energy_weight: 0.0`
- `size_penalty: 0.0`
- `switch_margin: 0.05`
- `gdop_xy: 10.0`, `gdop_3d: 12.0`

## C) Energy-first

Key changes from balanced:
- `energy_weight: 0.3`
- `size_penalty: 0.2`
- `v2_energy_mult: 4.0`
- `v2_size_penalty_mult: 3.0`
- keep `min_beacons_xy: 2`, `min_beacons_3d: 3` to avoid collapse.

## 4) Practical recommendations for publishable MC

- Use at least 30 seeds for final tables (`runs >= 30`).
- Keep seed list explicit in YAML for reproducibility.
- Start with `balanced`, then do one-factor sensitivity sweeps.
- Report both accuracy and efficiency:
  - RMSE / final error
  - NEES/NIS coverage
  - mean active beacons and switch count
  - energy savings vs all-4 baseline

## 5) Important current behavior

In the new MC pipeline, trajectory in `run_trial()` is currently hardcoded to spiral (`"trajectory": "spiral"`).
So YAML keys like `trajectory` are not used by `run_mc_new()` right now.
Use legacy mode if trajectory switching is required, or patch `run_trial()` to read a config field.
