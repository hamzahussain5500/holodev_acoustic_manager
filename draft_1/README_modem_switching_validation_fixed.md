# Modem Switching Validation (weighted & GDOP policies)

This document explains the modem_switching_validation_fixed.py harness: what it does, how the selector works (geometry, energy, mission), how to run it, and how to interpret outputs.

## Purpose
Validate acoustic modem switching policies by simulating an EKF run where the active beacon subset can change over time. The harness builds a selector_fn and passes it into current_acoustic_EKF_patched.py via run_single_trial so the EKF fuses ranges only from the selected beacons at each step.

## High-level flow
1) Parse CLI args (trajectory, targets, policy settings, energy model, mission phases). 
2) Build a selector_fn (GDOP-based or weighted multi-objective). Optional dropout windows can further mask beacons.
3) Call run_single_trial(...) with config_overrides including modem_selector_fn.
4) Collect timeseries (active set, GDOP, scores, SOC, errors), write CSV/JSON, and save plots.
5) Print selector transitions in the terminal when the active set changes (or at init).

## Policies
### GDOP policy (GeometryPolicySelector)
- Enumerates subsets within [min_beacons, max_beacons].
- Scores by GDOP + size_penalty.
- Feasible only if rank >= rank_req and GDOP <= threshold.
- Hysteresis: switch requires improvement by switch_margin and min_dwell_sec elapsed.

### Weighted policy (WeightedPolicySelector)
- Enumerates subsets (and optionally 0-beacon if allowed and gated).
- Feasibility: rank_req_xy/3d and gdop_thresh_xy/3d.
- Three normalized terms per candidate:
  - Observability f_obs: logdet(FIM) normalized across candidates.
  - Energy f_energy: uses power(base + n*beacon), battery_wh, drain_scale, predicted SOC.
  - Mission f_mission: aims for target uncertainty (target_unc_xy/3d); uses posterior approx (P_prior + FIM) or GDOP*σ_r proxy.
- Score: score = w_obs*f_obs + w_energy*f_energy + w_mission*f_mission.
- Weights come from mission_phase or phase_schedule (time-varying):
  - survey: (0.6, 0.2, 0.2)
  - cruise: (0.5, 0.3, 0.2)
  - transit: (0.3, 0.5, 0.2)
  - low_power: (0.2, 0.7, 0.1)
- Zero-beacon option: only if allow_zero_beacons and sqrt(trace(P)) <= off_unc_mult * target_unc.
- Hysteresis: min_dwell_sec plus score_margin for improvement; power_save_tol allows switching to fewer beacons if not worse than tolerance.
- SOC is stepped using the chosen subset’s power draw.

### V2 policy (AdaptiveModemManagerV2)
- Uses the external AdaptiveModemManagerV2 (multi-objective with uncertainty gating and dwell).
- Objective: minimize predicted posterior uncertainty + size penalty + rank deficit + energy penalty.
- Allows 0..4 beacons, honors size_penalty, switch_margin, and min_dwell_sec (converted to steps).
- Uncertainty gating: acoustics can turn off when sqrt(trace(P)) falls below off_unc_mult * target_unc (with hysteresis).
- Energy-aware: energy penalty scales with normalized modem power; penalties increase under low SOC.
- Battery depletion: if SOC reaches soc_min, acoustics are forced off (no power remains).
- If covariance is missing, seeds P with a target-uncertainty prior so the selector still works.

## Key parameters (CLI)
- Geometry: sigma-r, gdop-xy, gdop-3d, rank-req-xy, rank-req-3d, min-beacons-xy, min-beacons-3d, allow-zero-beacons, off-unc-mult, target-unc-xy, target-unc-3d.
- Energy: battery-wh, base-drain-w, beacon-drain-w, drain-scale, soc-init, soc-min, low-power-soc.
- Weights/phases: mission-phase, phase-schedule (phase:t0-t1,...). 
- Hysteresis: score-margin (weighted), power-save-tol (weighted), switch-margin (GDOP), min-dwell-sec, size-penalty (GDOP).
- Targets/dropout: targets, dropout (name:t0-t1).
- Run controls: duration, seed, traj, currents, make-plots, show-plots (block to view), outdir.
- V2 low-power tuning: v2-low-power-soc, v2-energy-mult, v2-size-penalty-mult, v2-rank-deficit-penalty, v2-rank-deficit-mult.
- Compatibility flags: target-unc (sets xy & 3d), energy-weight (scales energy term and V2 energy penalty).

## Outputs
- outdir/tag/config.json: run configuration (CLI + overrides).
- selector_meta.json: per-decision log (time, selected names, ranks, GDOP, score, SOC, phase).
- timeseries.csv: merged EKF timeseries and selector meta (active_count/set, GDOP, score components, SOC, pos_err, etc.).
- Quick plots: active_count, pos_err, GDOP, score vs time, SOC vs time (and EKF plots if make_plots=True).
- fig_paths.txt: emitted if any plots were saved; lists the saved figure paths. `--show-plots` will display them after saving.
- Terminal: prints when the active set initializes or changes: `[selector] t=XX.XXs active [...] -> [...]`.
- Plot fallbacks: if the EKF timeseries is missing or malformed, the harness will still attempt selector-only plots (n_active, score, SOC) and a minimal guard plot so runs are not plot-less.

## Example runs

**Weighted policy, time-varying phases**
```
python modem_switching_validation_fixed.py \
  --mode policy --policy-type weighted \
  --traj spiral --duration 120 --seed 0 \
  --targets usv1 usv2 usv3 usv4 \
  --battery-wh 10 --base-drain-w 15 --beacon-drain-w 10 --drain-scale 2 \
  --mission-phase cruise \
  --phase-schedule cruise:0-30,survey:30-60,transit:60-90,low_power:90-120 \
  --score-margin 0.0 --power-save-tol 0.0 --min-dwell-sec 1 \
  --gdop-xy 10 --gdop-3d 12 \
  --min-beacons-xy 3 --min-beacons-3d 3 \
  --allow-zero-beacons --target-unc-xy 5 --target-unc-3d 7 --off-unc-mult 2.0 \
  --make-plots
```

**GDOP policy with mild size penalty**
```
python modem_switching_validation_fixed.py \
  --mode policy --policy-type gdop \
  --traj lawnmower --duration 90 --seed 1 \
  --targets usv1 usv2 usv3 \
  --gdop-xy 12 --gdop-3d 14 --switch-margin 0.5 --min-dwell-sec 3 \
  --size-penalty 0.2 --min-beacons-xy 2 --min-beacons-3d 3 \
  --battery-wh 50 --base-drain-w 5 --beacon-drain-w 6 --drain-scale 1.0
```

**V2 policy (AdaptiveModemManagerV2) allowing 0..4 beacons**
```
python modem_switching_validation_fixed.py \
  --mode policy --policy-type v2 \
  --traj spiral --duration 120 --seed 0 \
  --targets usv1 usv2 usv3 usv4 \
  --min-beacons-xy 0 --min-beacons-3d 0 --allow-zero-beacons \
  --gdop-xy 15 --gdop-3d 18 --min-dwell-sec 2 --switch-margin 0.0 --size-penalty 0.0 \
  --target-unc-xy 5 --target-unc-3d 7 --off-unc-mult 2.5 \
  --battery-wh 10 --base-drain-w 15 --beacon-drain-w 8 --drain-scale 2 --energy-weight 2 \
  --v2-low-power-soc 0.3 --v2-energy-mult 3 --v2-size-penalty-mult 2 --v2-rank-deficit-penalty 10 --v2-rank-deficit-mult 1 \
  --show-plots
```

**Full switching spectrum (4→3→2→1→0) via staged dropouts**
```
python modem_switching_validation_fixed.py \
  --mode policy --policy-type weighted \
  --traj spiral --duration 90 --seed 31 \
  --min-dwell-sec 1 --score-margin 0.0 \
  --allow-zero-beacons --min-beacons-xy 0 --target-unc 0.1 --energy-weight 0 \
  --dropout usv4:18-90 usv3:36-90 usv2:54-90 usv1:72-90
```

**No-dropout switching (energy + observability trade-off)**
```
python modem_switching_validation_fixed.py \
  --mode policy --policy-type v2 \
  --traj spiral --duration 120 --seed 35 \
  --min-beacons-xy 2 --target-unc 0.5 --off-unc-mult 0.1 \
  --battery-wh 4 --base-drain-w 20 --beacon-drain-w 10 --drain-scale 3 \
  --energy-weight 0.05 --size-penalty 0.0 \
  --v2-low-power-soc 0.9 --v2-energy-mult 50 --v2-size-penalty-mult 5 \
  --v2-rank-deficit-penalty 10 --v2-rank-deficit-mult 1 \
  --min-dwell-sec 2 --switch-margin 0.0
```

## Practical tips
- To encourage switching down: increase w_energy (use low_power phase), increase beacon_drains or drain_scale, or allow_zero_beacons with a looser off_unc_mult.
- To favor more beacons: use survey weights, tighten gdop thresholds, raise min-beacons-xy/3d, or set power_save_tol=0.
- If no switching occurs, hysteresis may be too strict (lower score-margin / switch-margin and min-dwell-sec).
- If switching is too frequent (chatter), increase min-dwell-sec and/or switch-margin.
- For V2, lower v2-rank-deficit-penalty or v2-rank-deficit-mult to allow 1-beacon under low SOC.
- Check terminal logs to verify the selector is firing; check timeseries.csv for n_active and active_set over time.
- If plots are missing, open fig_paths.txt to see what was saved; the harness now emits selector-only plots when runner outputs are incomplete.
