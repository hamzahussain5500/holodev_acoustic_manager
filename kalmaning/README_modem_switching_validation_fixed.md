# Modem Switching Validation (manual + GDOP + weighted + v2)

This document explains the modem_switching_validation_fixed.py harness in detail: selector behavior, CLI options, fallback behavior, outputs, and how to interpret plots and logs.

## Purpose
Validate acoustic modem switching by running the EKF with a dynamic beacon subset. The harness builds a selector function that chooses the active beacons at each step and passes it into current_acoustic_EKF_patched.py via run_single_trial(...). The EKF then fuses only the ranges from those beacons.

## High-level flow
1) Parse CLI args (trajectory, targets, policy settings, energy model, mission phases, dropouts).
2) Build selector_fn (manual dropouts, GDOP policy, weighted policy, or v2 manager).
3) Wrap selector_fn to log decisions and print switch events/SOC warnings.
4) Call run_single_trial(...), requesting timeseries for later plots/CSV.
5) Save config, selector meta, CSV timeseries, figures, and a summary.json.

## Modes
### Manual mode
Manual mode does no policy optimization. It simply filters out beacons whose dropout window includes the current time.

Dropout specification format:
- usv2:30-45
- usv3:80-120

You can specify multiple windows per beacon by repeating the flag. Times are inclusive (t0 ≤ t ≤ t1). The active set equals the remaining targets after filtering.

### Policy mode
Policy mode uses one of three selectors: gdop, weighted, or v2.

## Policies
### GDOP policy (GeometryPolicySelector)
- Enumerates all subsets within [min_beacons, max_beacons].
- Scores by objective = GDOP + size_penalty * (#beacons).
- Feasible subsets: rank >= rank_req and GDOP <= threshold.
- If none feasible, falls back to best nonzero (or best any if all invalid).
- Hysteresis: min_dwell_sec and switch_margin; also allows power-saving switches if equal objective and fewer beacons.

### Weighted policy (WeightedPolicySelector)
- Enumerates subsets within [min_beacons, max_beacons]. Optionally includes 0-beacon only if allowed and uncertainty is low enough.
- Feasibility: rank_req_xy/3d and gdop_thresh_xy/3d.
- Three normalized terms per candidate:
  - Observability $f_{obs}$: normalized logdet(FIM).
  - Energy $f_{energy}$: uses base + n*beacon power, battery_wh, drain_scale, and predicted SOC.
  - Mission $f_{mission}$: target uncertainty (target_unc_xy/3d) from posterior approx $P_{post}$ or GDOP*σ_r proxy.
- Score: $score = w_{obs} f_{obs} + w_{energy} f_{energy} + w_{mission} f_{mission}$.
- Weights come from mission_phase or phase_schedule:
  - survey: (0.6, 0.2, 0.2)
  - cruise: (0.5, 0.3, 0.2)
  - transit: (0.3, 0.5, 0.2)
  - low_power: (0.2, 0.7, 0.1)
- Zero-beacon gating: only if allow_zero_beacons and $\sqrt{\mathrm{trace}(P)} \leq off\_unc\_mult \cdot target\_unc$.
- Hysteresis: min_dwell_sec, score_margin, and power_save_tol (switch to fewer beacons if nearly tied).
- SOC is updated each step using the chosen subset’s power draw.

### V2 policy (AdaptiveModemManagerV2)
- Uses AdaptiveModemManagerV2 from adaptive_modem_manager_v2.py.
- Objective: posterior uncertainty + size penalty + rank deficit penalty + energy penalty.
- Allows 0..4 beacons and supports uncertainty gating using off_unc_mult.
- Energy-aware: penalties grow when SOC is below v2_low_power_soc (scaled by v2_energy_mult, v2_size_penalty_mult, v2_rank_deficit_mult).
- Dwell and hysteresis: min_dwell_sec (converted to steps) and switch_margin.
- If covariance is missing, initializes P with target_unc to keep gating stable.

#### AdaptiveModemManagerV2 — detailed working
Inputs per decision:
- a_pos: AUV position (x,y,z)
- P: covariance (full EKF covariance with position in the first 3 states, or a 3x3 Ppos)
- available_ids: beacons currently available (dropouts handled here)
- depth_available: selects XY vs 3D geometry scoring

Core steps:
1) Uncertainty gating
  - Compute an uncertainty scalar from P (trace(P) or sqrt(trace(Ppos))).
  - If uncertainty is below off-threshold, acoustics can be gated off.
  - If above on-threshold, acoustics are enabled.
  - Hysteresis is controlled by off_unc_mult/on_unc_mult and gate_min_dwell_steps.

2) Subset enumeration
  - Enumerates all subsets between min_subset_size and max_subset_size (including 0 if allowed).
  - Computes rank, GDOP, FIM log-det, CRLB std for XY and 3D.

3) Objective scoring (lower is better)
  - Predict posterior trace using linearized update:
    P+ = P - P H^T (H P H^T + R)^{-1} H P
  - Score = trace(P+) + size_penalty * (#beacons) + rank_deficit_penalty * deficit + energy_penalty
  - Rank deficit penalizes subsets below rank 2 (XY) or rank 3 (3D).
  - Energy penalty uses a normalized power model (base + per-beacon draw).
  - When SOC <= low_power_soc, energy/size/rank penalties increase via low-power multipliers.

4) Switching hysteresis
  - If min_dwell_steps has not elapsed, hold the current subset.
  - Otherwise, switch only if best candidate improves by switch_margin.

Outputs:
- Selected IDs and SelectionMetrics (rank_xy/3d, gdop_xy/3d, fim_logdet_xy/3d, crlb_std, score, reason, soc).
- Reasons include: battery_depleted, uncertainty_gated_off, dwell_hold, switched, held_margin.

Energy model notes:
- Enabled when battery_wh > 0 and energy_weight > 0.
- If SOC <= soc_min, acoustics are forced off.

## Selector integration details
- The selector is wrapped to:
  - Print a line on first selection and on every change: [selector] t=... active [prev] -> [new].
  - Emit SOC warnings when SOC crosses low_power_soc or soc_min.
- The wrapper returns a payload dict (active_names/active_set, mode, ranks, GDOPs, n_active, soc) compatible with run_single_trial(...).

## Runner compatibility
- Preferred path: current_acoustic_EKF_patched.run_single_trial(...), which supports live switching.
- Fallback path: if run_single_trial is not available, it will call run_ekf_acoustics(...) with fixed targets and warn that mid-run switching is not supported.

## CLI reference
### Run controls
- --outdir: output root directory (default results_modem_switching).
- --duration: simulation duration in seconds.
- --seed: random seed.
- --traj: trajectory type (spiral, figure8, lawnmower, concentric).
- --currents: enable currents if supported by the runner.
- --make-plots: request EKF diagnostic plots from the runner.
- --show-plots: display plots after saving (blocks).

### Mode selection
- --mode: policy or manual.
- --policy-type: gdop, weighted, or v2 (only when mode=policy).

### Target selection + dropouts
- --targets: beacon names to consider (default usv1 usv2 usv3 usv4).
- --dropout: time windows like usv2:30-45 (repeatable).

### Geometry and observability
- --sigma-r: range noise std for geometry scoring.
- --gdop-xy / --gdop-3d: thresholds for feasibility.
- --rank-req-xy / --rank-req-3d: minimum rank requirement (weighted policy).
- --min-beacons-xy / --min-beacons-3d: minimum subset size for XY/3D.
- --allow-one-beacon: shortcut to set min-beacons-xy=1.
- --allow-zero-beacons: allow 0-beacon subset (with gating in weighted/v2).

### Hysteresis and size penalties
- --min-dwell-sec: minimum time between switches.
- --switch-margin: objective improvement required (GDOP policy + v2).
- --score-margin: score improvement required (weighted policy).
- --power-save-tol: allow fewer beacons if score is within tolerance (weighted policy).
- --size-penalty: per-beacon penalty in GDOP policy (and v2 if configured).

### Energy model (weighted + v2)
- --battery-wh: battery capacity in Wh.
- --base-drain-w: baseline power drain (W).
- --beacon-drain-w: per-beacon power draw (W).
- --drain-scale: scaling from power to SOC decrement.
- --soc-init / --soc-min: initial and minimum SOC.
- --low-power-soc: threshold for low_power phase in weighted policy.

### Mission targeting (weighted + v2 gating)
- --target-unc-xy / --target-unc-3d: target uncertainty for mission term.
- --off-unc-mult: allow turning acoustics off only if uncertainty is sufficiently low.
- --target-unc: compatibility flag to set both xy and 3d targets.
- --mission-phase: fixed phase when no schedule is provided.
- --phase-schedule: time-varying phases, e.g. survey:0-60,cruise:60-120.

### V2 low-power tuning
- --v2-low-power-soc: SOC threshold for low-power behavior.
- --v2-energy-mult: energy weight multiplier under low SOC.
- --v2-size-penalty-mult: size penalty multiplier under low SOC.
- --v2-rank-deficit-penalty: base rank deficit penalty.
- --v2-rank-deficit-mult: rank deficit multiplier under low SOC.

### Compatibility knobs
- --energy-weight: scales the weighted policy energy term and v2 energy penalty.

## Outputs
- outdir/tag/config.json: run configuration, CLI args, and overrides.
- outdir/tag/selector_meta.json: per-decision log (t, selected, rank, GDOP, score, SOC, phase).
- outdir/tag/timeseries.csv: merged EKF timeseries + selector meta (active_count/set, GDOP, scores, SOC, pos_err, etc.).
- outdir/tag/summary.json: rmse_pos and final_pos_err (if provided by runner).
- Quick figures: fig_active_count.png, fig_active_set.png, fig_pos_err.png, fig_gdop.png, fig_score_vs_time.png, fig_soc_vs_time.png.
- outdir/tag/fig_paths.txt: list of all saved figure paths.

Plot fallbacks:
- If the EKF timeseries is missing or malformed, the script will still attempt selector-only plots from selector_meta_log.
- If no plots were created, a minimal n_active plot is attempted to avoid empty outputs.

## Example runs

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
