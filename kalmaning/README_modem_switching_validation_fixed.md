# Modem Switching Validation — Full Reference

Harness: `modem_switching_validation_fixed.py`  
Comparative runner: `run_comparative_analysis.sh`  
V2 policy module: `adaptive_modem_manager_v2.py`  
EKF backend: `current_acoustic_EKF_patched.py`

---

## Table of Contents

1. [Purpose](#purpose)
2. [Architecture and high-level flow](#architecture-and-high-level-flow)
3. [Algorithm methodology](#algorithm-methodology)
   - [EKF state model](#ekf-state-model)
   - [Energy model](#energy-model)
   - [GDOP policy](#gdop-policy-geometrypolicyselector)
   - [Weighted policy](#weighted-policy-weightedpolicyselector)
   - [V2 posterior covariance policy](#v2-posterior-covariance-policy-adaptivemodemmanagerv2)
4. [Bug fixes (2026-03-31)](#bug-fixes-2026-03-31)
5. [CLI reference](#cli-reference)
6. [Reproducing published results](#reproducing-published-results)
   - [Quick single-policy runs](#quick-single-policy-runs)
   - [Full comparative analysis](#full-comparative-analysis)
7. [Output files](#output-files)
8. [Tuning guide](#tuning-guide)
9. [Validated comparative results](#validated-comparative-results)

---

## Purpose

Validate acoustic modem subset switching by running an EKF localizer with a dynamic beacon selection policy at each time step. The harness feeds only the ranges from the currently active subset into the EKF, enabling fair comparison of four strategies:

- **Manual** — deterministic staged dropout (baseline)
- **GDOP** — geometry-based selection with energy-triggered reduction
- **Weighted** — multi-objective mission-phase-aware scoring
- **V2** — posterior covariance approximation with uncertainty gating

---

## Architecture and high-level flow

```
CLI args
   |
   v
build selector_fn  ←  dropout windows / GeometryPolicySelector
                                       / WeightedPolicySelector
                                       / AdaptiveModemManagerV2
   |
   v
selector_fn_wrapped  ← prints switch events, SOC warnings
   |
   v
run_single_trial(selector_fn_wrapped, ...)  ← current_acoustic_EKF_patched.py
   |
   v
save: config.json, selector_meta.json, timeseries.csv, summary.json, figures
```

`run_single_trial` calls `selector_fn_wrapped(t, ekf_state, depth_available, target_info, covariance)` at every EKF tick (100 Hz). The selector returns a payload with `active_names`, rank/GDOP metadata, and SOC. Only ranges from active beacons are fused.

---

## Algorithm methodology

### EKF state model

Six-state constant-velocity model: `x = [px, py, pz, vx, vy, vz]`.

Measurements fused asynchronously:
- IMU — process noise propagation
- DVL — velocity updates
- Depth — pz measurement
- Acoustic ranges — range from AUV position to each active beacon

Range measurement model:  
`z = ||p_auv - p_beacon|| + noise`, where `noise ~ N(0, σ_r²)`

The linearized Jacobian row for beacon `i`:  
`H_i = (p_auv - p_i) / ||p_auv - p_i||`  (3D) or `[:2]` (XY)

### Energy model

Used by all three policies (once energy parameters are provided):

```
P(n) = P_base + n × P_beacon          [W]
ΔSOC = drain_scale × P(n) × Δt / (battery_wh × 3600)
SOC ← max(soc_min, SOC - ΔSOC)
```

Parameters for the comparative run (`battery_wh=10 Wh, P_base=4 W, P_beacon=3 W, drain_scale=20`):
- 4 active beacons: P = 16 W → SOC drains from 1.0 to ~0.05 over 180 s
- 2 active beacons: P = 10 W → SOC drains from 1.0 to ~0.05 over ~360 s
- Energy multipliers kick in below `low_power_soc` threshold

### GDOP policy (GeometryPolicySelector)

**Objective** (lower is better):

```
obj(S) = GDOP(S) + eff_penalty × |S|

where  eff_penalty = size_penalty                         if SOC > low_power_soc
                   = size_penalty × energy_size_mult      if SOC ≤ low_power_soc
```

**Subset GDOP** from Fisher Information Matrix:

```
J_xy = Σ_{i∈S}  H_i^T H_i / σ_r²
GDOP_xy = sqrt( trace(J_xy^{-1}) )
```

**Selection steps:**

1. Enumerate all subsets of size `[min_beacons, max_beacons]`.
2. Mark a subset *feasible* if: `rank(J) ≥ rank_req` AND `GDOP ≤ gdop_thresh`.
3. Choose the minimum-obj feasible subset; fall back to minimum-obj nonzero if none are feasible.
4. Hysteresis: hold current subset if `t - last_switch_t < min_dwell_sec`.
5. Switch if `obj_candidate < obj_current - switch_margin` (improved) OR if candidate has fewer beacons and `obj_candidate - obj_current ≤ switch_margin` (power save).

**Energy-triggered reduction:**  
When `SOC ≤ low_power_soc`, `eff_penalty = size_penalty × energy_size_mult` (default 3×). The increased penalty makes smaller subsets more competitive, eventually driving a switch.

### Weighted policy (WeightedPolicySelector)

**Score** (higher is better):

```
score(S) = w_obs × f_obs(S)  +  w_energy × f_energy(S)  +  w_mission × f_mission(S)
```

**Observability term** — normalized logdet of FIM:

```
f_obs(S) = ( logdet(J(S)) - logdet_min ) / ( logdet_max - logdet_min )
```

**Energy term** — predicted SOC after using subset `S` for one step, scaled by power saving:

```
energy_margin(S) = ( P(max_beacons) - P(|S|) ) / P(max_beacons)
f_energy(S) = clamp01( SOC_predicted × energy_margin )
```

**Mission term** — target uncertainty satisfaction:

```
u(S) = sqrt( trace( P+_{xy}(S) ) )          (uses posterior covariance approximation)
f_mission(S) = clamp01( 1 - |u(S) - target_unc_xy| / target_unc_xy )
```

**Phase weights** (w_obs, w_energy, w_mission):

| Phase | w_obs | w_energy | w_mission |
|-------|-------|----------|-----------|
| survey | 0.6 | 0.2 | 0.2 |
| cruise | 0.5 | 0.3 | 0.2 |
| transit | 0.3 | 0.5 | 0.2 |
| low_power | 0.2 | 0.7 | 0.1 |

Phase is determined by `phase_schedule` or forced to `low_power` when `SOC ≤ low_power_soc`.

**Switching hysteresis:**

- Hold if `t - last_switch_t < min_dwell_sec` (timer only resets on actual subset changes — not on every tick).
- Switch if `score_best ≥ score_current + score_margin` (improved).
- Also switch if candidate has fewer beacons and `score_best ≥ score_current - power_save_tol` (power save).

**Zero-beacon gating** (only when `--allow-zero-beacons`):  
Allow acoustics off only if `sqrt(trace(P_xy)) ≤ off_unc_mult × target_unc_xy`.

### V2 posterior covariance policy (AdaptiveModemManagerV2)

**Core objective** (lower is better):

```
score(S) = trace(P+_pos(S))  +  eff_size_pen × |S|
         + eff_rank_pen × rank_deficit(S)
         + energy_penalty(S)
```

**Posterior covariance approximation** (linearized Kalman update):

```
H  = jacobian of range measurements for subset S (rows: one per beacon)
R  = σ_r² × I
S_innov = H P H^T + R
P+ = P - P H^T S_innov^{-1} H P
trace(P+_pos) = trace( (P - K H P)[0:3, 0:3] )
```

This predicts how much the position covariance would shrink if subset `S` were used at the current step.

**Rank deficit penalty:**  
`rank_deficit(S) = max(0, rank_req - rank(J(S)))`  
Penalizes geometrically degenerate subsets (e.g., all beacons collinear).

**Energy penalty:**

```
energy_penalty(S) = energy_weight × P(|S|) / P(max_beacons)
```

**Low-power multipliers** (when `SOC ≤ low_power_soc`):

```
eff_size_pen      = size_penalty × size_penalty_low_power_mult
eff_rank_pen      = rank_deficit_penalty × rank_deficit_low_power_mult
eff_energy_weight = energy_weight × energy_weight_low_power_mult
```

**Uncertainty gating** (hysteresis):

```
off_threshold = off_unc_mult × target_unc_xy
on_threshold  = on_unc_mult  × target_unc_xy   (default: 1.25 × off_threshold)

If acoustics ON  and sqrt(trace(P_xy)) < off_threshold  →  gate OFF
If acoustics OFF and sqrt(trace(P_xy)) > on_threshold   →  gate ON
```

This allows the policy to completely disable acoustic updates when the EKF position estimate is already well-converged, saving energy, and re-enable them when the estimate drifts.

**Switching hysteresis:**

- Hold if `_dwell < min_dwell_steps` (step counter, not time — called at 100 Hz, so `dwell_steps = min_dwell_sec × 100`).
- Switch if `score_current - score_best ≥ switch_margin` (must genuinely improve).
- If `score_current - score_best < switch_margin`, hold (reason: `held_margin`).

**Initial state:**  
Pre-seeded with all beacons active so the first evaluation compares the full set against subsets, not against an empty set.

---

## Bug fixes (2026-03-31 → 2026-04-01)

Three bugs were found and fixed that prevented the policies from showing active switching.

### Bug 1 — V2: permanent dwell-hold lock

**File:** `adaptive_modem_manager_v2.py`, `select()` method, `held_margin` branch.

**Symptom:** Once the `held_margin` path fired (candidate did not improve enough to switch), `_dwell` was reset to 1, which is less than `min_dwell_steps`. On the very next call, the dwell check (`_dwell < min_dwell_steps`) fired again, resetting `_dwell` back to 1 — creating a permanent lock. The policy would never switch after the first held-margin event.

**Before:**
```python
# held_margin path:
self._dwell = 1   # BUG: causes dwell_hold on next call → infinite loop
```

**After:**
```python
# held_margin path:
self._dwell = self.min_dwell_steps  # correct: dwell is satisfied, re-evaluate next call
```

### Bug 2 — V2: initial 4→2 jump from empty seed

**File:** `adaptive_modem_manager_v2.py`, `__init__` and `reset()`.

**Symptom:** `_current_ids` started as `[]`. The first evaluation compared any nonzero subset against the empty set, whose "score" was `trace(P)` (large). With even a small energy weight, the 2-beacon subset had the lowest score (fewer active beacons = lower energy penalty), so V2 jumped immediately to 2 beacons regardless of geometry.

**Before:**
```python
self._current_ids: List[Union[str, int]] = []
```

**After:**
```python
self._current_ids: List[Union[str, int]] = list(self.beacon_positions.keys())
```

Same fix applied in `reset()`.

### Bug 3 — Weighted policy: dwell timer never expired

**File:** `modem_switching_validation_fixed.py`, `WeightedPolicySelector.decide()`.

**Symptom:** `last_switch_t = t` was assigned unconditionally at every call (not just on actual switches). Since the selector runs at 100 Hz, `t - last_switch_t ≈ 0.01 s`, which is always less than `min_dwell_sec`. The dwell check (`t - last_switch_t < min_dwell_sec`) always fired, blocking all switching. Only `battery_depleted` and `init` reasons ever fired.

**Before:**
```python
self.active_names = list(final_cand["names"])
self.last_switch_t = t   # BUG: always reset, dwell never expires
```

**After:**
```python
new_names = list(final_cand["names"])
if tuple(sorted(new_names)) != tuple(sorted(self.active_names)) or not self.active_names:
    self.last_switch_t = t   # only reset when the active set actually changes
self.active_names = new_names
```

### Enhancement — GDOP policy: energy-aware size penalty

**File:** `modem_switching_validation_fixed.py`, `GeometryPolicySelector`.

**Problem:** The GDOP policy had no energy model. With near-degenerate SBL geometry (18 m beacon cluster at ~450 m range), all subset sizes have nearly identical GDOP, so `size_penalty` alone was insufficient to drive switching. The policy would select the same subset throughout the entire run.

**Fix:** Added an `EnergyModel` to `GeometryPolicySelector`. When `SOC ≤ low_power_soc`, the effective size penalty is multiplied by `energy_size_mult` (default 3×), making smaller subsets more competitive and triggering an energy-driven switch.

Added fields to `PolicyParams`:
```python
battery_wh: float = 0.0        # enable energy model when > 0
base_drain_w: float = 0.0
beacon_drain_w: float = 0.0
drain_scale: float = 1.0
soc_init: float = 1.0
soc_min: float = 0.0
low_power_soc: float = 0.3     # SOC threshold for penalty scaling
energy_size_mult: float = 3.0  # size_penalty multiplier in low-power phase
```

Battery parameters are now forwarded from the CLI to `PolicyParams` when `--policy-type gdop` is used.

### Bug 5 — V2: rank deficit penalty incorrectly applied to 0-beacon subset

**File:** `adaptive_modem_manager_v2.py`, `select()` — dwell_hold score, candidate loop, and cur_score computation (3 locations).

**Symptom:** The 0-beacon (acoustics-off) subset has rank=0, so `deficit = max(0, target_rank - 0) = 2`. With `rank_deficit_penalty=5.0`, this adds `5.0 × 2 = 10.0` to the 0-beacon score, inflating it from ~1.72 to ~11.77. This made 0-beacon permanently non-competitive, blocking all acoustics-off states that the uncertainty gate would otherwise produce. After the gate itself fired, `held_margin` would immediately re-enable the prior beacon set because the 0-beacon "score" in the comparison was always ≫ the 2-beacon score.

**Before (all 3 locations):**
```python
deficit = max(0, target_rank - int(rank_now))
```

**After (all 3 locations):**
```python
# 0-beacon is intentional acoustics-off; exempt from rank deficit penalty
deficit = 0 if not subset else max(0, target_rank - int(rank_now))
```

### Bug 6 — V2: 0-beacon won immediately in subset scoring after rank deficit fix

**File:** `adaptive_modem_manager_v2.py`, `select()`.

**Symptom:** After fixing the rank deficit bug, the 0-beacon subset scored ~1.72 vs 2-beacon ~1.84 at any SOC. So 0-beacon won continuously in the scoring path, causing rapid on/off oscillation every ~15s (one dwell cycle).

**Fix:** Added `score_zero_below_soc` parameter. The 0-beacon subset is only included in candidate scoring when `SOC ≤ score_zero_below_soc`. Above that threshold, only the uncertainty gate can turn acoustics off. This separates two distinct mechanisms: gate-based off (precision-driven) and energy-based off (depletion-driven).

```python
# New parameter in __init__:
score_zero_below_soc: float = 0.0,

# In select(), candidate enumeration:
soc_now = self.energy.soc if self.energy is not None else 1.0
allow_zero_in_scoring = (self.min_subset_size == 0) and (soc_now <= self.score_zero_below_soc)
eff_min_size = 0 if allow_zero_in_scoring else max(1, self.min_subset_size)
candidates = self._enumerate_subsets_from(avail, eff_min_size)
```

### Bug 7 — V2: uncertainty gate fired immediately after subset switch

**File:** `adaptive_modem_manager_v2.py`, `select()`, switch branch.

**Symptom:** When the 4→2 switch fired at t~15s, the `_gate_dwell` counter had already accumulated 1500 steps (15s) from t=0 (the EKF converged early, so `unc_value < off_threshold` was true from the start). The `_gate_dwell` reset in the dwell_hold branch only fires during dwell-hold, not on subset switches. So after the switch, the gate immediately fired (0-step dwell) causing 2→0 within one second.

**Fix:** Reset `_gate_dwell = 0` in the switch branch, forcing the gate to wait a full `gate_min_dwell_steps` period before it can fire after any subset change.

```python
if (best_score + self.switch_margin) < cur_score:
    self._current_ids = list(best_subset)
    self._dwell = 0
    self._gate_dwell = 0  # NEW: prevent immediate re-gating after subset switch
    reason = "switched"
```

### Enhancement — Weighted policy: updated DEFAULT_PHASE_WEIGHTS for n=4 survey selection

**File:** `modem_switching_validation_fixed.py`, `DEFAULT_PHASE_WEIGHTS`.

**Problem:** With old survey weights `(0.6, 0.2, 0.2)`, the ratio `w_obs/w_energy = 3.0` was below the analytically-derived threshold of 4.2 needed for n=4 to beat n=2 in near-degenerate SBL geometry (where all subsets have nearly equal geometry). The policy would start at n=3 instead of n=4.

**Analysis:** In near-degenerate geometry, the energy margin advantage of n=2 over n=4 is `~0.375` (normalized), while the observation quality gap is only `~0.09`. For n=4 to win in survey phase: `w_obs × 0.09 > w_energy × 0.375`, requiring `w_obs/w_energy > 4.17`.

**Fix:** Updated survey weights to `(0.85, 0.08, 0.07)`, giving `w_obs/w_energy = 10.6 >> 4.2`. Also added `--v2-score-zero-below-soc` CLI argument.

```python
DEFAULT_PHASE_WEIGHTS = {
    "survey":    (0.85, 0.08, 0.07),  # w_obs/w_energy=10.6 → n=4 wins
    "cruise":    (0.60, 0.25, 0.15),
    "transit":   (0.25, 0.60, 0.15),  # energy-dominant → n=2 wins
    "low_power": (0.15, 0.75, 0.10),  # maximum energy conservation
}
```

---

## CLI reference

### Run controls

| Flag | Default | Description |
|------|---------|-------------|
| `--outdir` | `results_modem_switching` | Output root directory |
| `--duration` | — | Simulation duration (seconds) |
| `--seed` | — | Random seed |
| `--traj` | `spiral` | Trajectory: `spiral`, `figure8`, `lawnmower`, `concentric` |
| `--make-plots` | off | Request EKF diagnostic plots |
| `--show-plots` | off | Display plots interactively (blocks) |

### Mode selection

| Flag | Values | Description |
|------|--------|-------------|
| `--mode` | `policy`, `manual` | Selector mode |
| `--policy-type` | `gdop`, `weighted`, `v2` | Policy (when `--mode policy`) |

### Target selection and dropouts

| Flag | Description |
|------|-------------|
| `--targets` | Beacon names to consider (default: `usv1 usv2 usv3 usv4`) |
| `--dropout` | Dropout windows, e.g. `usv2:30-45` (repeatable) |

### Geometry and observability

| Flag | Default | Description |
|------|---------|-------------|
| `--sigma-r` | `0.5` | Range noise std (m), used in FIM/GDOP computation |
| `--gdop-xy` | `8.0` | XY GDOP feasibility threshold |
| `--gdop-3d` | `10.0` | 3D GDOP feasibility threshold |
| `--min-beacons-xy` | `2` | Minimum active beacons for XY mode |
| `--min-beacons-3d` | `3` | Minimum active beacons for 3D mode |
| `--allow-zero-beacons` | off | Allow 0-beacon (acoustics fully off) |

### Hysteresis and switching thresholds

| Flag | Default | Description |
|------|---------|-------------|
| `--min-dwell-sec` | `3.0` | Minimum time between switches (all policies) |
| `--switch-margin` | `0.05` | Objective improvement to switch (GDOP, V2) |
| `--score-margin` | `0.02` | Score improvement to switch (weighted) |
| `--power-save-tol` | `0.03` | Allow fewer-beacon switch within this score tolerance (weighted) |
| `--size-penalty` | `0.05` | Per-beacon penalty in GDOP objective |

### Energy model

| Flag | Default | Description |
|------|---------|-------------|
| `--battery-wh` | `100.0` | Battery capacity (Wh) |
| `--base-drain-w` | `0.0` | Baseline power draw (W) |
| `--beacon-drain-w` | `1.0` | Per-beacon power draw (W) |
| `--drain-scale` | `1.0` | Fractional SOC drain scaling factor |
| `--soc-init` | `1.0` | Initial SOC |
| `--soc-min` | `0.0` | Minimum SOC (forced off below this) |
| `--low-power-soc` | `0.3` | SOC threshold for low-power phase (weighted, GDOP) |

### Mission targeting (weighted + V2 gating)

| Flag | Default | Description |
|------|---------|-------------|
| `--target-unc-xy` | `0.5` | Target XY uncertainty (m) for mission term and gating |
| `--target-unc-3d` | `0.8` | Target 3D uncertainty (m) |
| `--off-unc-mult` | `0.7` | Gate off when `sqrt(trace(P_xy)) < mult × target_unc_xy` |
| `--mission-phase` | `cruise` | Fixed phase when no schedule is given |
| `--phase-schedule` | — | Time-varying phases, e.g. `survey:0-45,cruise:45-90,transit:90-135,low_power:135-180` |
| `--energy-weight` | — | Scale energy term weight (weighted) or energy penalty (V2) |

### V2 low-power tuning

| Flag | Default | Description |
|------|---------|-------------|
| `--v2-low-power-soc` | `0.5` | SOC threshold activating V2 low-power multipliers |
| `--v2-energy-mult` | `5.0` | Energy weight multiplier when SOC ≤ low-power threshold |
| `--v2-size-penalty-mult` | `1.0` | Size penalty multiplier in low-power phase |
| `--v2-rank-deficit-penalty` | `5.0` | Base rank deficit penalty |
| `--v2-rank-deficit-mult` | `0.5` | Rank deficit multiplier in low-power phase |
| `--v2-score-zero-below-soc` | `0.0` | Include 0-beacon in subset scoring only when SOC ≤ this value; otherwise only uncertainty gate can turn acoustics off |

---

## Reproducing published results

All results use: `--traj spiral --duration 180 --seed 0 --targets usv1 usv2 usv3 usv4`

Shared energy model calibration (SOC drains 1.0→~0.05 over 180 s with 2 active beacons):
```
--battery-wh 10.0 --base-drain-w 4.0 --beacon-drain-w 3.0 --drain-scale 20.0
--soc-init 1.0 --soc-min 0.05
```

### Quick single-policy runs

**1. Manual baseline — staged 4→3→2→1 dropout**

```bash
python3 modem_switching_validation_fixed.py \
  --outdir results_comparative/manual \
  --mode manual \
  --traj spiral --duration 180 --seed 0 \
  --targets usv1 usv2 usv3 usv4 \
  --dropout usv4:45-180 usv3:90-180 usv2:135-180 \
  --sigma-r 0.5 \
  --make-plots
```

Expected output:
```
[selector] t=0.01s  active=['usv1','usv2','usv3','usv4'] (initial)
[selector] t=45.00s active ['usv1','usv2','usv3','usv4'] -> ['usv1','usv2','usv3']
[selector] t=90.00s active ['usv1','usv2','usv3'] -> ['usv1','usv2']
[selector] t=135.00s active ['usv1','usv2'] -> ['usv1']
RMSE ≈ 1.200 m
```

---

**2. GDOP policy — geometry-based selection with energy-triggered reduction**

```bash
python3 modem_switching_validation_fixed.py \
  --outdir results_comparative/gdop \
  --mode policy --policy-type gdop \
  --traj spiral --duration 180 --seed 0 \
  --targets usv1 usv2 usv3 usv4 \
  --sigma-r 0.5 \
  --gdop-xy 30.0 --gdop-3d 40.0 \
  --min-beacons-xy 2 --min-beacons-3d 3 \
  --min-dwell-sec 15.0 \
  --switch-margin 2.0 \
  --size-penalty 0.5 \
  --battery-wh 10.0 --base-drain-w 4.0 --beacon-drain-w 3.0 \
  --drain-scale 20.0 --soc-init 1.0 --soc-min 0.05 \
  --low-power-soc 0.6 \
  --make-plots
```

Expected output:
```
[selector] t=0.01s  active=['usv1','usv2','usv4'] (initial)
[selector] t=55.40s active ['usv1','usv2','usv4'] -> ['usv2','usv4']
[WARN] t=55.40s SOC=0.600 below low_power threshold (0.600)
RMSE ≈ 0.962 m
```

The switch at t=55s is energy-triggered: when SOC drops to 0.6, the effective size penalty triples, making the 2-beacon subset the minimum-objective choice.

---

**3. Weighted policy — multi-objective with mission phase schedule**

```bash
python3 modem_switching_validation_fixed.py \
  --outdir results_comparative/weighted \
  --mode policy --policy-type weighted \
  --traj spiral --duration 180 --seed 0 \
  --targets usv1 usv2 usv3 usv4 \
  --sigma-r 0.5 \
  --gdop-xy 8.0 --gdop-3d 10.0 \
  --min-beacons-xy 2 --min-beacons-3d 3 \
  --min-dwell-sec 10.0 \
  --score-margin 0.03 \
  --power-save-tol 0.0 \
  --phase-schedule "survey:0-60,transit:60-120,low_power:120-180" \
  --target-unc-xy 0.5 --target-unc-3d 0.8 \
  --off-unc-mult 1.5 \
  --allow-zero-beacons \
  --battery-wh 10.0 --base-drain-w 4.0 --beacon-drain-w 3.0 \
  --drain-scale 20.0 --soc-init 1.0 --soc-min 0.05 \
  --low-power-soc 0.3 \
  --energy-weight 0.3 \
  --make-plots
```

Expected output:
```
[selector] t=0.01s   active=['usv1','usv2','usv3','usv4'] (initial)
[selector] t=60.00s  active ['usv1','usv2','usv3','usv4'] -> ['usv2','usv4']
[WARN]     t=135.02s SOC=0.050 below minimum threshold (0.050)
[selector] t=135.03s active ['usv2','usv4'] -> []
RMSE ≈ 1.02 m
```

Two switching events:
- **t=60s: 4→2** (survey→transit phase boundary; energy weight rises from 0.08 to 0.60, making 2-beacon the winning subset)
- **t=135s: 2→0** (battery fully depleted at SOC=soc_min)

**Parameter rationale:**
- Survey weights `(0.85, 0.08, 0.07)`: `w_obs/w_energy = 10.6 >> 4.2` threshold needed for n=4 to beat n=2 in near-degenerate geometry
- Transit weights `(0.25, 0.60, 0.15)`: energy-dominant; 2-beacon wins clearly
- `score_margin=0.03`: blocks minor score fluctuations from causing upward switches in low_power phase
- The 3-phase schedule (survey/transit/low_power) instead of 4-phase (survey/cruise/transit/low_power) prevents n=3 intermediate states; in near-degenerate geometry the n=3 window requires `2.88 < w_obs/w_energy < 2.97`, an impractically tight band

---

**4. V2 policy — posterior covariance approximation with uncertainty gating**

```bash
python3 modem_switching_validation_fixed.py \
  --outdir results_comparative/v2 \
  --mode policy --policy-type v2 \
  --traj spiral --duration 180 --seed 0 \
  --targets usv1 usv2 usv3 usv4 \
  --sigma-r 0.5 \
  --min-beacons-xy 0 --min-beacons-3d 0 \
  --allow-zero-beacons \
  --gdop-xy 8.0 --gdop-3d 10.0 \
  --min-dwell-sec 15.0 \
  --switch-margin 0.005 \
  --size-penalty 0.0 \
  --target-unc-xy 1.5 --target-unc-3d 2.0 \
  --off-unc-mult 0.55 \
  --battery-wh 10.0 --base-drain-w 4.0 --beacon-drain-w 3.0 \
  --drain-scale 20.0 --soc-init 1.0 --soc-min 0.05 \
  --energy-weight 0.195 \
  --v2-low-power-soc 0.45 \
  --v2-energy-mult 5.0 \
  --v2-size-penalty-mult 1.0 \
  --v2-rank-deficit-penalty 5.0 \
  --v2-rank-deficit-mult 0.5 \
  --v2-score-zero-below-soc 0.4 \
  --make-plots
```

Expected output:
```
[selector] t=0.01s   active=['usv1','usv2','usv3','usv4'] (initial)
[selector] t=15.01s  active ['usv1','usv2','usv3','usv4'] -> ['usv2','usv4']
[selector] t=99.00s  active ['usv2','usv4'] -> []
[selector] t=138.00s active [] -> ['usv2','usv4']
[selector] t=153.00s active ['usv2','usv4'] -> []
RMSE ≈ 1.29 m
```

Four switching events:
1. **t=15s: 4→2** (energy-driven: `energy_weight=0.195` drives 2-beacon after one dwell period; `min_dwell_sec=15s`)
2. **t=99s: 2→0** (uncertainty gate fires: EKF well-converged after prolonged 2-beacon operation; `sqrt(trace(P_xy)) < 0.55 × 1.5 = 0.825 m`)
3. **t=138s: 0→2** (uncertainty drifted above on-threshold `≈ 1.031 m`; EKF diverged while running open-loop)
4. **t=153s: 2→0** (energy-based off: `SOC ≤ score_zero_below_soc=0.4` admits 0-beacon into scoring; combined with `v2_low_power_soc=0.45` + `v2_energy_mult=5.0` raises energy penalty sharply)

**Parameter rationale for V2:**

The near-degenerate SBL geometry (18 m beacon cluster at ~450 m AUV range) means all subset sizes (2, 3, 4 beacons) have nearly equal `trace(P+_pos)` differences of only ~0.003 m². The effective differentiators are:

| Factor | Value | Effect |
|--------|-------|--------|
| `switch_margin=0.005` | ~5× smaller than `tr_post` diff | Allows geometry differences to trigger switches |
| `energy_weight=0.195` | Below 4→3 threshold (~0.198) | Energy drives 4→2 but not 4→3 (skips n=3) |
| `off_unc_mult=0.55` | off_thresh = 0.825 m | Gates off once EKF xy std drops well below target |
| `on_unc_mult` (auto) | on_thresh ≈ 1.031 m | Re-enables when estimate drifts |
| `score_zero_below_soc=0.4` | — | Separates gate-based off (precision) from energy-based off (depletion) |
| `gate_min_dwell_steps=1500` | = min_dwell_sec × 100 Hz | Prevents gate from firing immediately after a subset switch |

**How the three mechanisms interact:**

- **Energy mechanism** (t=15s): The 2-beacon score beats 4-beacon when `energy_weight × Δpower_norm > Δtr_post`, which happens after `min_dwell_sec=15s` of stable 4-beacon operation.
- **Uncertainty gate** (t=99s, t=138s): Runs on the EKF covariance directly. `_gate_dwell` must accumulate `gate_min_dwell_steps=1500` consecutive below-threshold steps before firing. After every subset switch, `_gate_dwell` resets to 0, so the gate cannot fire immediately.
- **Energy-based off** (t=153s): When `SOC ≤ 0.4`, the 0-beacon subset enters scoring. With `v2_energy_mult=5.0` active (SOC ≤ 0.45), the 0-beacon energy penalty (0.0) beats the 2-beacon penalty strongly enough to trigger a switch.

---

### Full comparative analysis

Run all four policies in one script with a final summary table:

```bash
cd kalmaning/
bash run_comparative_analysis.sh
```

Output directory structure:
```
results_comparative/
├── manual/   manual_spiral_seed0_T180/
├── gdop/     policy_spiral_seed0_T180/
├── weighted/ policy_spiral_seed0_T180/
└── v2/       policy_spiral_seed0_T180/
```

Each run directory contains: `config.json`, `selector_meta.json`, `timeseries.csv`, `summary.json`, and figures.

---

## Output files

| File | Contents |
|------|----------|
| `config.json` | Full CLI args, run tag, scenario parameters |
| `selector_meta.json` | Per-tick decision log: `t`, `selected`, `reason`, `rank`, `gdop`, `score`, `soc`, `phase` |
| `timeseries.csv` | Merged EKF + selector timeseries: `t`, `pos_err`, `active_count`, `active_set`, `gdop_xy`, `soc`, ... |
| `summary.json` | `rmse_pos`, `final_pos_err` |
| `fig_active_count.png` | Number of active beacons vs time |
| `fig_active_set.png` | Active beacon identities vs time (raster) |
| `fig_pos_err.png` | AUV position error vs time |
| `fig_score_vs_time.png` | Policy scores vs time (weighted/V2 only) |
| `fig_soc_vs_time.png` | Battery SOC vs time (policies with energy model) |
| `fig_gdop.png` | GDOP vs time (manual/GDOP modes) |
| `fig_paths.txt` | List of all saved figure paths |

---

## Tuning guide

**No switching occurs:**
- Check `min_dwell_sec` is not too large relative to run duration
- Reduce `switch_margin` / `score_margin` (default 0.05/0.02 — for near-degenerate geometry may need 0.005)
- For GDOP: verify battery parameters are set (`--battery-wh > 0`) and `low_power_soc` is reachable within the run duration
- For V2: check `off_unc_mult × target_unc_xy` is close to the actual EKF uncertainty; use `selector_meta.json` to inspect `soc` and `reason` fields

**Too much switching (chattering):**
- Increase `min_dwell_sec` (10–15 s recommended)
- Increase `score_margin` / `switch_margin`
- For weighted: increase `power_save_tol` to `0.05–0.1`

**V2 starts with 0 beacons:**
- `off_unc_mult × target_unc_xy` is too high — EKF uncertainty at start is already below the off-threshold
- Reduce `off_unc_mult` or increase `target_unc_xy` so the gate does not fire immediately
- Typical EKF XY std at initialization: 1–3 m; set `off_unc_mult × target_unc_xy` > 3 m to keep acoustics on at start

**GDOP policy holds one subset throughout:**
- Ensure battery parameters are provided and `drain_scale` is calibrated so SOC crosses `low_power_soc` during the run
- Increase `low_power_soc` (e.g., 0.6–0.7) so the threshold is hit earlier
- Verify `energy_size_mult` (PolicyParams default: 3.0) is large enough to flip the minimum-objective subset

**Weighted policy shows no phase-driven switches:**
- Verify `phase_schedule` spans the full run duration
- Confirm `energy_weight` scales the energy term enough (use `--energy-weight 0.3`)
- Check that `min_dwell_sec` is shorter than the phase duration (e.g., 8 s < 45 s phase length)

---

## Validated comparative results

Scenario: `--traj spiral --duration 180 --seed 0 --targets usv1 usv2 usv3 usv4`  
Geometry: 18 m SBL beacon cluster at ~450 m AUV range (near-degenerate)  
Energy: 10 Wh battery, P_base=4 W, P_beacon=3 W, drain_scale=20

| Policy | RMSE (m) | Switches | Active-set counts | Switch events |
|--------|----------|----------|-------------------|---------------|
| **Manual** | 0.863 | 3 | {1, 2, 3, 4} | t=45s: 4→3, t=90s: 3→2, t=135s: 2→1 |
| **GDOP** | 0.792 | 1 | {2, 3} | t=55s: 3→2 (energy, SOC=0.60) |
| **Weighted** | 1.017 | 2 | {0, 2, 4} | t=60s: 4→2 (phase), t=135s: 2→0 (battery) |
| **V2** | 1.287 | **4** | {0, 2, 4} | t=15s: 4→2 (energy), t=99s: 2→0 (gate), t=138s: 0→2 (drift), t=153s: 2→0 (energy) |

Validated scenario: `--traj spiral --duration 180 --seed 0`, 18 m SBL cluster at ~450 m range, 10 Wh battery.

**Key observations:**

- V2 demonstrates the richest switching behavior: 4 distinct events driven by 3 independent mechanisms (energy scoring, uncertainty gate, energy-based off at low SOC). It is the only policy that autonomously re-enables acoustics after a gate-off event.
- Weighted policy now starts at n=4 (survey phase, `w_obs/w_energy=10.6`) and cleanly transitions to n=2 at the survey→transit boundary (t=60s), then off at battery depletion.
- GDOP shows a single energy-triggered switch (3→2 at SOC=0.60) as designed.
- Manual baseline provides the staged-dropout control: deterministic 4→3→2→1 with no navigation feedback.
- Higher RMSE for V2 and Weighted reflects periods with 0 active beacons (EKF runs open-loop); Manual and GDOP keep at least 2 beacons active throughout.
- The uncertainty gate in V2 achieves genuine energy savings: acoustics are off for ~39s (t=99s to t=138s), with the EKF relying on DVL+IMU+depth only.

Figures saved under `results_comparative/<policy>/<run-tag>/`.
