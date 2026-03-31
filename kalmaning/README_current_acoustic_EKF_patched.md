# current_acoustic_EKF_patched.py — Complete Guide

This script runs a single HoloOcean EKF simulation that fuses IMU, DVL, depth, and acoustic ranges. It supports multiple acoustic beacons, optional modem selection, and produces diagnostics such as RMSE, NEES/NIS, and plots when run directly.

## Validation summary — bugs found and fixes applied

Six issues were identified by running NEES/NIS consistency tests across multiple seeds and comparing EKF outputs against ground truth.

### 1. `DEPTH_HZ` was 50 Hz — should be 100 Hz
`DepthSensor` has no `Hz` field in `usv_auv_100_imu.json`, so HoloOcean defaults it to `ticks_per_sec = 100`. The code was scheduling depth updates at half the actual rate, causing missed measurements.
**Fix:** `DEPTH_HZ = 100`.

### 2. `Q_POS_STD` was 0.10 — should be 0.0
The EKF uses `Q = B·Qa·Bᵀ` (IMU acceleration drives process noise), which already injects position diffusion through the double-integration structure of `B`. Adding a separate `q_pos_std` term inflated `P` beyond what the model warranted, making the filter over-confident about uncertainty growth.
**Fix:** `Q_POS_STD = 0.0`.

### 3. `Q_VEL_STD` was 0.22 — tuned to 0.3 m/s²
`Q_VEL_STD` represents unmodelled vehicle manoeuvre uncertainty (not IMU sensor noise). A sweep showed `Q_VEL_STD = 0.22` gave `NEES_3D ≈ 0.6` (over-confident filter). Setting it to 0.30 gives `NEES_3D ≈ 1.0` for DVL + Depth alone — the correct baseline before acoustic updates.
**Fix:** `Q_VEL_STD = 0.3`.

### 4. `ACOUSTIC_RANGE_STD` was 0.03 m — set to 0.30 m
The JSON `DistanceSigma` fields (0.1 / 0.3) are **ignored** by HoloOcean's `AcousticBeaconSensor`. Empirical measurement showed actual range noise std ≈ 0.0001 m (near-zero). However, all 4 USV beacons cluster within 18 m at ≈ 450 m range, giving a near-rank-1 FIM with almost no lateral observability. Setting `R_acoustic` to the actual noise (0.03 m) caused `P` to collapse in poorly-observed directions while the true position error remained large, producing NEES blow-up (observed NEES_3D > 700 in runs with Q=0.03). A 0.30 m floor acts as geometry-informed regularisation.
**Fix:** `ACOUSTIC_RANGE_STD = 0.30`.

### 5. DVL R matrix was isotropic — made anisotropic
The original code set `R_dvl = diag([std², std², std²])` using a single `DVL_VEL_STD`. Empirical measurement showed XY-axis DVL noise std ≈ 0.44 m/s but Z-axis noise std ≈ 0.13 m/s (≈ 3× smaller). Using the same value for all three axes over-weighted Z velocity measurements (Z noise underestimated by 3×), causing incorrect velocity covariance.

The suggested fix of changing `DVL_VEL_STD` from 0.45 → 0.24 (based on the JSON `VelSigma: 0.24` beam value) was **incorrect** — the JSON value is per acoustic-beam, not per world-frame axis. HoloOcean's DVL outputs processed world-frame velocity, and empirical XY std was confirmed at ≈ 0.44–0.45 m/s.

**Fix:** `DVL_VEL_STD = 0.45` (XY, unchanged), `DVL_VEL_STD_Z = 0.13` (Z, new), `R_dvl = diag([0.45², 0.45², 0.13²])`.

### 6. Delayed acoustic update used wrong beacon position
When an acoustic response arrived with latency > 0 ticks, the code looked up the USV beacon position from the **current** tick's state — not the position at the time the ranging request was sent (`obs_tick`). If USVs are moving, this introduces a systematic range innovation error proportional to USV displacement during the round-trip.

**Fix:** The tick-history buffer now stores a `beacon_positions` snapshot at each tick. When a delayed response arrives, the beacon position is looked up at `obs_tick`. Falls back to current position if history is unavailable.

---

## What the script does (step‑by‑step)

### 1) Global configuration
- Defines sensor rates, noise parameters, scenario and agent names, and AUV/USV beacon IDs.
- Sets default simulation duration and target depth/yaw.
- Provides helper utilities to reset acoustic sensor state and query beacon IDs/statuses.

### 2) Acoustic scheduler (round‑robin)
- `AcousticRoundRobin` manages asynchronous acoustic ranging:
  - Only one request in flight at a time.
  - Sends a request every `period_ticks` if modems are idle.
  - Polls sensor messages for `MSG_RESPX` replies.
  - Tracks request/response timing (`request_tick`, `response_tick`, `latency_ticks`).
  - Avoids calling `env.tick()` inside the scheduler (keeps simulation time consistent).

### 3) Core EKF run: `run_ekf_acoustics(...)`

This function is the heart of the simulation:

- **Builds a runtime config** (merged with `sim_config`) including:
  - Sensor usage (DVL, depth, acoustic)
  - Trajectory parameters
  - Noise settings and optional extra noise
  - Modem selection function (adaptive switching)

- **Initializes the HoloOcean environment** with the chosen scenario.

- **Builds trajectory waypoints** using `trajectory.py` (`spiral`, `lawnmower`, `concentric`, `figure8`).

- **Main loop (one EKF heartbeat = one env.step)**
  1. Send a 6‑DOF target waypoint to the vehicle.
  2. Read IMU, DVL, depth, pose, and location sensors.
  3. Compute dynamic `dt_step` from simulation time and set `ekf.dt = dt_step`.
  4. Perform the **EKF predict** step using IMU acceleration.
  5. Asynchronous, multi-rate updates using per-sensor due scheduling:
    - DVL velocity update when DVL is due
    - Depth update when depth is due
    - Acoustic updates when responses arrive
  6. Acoustic latency handling:
    - `acoustic_latency_policy="skip"` drops delayed responses above threshold
    - `acoustic_latency_policy="apply_current"` applies on current state (no rewind)
    - `acoustic_latency_policy="rewind"` applies delayed acoustic via fixed-lag rewind/replay
  5. Log NEES/NIS and time‑series state.
  6. Handle waypoint switching when the AUV reaches a waypoint.

- **Modem selection (optional)**
  - If a selector function is provided (adaptive policy), it selects active beacons each tick.
  - Ranges from inactive beacons are ignored.

- **Returns time series** arrays for positions, covariance, NEES/NIS, active beacons, GDOP (if applicable), and mode.

### 4) Per‑run wrapper: `run_single_trial(...)`

- Sets random seed for reproducibility.
- Builds a base config and applies overrides.
- Calls `run_ekf_acoustics`.
- Computes metrics:
  - RMSE (position/velocity)
  - Final position error
  - NEES/NIS consistency tests
  - Total distance traveled (true/estimated)
- Optionally returns full time series for Monte Carlo aggregation.
- Returns a picklable config snapshot (callables stripped) so process-pool Monte Carlo runs work without pickle errors.

### 5) Plotting utilities

`plot_ekf_outputs(...)` generates:
- XY trajectory with covariance ellipses
- Z vs time
- Position error vs time
- NEES/NIS time series with chi‑square bounds
- Per‑axis 95% uncertainty
- 3D uncertainty ellipsoid

These plots are shown when running the script directly.

## How to run

### Basic run (with plots)

```bash
/usr/bin/python current_acoustic_EKF_patched.py --trajectory spiral --target-name usv1 --target-name usv2
```

### Enable verbose logging

```bash
/usr/bin/python current_acoustic_EKF_patched.py --trajectory spiral --verbose
```

### Headless run (no viewport)

```bash
/usr/bin/python current_acoustic_EKF_patched.py --trajectory spiral
```

### Show HoloOcean viewport

```bash
/usr/bin/python current_acoustic_EKF_patched.py --trajectory spiral --show-viewport
```

## Tuned noise parameters (empirically validated)

All values below were determined by running the simulator and measuring actual sensor outputs against ground truth. Do not change them without re-running the validation sweep.

| Constant | Value | How determined |
|---|---|---|
| `DEPTH_HZ` | `100` Hz | `DepthSensor` has no `Hz` field in `usv_auv_100_imu.json` — HoloOcean defaults to `ticks_per_sec` (100 Hz). Previously wrong at 50 Hz. |
| `Q_POS_STD` | `0.0` | Position random-walk is redundant: the IMU-driven `Q = B·Qa·Bᵀ` already provides position diffusion. Setting this to non-zero inflates `P` unnecessarily. |
| `Q_VEL_STD` | `0.3 m/s²` | Represents unmodelled vehicle manoeuvre uncertainty (not IMU noise). Tuned empirically: gives `NEES_3D ≈ 1.0` for DVL+Depth alone (no acoustics). |
| `DVL_VEL_STD` | `0.45 m/s` (XY) | Empirically measured from simulator: per-axis XY noise std ≈ 0.44 m/s. |
| `DVL_VEL_STD_Z` | `0.13 m/s` (Z) | Empirically measured: Z-axis DVL noise is ≈ 3× smaller than XY. `R_dvl` is now anisotropic: `diag([0.45², 0.45², 0.13²])`. |
| `DEPTH_STD` | `0.03 m` | Confirmed from simulator: matches `"Sigma": 0.03` in JSON. |
| `ACOUSTIC_RANGE_STD` | `0.30 m` | Simulator returns near-zero acoustic noise (std ≈ 0.0001 m). The 0.30 m floor is a geometry-informed regularisation: all 4 USV beacons cluster within 18 m at ≈ 450 m range from the AUV, giving a near-rank-1 FIM (poor lateral observability). Without this floor the filter collapses `P` in directions it cannot genuinely observe. Validated: gives `NEES_3D ≈ 3.0` (consistent) across 5 seeds. |

> **Important — `DistanceSigma` in JSON is ignored by HoloOcean acoustic sensor:**
> The `DistanceSigma` fields in `usv_auv_100_imu.json` (AUV: 0.1, USVs: 0.3) do not appear
> to inject noise into the range measurements returned by `AcousticBeaconSensor`.
> Measured range error std is ≈ 0.0001 m. Set `ACOUSTIC_RANGE_STD` based on your
> geometry/linearisation requirements, not the JSON sigma values.

## Key configuration fields (used in `config_overrides`)

You can pass these via `run_single_trial(..., config_overrides=...)`:

- `trajectory`: `spiral`, `lawnmower`, `concentric`, `figure8`
- `duration_sec`: simulation length
- `enable_dvl`, `enable_depth`, `enable_acoustic`
- `use_all_acoustic`: enable all beacons
- `modem_selector_fn`: adaptive selector callback
- `acoustic_period_ticks`: acoustic request rate (total across round-robin targets)
- `dvl_measurement_std`: DVL XY-axis velocity noise std (m/s), default `DVL_VEL_STD = 0.45`
- `dvl_measurement_std_z`: DVL Z-axis velocity noise std (m/s), default `DVL_VEL_STD_Z = 0.13`
- `depth_measurement_std`: depth sensor noise std (m), default `DEPTH_STD = 0.03`
- `acoustic_measurement_std`: acoustic range noise std (m), default `ACOUSTIC_RANGE_STD = 0.30`
- `imu_accel_extra_std`, `imu_bias_rw_std`, `dvl_extra_std`, `depth_extra_std`, `range_extra_std`
- `q_pos_std`, `q_vel_std`, `p_pos_std_init`, `p_vel_std_init`
- `ekf_mode`: `"asynchronous"` (documentary mode flag)
- `acoustic_latency_policy`: `"skip"`, `"apply_current"`, or `"rewind"`
- `acoustic_latency_max_sec`: max accepted acoustic age before skip (when policy is `skip`)
- `acoustic_rewind_buffer_sec`: fixed-lag history length used by rewind/replay

## Notes and tips

- The EKF uses **one `env.step()` per heartbeat** with dynamic `dt` for time consistency.
- DVL/depth are applied asynchronously via due-tick scheduling (multi-rate behavior).
- Acoustic responses are processed asynchronously in the main loop.
- **DVL R matrix is anisotropic**: Z-axis noise (0.13 m/s) is modelled separately from XY (0.45 m/s). Override with `dvl_measurement_std_z` if needed.
- **Delayed acoustic beacon position fix**: when an acoustic response arrives with latency > 0, the USV beacon position used for the range innovation is looked up from the tick-history buffer at `obs_tick` (the time the request was sent), not the current tick. This prevents a systematic error if USVs are moving. Falls back to current-tick position if history is unavailable.
- Delayed acoustic rewind/replay is implemented as a fixed-lag buffer over recent ticks.
- If delayed data is older than the retained buffer window, the implementation falls back to current-state update behavior.
- Use `return_timeseries=True` for Monte Carlo analysis and plotting.
- If you use adaptive modem selection, ensure beacons are named consistently (`usv1..usv4`).
- No `holoocean_uuid` wiring is needed; the environment manages IDs internally.

## Known structural limitation — beacon geometry

The 4 USV beacons in `usv_auv_100_imu.json` are positioned within an 18 m cluster at approximately 450 m from the AUV spawn point:

| Beacon | Position |
|---|---|
| usv1 | `[0, -660, 0]` |
| usv2 | `[15, -660, 0]` |
| usv3 | `[10, -650, 0]` |
| usv4 | `[0, -650, -10]` |

All four beacons subtend less than 3° of angle from the AUV. The FIM is near-rank-1 — acoustics can constrain range toward the cluster well but provide almost no lateral observability. DVL prevents velocity drift; acoustics correct range-direction position bias. Spreading the USVs further apart would be the single biggest improvement to positioning accuracy.

## Dependencies

- Python
- numpy, matplotlib
- HoloOcean (with the scenario `usv_auv_100_imu`)

## Where to look for related logic

- EKF math: `kalman_utils.py`
- Trajectories: `trajectory.py`
- Consistency metrics: `validation_metrics.py`
- Modem selection policies: `modem_switching_validation_fixed.py`, `adaptive_modem_manager_v2.py`
