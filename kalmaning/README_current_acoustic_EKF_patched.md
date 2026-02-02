# current_acoustic_EKF_patched.py — Complete Guide

This script runs a single HoloOcean EKF simulation that fuses IMU, DVL, depth, and acoustic ranges. It supports multiple acoustic beacons, optional modem selection, and produces diagnostics such as RMSE, NEES/NIS, and plots when run directly.

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

- **Main loop (one EKF step = one env.step)**
  1. Send a 6‑DOF target waypoint to the vehicle.
  2. Read IMU, DVL, depth, pose, and location sensors.
  3. Perform the **EKF predict** step using IMU acceleration.
  4. Conditionally update with:
     - DVL velocity
     - Depth measurement
     - Acoustic ranges (when responses arrive)
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

## Key configuration fields (used in `config_overrides`)

You can pass these via `run_single_trial(..., config_overrides=...)`:

- `trajectory`: `spiral`, `lawnmower`, `concentric`, `figure8`
- `duration_sec`: simulation length
- `enable_dvl`, `enable_depth`, `enable_acoustic`
- `use_all_acoustic`: enable all beacons
- `modem_selector_fn`: adaptive selector callback
- `acoustic_period_ticks`: acoustic request rate
- `dvl_measurement_std`, `depth_measurement_std`, `acoustic_measurement_std`
- `imu_accel_extra_std`, `imu_bias_rw_std`, `dvl_extra_std`, `depth_extra_std`, `range_extra_std`
- `q_pos_std`, `q_vel_std`, `p_pos_std_init`, `p_vel_std_init`

## Notes and tips

- The EKF uses **one `env.step()` per iteration** for time consistency.
- Acoustic responses are processed asynchronously in the main loop.
- Use `return_timeseries=True` for Monte Carlo analysis and plotting.
- If you use adaptive modem selection, ensure beacons are named consistently (`usv1..usv4`).
- No `holoocean_uuid` wiring is needed; the environment manages IDs internally.

## Dependencies

- Python
- numpy, matplotlib
- HoloOcean (with the scenario `usv_auv_100_imu`)

## Where to look for related logic

- EKF math: `kalman_utils.py`
- Trajectories: `trajectory.py`
- Consistency metrics: `validation_metrics.py`
- Modem selection policies: `modem_switching_validation_fixed.py`, `adaptive_modem_manager_v2.py`
