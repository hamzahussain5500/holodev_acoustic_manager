# HoloOcean EKF + Acoustic Ranging Workspace

This workspace implements an Extended Kalman Filter (EKF) for an AUV in HoloOcean, fusing IMU, DVL, Depth, and Acoustic range measurements. It includes a reusable acoustic ranging utility and five comparable experiment runs to quantify the effect of different sensor combinations.

## Repository Layout
- `kalmaning/current_acoustic_EKF_patched.py`: Main EKF runner, keyboard control, plotting, and scenario orchestration. Compares five configurations.
- `kalmaning/kalman_utils.py`: Shared EKF utilities (EKF class, RMSE helper).
- `kalmaning/four_ranges.py`: Importable acoustic ranging utility (`collect_acoustic_ranges`) + robust `safe_tick`.
- Other top-level scripts (e.g., `examples.py`, `manual.py`) are unrelated demos.

## Scenario Requirements
The code assumes a HoloOcean scenario named `usv_auv` with:
- AUV agent name `auv` with Acoustic Beacon ID `0`.
- USV agents:
  - `usv1` with Acoustic Beacon ID `1`
  - `usv2` with Acoustic Beacon ID `2`
- Sensors on AUV: `IMUSensor`, `DVLSensor`, `DepthSensor`, `PoseSensor`, `AcousticBeaconSensor`.
- Sensors on USVs: `PoseSensor` or `LocationSensor` (for beacon position), `AcousticBeaconSensor`.

Update `SCENARIO_NAME`, agent names, and beacon IDs in `kalmaning/current_acoustic_EKF_patched.py` if your scenario differs.

## Dependencies
- Python 3.9+
- `holoocean` (environment, agents, sensors)
- `numpy`, `matplotlib`, `pynput`

Install:
```bash
pip install numpy matplotlib pynput
# Install HoloOcean per official docs (varies by platform)
```

## How It Works
### EKF State and Updates (`kalman_utils.py`)
- State: 6D `[x, y, z, vx, vy, vz]`.
- Predict: converts IMU body acceleration to world frame, subtracts gravity, integrates motion.
- Linear updates:
  - DVL: body-frame velocity → world-frame via pose rotation; updates `[vx, vy, vz]`.
  - Depth: updates `z`.
- Nonlinear update:
  - Acoustic range: scalar distance to a known beacon position; standard EKF range measurement update.
- `compute_rmse(true, est)` reports total and per-axis RMSE for positions/velocities.

### Acoustic Ranging and Scheduling (`current_acoustic_EKF_patched.py`)
- Periodic non-blocking ranging via an internal round-robin scheduler; sends `MSG_REQX` only when modems are idle.
- Single-tick discipline: exactly one `env.tick()` per EKF iteration; acoustic send/poll does not call `tick()` internally.
- On `MSG_RESPX`, parses range and fuses with `ekf.update_range(range_m, beacon_pos, R)` using the USV’s current position.
- Optional utility `kalmaning/four_ranges.py` remains available for standalone ranging helpers, but the main script uses its own scheduler.

### Currents Model (`current_acoustic_EKF_patched.py`)
- `vortex_field(location)`: XY swirl field used as example ocean current.
- `apply_currents(env, state, clock)`: applies currents to vehicles listed in `VEHICLES_FOR_CURRENTS` (default AUV only). Draws a debug vector field once.

### Keyboard Control (`current_acoustic_EKF_patched.py`)
- `i/k`: forward/backward
- `j/l`: yaw
- `w/s`: vertical thrust
- `a/d`: roll

## Experiments (`current_acoustic_EKF_patched.py`)
`main()` runs five configurations via `run_ekf_with_control`:
1. IMU-only
2. IMU + DVL
3. IMU + DVL + Depth
4. IMU + DVL + Depth + Acoustic_1 (one beacon)
5. IMU + DVL + Depth + Acoustic_2 (two beacons)

Acoustic flags and blocks:
- `use_acoustic_update_1`: Collects a single range (`num_targets=1`) from `USV_BEACON_ID` (usv1).
- `use_acoustic_update_2`: Collects two ranges (`num_targets=2`) from `[USV_BEACON_ID, USV2_BEACON_ID]` (usv1 + usv2), with `max_wait_seconds` and `cooldown_ticks` tuned to reduce overlap.

Outputs:
- Printed metrics (RMSE, final position error) per configuration.
- Plots: XY trajectory + uncertainty ellipses; Z vs time; Position error vs time; 3D uncertainty “bubble” per configuration.

## Running the Experiments
From workspace root:
```bash
python3 kalmaning/current_acoustic_EKF_patched.py
```
The script disables HoloOcean viewport (`show_viewport=False`), relies on keyboard inputs, and shows plots at the end.

## CLI Usage
- Run default:
```bash
python3 kalmaning/current_acoustic_EKF_patched.py
```
- Target specific USVs (repeatable):
```bash
python3 kalmaning/current_acoustic_EKF_patched.py --target-name usv1 --target-name usv3
```
- Verbose modem/ranging logs:
```bash
python3 kalmaning/current_acoustic_EKF_patched.py --verbose
```

## Configuration
- Scenario and agents: `SCENARIO_NAME`, `AUV_NAME`, `USV_1_NAME`..`USV_4_NAME`.
- Acoustic IDs and scheduling: `AUV_BEACON_ID`, `USV_BEACON_ID_1..4`, `ACOUSTIC_UPDATE_PERIOD_TICKS`, `timeout_ticks`.
- Noise levels: process `Q_POS_STD`, `Q_VEL_STD`; initial covariance `P_POS_STD_INIT`, `P_VEL_STD_INIT`; DVL `DVL_VEL_STD`; Depth `DEPTH_STD`; Acoustic `ACOUSTIC_RANGE_STD`.
- Currents: toggle `USE_CURRENTS`, vehicles `VEHICLES_FOR_CURRENTS`, field parameters in `vortex_field()`.

## Inputs & Outputs
- Inputs (HoloOcean sensors): `IMUSensor`, `DVLSensor`, `DepthSensor`, `PoseSensor`, `LocationSensor`, `AcousticBeaconSensor`.
- Console outputs: position/velocity RMSE (total and per-axis), final position error.
- Figures: XY with 95% covariance ellipses, Z vs time, position error vs time, final 3D uncertainty ellipsoid.

## Troubleshooting
- No beacons: if selected USV beacons aren’t present in the scenario, acoustic fusion is auto-disabled; verify `env.beacons_id`.
- Modem busy: sends are skipped unless both modems are idle; use `--verbose` to inspect statuses and timeouts.
- Time sync: do not add `env.tick()` calls inside acoustic logic; the scheduler adheres to single-tick updates.
- Currents: disable `USE_CURRENTS` for an unforced trajectory if needed.

## Future Work
- CLI flags: expose configuration toggles (enable/disable DVL/Depth/Acoustic, set periods, waits).
- Multi-beacon scheduling: formalize a scheduler to interleave targets without overlap; add diagnostics on acoustic channel state.
- Data logging: export CSV of measurements, states, and covariances for offline analysis.
- Sensor models: tune noise parameters and include IMU bias estimation.
- Triangulation: with two ranges, optionally perform multi-lateration or batch updates in the EKF for stronger constraints.
- Tests: add unit and integration tests for `collect_acoustic_ranges` timing and EKF update correctness.

## Contribution Notes
- Keep changes minimal and focused; follow current style.
- When adding sensors or beacons, update mappings in `current_acoustic_EKF_patched.py` and scenario.
- Prefer modifying `collect_acoustic_ranges` for messaging/timing improvements rather than duplicating logic.

---
If you want, we can implement the staggered scheduling and pre-ping idle in code to fully suppress the “still transmitting” warning.
