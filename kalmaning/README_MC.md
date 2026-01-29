# HoloOcean EKF Monte Carlo Harness (Detailed Guide)

This guide explains what happens inside the Monte Carlo runner, how data flows through the EKF, and how to run experiments using the YAML configuration workflow.

## What the Monte Carlo runner does (high-level)

`monte_carlo_runner.py` repeatedly runs the EKF simulation under controlled random seeds and aggregates performance metrics.

**Core loop:**
1. Load configuration from YAML (`mc_config.yaml`) and apply minimal CLI overrides (`--outdir`, `--duration`, `--runs`).
2. Generate common random streams per seed (initial state perturbations and sensor noise) so all algorithms share the same randomness.
3. Run each algorithm for every seed, collecting time-series and summary metrics.
4. Write per-run outputs (CSV, JSON) and per-experiment plots/tables (CDFs, boxplots, CI bands, energy/accuracy trade-offs).
5. Print “main findings” with RMSE, NEES/NIS coverage, energy savings, and consistency notes.

## Algorithms compared

The runner evaluates three baseline variants plus adaptive modem selection:

- `imu_dvl_depth` — no acoustics (DVL+depth only).
- `imu_dvl_depth_all4` — acoustics always on, all 4 beacons.
- `adaptive` — uses a modem selector; policy configured by `adaptive_policy`.

Adaptive policies (choose one in YAML or run explicitly):
- `gdop` — geometry-based (rank/GDOP + dwell/margin).
- `weighted` — mission-aware multi-objective scoring.
- `v2` — uncertainty + energy-aware selector with SOC gating.

You can also run `adaptive_gdop`, `adaptive_weighted`, or `adaptive_v2` explicitly in `algorithms`.

## Data flow (detailed)

**Entry point**: `run_mc_new(args)`

- Reads configuration (YAML + CLI).
- For each `seed` and `algorithm`:
  - Builds a per-run `config_overrides` dict for the EKF.
  - Calls `run_single_trial(...)` in `current_acoustic_EKF_patched.py`.
  - Collects:
    - RMSE and final error
    - NEES/NIS coverage
    - time-series (positions, covariance, NIS logs, active beacons, GDOP)
    - energy usage estimates
    - selector metadata (for adaptive policies)

**Outputs** are written under the run’s `--outdir`:

- `configs/config.json` — resolved settings for traceability.
- `trials/seed_####/<algorithm>/timeseries.csv` — per-step time-series.
- `trials/seed_####/<algorithm>/trial_summary.json` — per-run summary.
- `tables/per_run_metrics.csv` — row per run with summary metrics.
- `tables/summary_metrics.csv` — aggregated algorithm summaries.
- `tables/policy_summary.csv` — policy-level summary including CI, energy savings.
- `figures/*.png` — plots (CDFs, boxplots, CI bands, energy, GDOP, etc.).

## Trajectory handling

The EKF uses `trajectory.py` to generate waypoint paths (spiral, lawnmower, concentric, figure8). The Monte Carlo runner passes trajectory choice via EKF overrides. The saved plot
`figures/traj_xy.png` shows the ground-truth path once per run.

## Consistency metrics (NEES/NIS)

- **NEES (state)**: compares estimation error vs covariance. Expected bounds derived from chi-square distribution.
- **NIS (measurement)**: compares innovation vs measurement covariance.

The runner reports coverage percentages and plots mean series with 95% CI bands.

## Energy and switching metrics

The adaptive policies track:
- active beacon count vs time
- estimated SOC and energy usage
- switch counts

These are saved to tables and plotted as CI bands.

## Configuration workflow (YAML first)

Only three CLI flags remain:

- `--outdir` (output folder)
- `--duration` (seconds)
- `--runs` (number of runs)

All other settings are loaded from `mc_config.yaml`.

### Example config: `mc_config.yaml`

Key settings include:
- `algorithms` and `adaptive_policy`
- `seeds` (null means auto range)
- sensor noise and random stream stds
- energy model (`battery_wh`, `base_drain_w`, `beacon_drain_w`)
- adaptive policy parameters (`min_dwell_sec`, `switch_margin`, etc.)

## How to run (examples)

### 1) Basic run (YAML config only)
```bash
/usr/bin/python monte_carlo_runner.py --outdir results_monte_carlo/spiral_T10_N5 --duration 10 --runs 5
```

### 2) Explicit adaptive policies in one run
```bash
/usr/bin/python monte_carlo_runner.py --outdir results_monte_carlo/spiral_T30_N5 --duration 30 --runs 5
```
In `mc_config.yaml`, set:
```yaml
algorithms:
  - adaptive_gdop
  - adaptive_weighted
  - adaptive_v2
```

### 3) Monte Carlo with fixed seeds
```yaml
seeds: [0, 1, 2, 3, 4]
```
```bash
/usr/bin/python monte_carlo_runner.py --outdir results_monte_carlo/spiral_fixed --duration 20 --runs 5
```

## Where to find results

After a run, check:
- `figures/` for plots (CDFs, CI bands, energy/accuracy, GDOP, FIM logdet)
- `tables/` for CSV/MD summaries
- `trials/` for per-seed timeseries and metadata

## Single-run EKF (for debugging)

```bash
/usr/bin/python current_acoustic_EKF_patched.py --trajectory spiral --target-name usv1 --target-name usv2
```

## Tips

- Use small durations for quick iteration; increase for publication-quality results.
- If you want more switching in v2, reduce `min_dwell_sec` and `switch_margin`, and increase `energy_weight`.
- Keep `seeds` explicit for reproducibility in comparisons.
