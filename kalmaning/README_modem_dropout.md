# Modem Dropout Test — Complete Guide

`modem_dropout_test.py` is a focused diagnostic runner that stresses the EKF under **planned modem outages**. It runs a **single controlled trial**, applies a dropout schedule across beacons, and produces plots and summaries that highlight how the estimator behaves under partial or complete acoustic loss.

---

## What the script does (step‑by‑step)

1. **Parse CLI arguments**
   - Duration, trajectory type, seed
   - Targets (which modems are considered)
   - Dropout schedule (manual string or built‑in overlap schedule)

2. **Build dropout windows**
   - Manual input like `usv2:30-60;usv3:80-110` is parsed into per‑modem time spans.
   - `--use-default-overlaps` generates a schedule with:
     - early pairwise overlaps
     - mid‑run rolling triple overlaps
     - short full‑outage window (all modems down)

3. **Run one EKF trial**
   - Calls `run_single_trial(...)` from `current_acoustic_EKF_patched.py`.
   - The trial is executed with the configured trajectory and targets.
   - Time‑series data (truth, estimate, covariance, NEES/NIS, gating decisions) is returned.

4. **Post‑process metrics**
   - Position error statistics (RMSE, median, p90/p95, max)
   - Uncertainty proxy: $\sqrt{\mathrm{trace}(P_{pos})}$
   - NEES and NIS summaries
   - Acoustic gating counts (used vs skipped)
   - Uptime/downtime per modem

5. **Write outputs**
   - CSV/JSON summaries
   - Plots with dropout shading and phase‑wise medians

---

## Key arguments

- `--duration-sec` (float): simulation length (seconds)
- `--trajectory`: `lawnmower | spiral | concentric | figure8`
- `--targets`: list of beacons to include (default: all `usv1..usv4`)
- `--seed`: RNG seed
- `--dropout`: manual schedule, e.g. `"usv2:30-60;usv3:80-110"`
- `--use-default-overlaps`: use built‑in schedule with overlapping outages

**Important:** If `--use-default-overlaps` is set, `--dropout` is ignored.

---

## Output structure

Results are saved under:

```
results_modem_dropout/<timestamp>_seed<seed>/
```

### Files

- `summary.json`
  - Seed, runtime, RMSEs, final error
  - Dropout intervals and total disabled time per modem
  - NEES/NIS summaries
- `timeseries.csv`
  - Truth/estimate positions
  - Error norms
  - Covariance diagonals
  - NEES/NIS series
  - Modem enabled/disabled flags

### Plots (in `plots/`)

Each plot is saved as `.png`, `.pdf`, and `.svg`.

- `pos_err_vs_time.png`
  - position error time series with 10 s rolling median
  - per-modem dropout swimlanes (one row per acoustic modem) below the main axis
- `position_error_main.png`
  - raw error and rolling median
  - dropout shading for usv2 and usv4 only + per‑phase medians
- `position_error_raw.png`
  - raw (downsampled) error with dropout shading
- `position_error_cdf.png`
  - empirical CDF of position error split by phase (baseline / dropout / overlap)
- `uncertainty_vs_time.png`
  - $\sqrt{\mathrm{trace}(P_{pos})}$ over time with dropout swimlanes
- `nis_acoustic_vs_time.png`
  - acoustic NIS time series with 95% chi-squared bounds
  - gated (skipped) measurements shown as scatter
  - per-modem dropout shading
- `xy_traj.png`
  - ground truth vs EKF estimate (XY plane)

---

## Terminal output (summary metrics)

At the end of the run the script prints:

- **Position error**: RMSE, median, p90, p95, max
- **Uncertainty**: median/p90/p95 of $\sqrt{\mathrm{trace}(P_{pos})}$
- **NEES** (pos and full): mean/median/p90/p95 and % inside 95% bounds
- **NIS** (acoustic used vs skipped)
- **Uptime/downtime** per modem

---

## Example usage

### 1) Built‑in overlap schedule

```bash
/usr/bin/python modem_dropout_test.py --duration-sec 900 --use-default-overlaps
```

### 2) Manual schedule (quote required)

```bash
/usr/bin/python modem_dropout_test.py \
  --duration-sec 900 \
  --targets usv2 usv4 \
  --dropout "usv2:200-400;usv4:350-550"
```

### 3) Short sanity run

```bash
/usr/bin/python modem_dropout_test.py --duration-sec 120 --seed 0 --trajectory spiral
```

---

## Notes and best practices

- Quote the `--dropout` string so the shell doesn’t split on semicolons.
- Use short durations for quick debugging; scale up for publication‑quality plots.
- Compare `position_error_main.png` vs `uncertainty_vs_time.png` to see whether error grows when modems are down.
- If you want to isolate a single modem’s impact, disable the others via `--targets` and drop only one.

---

## Where the logic lives

- EKF simulation: `current_acoustic_EKF_patched.py` (`run_single_trial`)
- Trajectories: `trajectory.py`
- Consistency metrics: `validation_metrics.py`
- Dropout parsing and plotting: `modem_dropout_test.py`

---

## Validation results (2026-03-31)

Tested with `--seed 0 --duration-sec 60`, three configurations: no dropout, manual schedule `usv2:10-40;usv4:20-50`, and `--use-default-overlaps`.

| Config | RMSE [m] | median err [m] | NEES(pos) mean/dof | Gating used% |
|---|---|---|---|---|
| No dropout | 0.372 | 0.308 | 2.88 / 3 = 0.96 | 100% |
| usv2+usv4 dropout | 0.301 | 0.289 | 2.57 / 3 = 0.86 | 76% |
| Default overlaps | 0.367 | 0.324 | 2.85 / 3 = 0.95 | 62% |

NEES(pos) mean/dof close to 1.0 indicates a well-calibrated filter. All runs completed without errors.

---

## Known issues fixed (2026-03-31)

### Bug 1: `plot_pos_err()` and `plot_nis()` were never called from `main()`

Both functions were fully implemented but never invoked, so `pos_err_vs_time.png` and `nis_acoustic_vs_time.png` were silently absent from every run. Fixed by adding the calls to `main()`.

### Bug 2: Dead assignment in `build_phase_masks()` (line 166)

The line `overlap = m1 | m2 | m3 | m4` was immediately overwritten by the pairwise-AND expression on the next line. The first assignment was dead code. Removed to prevent confusion about what "overlap" means (it correctly means two or more modems simultaneously down, not any modem down).
