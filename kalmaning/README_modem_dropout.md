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

- `position_error_main.png`
  - raw error and rolling median
  - dropout shading + per‑phase medians
- `position_error_raw.png`
  - raw error with dropout shading
- `position_error_cdf.png`
  - CDF of error by phase
- `uncertainty_vs_time.png`
  - $\sqrt{\mathrm{trace}(P_{pos})}$ over time with dropouts
- `xy_traj.png`
  - ground truth vs EKF estimate (XY)

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
