# FIM / GDOP Geometry Runner (Complete Guide)

This script (`fim_gdop_runner.py`) evaluates acoustic beacon geometry along a trajectory and produces paper‑ready metrics and plots: FIM log‑det, GDOP, CRLB, rank, and best‑subset tracking. It is intended for geometry/observability analysis and does not run the EKF.

## What happens inside `fim_gdop_runner.py`

At a high level, the runner does the following:

1. **Load trajectory**
   - If `--data` is provided, it loads a trajectory from NPZ or CSV.
   - Otherwise, it generates a default 3D spiral using `trajectory.py`.

2. **Load beacon positions**
   - If `--beacons-csv` is provided, it reads x,y,z from that file.
   - Otherwise, it uses the default 4‑beacon layout.
   - If an NPZ input includes `beacon_positions`, those are used (static or time‑varying).

3. **Optional decimation**
   - Trajectory samples can be decimated with `--decimate` to reduce compute and smooth output.

4. **Enumerate subsets**
   - For each time step, the script evaluates every beacon subset of size 2 and 3 (and optionally more).
   - For each subset, it builds the range Jacobian `H` and (if needed) the horizontal Jacobian `H_xy`.

5. **Compute geometry metrics**
   - FIM: $F = H^T R^{-1} H$
   - Rank, log‑det($F$)
   - GDOP: $\sqrt{\mathrm{trace}(F^{-1})}$
   - CRLB standard deviations: $\sqrt{\mathrm{diag}(F^{-1})}$

6. **Track "best" subsets**
   - If `--mode logdet`, choose the subset with maximum log‑det.
   - If `--mode gdop`, choose the subset with minimum GDOP.
   - For 2D (XY) mode, only subsets with rank ≥ 2 are eligible.

7. **Write outputs**
   - NPZ time series with metrics for all/beacons and best‑subset tracking
   - CSV summaries for 3D and 2D
   - Plots: log‑det, GDOP, rank, CRLB vs time
   - A short output README with file descriptions

## Key metrics and interpretation

- **Rank(H)**: observability. 3D needs rank 3, 2D (xy) needs rank 2.
- **FIM log‑det**: larger is better (more information volume).
- **GDOP**: lower is better. `inf` means singular or ill‑conditioned.
- **CRLB stds**: sqrt of diagonal of $F^{-1}$; lower is better.

## Inputs

### `--data` (NPZ or CSV)

- **NPZ** keys:
  - `trajectory_xyz` (T,3)
  - `beacon_positions` (N,3) or (T,N,3)

- **CSV**:
  - First three columns are trajectory x,y,z
  - Beacon positions fall back to defaults unless an NPZ includes them

### `--beacons-csv`

CSV with columns x,y,z (header optional). Overrides defaults.

## Outputs (written under `--out`)

- `metrics_timeseries.npz` — all‑beacon metrics, best subsets, optional per‑config metrics
- `summary.csv` — 3D stats (GDOP min/median/max, mean rank, invertible %)
- `summary_2d.csv` — 2D stats (observable %, GDOP_xy median/p95, logdet_xy)
- `best_subset_timeseries.csv` — best‑2/3 GDOP/logdet/rank
- `best_subset_timeseries_2d.csv` — 2D best subsets with sigma_x/y
- Plots:
  - `fig_fim_logdet_vs_time.png`
  - `fig_gdop_vs_time.png`
  - `fig_rank_vs_time.png`
  - `fig_crlb_pos_std_vs_time.png`
  - 2D counterparts (`*_xy_*`)
- `README_geometry.md` — auto‑generated output summary

## CLI usage

Basic run:
```bash
/usr/bin/python fim_gdop_runner.py \
  --out results_fim_gdop \
  --sigma-r 0.1 \
  --decimate 10 \
  --mode gdop
```

Use log‑det selection instead of GDOP:
```bash
/usr/bin/python fim_gdop_runner.py \
  --out results_fim_gdop_logdet \
  --sigma-r 0.1 \
  --decimate 5 \
  --mode logdet
```

Use a custom trajectory CSV and beacon layout:
```bash
/usr/bin/python fim_gdop_runner.py \
  --out results_fim_gdop_custom \
  --data traj.csv \
  --beacons-csv beacon_positions.csv \
  --sigma-r 0.25 \
  --mode gdop
```

Save all per‑configuration time series (larger output):
```bash
/usr/bin/python fim_gdop_runner.py \
  --out results_fim_gdop_full \
  --save-all-configs \
  --decimate 2
```

## Recommendations

- Prefer `--mode logdet` for information‑volume optimality; use `--mode gdop` for DOP‑style reporting.
- 2‑beacon 3D subsets are typically rank‑deficient; expect large/infinite GDOP. Use 3‑beacon subsets for 3D.
- Use `--decimate 1–10` for smooth plots; larger values reduce compute at the cost of fidelity.

## Dependencies

- Python
- numpy
- matplotlib (Agg backend, no display required)

## Where the logic lives

- Geometry + metrics: `sbl_geometry.py` (`SBLConfigurationAnalyzer`)
- Trajectory generation: `trajectory.py`
- CLI / I/O / plotting: `fim_gdop_runner.py`

## Known issues and validation notes

### Bug fixed (2026-03-31): `AttributeError` in `write_best_subset_csv_2d()`

**Root cause:** In `write_best_subset_csv_2d()` (previously lines 368–369), the code accessed `.size` directly on `rec.crlb_diags_xy[idx]`, assuming it was always a numpy array. However, `SBLConfigurationAnalyzer.analyze_trajectory()` stores `crlb_diags_xy` as a `List[np.ndarray]` by default — only converting to a 2D numpy array when `all_configs_metrics=True` (triggered by `--save-all-configs`). Without the flag, `rec.crlb_diags_xy[idx]` returns a plain Python list, which has no `.size` attribute, causing an `AttributeError` on every normal run.

**Fix:** The row element is now wrapped with `np.asarray()` before accessing `.size`, making the code robust regardless of whether `--save-all-configs` is set.

**Impact:** All runs without `--save-all-configs` were broken for the 2D best-subset CSV output. This affected the default code path.

### Data type behaviour to be aware of

`config_records` values (`ConfigMetrics`) hold list fields by default. When `--save-all-configs` is passed, `analyze_trajectory()` converts them to numpy arrays in-place. Functions that consume `config_records` fields use `np.asarray()` wrappers so they work in both modes.

### Default beacon geometry

The built-in 4-beacon layout spans only ~15 m in x and ~10 m in y at z=0 / z=−10 m:

```
[  0.0, -660.0,   0.0]
[ 15.0, -660.0,   0.0]
[ 10.0, -650.0,   0.0]
[  0.0, -650.0, -10.0]
```

This is a very tight, nearly co-planar cluster. The default spiral trajectory (center 200, −200; radii 20–50 m; z from −5 to −150 m) is far from this cluster, so expect near-singular geometry (GDOP → ∞, rank < 3) in default runs. Use `--beacons-csv` to supply a realistic deployment.
