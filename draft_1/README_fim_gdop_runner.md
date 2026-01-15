# FIM / GDOP geometry runner

This script (`fim_gdop_runner.py`) evaluates acoustic beacon geometries over a trajectory and produces paper-ready metrics (FIM log-det, GDOP, CRLB, ranks) plus plots and CSV/NPZ artifacts.

## What it does
- Loads a trajectory (synthetic spiral by default, or provided via NPZ/CSV).
- Loads beacon positions (default 4-beacon static layout, or from `--beacons-csv`, or from NPZ).
- For every trajectory sample (optionally decimated) and every subset size 2 and 3:
  - Builds the range Jacobian `H` (and `H_xy` for horizontal-only).
  - Computes FIM `F = H^T R^-1 H` (and `F_xy`).
  - Derives rank, log-det(`F`), GDOP (sqrt(trace(F^-1))), and CRLB diagonals.
  - Tracks best subsets according to the selection mode.
- Writes metrics to NPZ and summary CSVs; renders plots; prints headline stats to the terminal.

## Key metrics and interpretation
- **Rank(H)**: observability. 3D needs rank 3; 2D (xy) needs rank 2.
- **FIM log-det**: larger is better geometry (tighter information volume).
- **GDOP**: lower is better. `inf` means singular/ill-conditioned.
- **CRLB stds**: sqrt of the diagonal of F^-1; lower means tighter bounds.

## "Best" subset selection
- Controlled by `--mode`:
  - `logdet`: pick subset with maximum log-det(F) (or log-det(F_xy) in 2D).
  - `gdop`: pick subset with minimum GDOP (or GDOP_xy in 2D).
- Eligibility:
  - 3D best-2 / best-3: all subsets are considered; rank may be <3 for 2-beacon 3D (expect GDOP to blow up).
  - 2D best-2 / best-3: subsets must be observable (rank(H_xy) >= 2) to be considered.
- Stored in the result dict as `best_2`, `best_3`, `best_2_xy`, `best_3_xy` (each carries idx, gdop, logdet, rank over time).

## Inputs
- `--beacons-csv`: CSV with columns x,y,z (header optional). Overrides defaults.
- `--data`: NPZ or CSV.
  - NPZ keys: `trajectory_xyz` (T,3), `beacon_positions` (N,3) or (T,N,3).
  - CSV: first three columns are trajectory x,y,z; beacons fall back to default unless NPZ is used.
- Default trajectory: 3D spiral (600 points, truncated to `--decimate` cadence).
- Default beacons: fixed 4-beacon layout from user-provided coordinates.

## Outputs (written to --out)
- `metrics_timeseries.npz`: all-beacon metrics, best subsets, optional per-config time series (when `--save-all-configs`).
- `summary.csv`: per-config 3D stats (min/median/max GDOP, mean rank, invertibility pct).
- `summary_2d.csv`: per-config 2D stats (observable fraction, median/p95 GDOP_xy, logdet_xy, mean sigma_x/y, invertibility pct).
- `best_subset_timeseries.csv`: time-indexed best-2/3 (3D) GDOP/logdet/rank.
- `best_subset_timeseries_2d.csv`: same for 2D with positions and sigma_x/y.
- Plots: `fig_fim_logdet_vs_time.png`, `fig_gdop_vs_time.png`, `fig_rank_vs_time.png`, `fig_crlb_pos_std_vs_time.png`, and 2D counterparts (`*_xy_*`).
- `README_geometry.md`: brief description of outputs (auto-written).
- Terminal: printed headline stats for quick reporting.

## CLI
```
python fim_gdop_runner.py \
  --out results_fim_gdop \
  --sigma-r 0.1 \
  --decimate 10 \
  --mode gdop        # or logdet
  [--configs-max-r N]
  [--save-all-configs]
  [--data data.npz | traj.csv]
  [--beacons-csv beacon_positions.csv]
```

## Notes and recommendations
- Use `--mode logdet` when you prefer volume-based optimality; `--mode gdop` when you prefer dilution-of-precision directly.
- 2-beacon 3D is generally rank-deficient; expect huge GDOP. Use 3-beacon (or all) for 3D or switch to 2D mode when depth is known.
- `--decimate` reduces compute; keep small (1-10) for smoother time series.
- `--save-all-configs` produces large NPZ files; enable only when you need per-config time series for post-analysis.

## Dependencies
- Python, numpy, matplotlib. No display needed (Agg backend).

## Where logic lives
- Analyzer and metric computation: `sbl_geometry.py` (`SBLConfigurationAnalyzer`).
- Trajectory construction: `trajectory.py` (spiral helper).
- CLI and I/O glue: `fim_gdop_runner.py` (this runner).
