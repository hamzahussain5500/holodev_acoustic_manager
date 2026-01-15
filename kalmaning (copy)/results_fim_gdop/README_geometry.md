Geometry outputs (3D and 2D horizontal-only)

- 3D metrics: rank(H), log det(F), GDOP, CRLB diag for full xyz.
- 2D metrics: rank(H_xy), log det(F_xy), GDOP_xy, CRLB_xy std (sigma_x, sigma_y).
- Best subsets: chosen independently for 3D and 2D (best 2, best 3 by log-det or GDOP policy).
- Why 2D: when depth is known from a Depth sensor, acoustics mainly constrain x–y, so 2 beacons can still provide good horizontal geometry.
- Observability: 2D observable if rank(H_xy) >= 2; otherwise GDOP_xy=inf and CRLB_xy is NaN.

Files:
- summary.csv (3D), summary_2d.csv (2D)
- best_subset_timeseries.csv (3D), best_subset_timeseries_2d.csv (2D)
- metrics_timeseries.npz (both 3D/2D arrays)
- fig_*png: logdet, GDOP, rank, CRLB for 3D and 2D.
