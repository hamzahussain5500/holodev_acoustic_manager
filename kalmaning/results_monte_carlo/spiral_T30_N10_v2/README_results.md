# Monte Carlo Results

- trajectory: spiral
- currents: disabled
- duration: 30.0s
- runs: 10
- seeds: 0,1,2,3,4,5,6,7,8,9
- algorithms: imu_dvl_depth, imu_dvl_depth_all4, adaptive

## Figures
- final_error_cdf_overlay.png: CDF of final position error across algorithms.
- rmse_boxplot_overlay.png: RMSE distribution across algorithms.
- energy_boxplot.png: Acoustic energy distribution across algorithms.
- energy_vs_error.png: Energy vs final error scatter.
- mean_nees_pos.png: Mean NEES over time with 95% bounds.
- mean_nis_acoustic.png: Mean acoustic NIS over time (when applicable).
- active_count_hist_adaptive.png: % time in each active-beacon count.
- switches_hist_adaptive.png: Switching events per run.

## Metrics to cite
- mean/median final position error
- position RMSE distribution
- energy_Wh savings vs baselines
- switching statistics (mean switches/run, % time at k beacons)
