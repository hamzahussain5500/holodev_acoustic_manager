# Monte Carlo Results

- trajectory: spiral
- currents: disabled
- duration: 60.0s
- runs: 5
- seeds: 0,1,2,3,4
- algorithms: imu_dvl_depth, imu_dvl_depth_all4, adaptive

## Figures
- final_error_cdf_overlay.png: CDF of final position error across algorithms.
- rmse_boxplot_overlay.png: RMSE distribution across algorithms.
- energy_boxplot.png: Acoustic energy distribution across algorithms.
- energy_vs_error.png: Energy vs final error scatter.
- mean_nees_pos.png: Mean NEES over time with 95% bounds.
- mean_nis_acoustic.png: Mean acoustic NIS over time (when applicable).
- pos_error_vs_time_ci.png: Position error vs time with 95% CI.
- nees_vs_time_ci.png: NEES vs time with 95% CI and bounds.
- nis_acoustic_vs_time_ci.png: Acoustic NIS vs time with 95% CI and bounds.
- active_count_vs_time_ci.png: Active beacon count vs time with 95% CI.
- soc_vs_time_ci.png: SOC vs time with 95% CI.
- energy_vs_time_ci.png: Energy vs time with 95% CI.
- gdop_vs_time_ci.png: GDOP vs time (adaptive policies).
- fim_logdet_vs_time_ci.png: FIM logdet vs time (adaptive policies).
- rmse_vs_crlb.png: RMSE vs CRLB efficiency scatter.
- active_count_hist_adaptive.png: % time in each active-beacon count.
- switches_hist_adaptive.png: Switching events per run.

## Metrics to cite
- mean/median final position error
- position RMSE distribution
- energy_Wh savings vs baselines
- switching statistics (mean switches/run, % time at k beacons)
- policy_summary.csv: RMSE/CRLB/consistency/energy summary per policy.
