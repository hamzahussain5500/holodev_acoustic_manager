# Monte Carlo Results

- trajectory: spiral
- currents: disabled
- duration: 600.0s
- runs: 40
- seeds: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39
- algorithms: imu_dvl_depth, imu_dvl_depth_all4, adaptive

## Figures
- final_error_cdf_overlay.pdf: CDF of final position error across algorithms.
- rmse_boxplot_overlay.pdf: RMSE distribution across algorithms.
- energy_boxplot.pdf: Acoustic energy distribution across algorithms.
- energy_vs_error.pdf: Energy vs final error scatter.
- mean_nees_pos.pdf: Mean NEES over time with 95% bounds.
- mean_nis_acoustic.pdf: Mean acoustic NIS over time (when applicable).
- pos_error_vs_time_ci.pdf: Position error vs time with 95% CI.
- nees_vs_time_ci.pdf: NEES vs time with 95% CI and bounds.
- nis_acoustic_vs_time_ci.pdf: Acoustic NIS vs time with 95% CI and bounds.
- active_count_vs_time_ci.pdf: Active beacon count vs time with 95% CI.
- soc_vs_time_ci.pdf: SOC vs time with 95% CI.
- energy_vs_time_ci.pdf: Energy vs time with 95% CI.
- gdop_vs_time_ci.pdf: GDOP vs time (adaptive policies).
- fim_logdet_vs_time_ci.pdf: FIM logdet vs time (adaptive policies).
- rmse_vs_crlb.pdf: RMSE vs CRLB efficiency scatter.
- active_count_hist_adaptive.pdf: % time in each active-beacon count.
- switches_hist_adaptive.pdf: Switching events per run.

## Metrics to cite
- mean/median final position error
- position RMSE distribution
- energy_Wh savings vs baselines
- switching statistics (mean switches/run, % time at k beacons)
- policy_summary.csv: RMSE/CRLB/consistency/energy summary per policy.
