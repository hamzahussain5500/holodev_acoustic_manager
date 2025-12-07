# holodev

This repository contains examples and experiments using HoloOcean (AUV/USV simulator) with
EKF-based sensor fusion and small utility scripts. The main code for filtering and
experiments lives in the `klamaning/` folder.

**Project highlights**
- **EKF experiments:** Multi-sensor EKF fusion (IMU, DVL, Depth, Acoustic) with visualization.
- **Interactive control:** Keyboard thruster control examples for manually driving agents in-sim.
- **Utilities & examples:** Helper functions for uncertainty visualization and various HoloOcean examples.

**Contents**
- **`klamaning/`**: main experiments and utilities
	- `currents.py` — full EKF demo comparing 4 sensor configurations (IMU / IMU+DVL / IMU+DVL+Depth / IMU+DVL+Depth+Acoustic). Produces trajectory plots and uncertainty visualizations.
	- `compare.py` — simpler IMU vs IMU+DVL comparison with RMSE and plots.
	- `imu.py` — small example that reads IMU, logs acceleration history and saves `accel_history.csv` + `accel_plot.png`.
	- `uncertainty_utils.py` — helpers for covariance ellipses/ellipsoids used by plotting code.
	- `ekf.ipynb`, `imu_dvl.ipynb` — notebooks for interactive analysis.
- Top-level scripts: quick examples and scenarios
	- `examples.py` — assorted HoloOcean usage examples and dynamics demos.
	- `currents.py`, `kalmaning.py`, `manual.py`, `try.py` — small demos for currents, keyboard control, acoustic messages.
	- `four_ranges.py`, `accel_history.csv` — other utilities and sample data.

**Requirements**
- **Python:** 3.8+ recommended
- **Key packages:** `holoocean`, `numpy`, `matplotlib`, `pynput`.
- Install (example):

```bash
pip install numpy matplotlib pynput
# Install HoloOcean per its docs (may be via pip or local package/setup specific to your environment)
```

Note: `holoocean` is an external simulator — follow its installation and scenario setup instructions before running the examples.

**Quick start**
- Run the main EKF comparison (full demo):

```bash
python klamaning/currents.py
```

- Run the lighter comparison (IMU vs IMU+DVL):

```bash
python klamaning/compare.py
```

- Collect IMU acceleration history + plot:

```bash
python klamaning/imu.py
# produces `accel_history.csv` and `accel_plot.png`
```

**Keyboard controls (common mapping)**
- **Forward / Back:** `i` / `k` (applies forward/back thrust)
- **Yaw left / right:** `j` / `l`
- **Up / Down (vertical):** `w` / `s`
- **Roll left / right:** `a` / `d`
- **Quit example loops:** typically `q` (varies by script)

**Configuration tips**
- Many scripts use the `SCENARIO_NAME` or hard-coded scenario like `blue_rov` or `usv_auv`. Edit the top of the script if your scenario name differs.
- Noise / filter parameters are tunable constants near the top of the EKF scripts (`Q_*`, `P_*`, `*_STD` values).

**Outputs**
- `accel_history.csv` — example saved in the repo (produced by `klamaning/imu.py` when run).
- Plots — scripts call `matplotlib.pyplot.show()` and may save figures (e.g., `accel_plot.png`).

**Notebooks**
- `klamaning/ekf.ipynb` and `klamaning/imu_dvl.ipynb` are available for interactive exploration and visualizing results.

**Contributing / next steps**
- If you want, I can:
	- run a smoke test (if `holoocean` is installed here),
	- add a `requirements.txt` or `pyproject.toml`,
	- standardize scenario configuration into a single config file.

If you'd like any of those, tell me which and I will proceed.
