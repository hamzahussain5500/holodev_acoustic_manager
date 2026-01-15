"""Geometry-only FIM/GDOP/CRLB analyzer for single-beacon localization (SBL).

Usage (standalone): see fim_gdop_runner.py for CLI integration.
"""
import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ConfigMetrics:
    cfg: Tuple[int, ...]
    ranks: List[int]
    gdops: List[float]
    logdets: List[float]
    invertible: List[bool]
    crlb_diags: List[np.ndarray]
    ranks_xy: List[int]
    gdops_xy: List[float]
    logdets_xy: List[float]
    invertible_xy: List[bool]
    crlb_diags_xy: List[np.ndarray]


class SBLConfigurationAnalyzer:
    """Evaluate observability/FIM/GDOP/CRLB for beacon subsets along a trajectory."""

    def __init__(self, rank_tol: float = 1e-10):
        self.rank_tol = rank_tol

    def generate_configurations(self, n: int, min_r: int = 2, max_r: Optional[int] = None) -> List[Tuple[int, ...]]:
        max_r = n if max_r is None else min(max_r, n)
        configs: List[Tuple[int, ...]] = []
        for r in range(min_r, max_r + 1):
            configs.extend(itertools.combinations(range(n), r))
        return configs

    def compute_H(self, target_pos: np.ndarray, beacon_pos_subset: np.ndarray) -> np.ndarray:
        target = np.asarray(target_pos).reshape(3,)
        beacons = np.asarray(beacon_pos_subset)
        H_rows = []
        for b in beacons:
            diff = target - b
            dist = np.linalg.norm(diff)
            if dist < 1e-9:
                H_rows.append(np.zeros(3))
            else:
                H_rows.append(diff / dist)
        return np.vstack(H_rows) if H_rows else np.zeros((0, 3))

    def jacobian_H(self, target_pos: np.ndarray, beacon_pos_subset: np.ndarray) -> np.ndarray:
        return self.compute_H(target_pos, beacon_pos_subset)

    def fim(self, H: np.ndarray, sigma_r: float) -> np.ndarray:
        if H.size == 0:
            return np.zeros((3, 3))
        inv_R = 1.0 / (sigma_r * sigma_r)
        return H.T @ (inv_R * H)

    def fim_xy(self, H_xy: np.ndarray, sigma_r: float) -> np.ndarray:
        if H_xy.size == 0:
            return np.zeros((2, 2))
        inv_R = 1.0 / (sigma_r * sigma_r)
        return H_xy.T @ (inv_R * H_xy)

    def rank_of_H(self, H: np.ndarray, tol: Optional[float] = None) -> int:
        tol = self.rank_tol if tol is None else tol
        if H.size == 0:
            return 0
        return int(np.linalg.matrix_rank(H, tol=tol))

    def gdop_from_F(self, F: np.ndarray) -> float:
        try:
            Finv = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            return math.inf
        if not np.isfinite(Finv).all():
            return math.inf
        tr = float(np.trace(Finv))
        if tr <= 0.0:
            return math.inf
        return float(math.sqrt(tr))

    def crlb_std_from_F(self, F: np.ndarray) -> Optional[np.ndarray]:
        try:
            Finv = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(Finv).all():
            return None
        diag = np.diag(Finv).copy()
        diag[diag < 0.0] = np.nan
        return np.sqrt(diag)

    def crlb_from_F(self, F: np.ndarray) -> Optional[np.ndarray]:
        try:
            Finv = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            return None
        return Finv

    def _logdet_F(self, F: np.ndarray) -> float:
        sign, ld = np.linalg.slogdet(F)
        if sign <= 0:
            return -math.inf
        return float(ld)

    def analyze_trajectory(
        self,
        trajectory_xyz: np.ndarray,
        beacon_positions: np.ndarray,
        sigma_r: float = 0.1,
        mode: str = "logdet",
        decimate: int = 1,
        min_r: int = 2,
        max_r: Optional[int] = None,
        all_configs_metrics: bool = False,
    ) -> Dict[str, any]:
        traj = np.asarray(trajectory_xyz)
        if traj.ndim != 2 or traj.shape[1] != 3:
            raise ValueError("trajectory_xyz must have shape (T,3)")
        beacons = np.asarray(beacon_positions)
        if beacons.ndim == 2:
            n_beacons = beacons.shape[0]
        elif beacons.ndim == 3:
            n_beacons = beacons.shape[1]
            if beacons.shape[0] < traj.shape[0]:
                raise ValueError("time-varying beacon_positions must have length >= trajectory")
        else:
            raise ValueError("beacon_positions must have shape (N,3) or (T,N,3)")

        configs = self.generate_configurations(n_beacons, min_r=min_r, max_r=max_r)

        t_idx: List[int] = []
        all_rank: List[int] = []
        all_logdet: List[float] = []
        all_gdop: List[float] = []
        all_crlb_diag: List[np.ndarray] = []
        all_rank_xy: List[int] = []
        all_logdet_xy: List[float] = []
        all_gdop_xy: List[float] = []
        all_crlb_diag_xy: List[np.ndarray] = []

        cfg_records: Dict[Tuple[int, ...], ConfigMetrics] = {
            cfg: ConfigMetrics(
                cfg=cfg,
                ranks=[], gdops=[], logdets=[], invertible=[], crlb_diags=[],
                ranks_xy=[], gdops_xy=[], logdets_xy=[], invertible_xy=[], crlb_diags_xy=[],
            )
            for cfg in configs
        }

        best2_idx: List[Optional[Tuple[int, ...]]] = []
        best2_gdop: List[float] = []
        best2_logdet: List[float] = []
        best2_rank: List[int] = []

        best2_idx_xy: List[Optional[Tuple[int, ...]]] = []
        best2_gdop_xy: List[float] = []
        best2_logdet_xy: List[float] = []
        best2_rank_xy: List[int] = []

        best3_idx: List[Optional[Tuple[int, ...]]] = []
        best3_gdop: List[float] = []
        best3_logdet: List[float] = []
        best3_rank: List[int] = []

        best3_idx_xy: List[Optional[Tuple[int, ...]]] = []
        best3_gdop_xy: List[float] = []
        best3_logdet_xy: List[float] = []
        best3_rank_xy: List[int] = []

        for k in range(0, traj.shape[0], max(1, int(decimate))):
            target = traj[k]
            beacons_k = beacons if beacons.ndim == 2 else beacons[k]
            t_idx.append(k)

            # All-beacon metrics
            H_all = self.jacobian_H(target, beacons_k)
            F_all = self.fim(H_all, sigma_r)
            rank_all = self.rank_of_H(H_all)
            logdet_all = self._logdet_F(F_all)
            gdop_all = self.gdop_from_F(F_all)
            crlb_all = self.crlb_from_F(F_all)

            H_all_xy = H_all[:, :2]
            F_all_xy = self.fim_xy(H_all_xy, sigma_r)
            rank_all_xy = self.rank_of_H(H_all_xy)
            logdet_all_xy = self._logdet_F(F_all_xy)
            gdop_all_xy = self.gdop_from_F(F_all_xy)
            crlb_all_xy = self.crlb_std_from_F(F_all_xy)

            all_rank.append(rank_all)
            all_logdet.append(logdet_all)
            all_gdop.append(gdop_all)
            all_crlb_diag.append(np.diag(crlb_all) if crlb_all is not None else np.full(3, np.nan))

            all_rank_xy.append(rank_all_xy)
            all_logdet_xy.append(logdet_all_xy)
            all_gdop_xy.append(gdop_all_xy)
            if crlb_all_xy is None or crlb_all_xy.size < 2:
                all_crlb_diag_xy.append(np.full(2, np.nan))
            else:
                all_crlb_diag_xy.append(crlb_all_xy[:2])

            # Per-configuration metrics
            for cfg in configs:
                subset = beacons_k[list(cfg)]
                H = self.jacobian_H(target, subset)
                F = self.fim(H, sigma_r)
                r = self.rank_of_H(H)
                ld = self._logdet_F(F)
                g = self.gdop_from_F(F)
                crlb = self.crlb_from_F(F)
                cfg_rec = cfg_records[cfg]
                cfg_rec.ranks.append(r)
                cfg_rec.logdets.append(ld)
                cfg_rec.gdops.append(g)
                cfg_rec.invertible.append(np.isfinite(g) and g != math.inf)
                cfg_rec.crlb_diags.append(np.diag(crlb) if crlb is not None else np.full(3, np.nan))

                H_xy = H[:, :2]
                F_xy = self.fim_xy(H_xy, sigma_r)
                r_xy = self.rank_of_H(H_xy)
                ld_xy = self._logdet_F(F_xy)
                g_xy = self.gdop_from_F(F_xy)
                crlb_xy = self.crlb_std_from_F(F_xy)
                cfg_rec.ranks_xy.append(r_xy)
                cfg_rec.logdets_xy.append(ld_xy)
                cfg_rec.gdops_xy.append(g_xy)
                cfg_rec.invertible_xy.append(np.isfinite(g_xy) and g_xy != math.inf)
                if crlb_xy is None or crlb_xy.size < 2:
                    cfg_rec.crlb_diags_xy.append(np.full(2, np.nan))
                else:
                    cfg_rec.crlb_diags_xy.append(crlb_xy[:2])

            # Select best subsets
            def choose_best(cfgs: Iterable[Tuple[int, ...]]) -> Tuple[Optional[Tuple[int, ...]], float, float, int]:
                best_cfg = None
                best_ld = -math.inf
                best_g = math.inf
                best_rank = 0
                for cfg in cfgs:
                    rec = cfg_records[cfg]
                    ld = rec.logdets[-1]
                    g = rec.gdops[-1]
                    r = rec.ranks[-1]
                    if mode == "logdet":
                        if ld > best_ld:
                            best_ld = ld
                            best_g = g
                            best_cfg = cfg
                            best_rank = r
                    else:  # gdop mode
                        if g < best_g:
                            best_g = g
                            best_ld = ld
                            best_cfg = cfg
                            best_rank = r
                return best_cfg, best_g, best_ld, best_rank

            def choose_best_xy(cfgs: Iterable[Tuple[int, ...]]) -> Tuple[Optional[Tuple[int, ...]], float, float, int]:
                best_cfg = None
                best_ld = -math.inf
                best_g = math.inf
                best_rank = 0
                for cfg in cfgs:
                    rec = cfg_records[cfg]
                    ld = rec.logdets_xy[-1]
                    g = rec.gdops_xy[-1]
                    r = rec.ranks_xy[-1]
                    if r < 2:
                        continue
                    if mode == "logdet":
                        if ld > best_ld:
                            best_ld = ld
                            best_g = g
                            best_cfg = cfg
                            best_rank = r
                    else:
                        if g < best_g:
                            best_g = g
                            best_ld = ld
                            best_cfg = cfg
                            best_rank = r
                return best_cfg, best_g, best_ld, best_rank

            cfg_2 = [c for c in configs if len(c) == 2]
            cfg_3 = [c for c in configs if len(c) == 3]

            b2, g2, ld2, r2 = choose_best(cfg_2) if cfg_2 else (None, math.inf, -math.inf, 0)
            b3, g3, ld3, r3 = choose_best(cfg_3) if cfg_3 else (None, math.inf, -math.inf, 0)

            b2_xy, g2_xy, ld2_xy, r2_xy = choose_best_xy(cfg_2) if cfg_2 else (None, math.inf, -math.inf, 0)
            b3_xy, g3_xy, ld3_xy, r3_xy = choose_best_xy(cfg_3) if cfg_3 else (None, math.inf, -math.inf, 0)

            best2_idx.append(b2)
            best2_gdop.append(g2)
            best2_logdet.append(ld2)
            best2_rank.append(r2)

            best2_idx_xy.append(b2_xy)
            best2_gdop_xy.append(g2_xy)
            best2_logdet_xy.append(ld2_xy)
            best2_rank_xy.append(r2_xy)

            best3_idx.append(b3)
            best3_gdop.append(g3)
            best3_logdet.append(ld3)
            best3_rank.append(r3)

            best3_idx_xy.append(b3_xy)
            best3_gdop_xy.append(g3_xy)
            best3_logdet_xy.append(ld3_xy)
            best3_rank_xy.append(r3_xy)

        result: Dict[str, any] = {
            "t_idx": np.asarray(t_idx),
            "all_beacons": {
                "rank": np.asarray(all_rank),
                "logdet": np.asarray(all_logdet),
                "gdop": np.asarray(all_gdop),
                "crlb_diag": np.vstack(all_crlb_diag),
            },
            "all_beacons_xy": {
                "rank": np.asarray(all_rank_xy),
                "logdet": np.asarray(all_logdet_xy),
                "gdop": np.asarray(all_gdop_xy),
                "crlb_diag": np.vstack(all_crlb_diag_xy),
            },
            "best_2": {
                "idx": np.asarray(best2_idx, dtype=object),
                "gdop": np.asarray(best2_gdop),
                "logdet": np.asarray(best2_logdet),
                "rank": np.asarray(best2_rank),
            },
            "best_2_xy": {
                "idx": np.asarray(best2_idx_xy, dtype=object),
                "gdop": np.asarray(best2_gdop_xy),
                "logdet": np.asarray(best2_logdet_xy),
                "rank": np.asarray(best2_rank_xy),
            },
            "best_3": {
                "idx": np.asarray(best3_idx, dtype=object),
                "gdop": np.asarray(best3_gdop),
                "logdet": np.asarray(best3_logdet),
                "rank": np.asarray(best3_rank),
            },
            "best_3_xy": {
                "idx": np.asarray(best3_idx_xy, dtype=object),
                "gdop": np.asarray(best3_gdop_xy),
                "logdet": np.asarray(best3_logdet_xy),
                "rank": np.asarray(best3_rank_xy),
            },
            "config_records": cfg_records,
            "mode": mode,
            "sigma_r": sigma_r,
            "decimate": decimate,
        }

        if all_configs_metrics:
            # Convert lists to numpy for convenience
            for rec in cfg_records.values():
                rec.ranks = np.asarray(rec.ranks)
                rec.gdops = np.asarray(rec.gdops)
                rec.logdets = np.asarray(rec.logdets)
                rec.invertible = np.asarray(rec.invertible, dtype=bool)
                rec.crlb_diags = np.vstack(rec.crlb_diags)
                rec.ranks_xy = np.asarray(rec.ranks_xy)
                rec.gdops_xy = np.asarray(rec.gdops_xy)
                rec.logdets_xy = np.asarray(rec.logdets_xy)
                rec.invertible_xy = np.asarray(rec.invertible_xy, dtype=bool)
                rec.crlb_diags_xy = np.vstack(rec.crlb_diags_xy)
        return result


def _self_test():
    """Minimal self-test to sanity-check rank/GDOP behavior."""
    analyzer = SBLConfigurationAnalyzer()
    # Collinear beacons (poor geometry)
    beacons_line = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ])
    traj = np.array([[5.0, 5.0, 0.0], [10.0, 5.0, 0.0]])
    res = analyzer.analyze_trajectory(traj, beacons_line, sigma_r=1.0, max_r=3)
    assert np.all(res["all_beacons"]["rank"] < 3), "Collinear case should be rank-deficient"
    assert np.isinf(res["all_beacons"]["gdop"][0]), "GDOP should be infinite when F is singular"

    # Good 3D geometry (tetrahedron)
    beacons_good = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    traj2 = np.array([[2.0, 2.0, 2.0]])
    res2 = analyzer.analyze_trajectory(traj2, beacons_good, sigma_r=1.0, max_r=4)
    assert res2["all_beacons"]["rank"][0] == 3, "Tetrahedron should be full rank for position"
    assert np.isfinite(res2["all_beacons"]["gdop"][0]), "GDOP should be finite for good geometry"
    print("Self-test passed.")


if __name__ == "__main__":
    _self_test()
