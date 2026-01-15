import numpy as np
import unittest

from adaptive_modem_manager_v2 import AdaptiveModemManagerV2


class AdaptiveModemManagerV2Tests(unittest.TestCase):
    def setUp(self):
        self.beacons = {
            "b1": np.array([0.0, 0.0, 0.0]),
            "b2": np.array([10.0, 0.0, 0.0]),
            "b3": np.array([0.0, 10.0, 0.0]),
            "b4": np.array([0.0, 0.0, -10.0]),
        }
        self.manager = AdaptiveModemManagerV2(
            beacon_positions=self.beacons,
            sigma_r=0.5,
            size_penalty=0.0,
            rank_deficit_penalty=5.0,
            min_dwell_steps=2,
            switch_margin=1e-3,
            enable_uncertainty_gate=True,
            gate_on_threshold=2.0,
            gate_off_threshold=1.0,
            min_subset_size=1,
            max_subset_size=3,
            prefer_smaller=True,
        )
        self.P_good = np.diag([0.4, 0.4, 0.4])
        self.P_bad = np.diag([5.0, 5.0, 5.0])

    def test_selects_nonempty_when_enabled(self):
        ids, metrics = self.manager.select(
            t=0.0,
            a_pos=np.array([1.0, 1.0, -1.0]),
            P=self.P_bad,
            available_ids=list(self.beacons.keys()),
            depth_available=True,
        )
        self.assertGreaterEqual(len(ids), 1)
        self.assertTrue(metrics.acoustics_enabled)

    def test_uncertainty_gate_turns_off(self):
        # First enable, then reduce uncertainty so gate turns off
        ids, _ = self.manager.select(
            t=0.0,
            a_pos=np.zeros(3),
            P=self.P_bad,
            available_ids=list(self.beacons.keys()),
            depth_available=True,
        )
        self.assertTrue(len(ids) > 0)

        ids2, metrics2 = self.manager.select(
            t=1.0,
            a_pos=np.zeros(3),
            P=self.P_good,
            available_ids=list(self.beacons.keys()),
            depth_available=True,
        )
        self.assertEqual(len(ids2), 0)
        self.assertFalse(metrics2.acoustics_enabled)
        self.assertEqual(metrics2.reason, "uncertainty_gated_off")

    def test_dwell_holds_selection(self):
        # force a selection
        ids, metrics = self.manager.select(
            t=0.0,
            a_pos=np.array([1.0, 1.0, -1.0]),
            P=self.P_bad,
            available_ids=list(self.beacons.keys()),
            depth_available=True,
        )
        # Immediately request again with same availability; should hold due to dwell
        ids2, metrics2 = self.manager.select(
            t=0.1,
            a_pos=np.array([2.0, 2.0, -1.0]),
            P=self.P_bad,
            available_ids=list(self.beacons.keys()),
            depth_available=True,
        )
        self.assertEqual(ids2, ids)
        self.assertEqual(metrics2.reason, "dwell_hold")


if __name__ == "__main__":
    unittest.main()
