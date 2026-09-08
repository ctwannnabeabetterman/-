"""阶段 5：多轮时钟漂移和随机游走跟踪测试。"""

from __future__ import annotations

import unittest

import numpy as np

from clock_model import LocalClock
from config import ChannelConfig, ClockTrackingConfig, TwoWayConfig, WaveformConfig
from experiments import run_clock_tracking
from lut_calibration import build_qls_lut


class ClockTrackingTests(unittest.TestCase):
    """验证周期双向同步对偏差、频率漂移和随机游走的跟踪。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.waveform = WaveformConfig(pulse_duration_s=1e-6)
        cls.calibration = build_qls_lut(cls.waveform, grid_points=401)
        cls.link = ChannelConfig(propagation_delay_s=50.25e-9, snr_db=None)
        cls.exchange = TwoWayConfig(
            tx1_local_time_s=1e-3,
            processing_delay_s=10e-6,
            coarse_up_delay_s=50e-9,
            coarse_down_delay_s=50e-9,
        )

    def _run(
        self,
        interval_s: float,
        frequency_offset: float = 0.0,
        random_walk: float = 0.0,
        seed: int = 51,
    ):
        return run_clock_tracking(
            waveform_config=self.waveform,
            two_way_config=self.exchange,
            tracking_config=ClockTrackingConfig(
                rounds=12,
                sync_interval_s=interval_s,
                correction_gain=1.0,
                random_walk_std_s_per_sqrt_s=random_walk,
            ),
            ap0_clock=LocalClock(),
            ap1_clock=LocalClock(
                offset_s=100e-9,
                fractional_frequency_offset=frequency_offset,
            ),
            up_link=self.link,
            down_link=self.link,
            calibration=self.calibration,
            rng=np.random.default_rng(seed),
        )

    def test_static_offset_remains_corrected_after_first_round(self) -> None:
        result = self._run(interval_s=50e-3)

        self.assertLess(float(np.max(np.abs(result.residual_after_s))), 5e-12)
        self.assertLess(float(np.max(np.abs(result.applied_correction_s[1:]))), 5e-12)

    def test_periodic_sync_tracks_constant_clock_rate_error(self) -> None:
        result = self._run(interval_s=50e-3, frequency_offset=1e-8)

        self.assertLess(float(np.max(np.abs(result.residual_after_s))), 20e-12)
        expected_drift_per_round_s = 1e-8 * 50e-3
        self.assertLess(
            abs(float(np.median(-result.applied_correction_s[2:])) - expected_drift_per_round_s),
            20e-12,
        )

    def test_random_walk_is_reproducible_for_fixed_seed(self) -> None:
        first = self._run(interval_s=50e-3, random_walk=50e-12, seed=52)
        second = self._run(interval_s=50e-3, random_walk=50e-12, seed=52)

        np.testing.assert_array_equal(first.random_walk_accumulated_s, second.random_walk_accumulated_s)
        np.testing.assert_array_equal(first.residual_after_s, second.residual_after_s)

    def test_shorter_sync_interval_reduces_pre_update_drift(self) -> None:
        fast = self._run(interval_s=10e-3, frequency_offset=2e-8)
        slow = self._run(interval_s=100e-3, frequency_offset=2e-8)
        fast_rms = float(np.sqrt(np.mean(fast.residual_before_s[2:] ** 2)))
        slow_rms = float(np.sqrt(np.mean(slow.residual_before_s[2:] ** 2)))

        self.assertLess(fast_rms, 0.2 * slow_rms)


if __name__ == "__main__":
    unittest.main()
