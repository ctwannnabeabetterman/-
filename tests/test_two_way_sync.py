"""阶段 4：基于两次波形测量的双向时间传递测试。"""

from __future__ import annotations

import unittest

import numpy as np

from clock_model import LocalClock
from config import ChannelConfig, TwoWayConfig, WaveformConfig
from lut_calibration import build_qls_lut
from two_way_sync import estimate_two_way, simulate_two_way_exchange


class TwoWayTimeTransferTests(unittest.TestCase):
    """验证四时间戳公式、符号和对称链路假设。"""

    def setUp(self) -> None:
        self.waveform_config = WaveformConfig(pulse_duration_s=1e-6)
        self.calibration = build_qls_lut(self.waveform_config, grid_points=401)

    def _exchange(
        self,
        processing_delay_s: float,
        up_delay_s: float = 50.25e-9,
        down_delay_s: float = 50.25e-9,
    ):
        ap0 = LocalClock(offset_s=0.0)
        ap1 = LocalClock(offset_s=100e-9)
        configuration = TwoWayConfig(
            tx1_local_time_s=1e-3,
            processing_delay_s=processing_delay_s,
            coarse_up_delay_s=50e-9,
            coarse_down_delay_s=50e-9,
            gate_half_width_samples=2.5,
        )
        observation = simulate_two_way_exchange(
            waveform_config=self.waveform_config,
            two_way_config=configuration,
            ap0_clock=ap0,
            ap1_clock=ap1,
            up_link=ChannelConfig(propagation_delay_s=up_delay_s, snr_db=None),
            down_link=ChannelConfig(propagation_delay_s=down_delay_s, snr_db=None),
            calibration=self.calibration,
            rng=np.random.default_rng(41),
        )
        return observation, estimate_two_way(observation)

    def test_symmetric_exchange_recovers_clock_correction_and_propagation_delay(self) -> None:
        observation, estimate = self._exchange(processing_delay_s=20e-6)

        self.assertLess(abs(estimate.clock_correction_s + 100e-9), 5e-12)
        self.assertLess(abs(estimate.ap1_offset_estimate_s - 100e-9), 5e-12)
        self.assertLess(abs(estimate.symmetric_propagation_delay_s - 50.25e-9), 5e-12)
        self.assertTrue(observation.up_measurement.raw_estimate.qls_valid)
        self.assertTrue(observation.down_measurement.raw_estimate.qls_valid)

    def test_processing_delay_cancels_from_estimate(self) -> None:
        estimates = [
            self._exchange(processing_delay_s=value)[1]
            for value in (1e-6, 10e-6, 100e-6, 1e-3)
        ]

        corrections = np.array([value.clock_correction_s for value in estimates])
        propagation = np.array([value.symmetric_propagation_delay_s for value in estimates])
        self.assertLess(float(np.ptp(corrections)), 5e-12)
        self.assertLess(float(np.ptp(propagation)), 5e-12)

    def test_asymmetric_link_appears_as_half_delay_difference_in_clock_result(self) -> None:
        up_delay_s = 50.25e-9
        down_delay_s = 54.25e-9
        _, estimate = self._exchange(
            processing_delay_s=20e-6,
            up_delay_s=up_delay_s,
            down_delay_s=down_delay_s,
        )
        expected_correction_s = -100e-9 + (up_delay_s - down_delay_s) / 2.0

        self.assertLess(abs(estimate.clock_correction_s - expected_correction_s), 5e-12)
        self.assertLess(
            abs(
                estimate.symmetric_propagation_delay_s
                - (up_delay_s + down_delay_s) / 2.0
            ),
            5e-12,
        )

    def test_only_estimated_correction_is_applied_to_secondary_clock(self) -> None:
        ap0 = LocalClock(offset_s=0.0)
        ap1 = LocalClock(offset_s=125e-9)
        configuration = TwoWayConfig(
            tx1_local_time_s=1e-3,
            processing_delay_s=10e-6,
            coarse_up_delay_s=50e-9,
            coarse_down_delay_s=50e-9,
        )
        observation = simulate_two_way_exchange(
            waveform_config=self.waveform_config,
            two_way_config=configuration,
            ap0_clock=ap0,
            ap1_clock=ap1,
            up_link=ChannelConfig(propagation_delay_s=50.25e-9, snr_db=None),
            down_link=ChannelConfig(propagation_delay_s=50.25e-9, snr_db=None),
            calibration=self.calibration,
            rng=np.random.default_rng(42),
        )
        estimate = estimate_two_way(observation)

        ap1.apply_time_correction(estimate.clock_correction_s)

        self.assertEqual(ap1.time_correction_s, estimate.clock_correction_s)
        self.assertLess(abs(ap1.read_time(2e-3) - ap0.read_time(2e-3)), 5e-12)


if __name__ == "__main__":
    unittest.main()
